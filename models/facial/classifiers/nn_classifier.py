import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
NN_MODEL_PATH = CHECKPOINT_PATH / "nn_classifier.pth"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "nn_label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.75

_mtcnn = None
_resnet = None
_nn_classifier = None
_label_encoder = None


class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=None):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes if num_classes else 1)
        self.relu = nn.ReLU()

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        use_eval_mode = batch_size == 1 and self.training

        if use_eval_mode:
            self.bn1.eval()
            self.bn2.eval()

        embeddings = self.fc1(embeddings)
        embeddings = self.bn1(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout1(embeddings)
        embeddings = self.fc2(embeddings)
        embeddings = self.bn2(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout2(embeddings)
        logits = self.fc3(embeddings)

        if use_eval_mode:
            self.bn1.train()
            self.bn2.train()

        return logits


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
    return _mtcnn


def get_resnet():
    global _resnet
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return _resnet


def extract_embedding(image_path):
    resnet = get_resnet()

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    if image_array.shape != (160, 160, 3):
        mtcnn = get_mtcnn()
        cropped_face_tensor = mtcnn(image)
        if cropped_face_tensor is None:
            raise ValueError(f"No faces found in image {image_path}")
        face_tensor = cropped_face_tensor
    else:
        image_array_normalized = image_array.astype(np.float32) / 255.0
        face_tensor = torch.from_numpy(image_array_normalized).permute(2, 0, 1)

    face_tensor = face_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(face_tensor)

    return embedding.squeeze().numpy()


def extract_embeddings(data_dir):
    embeddings_list = []
    labels_list = []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(person_dir, image_name)
            try:
                embedding = extract_embedding(image_path)
                embeddings_list.append(embedding)
                labels_list.append(person_name)
            except Exception:
                continue
    return np.array(embeddings_list), np.array(labels_list)


def train_nn_classifier(
    X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001
):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train)
    num_classes = len(label_encoder.classes_)

    embedding_dim = X_train.shape[1]
    classifier_model = FaceClassifier(
        embedding_dim=embedding_dim, num_classes=num_classes
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)
    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    train_dataset = EmbeddingDataset(X_train, encoded_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model = classifier_model.to(device)

    print(f"Entrenando clasificador Neural Network en {device}...")
    print(f"Clases: {num_classes}, Épocas: {epochs}, Batch size: {batch_size}")

    classifier_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = classifier_model(batch_embeddings)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        learning_rate_scheduler.step(average_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, LR: {current_lr:.6f}"
            )

    classifier_model.eval()
    torch.save(classifier_model.state_dict(), NN_MODEL_PATH)

    import pickle

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Modelo Neural Network entrenado y guardado en {NN_MODEL_PATH}")

    return classifier_model, label_encoder


def load_nn_classifier():
    global _nn_classifier, _label_encoder

    if _nn_classifier is None or _label_encoder is None:
        if NN_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            import pickle

            with open(LABEL_ENCODER_PATH, "rb") as f:
                _label_encoder = pickle.load(f)
            num_classes = len(_label_encoder.classes_)

            embedding_dim = 512
            _nn_classifier = FaceClassifier(
                embedding_dim=embedding_dim, num_classes=num_classes
            )
            _nn_classifier.load_state_dict(
                torch.load(NN_MODEL_PATH, map_location="cpu")
            )
            _nn_classifier.eval()
        else:
            raise ValueError(
                "Neural network classifier not trained. Please train the model first."
            )

    return _nn_classifier, _label_encoder


def identify_face(query_embedding):
    classifier_model, label_encoder = load_nn_classifier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model = classifier_model.to(device)

    query_embedding_tensor = torch.FloatTensor(query_embedding).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier_model(query_embedding_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence_score, predicted_index = torch.max(probabilities, 1)

    confidence_value = confidence_score.item()
    predicted_class_index = predicted_index.item()

    if confidence_value < CONFIDENCE_THRESHOLD:
        return "unknown", float(confidence_value)

    identified_person = label_encoder.inverse_transform([predicted_class_index])[0]
    return identified_person, float(confidence_value)
