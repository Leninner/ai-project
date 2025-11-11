import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
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
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "nn_label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.6

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
        embeddings = self.fc1(embeddings)
        embeddings = self.bn1(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout1(embeddings)
        embeddings = self.fc2(embeddings)
        embeddings = self.bn2(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout2(embeddings)
        logits = self.fc3(embeddings)
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
        _resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return _resnet

def extract_embedding(image_path):
    mtcnn = get_mtcnn()
    resnet = get_resnet()
    
    image = Image.open(image_path).convert('RGB')
    cropped_face_tensor = mtcnn(image)
    
    if cropped_face_tensor is None:
        raise ValueError(f"No faces found in image {image_path}")
    
    cropped_face_tensor = cropped_face_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(cropped_face_tensor)
    
    return embedding.squeeze().numpy()

def extract_embeddings(data_dir):
    embeddings_list = []
    labels_list = []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(person_dir, image_name)
            try:
                embedding = extract_embedding(image_path)
                embeddings_list.append(embedding)
                labels_list.append(person_name)
            except Exception:
                continue
    return np.array(embeddings_list), np.array(labels_list)

def train_nn_classifier(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train)
    num_classes = len(label_encoder.classes_)
    
    embedding_dim = X_train.shape[1]
    classifier_model = FaceClassifier(embedding_dim=embedding_dim, num_classes=num_classes)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)
    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    train_dataset = EmbeddingDataset(X_train, encoded_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_model = classifier_model.to(device)
    
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
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")
    
    classifier_model.eval()
    torch.save(classifier_model.state_dict(), NN_MODEL_PATH)
    
    import joblib
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    return classifier_model, label_encoder

def load_nn_classifier():
    global _nn_classifier, _label_encoder
    
    if _nn_classifier is None or _label_encoder is None:
        if NN_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            import joblib
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
            num_classes = len(_label_encoder.classes_)
            
            embedding_dim = 512
            _nn_classifier = FaceClassifier(embedding_dim=embedding_dim, num_classes=num_classes)
            _nn_classifier.load_state_dict(torch.load(NN_MODEL_PATH, map_location='cpu'))
            _nn_classifier.eval()
        else:
            raise ValueError("Neural network classifier not trained. Please train the model first.")
    
    return _nn_classifier, _label_encoder

def identify_face(query_embedding):
    classifier_model, label_encoder = load_nn_classifier()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

if __name__ == "__main__":
    print("Extrayendo embeddings faciales...")
    embeddings, labels = extract_embeddings(DATA_DIR)

    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

    print("Entrenando clasificador de red neuronal...")
    trained_classifier, label_encoder = train_nn_classifier(embeddings_train, labels_train, epochs=100, batch_size=32)
    print(f"Modelo de red neuronal entrenado y guardado en {NN_MODEL_PATH}")

    print("Identificando personas en conjunto de prueba...")
    predictions = []
    confidence_scores = []

    for test_embedding in embeddings_test:
        identified_person, confidence = identify_face(test_embedding)
        predictions.append(identified_person)
        confidence_scores.append(confidence)

    predictions = np.array(predictions)
    
    unknown_mask = predictions == "unknown"
    known_mask = ~unknown_mask
    
    print(f"\n--- Resultados de clasificación NN (Umbral: {CONFIDENCE_THRESHOLD}) ---")
    print(f"Personas conocidas identificadas: {np.sum(known_mask)}/{len(labels_test)}")
    print(f"Personas etiquetadas como desconocidas: {np.sum(unknown_mask)}/{len(labels_test)}")
    
    if np.sum(known_mask) > 0:
        labels_test_known = labels_test[known_mask]
        predictions_known = predictions[known_mask]
        
        accuracy_known = accuracy_score(labels_test_known, predictions_known)
        print(f"\n--- Precisión en personas conocidas: {accuracy_known:.4f} ---")
        
        unique_persons = np.unique(np.concatenate([labels_test_known, predictions_known]))
        print("\n--- Reporte de clasificación (personas conocidas) ---")
        print(classification_report(labels_test_known, predictions_known, labels=unique_persons, target_names=unique_persons))

    confidence_scores = np.array(confidence_scores)
    print(f"\n--- Estadísticas de confianza ---")
    print(f"Confianza promedio: {np.mean(confidence_scores):.4f}")
    print(f"Confianza mínima: {np.min(confidence_scores):.4f}")
    print(f"Confianza máxima: {np.max(confidence_scores):.4f}")
    print(f"Confianza promedio (conocidas): {np.mean(confidence_scores[known_mask]):.4f}" if np.sum(known_mask) > 0 else "")
    print(f"Confianza promedio (desconocidas): {np.mean(confidence_scores[unknown_mask]):.4f}" if np.sum(unknown_mask) > 0 else "")

    all_labels = np.unique(np.concatenate([labels_test, predictions]))
    confusion_matrix_data = confusion_matrix(labels_test, predictions, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix_data, interpolation='nearest', cmap='Greens')
    plt.title(f"Matriz de confusión - FaceNet + Neural Network (Umbral: {CONFIDENCE_THRESHOLD})")
    plt.colorbar()
    tick_marks = np.arange(len(all_labels))
    plt.xticks(tick_marks, all_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, all_labels)
    plt.ylabel("Persona real")
    plt.xlabel("Persona identificada")
    threshold_value = confusion_matrix_data.max() / 2.
    for row_idx in range(confusion_matrix_data.shape[0]):
        for col_idx in range(confusion_matrix_data.shape[1]):
            plt.text(col_idx, row_idx, format(confusion_matrix_data[row_idx, col_idx], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix_data[row_idx, col_idx] > threshold_value else "black")
    plt.tight_layout()
    
    confusion_matrix_path = Path(__file__).parent.parent / "confusion_matrix_nn.png"
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    print(f"\nMatriz de confusión guardada en: {confusion_matrix_path}")
    plt.close()

