import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['soundfile']
    torchaudio.list_audio_backends = list_audio_backends

from speechbrain.inference import EncoderClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
NN_MODEL_PATH = CHECKPOINT_PATH / "nn_classifier.pth"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "nn_label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.75
MODEL = "speechbrain/spkrec-ecapa-voxceleb"

_encoder = None
_nn_classifier = None
_label_encoder = None

class VoiceClassifier(nn.Module):
    def __init__(self, embedding_dim=192, num_classes=None):
        super(VoiceClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes if num_classes else 1)
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

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = EncoderClassifier.from_hparams(source=MODEL)
    return _encoder

def extract_embedding(audio_path):
    encoder = get_encoder()
    audio_data, sample_rate = sf.read(audio_path)
    signal = torch.from_numpy(audio_data).float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    emb = encoder.encode_batch(signal)
    return emb.squeeze().detach().numpy()

def extract_embeddings(data_dir):
    embeddings_list = []
    labels_list = []
    for speaker in os.listdir(data_dir):
        speaker_dir = os.path.join(data_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for file in os.listdir(speaker_dir):
            if not file.endswith(".wav"):
                continue
            audio_path = os.path.join(speaker_dir, file)
            try:
                embedding = extract_embedding(audio_path)
                embeddings_list.append(embedding)
                labels_list.append(speaker)
            except Exception:
                continue
    return np.array(embeddings_list), np.array(labels_list)

def train_nn_classifier(embeddings_train, labels_train, epochs=100, batch_size=32, learning_rate=0.001):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_train)
    num_classes = len(label_encoder.classes_)
    
    embedding_dim = embeddings_train.shape[1]
    classifier_model = VoiceClassifier(embedding_dim=embedding_dim, num_classes=num_classes)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)
    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
    
    train_dataset = EmbeddingDataset(embeddings_train, encoded_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}, LR: {current_lr:.6f}")
    
    classifier_model.eval()
    torch.save(classifier_model.state_dict(), NN_MODEL_PATH)
    
    import joblib
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    print(f"Modelo Neural Network entrenado y guardado en {NN_MODEL_PATH}")
    
    return classifier_model, label_encoder

def load_nn_classifier():
    global _nn_classifier, _label_encoder
    
    if _nn_classifier is None or _label_encoder is None:
        if NN_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            import joblib
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
            num_classes = len(_label_encoder.classes_)
            
            embedding_dim = 192
            _nn_classifier = VoiceClassifier(embedding_dim=embedding_dim, num_classes=num_classes)
            _nn_classifier.load_state_dict(torch.load(NN_MODEL_PATH, map_location='cpu'))
            _nn_classifier.eval()
        else:
            raise ValueError("Neural network classifier not trained. Please train the model first.")
    
    return _nn_classifier, _label_encoder

def identify_speaker(query_embedding):
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
    
    identified_speaker = label_encoder.inverse_transform([predicted_class_index])[0]
    return identified_speaker, float(confidence_value)
