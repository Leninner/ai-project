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
CONFIDENCE_THRESHOLD = 0.3
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
    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    train_dataset = EmbeddingDataset(embeddings_train, encoded_labels)
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

if __name__ == "__main__":
    print("Cargando modelo de SpeechBrain...")
    get_encoder()
    
    print("Extrayendo embeddings de voz...")
    embeddings, labels = extract_embeddings(DATA_DIR)

    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

    print("Entrenando clasificador de red neuronal...")
    trained_classifier, label_encoder = train_nn_classifier(embeddings_train, labels_train, epochs=100, batch_size=32)
    print(f"Modelo de red neuronal entrenado y guardado en {NN_MODEL_PATH}")

    print("Identificando hablantes en conjunto de prueba...")
    predictions = []
    confidence_scores = []

    for test_embedding in embeddings_test:
        identified_speaker, confidence = identify_speaker(test_embedding)
        predictions.append(identified_speaker)
        confidence_scores.append(confidence)

    predictions = np.array(predictions)
    
    unknown_mask = predictions == "unknown"
    known_mask = ~unknown_mask
    
    print(f"\n--- Resultados de clasificación NN (Umbral: {CONFIDENCE_THRESHOLD}) ---")
    print(f"Usuarios conocidos identificados: {np.sum(known_mask)}/{len(labels_test)}")
    print(f"Usuarios etiquetados como desconocidos: {np.sum(unknown_mask)}/{len(labels_test)}")
    
    if np.sum(known_mask) > 0:
        labels_test_known = labels_test[known_mask]
        predictions_known = predictions[known_mask]
        
        accuracy_known = accuracy_score(labels_test_known, predictions_known)
        print(f"\n--- Precisión en usuarios conocidos: {accuracy_known:.4f} ---")
        
        unique_speakers = np.unique(np.concatenate([labels_test_known, predictions_known]))
        print("\n--- Reporte de clasificación (usuarios conocidos) ---")
        print(classification_report(labels_test_known, predictions_known, labels=unique_speakers, target_names=unique_speakers))

    confidence_scores = np.array(confidence_scores)
    print(f"\n--- Estadísticas de confianza ---")
    print(f"Confianza promedio: {np.mean(confidence_scores):.4f}")
    print(f"Confianza mínima: {np.min(confidence_scores):.4f}")
    print(f"Confianza máxima: {np.max(confidence_scores):.4f}")
    print(f"Confianza promedio (conocidos): {np.mean(confidence_scores[known_mask]):.4f}" if np.sum(known_mask) > 0 else "")
    print(f"Confianza promedio (desconocidos): {np.mean(confidence_scores[unknown_mask]):.4f}" if np.sum(unknown_mask) > 0 else "")

    all_labels = np.unique(np.concatenate([labels_test, predictions]))
    confusion_matrix_data = confusion_matrix(labels_test, predictions, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix_data, interpolation='nearest', cmap='Greens')
    plt.title(f"Matriz de confusión - SpeechBrain + Neural Network (Umbral: {CONFIDENCE_THRESHOLD})")
    plt.colorbar()
    tick_marks = np.arange(len(all_labels))
    plt.xticks(tick_marks, all_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, all_labels)
    plt.ylabel("Hablante real")
    plt.xlabel("Hablante identificado")
    threshold_value = confusion_matrix_data.max() / 2.
    for row_idx in range(confusion_matrix_data.shape[0]):
        for col_idx in range(confusion_matrix_data.shape[1]):
            plt.text(col_idx, row_idx, format(confusion_matrix_data[row_idx, col_idx], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix_data[row_idx, col_idx] > threshold_value else "black")
    plt.tight_layout()
    
    confusion_matrix_path = Path(__file__).parent.parent / "confusion_matrix_nn_voice.png"
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    print(f"\nMatriz de confusión guardada en: {confusion_matrix_path}")
    plt.close()

