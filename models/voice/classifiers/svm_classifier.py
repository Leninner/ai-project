import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchaudio
import soundfile as sf

if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['soundfile']
    torchaudio.list_audio_backends = list_audio_backends

from speechbrain.inference import EncoderClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
SVM_MODEL_PATH = CHECKPOINT_PATH / "svm_classifier.joblib"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.3
MODEL = "speechbrain/spkrec-ecapa-voxceleb"

_encoder = None
_svm_classifier = None
_label_encoder = None

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

def train_svm_classifier(embeddings_train, labels_train):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_train)
    num_classes = len(label_encoder.classes_)
    
    print(f"Entrenando clasificador SVM...")
    print(f"Clases: {num_classes}, Muestras: {len(embeddings_train)}")
    
    svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
    svm_classifier.fit(embeddings_train, encoded_labels)
    
    joblib.dump(svm_classifier, SVM_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    print(f"Modelo SVM entrenado y guardado en {SVM_MODEL_PATH}")
    
    return svm_classifier, label_encoder

def load_svm_classifier():
    global _svm_classifier, _label_encoder
    
    if _svm_classifier is None or _label_encoder is None:
        if SVM_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            _svm_classifier = joblib.load(SVM_MODEL_PATH)
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        else:
            raise ValueError("SVM classifier not trained. Please train the model first.")
    
    return _svm_classifier, _label_encoder

def identify_speaker(query_embedding):
    svm_classifier, label_encoder = load_svm_classifier()
    
    query_embedding_reshaped = query_embedding.reshape(1, -1)
    probabilities = svm_classifier.predict_proba(query_embedding_reshaped)[0]
    predicted_class = svm_classifier.predict(query_embedding_reshaped)[0]
    
    confidence = np.max(probabilities)
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", float(confidence)
    
    identified_speaker = label_encoder.inverse_transform([predicted_class])[0]
    return identified_speaker, float(confidence)

if __name__ == "__main__":
    get_encoder()
    embeddings, labels = extract_embeddings(DATA_DIR)
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
    svm_classifier, label_encoder = train_svm_classifier(embeddings_train, labels_train)
    predictions = []
    confidence_scores = []

    for test_embedding in embeddings_test:
        identified_speaker, confidence = identify_speaker(test_embedding)
        predictions.append(identified_speaker)
        confidence_scores.append(confidence)

    predictions = np.array(predictions)
    unknown_mask = predictions == "unknown"
    known_mask = ~unknown_mask
    all_labels = np.unique(np.concatenate([labels_test, predictions]))
    confusion_matrix_data = confusion_matrix(labels_test, predictions, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix_data, interpolation='nearest', cmap='Blues')
    plt.title(f"Matriz de confusión - SpeechBrain + SVM (Umbral: {CONFIDENCE_THRESHOLD})")
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
    confusion_matrix_path = Path(__file__).parent.parent / "confusion_matrix_svm_voice.png"
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()

