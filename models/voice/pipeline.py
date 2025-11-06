import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['soundfile']
    torchaudio.list_audio_backends = list_audio_backends

from speechbrain.inference import EncoderClassifier

DATA_DIR = "data"
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints"
SVM_MODEL_PATH = CHECKPOINT_PATH / "svm_classifier.joblib"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.3

_encoder = None
_svm_classifier = None
_label_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = EncoderClassifier.from_hparams(source=MODEL)
    return _encoder

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_embedding(audio_path):
    encoder = get_encoder()
    audio_data, sample_rate = sf.read(audio_path)
    signal = torch.from_numpy(audio_data).float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    emb = encoder.encode_batch(signal)
    return emb.squeeze().detach().numpy()

def extract_embeddings(data_dir):
    X, y = [], []
    for speaker in os.listdir(data_dir):
        spk_dir = os.path.join(data_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue
        for file in os.listdir(spk_dir):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(spk_dir, file)
            emb = extract_embedding(path)
            X.append(emb)
            y.append(speaker)
    return np.array(X), np.array(y)

def build_reference_embeddings(X_train, y_train):
    speaker_embeddings = defaultdict(list)
    for emb, speaker in zip(X_train, y_train):
        speaker_embeddings[speaker].append(emb)
    
    reference_embeddings = {}
    for speaker, embeddings in speaker_embeddings.items():
        reference_embeddings[speaker] = np.mean(embeddings, axis=0)
    
    return reference_embeddings

def train_svm_classifier(X_train, y_train):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
    svm_classifier.fit(X_train, y_encoded)
    
    joblib.dump(svm_classifier, SVM_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
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

def identify_speaker(query_embedding, reference_embeddings=None):
    try:
        svm_classifier, label_encoder = load_svm_classifier()
    except ValueError:
        if reference_embeddings is None:
            return "unknown", 0.0
        
        best_similarity = -1.0
        identified_speaker = None
        
        for speaker, ref_emb in reference_embeddings.items():
            similarity = cosine_similarity(query_embedding, ref_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                identified_speaker = speaker
        
        if best_similarity < CONFIDENCE_THRESHOLD:
            return "unknown", best_similarity
        
        return identified_speaker, best_similarity
    
    query_embedding_reshaped = query_embedding.reshape(1, -1)
    probabilities = svm_classifier.predict_proba(query_embedding_reshaped)[0]
    predicted_class = svm_classifier.predict(query_embedding_reshaped)[0]
    
    confidence = np.max(probabilities)
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", float(confidence)
    
    identified_speaker = label_encoder.inverse_transform([predicted_class])[0]
    return identified_speaker, float(confidence)

def identify_speaker_legacy(query_embedding, reference_embeddings):
    best_similarity = -1.0
    identified_speaker = None
    
    for speaker, ref_emb in reference_embeddings.items():
        similarity = cosine_similarity(query_embedding, ref_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            identified_speaker = speaker
    
    if best_similarity < CONFIDENCE_THRESHOLD:
        return "unknown", best_similarity
    
    return identified_speaker, best_similarity

if __name__ == "__main__":
    print("Cargando modelo de SpeechBrain...")
    get_encoder()
    
    print("Extrayendo embeddings de voz...")
    X, y = extract_embeddings(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Entrenando clasificador SVM...")
    svm_classifier, label_encoder = train_svm_classifier(X_train, y_train)
    print(f"Modelo SVM entrenado y guardado en {SVM_MODEL_PATH}")

    print("Identificando hablantes en conjunto de prueba...")
    y_pred = []
    confidences = []

    for query_emb in X_test:
        identified, confidence = identify_speaker(query_emb)
        y_pred.append(identified)
        confidences.append(confidence)

    y_pred = np.array(y_pred)
    
    unknown_mask = y_pred == "unknown"
    known_mask = ~unknown_mask
    
    print(f"\n--- Resultados de clasificación (Umbral: {CONFIDENCE_THRESHOLD}) ---")
    print(f"Usuarios conocidos identificados: {np.sum(known_mask)}/{len(y_test)}")
    print(f"Usuarios etiquetados como desconocidos: {np.sum(unknown_mask)}/{len(y_test)}")
    
    if np.sum(known_mask) > 0:
        y_test_known = y_test[known_mask]
        y_pred_known = y_pred[known_mask]
        
        accuracy_known = accuracy_score(y_test_known, y_pred_known)
        print(f"\n--- Precisión en usuarios conocidos: {accuracy_known:.4f} ---")
        
        unique_speakers = np.unique(np.concatenate([y_test_known, y_pred_known]))
        print("\n--- Reporte de clasificación (usuarios conocidos) ---")
        print(classification_report(y_test_known, y_pred_known, labels=unique_speakers, target_names=unique_speakers))

    confidences = np.array(confidences)
    print(f"\n--- Estadísticas de confianza ---")
    print(f"Confianza promedio: {np.mean(confidences):.4f}")
    print(f"Confianza mínima: {np.min(confidences):.4f}")
    print(f"Confianza máxima: {np.max(confidences):.4f}")
    print(f"Confianza promedio (conocidos): {np.mean(confidences[known_mask]):.4f}" if np.sum(known_mask) > 0 else "")
    print(f"Confianza promedio (desconocidos): {np.mean(confidences[unknown_mask]):.4f}" if np.sum(unknown_mask) > 0 else "")

    all_labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Matriz de confusión - SpeechBrain + SVM (Umbral: {CONFIDENCE_THRESHOLD})")
    plt.colorbar()
    tick_marks = np.arange(len(all_labels))
    plt.xticks(tick_marks, all_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, all_labels)
    plt.ylabel("Hablante real")
    plt.xlabel("Hablante identificado")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()
