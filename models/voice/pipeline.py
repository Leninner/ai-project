import os
import numpy as np
import argparse
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
CONFIDENCE_THRESHOLD = 0.75

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

def identify_speaker(query_embedding, classifier_type: str = "svm"):
    try:
        from .classifiers.classifier_strategy import create_classifier_strategy, VoiceIdentifier
    except ImportError:
        import sys
        from pathlib import Path
        voice_path = Path(__file__).parent
        if str(voice_path) not in sys.path:
            sys.path.insert(0, str(voice_path))
        from classifiers.classifier_strategy import create_classifier_strategy, VoiceIdentifier
    
    strategy = create_classifier_strategy(classifier_type)
    voice_identifier = VoiceIdentifier(strategy)
    return voice_identifier.identify_speaker(query_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate voice recognition classifier')
    parser.add_argument(
        '--classifier', '-c',
        type=str,
        choices=['svm', 'nn'],
        default='svm',
        help='Classifier type to use: svm (Support Vector Machine) or nn (Neural Network). Default: svm'
    )
    args = parser.parse_args()
    
    classifier_type = args.classifier
    
    get_encoder()
    X, y = extract_embeddings(DATA_DIR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    try:
        from .classifiers.classifier_strategy import create_classifier_strategy, VoiceIdentifier
    except ImportError:
        import sys
        from pathlib import Path
        voice_path = Path(__file__).parent
        if str(voice_path) not in sys.path:
            sys.path.insert(0, str(voice_path))
        from classifiers.classifier_strategy import create_classifier_strategy, VoiceIdentifier
    
    strategy = create_classifier_strategy(classifier_type)
    voice_identifier = VoiceIdentifier(strategy)
    
    voice_identifier.train(X_train, y_train)

    print("")
    print("─" * 60)
    print("🔍 Identificando hablantes en conjunto de prueba...")
    print("─" * 60)
    y_pred = []
    confidences = []

    for query_emb in X_test:
        identified, confidence = voice_identifier.identify_speaker(query_emb)
        y_pred.append(identified)
        confidences.append(confidence)

    y_pred = np.array(y_pred)
    
    unknown_mask = y_pred == "unknown"
    known_mask = ~unknown_mask
    
    classifier_name = "SVM" if classifier_type == "svm" else "Neural Network"
    print("")
    print("═" * 60)
    print(f"📊 Resultados de Clasificación - {classifier_name}")
    print(f"   └─ Umbral de confianza: {CONFIDENCE_THRESHOLD:.2%}")
    print("═" * 60)
    print(f"   ├─ Usuarios conocidos identificados: {np.sum(known_mask)}/{len(y_test)}")
    print(f"   └─ Usuarios etiquetados como desconocidos: {np.sum(unknown_mask)}/{len(y_test)}")
    
    if np.sum(known_mask) > 0:
        y_test_known = y_test[known_mask]
        y_pred_known = y_pred[known_mask]
        
        accuracy_known = accuracy_score(y_test_known, y_pred_known)
        print("")
        print("─" * 60)
        print(f"✓ Precisión en usuarios conocidos: {accuracy_known:.2%}")
        print("─" * 60)
        
        unique_speakers = np.unique(np.concatenate([y_test_known, y_pred_known]))
        print("")
        print("📋 Reporte de Clasificación (usuarios conocidos)")
        print("─" * 60)
        print(classification_report(y_test_known, y_pred_known, labels=unique_speakers, target_names=unique_speakers))

    confidences = np.array(confidences)
    print("")
    print("─" * 60)
    print("📈 Estadísticas de Confianza")
    print("─" * 60)
    print(f"   ├─ Confianza promedio: {np.mean(confidences):.2%}")
    print(f"   ├─ Confianza mínima: {np.min(confidences):.2%}")
    print(f"   └─ Confianza máxima: {np.max(confidences):.2%}")
    if np.sum(known_mask) > 0:
        print(f"   ├─ Confianza promedio (conocidos): {np.mean(confidences[known_mask]):.2%}")
    if np.sum(unknown_mask) > 0:
        print(f"   └─ Confianza promedio (desconocidos): {np.mean(confidences[unknown_mask]):.2%}")

    all_labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues' if classifier_type == 'svm' else 'Greens')
    plt.title(f"Matriz de confusión - SpeechBrain + {classifier_name} (Umbral: {CONFIDENCE_THRESHOLD})")
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
    
    confusion_matrix_filename = f"confusion_matrix_voice_{classifier_type}.png"
    confusion_matrix_path = Path(__file__).parent / confusion_matrix_filename
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    print("")
    print("─" * 60)
    print(f"💾 Matriz de confusión guardada")
    print(f"   └─ {confusion_matrix_path}")
    print("─" * 60)
    print("")
    plt.close()
