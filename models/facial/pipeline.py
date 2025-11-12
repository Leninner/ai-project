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
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DATA_DIR = "data"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints"
SVM_MODEL_PATH = CHECKPOINT_PATH / "svm_classifier.joblib"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.75

_mtcnn = None
_resnet = None
_svm_classifier = None
_label_encoder = None

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
    
    img = Image.open(image_path).convert('RGB')
    img_cropped = mtcnn(img)
    
    if img_cropped is None:
        raise ValueError(f"No faces found in image {image_path}")
    
    img_cropped = img_cropped.unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(img_cropped)
    
    return embedding.squeeze().numpy()

def extract_embeddings(data_dir):
    X, y = [], []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(person_dir, image_name)
            try:
                emb = extract_embedding(image_path)
                X.append(emb)
                y.append(person_name)
            except Exception:
                continue
    return np.array(X), np.array(y)

def identify_face(query_embedding, classifier_type: str = "svm"):
    try:
        from .classifiers.classifier_strategy import create_classifier_strategy, FaceIdentifier
    except ImportError:
        import sys
        from pathlib import Path
        facial_path = Path(__file__).parent
        if str(facial_path) not in sys.path:
            sys.path.insert(0, str(facial_path))
        from classifiers.classifier_strategy import create_classifier_strategy, FaceIdentifier
    
    strategy = create_classifier_strategy(classifier_type)
    face_identifier = FaceIdentifier(strategy)
    return face_identifier.identify_face(query_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate facial recognition classifier')
    parser.add_argument(
        '--classifier', '-c',
        type=str,
        choices=['svm', 'nn'],
        default='svm',
        help='Classifier type to use: svm (Support Vector Machine) or nn (Neural Network). Default: svm'
    )
    args = parser.parse_args()
    
    classifier_type = args.classifier
    
    X, y = extract_embeddings(DATA_DIR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    try:
        from .classifiers.classifier_strategy import create_classifier_strategy, FaceIdentifier
    except ImportError:
        import sys
        from pathlib import Path
        facial_path = Path(__file__).parent
        if str(facial_path) not in sys.path:
            sys.path.insert(0, str(facial_path))
        from classifiers.classifier_strategy import create_classifier_strategy, FaceIdentifier
    
    strategy = create_classifier_strategy(classifier_type)
    face_identifier = FaceIdentifier(strategy)
    
    face_identifier.train(X_train, y_train)

    print("")
    print("─" * 60)
    print("🔍 Identificando personas en conjunto de prueba...")
    print("─" * 60)
    y_pred = []
    confidences = []

    for query_emb in X_test:
        identified, confidence = face_identifier.identify_face(query_emb)
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
    print(f"   ├─ Personas conocidas identificadas: {np.sum(known_mask)}/{len(y_test)}")
    print(f"   └─ Personas etiquetadas como desconocidas: {np.sum(unknown_mask)}/{len(y_test)}")
    
    if np.sum(known_mask) > 0:
        y_test_known = y_test[known_mask]
        y_pred_known = y_pred[known_mask]
        
        accuracy_known = accuracy_score(y_test_known, y_pred_known)
        print("")
        print("─" * 60)
        print(f"✓ Precisión en personas conocidas: {accuracy_known:.2%}")
        print("─" * 60)
        
        unique_persons = np.unique(np.concatenate([y_test_known, y_pred_known]))
        print("")
        print("📋 Reporte de Clasificación (personas conocidas)")
        print("─" * 60)
        print(classification_report(y_test_known, y_pred_known, labels=unique_persons, target_names=unique_persons))

    confidences = np.array(confidences)
    print("")
    print("─" * 60)
    print("📈 Estadísticas de Confianza")
    print("─" * 60)
    print(f"   ├─ Confianza promedio: {np.mean(confidences):.2%}")
    print(f"   ├─ Confianza mínima: {np.min(confidences):.2%}")
    print(f"   └─ Confianza máxima: {np.max(confidences):.2%}")
    if np.sum(known_mask) > 0:
        print(f"   ├─ Confianza promedio (conocidas): {np.mean(confidences[known_mask]):.2%}")
    if np.sum(unknown_mask) > 0:
        print(f"   └─ Confianza promedio (desconocidas): {np.mean(confidences[unknown_mask]):.2%}")

    all_labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues' if classifier_type == 'svm' else 'Greens')
    plt.title(f"Matriz de confusión - FaceNet + {classifier_name} (Umbral: {CONFIDENCE_THRESHOLD})")
    plt.colorbar()
    tick_marks = np.arange(len(all_labels))
    plt.xticks(tick_marks, all_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, all_labels)
    plt.ylabel("Persona real")
    plt.xlabel("Persona identificada")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    
    confusion_matrix_filename = f"confusion_matrix_facial_{classifier_type}.png"
    confusion_matrix_path = Path(__file__).parent / confusion_matrix_filename
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    print("")
    print("─" * 60)
    print(f"💾 Matriz de confusión guardada")
    print(f"   └─ {confusion_matrix_path}")
    print("─" * 60)
    print("")
    plt.close()
