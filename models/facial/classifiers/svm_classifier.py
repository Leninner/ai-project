import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
SVM_MODEL_PATH = CHECKPOINT_PATH / "svm_classifier.joblib"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.6

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

def identify_face(query_embedding):
    svm_classifier, label_encoder = load_svm_classifier()
    
    query_embedding_reshaped = query_embedding.reshape(1, -1)
    probabilities = svm_classifier.predict_proba(query_embedding_reshaped)[0]
    predicted_class = svm_classifier.predict(query_embedding_reshaped)[0]
    
    confidence = np.max(probabilities)
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", float(confidence)
    
    identified_person = label_encoder.inverse_transform([predicted_class])[0]
    return identified_person, float(confidence)

if __name__ == "__main__":
    print("Extrayendo embeddings faciales...")
    X, y = extract_embeddings(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Entrenando clasificador SVM...")
    svm_classifier, label_encoder = train_svm_classifier(X_train, y_train)
    print(f"Modelo SVM entrenado y guardado en {SVM_MODEL_PATH}")

    print("Identificando personas en conjunto de prueba...")
    y_pred = []
    confidences = []

    for query_emb in X_test:
        identified, confidence = identify_face(query_emb)
        y_pred.append(identified)
        confidences.append(confidence)

    y_pred = np.array(y_pred)
    
    unknown_mask = y_pred == "unknown"
    known_mask = ~unknown_mask
    
    print(f"\n--- Resultados de clasificación SVM (Umbral: {CONFIDENCE_THRESHOLD}) ---")
    print(f"Personas conocidas identificadas: {np.sum(known_mask)}/{len(y_test)}")
    print(f"Personas etiquetadas como desconocidas: {np.sum(unknown_mask)}/{len(y_test)}")
    
    if np.sum(known_mask) > 0:
        y_test_known = y_test[known_mask]
        y_pred_known = y_pred[known_mask]
        
        accuracy_known = accuracy_score(y_test_known, y_pred_known)
        print(f"\n--- Precisión en personas conocidas: {accuracy_known:.4f} ---")
        
        unique_persons = np.unique(np.concatenate([y_test_known, y_pred_known]))
        print("\n--- Reporte de clasificación (personas conocidas) ---")
        print(classification_report(y_test_known, y_pred_known, labels=unique_persons, target_names=unique_persons))

    confidences = np.array(confidences)
    print(f"\n--- Estadísticas de confianza ---")
    print(f"Confianza promedio: {np.mean(confidences):.4f}")
    print(f"Confianza mínima: {np.min(confidences):.4f}")
    print(f"Confianza máxima: {np.max(confidences):.4f}")
    print(f"Confianza promedio (conocidas): {np.mean(confidences[known_mask]):.4f}" if np.sum(known_mask) > 0 else "")
    print(f"Confianza promedio (desconocidas): {np.mean(confidences[unknown_mask]):.4f}" if np.sum(unknown_mask) > 0 else "")

    all_labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Matriz de confusión - FaceNet + SVM (Umbral: {CONFIDENCE_THRESHOLD})")
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
    
    confusion_matrix_path = Path(__file__).parent.parent / "confusion_matrix_svm.png"
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    print(f"\nMatriz de confusión guardada en: {confusion_matrix_path}")
    plt.close()

