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
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "label_encoder.pkl"
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
    resnet = get_resnet()
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    if img_array.shape != (160, 160, 3):
        mtcnn = get_mtcnn()
        img_cropped = mtcnn(img)
        if img_cropped is None:
            raise ValueError(f"No faces found in image {image_path}")
        img_tensor = img_cropped
    else:
        img_array_normalized = img_array.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array_normalized).permute(2, 0, 1)
    
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(img_tensor)
    
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
    num_classes = len(label_encoder.classes_)
    
    print(f"Entrenando clasificador SVM...")
    print(f"Clases: {num_classes}, Muestras: {len(X_train)}")
    
    svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
    svm_classifier.fit(X_train, y_encoded)
    
    joblib.dump(svm_classifier, SVM_MODEL_PATH)
    import pickle
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Modelo SVM entrenado y guardado en {SVM_MODEL_PATH}")
    
    return svm_classifier, label_encoder

def load_svm_classifier():
    global _svm_classifier, _label_encoder
    
    if _svm_classifier is None or _label_encoder is None:
        if SVM_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            _svm_classifier = joblib.load(SVM_MODEL_PATH)
            import pickle
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                _label_encoder = pickle.load(f)
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

