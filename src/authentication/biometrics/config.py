import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class BiometricConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    MODELS_DIR = BASE_DIR / "models"

    VOICE_CHECKPOINTS_DIR = MODELS_DIR / "voice" / "checkpoints"
    VOICE_NN_MODEL_PATH = VOICE_CHECKPOINTS_DIR / "nn_classifier.pth"
    VOICE_NN_LABEL_ENCODER_PATH = VOICE_CHECKPOINTS_DIR / "nn_label_encoder.pkl"
    VOICE_CONFIDENCE_THRESHOLD = 0.75
    VOICE_ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    VOICE_CLASSIFIER_TYPE = os.getenv("VOICE_CLASSIFIER_TYPE", "nn").lower()

    FACIAL_CHECKPOINTS_DIR = MODELS_DIR / "facial" / "checkpoints"
    FACIAL_NN_MODEL_PATH = FACIAL_CHECKPOINTS_DIR / "nn_classifier.pth"
    FACIAL_NN_LABEL_ENCODER_PATH = FACIAL_CHECKPOINTS_DIR / "nn_label_encoder.pkl"
    FACIAL_CNN_MODEL_PATH = FACIAL_CHECKPOINTS_DIR / "cnn_classifier.keras"
    FACIAL_CNN_LABEL_ENCODER_PATH = FACIAL_CHECKPOINTS_DIR / "cnn_label_encoder.pkl"
    FACIAL_EMBEDDING_MODEL_PATH = FACIAL_CHECKPOINTS_DIR / "embedding_model.keras"
    FACIAL_EMBEDDING_DATABASE_PATH = FACIAL_CHECKPOINTS_DIR / "embedding_database.pkl"
    FACIAL_CONFIDENCE_THRESHOLD = 0.75
    FACIAL_SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold for embeddings
    FACIAL_CLASSIFIER_TYPE = os.getenv("FACIAL_CLASSIFIER_TYPE", "nn").lower()

