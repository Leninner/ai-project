import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / '.env')


class BiometricConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    
    VOICE_CHECKPOINTS_DIR = MODELS_DIR / 'voice' / 'checkpoints'
    VOICE_SVM_MODEL_PATH = VOICE_CHECKPOINTS_DIR / 'svm_classifier.joblib'
    VOICE_SVM_LABEL_ENCODER_PATH = VOICE_CHECKPOINTS_DIR / 'label_encoder.joblib'
    VOICE_NN_MODEL_PATH = VOICE_CHECKPOINTS_DIR / 'nn_classifier.pth'
    VOICE_NN_LABEL_ENCODER_PATH = VOICE_CHECKPOINTS_DIR / 'nn_label_encoder.joblib'
    VOICE_CONFIDENCE_THRESHOLD = 0.75
    VOICE_ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    VOICE_CLASSIFIER_TYPE = os.getenv('VOICE_CLASSIFIER_TYPE', 'svm').lower()
    
    FACIAL_CHECKPOINTS_DIR = MODELS_DIR / 'facial' / 'checkpoints'
    FACIAL_SVM_MODEL_PATH = FACIAL_CHECKPOINTS_DIR / 'svm_classifier.joblib'
    FACIAL_SVM_LABEL_ENCODER_PATH = FACIAL_CHECKPOINTS_DIR / 'label_encoder.joblib'
    FACIAL_NN_MODEL_PATH = FACIAL_CHECKPOINTS_DIR / 'nn_classifier.pth'
    FACIAL_NN_LABEL_ENCODER_PATH = FACIAL_CHECKPOINTS_DIR / 'nn_label_encoder.joblib'
    FACIAL_CONFIDENCE_THRESHOLD = 0.75
    FACIAL_CLASSIFIER_TYPE = os.getenv('FACIAL_CLASSIFIER_TYPE', 'svm').lower()

