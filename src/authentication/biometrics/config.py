from pathlib import Path


class BiometricConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    
    VOICE_CHECKPOINTS_DIR = MODELS_DIR / 'voice' / 'checkpoints'
    VOICE_SVM_MODEL_PATH = VOICE_CHECKPOINTS_DIR / 'svm_classifier.joblib'
    VOICE_LABEL_ENCODER_PATH = VOICE_CHECKPOINTS_DIR / 'label_encoder.joblib'
    VOICE_CONFIDENCE_THRESHOLD = 0.3
    VOICE_ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    
    FACIAL_CHECKPOINTS_DIR = MODELS_DIR / 'facial' / 'checkpoints'
    FACIAL_SVM_MODEL_PATH = FACIAL_CHECKPOINTS_DIR / 'svm_classifier.joblib'
    FACIAL_LABEL_ENCODER_PATH = FACIAL_CHECKPOINTS_DIR / 'label_encoder.joblib'
    FACIAL_CONFIDENCE_THRESHOLD = 0.75

