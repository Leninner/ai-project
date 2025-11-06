import joblib
from abc import ABC, abstractmethod
from typing import Tuple, Any
from pathlib import Path

from .config import BiometricConfig


class BaseModelLoader(ABC):
    def __init__(self, svm_model_path: Path, label_encoder_path: Path):
        self.svm_model_path = svm_model_path
        self.label_encoder_path = label_encoder_path
        self._svm_classifier = None
        self._label_encoder = None
    
    @abstractmethod
    def _validate_paths(self) -> None:
        pass
    
    def load(self) -> Tuple[Any, Any]:
        if self._svm_classifier is None or self._label_encoder is None:
            self._validate_paths()
            self._svm_classifier = joblib.load(self.svm_model_path)
            self._label_encoder = joblib.load(self.label_encoder_path)
            
            # Validar que el modelo tiene al menos una clase
            if hasattr(self._label_encoder, 'classes_'):
                if len(self._label_encoder.classes_) == 0:
                    raise ValueError(f"Model at {self.svm_model_path} has no classes")
        
        return self._svm_classifier, self._label_encoder


class VoiceModelLoader(BaseModelLoader):
    def __init__(self):
        super().__init__(
            BiometricConfig.VOICE_SVM_MODEL_PATH,
            BiometricConfig.VOICE_LABEL_ENCODER_PATH
        )
    
    def _validate_paths(self) -> None:
        if not self.svm_model_path.exists() or not self.label_encoder_path.exists():
            raise ValueError("Voice models not found. Please train the models first.")


class FacialModelLoader(BaseModelLoader):
    def __init__(self):
        super().__init__(
            BiometricConfig.FACIAL_SVM_MODEL_PATH,
            BiometricConfig.FACIAL_LABEL_ENCODER_PATH
        )
    
    def _validate_paths(self) -> None:
        if not self.svm_model_path.exists() or not self.label_encoder_path.exists():
            raise ValueError("Facial models not found. Please train the models first.")

