import joblib
import torch
import torch.nn as nn
import sys
from abc import ABC, abstractmethod
from typing import Tuple, Any
from pathlib import Path

from .config import BiometricConfig


class BaseModelLoader(ABC):
    def __init__(self, classifier_type: str = "nn"):
        self.classifier_type = classifier_type.lower()
        if self.classifier_type not in ['nn', 'cnn']:
            raise ValueError(f"Invalid classifier type: {classifier_type}. Must be 'nn', or 'cnn'")
        self._model = None
        self._label_encoder = None
    
    @abstractmethod
    def _get_nn_paths(self) -> Tuple[Path, Path]:
        pass
    
    @abstractmethod
    def _create_nn_model(self, num_classes: int) -> nn.Module:
        pass
    
    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass
    
    @abstractmethod
    def _get_cnn_paths(self) -> Tuple[Path, Path]:
        pass
    
    @abstractmethod
    def _load_cnn_model(self) -> Any:
        pass
    
    @abstractmethod
    def _validate_paths(self) -> None:
        pass
    
    def load(self) -> Tuple[Any, Any]:
        if self._model is None or self._label_encoder is None:
            self._validate_paths()
            
            if self.classifier_type == 'cnn':
                cnn_model_path, label_encoder_path = self._get_cnn_paths()
                self._model = self._load_cnn_model()
                import pickle
                with open(label_encoder_path, 'rb') as f:
                    self._label_encoder = pickle.load(f)
            else:
                nn_model_path, label_encoder_path = self._get_nn_paths()
                import pickle
                with open(label_encoder_path, 'rb') as f:
                    self._label_encoder = pickle.load(f)
                num_classes = len(self._label_encoder.classes_)
                self._model = self._create_nn_model(num_classes)
                self._model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
                self._model.eval()
            
            if hasattr(self._label_encoder, 'classes_'):
                if len(self._label_encoder.classes_) == 0:
                    raise ValueError(f"Model has no classes")
        
        return self._model, self._label_encoder


class VoiceModelLoader(BaseModelLoader):
    def __init__(self, classifier_type: str = None):
        classifier_type = classifier_type or BiometricConfig.VOICE_CLASSIFIER_TYPE
        super().__init__(classifier_type)
    
    def _get_nn_paths(self) -> Tuple[Path, Path]:
        return BiometricConfig.VOICE_NN_MODEL_PATH, BiometricConfig.VOICE_NN_LABEL_ENCODER_PATH
    
    def _get_cnn_paths(self) -> Tuple[Path, Path]:
        raise NotImplementedError("CNN classifier not supported for voice recognition")
    
    def _load_cnn_model(self) -> Any:
        raise NotImplementedError("CNN classifier not supported for voice recognition")
    
    def _get_embedding_dim(self) -> int:
        return 192
    
    def _create_nn_model(self, num_classes: int) -> nn.Module:
        models_dir = BiometricConfig.MODELS_DIR
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir.parent))
        
        try:
            from models.voice.classifiers.nn_classifier import VoiceClassifier
        except ImportError:
            import importlib.util
            nn_classifier_path = models_dir / 'voice' / 'classifiers' / 'nn_classifier.py'
            spec = importlib.util.spec_from_file_location("nn_classifier", nn_classifier_path)
            nn_classifier_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn_classifier_module)
            VoiceClassifier = nn_classifier_module.VoiceClassifier
        
        return VoiceClassifier(embedding_dim=self._get_embedding_dim(), num_classes=num_classes)
    
    def _validate_paths(self) -> None:
        if self.classifier_type == 'cnn':
            raise NotImplementedError("CNN classifier not supported for voice recognition")
        else:
            nn_model_path, label_encoder_path = self._get_nn_paths()
            if not nn_model_path.exists() or not label_encoder_path.exists():
                raise ValueError("Voice Neural Network models not found. Please train the models first.")


class FacialModelLoader(BaseModelLoader):
    def __init__(self, classifier_type: str = None):
        classifier_type = classifier_type or BiometricConfig.FACIAL_CLASSIFIER_TYPE
        super().__init__(classifier_type)
    
    def _get_nn_paths(self) -> Tuple[Path, Path]:
        return BiometricConfig.FACIAL_NN_MODEL_PATH, BiometricConfig.FACIAL_NN_LABEL_ENCODER_PATH
    
    def _get_cnn_paths(self) -> Tuple[Path, Path]:
        return BiometricConfig.FACIAL_CNN_MODEL_PATH, BiometricConfig.FACIAL_CNN_LABEL_ENCODER_PATH
    
    def _load_cnn_model(self) -> Any:
        models_dir = BiometricConfig.MODELS_DIR
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir.parent))
        
        try:
            from models.facial.classifiers import cnn_classifier
        except ImportError:
            import importlib.util
            cnn_classifier_path = models_dir / 'facial' / 'classifiers' / 'cnn_classifier.py'
            spec = importlib.util.spec_from_file_location("cnn_classifier", cnn_classifier_path)
            cnn_classifier_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cnn_classifier_module)
            cnn_classifier = cnn_classifier_module
        
        model, _ = cnn_classifier.load_cnn_classifier()
        return model
    
    def _get_embedding_dim(self) -> int:
        return 512
    
    def _create_nn_model(self, num_classes: int) -> nn.Module:
        models_dir = BiometricConfig.MODELS_DIR
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir.parent))
        
        try:
            from models.facial.classifiers.nn_classifier import FaceClassifier
        except ImportError:
            import importlib.util
            nn_classifier_path = models_dir / 'facial' / 'classifiers' / 'nn_classifier.py'
            spec = importlib.util.spec_from_file_location("nn_classifier", nn_classifier_path)
            nn_classifier_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn_classifier_module)
            FaceClassifier = nn_classifier_module.FaceClassifier
        
        return FaceClassifier(embedding_dim=self._get_embedding_dim(), num_classes=num_classes)
    
    def _validate_paths(self) -> None:
        if self.classifier_type == 'cnn':
            cnn_model_path, label_encoder_path = self._get_cnn_paths()
            if not cnn_model_path.exists() or not label_encoder_path.exists():
                raise ValueError("Facial CNN models not found. Please train the models first.")
        else:
            nn_model_path, label_encoder_path = self._get_nn_paths()
            if not nn_model_path.exists() or not label_encoder_path.exists():
                raise ValueError("Facial Neural Network models not found. Please train the models first.")

