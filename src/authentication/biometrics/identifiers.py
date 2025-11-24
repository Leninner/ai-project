from typing import Tuple
import numpy as np
import torch

from .config import BiometricConfig
from .model_loaders import VoiceModelLoader, FacialModelLoader
from .embedding_extractors import VoiceEmbeddingExtractor, FacialEmbeddingExtractor


class BaseIdentifier:
    def __init__(self, model_loader, embedding_extractor, confidence_threshold: float):
        self.model_loader = model_loader
        self.embedding_extractor = embedding_extractor
        self.confidence_threshold = confidence_threshold

    def _predict(self, embedding: np.ndarray) -> Tuple[str, float]:
        model, label_encoder = self.model_loader.load()
        classifier_type = self.model_loader.classifier_type

        if classifier_type == "cnn":
            raise NotImplementedError("CNN prediction should use _predict_cnn method")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            query_embedding_tensor = (
                torch.FloatTensor(embedding).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                logits = model(query_embedding_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence_score, predicted_index = torch.max(probabilities, 1)

            confidence_value = confidence_score.item()
            predicted_class_index = predicted_index.item()

            if confidence_value < self.confidence_threshold:
                return "unknown", float(confidence_value)

            identified = label_encoder.inverse_transform([predicted_class_index])[0]
            return identified, float(confidence_value)

    def _predict_cnn(self, image_path: str) -> Tuple[str, float]:
        model, label_encoder = self.model_loader.load()
        classifier_type = self.model_loader.classifier_type

        if classifier_type != "cnn":
            raise ValueError("_predict_cnn can only be used with CNN classifier")

        import sys
        from pathlib import Path

        models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir.parent))

        try:
            from models.facial.classifiers import cnn_classifier
        except ImportError:
            import importlib.util

            cnn_classifier_path = (
                models_dir / "facial" / "classifiers" / "cnn_classifier.py"
            )
            spec = importlib.util.spec_from_file_location(
                "cnn_classifier", cnn_classifier_path
            )
            cnn_classifier_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cnn_classifier_module)
            cnn_classifier = cnn_classifier_module

        return cnn_classifier.identify_face_from_image(image_path)


class VoiceIdentifier(BaseIdentifier):
    def __init__(self):
        super().__init__(
            VoiceModelLoader(),
            VoiceEmbeddingExtractor(),
            BiometricConfig.VOICE_CONFIDENCE_THRESHOLD,
        )

    def identify(self, audio_path: str) -> Tuple[str, float]:
        embedding = self.embedding_extractor.extract(audio_path)
        embedding_array = np.array(embedding)
        return self._predict(embedding_array)


class FacialIdentifier(BaseIdentifier):
    def __init__(self):
        super().__init__(
            FacialModelLoader(),
            FacialEmbeddingExtractor(),
            BiometricConfig.FACIAL_CONFIDENCE_THRESHOLD,
        )

    def identify(self, image_path: str) -> Tuple[str, float]:
        classifier_type = self.model_loader.classifier_type

        if classifier_type == "cnn":
            return self._predict_cnn(image_path)
        else:
            embedding = self.embedding_extractor.extract(image_path)
            embedding_array = np.array(embedding)
            return self._predict(embedding_array)
