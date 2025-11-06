from typing import Tuple
import numpy as np

from .config import BiometricConfig
from .model_loaders import VoiceModelLoader, FacialModelLoader
from .embedding_extractors import VoiceEmbeddingExtractor, FacialEmbeddingExtractor


class BaseIdentifier:
    def __init__(self, model_loader, embedding_extractor, confidence_threshold: float):
        self.model_loader = model_loader
        self.embedding_extractor = embedding_extractor
        self.confidence_threshold = confidence_threshold
    
    def _predict(self, embedding: np.ndarray) -> Tuple[str, float]:
        svm_classifier, label_encoder = self.model_loader.load()
        
        query_embedding_reshaped = embedding.reshape(1, -1)
        probabilities = svm_classifier.predict_proba(query_embedding_reshaped)[0]
        predicted_class = svm_classifier.predict(query_embedding_reshaped)[0]
        
        confidence = np.max(probabilities)
        
        if confidence < self.confidence_threshold:
            return "unknown", float(confidence)
        
        identified = label_encoder.inverse_transform([predicted_class])[0]
        return identified, float(confidence)


class VoiceIdentifier(BaseIdentifier):
    def __init__(self):
        super().__init__(
            VoiceModelLoader(),
            VoiceEmbeddingExtractor(),
            BiometricConfig.VOICE_CONFIDENCE_THRESHOLD
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
            BiometricConfig.FACIAL_CONFIDENCE_THRESHOLD
        )
    
    def identify(self, image_path: str) -> Tuple[str, float]:
        embedding = self.embedding_extractor.extract(image_path)
        embedding_array = np.array(embedding)
        return self._predict(embedding_array)

