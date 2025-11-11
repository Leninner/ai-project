import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class ClassifierStrategy(ABC):
    @abstractmethod
    def train(self, embeddings_train: np.ndarray, labels_train: np.ndarray) -> Tuple:
        pass
    
    @abstractmethod
    def identify_face(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class SVMClassifierStrategy(ClassifierStrategy):
    def __init__(self):
        from . import svm_classifier
        self.svm_module = svm_classifier
    
    def train(self, embeddings_train: np.ndarray, labels_train: np.ndarray) -> Tuple:
        return self.svm_module.train_svm_classifier(embeddings_train, labels_train)
    
    def identify_face(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        return self.svm_module.identify_face(query_embedding)
    
    def get_name(self) -> str:
        return "SVM"

class NeuralNetworkClassifierStrategy(ClassifierStrategy):
    def __init__(self):
        from . import nn_classifier
        self.nn_module = nn_classifier
    
    def train(self, embeddings_train: np.ndarray, labels_train: np.ndarray) -> Tuple:
        return self.nn_module.train_nn_classifier(embeddings_train, labels_train, epochs=100, batch_size=32)
    
    def identify_face(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        return self.nn_module.identify_face(query_embedding)
    
    def get_name(self) -> str:
        return "Neural Network"

class FaceIdentifier:
    def __init__(self, classifier_strategy: ClassifierStrategy):
        self.classifier_strategy = classifier_strategy
    
    def set_strategy(self, classifier_strategy: ClassifierStrategy):
        self.classifier_strategy = classifier_strategy
    
    def train(self, embeddings_train: np.ndarray, labels_train: np.ndarray):
        return self.classifier_strategy.train(embeddings_train, labels_train)
    
    def identify_face(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        return self.classifier_strategy.identify_face(query_embedding)
    
    def get_classifier_name(self) -> str:
        return self.classifier_strategy.get_name()

def create_classifier_strategy(classifier_type: str) -> ClassifierStrategy:
    classifier_type_lower = classifier_type.lower()
    if classifier_type_lower == "svm":
        return SVMClassifierStrategy()
    elif classifier_type_lower in ["nn", "neural_network", "neuralnetwork"]:
        return NeuralNetworkClassifierStrategy()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'svm' or 'nn'")

