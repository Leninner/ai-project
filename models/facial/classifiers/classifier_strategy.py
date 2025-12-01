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


class NeuralNetworkClassifierStrategy(ClassifierStrategy):
    def __init__(self):
        from . import nn_classifier

        self.nn_module = nn_classifier

    def train(self, embeddings_train: np.ndarray, labels_train: np.ndarray) -> Tuple:
        return self.nn_module.train_nn_classifier(
            embeddings_train, labels_train, epochs=100, batch_size=32
        )

    def identify_face(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        return self.nn_module.identify_face(query_embedding)

    def get_name(self) -> str:
        return "Neural Network"


class CNNClassifierStrategy(ClassifierStrategy):
    def __init__(self):
        from . import cnn_classifier

        self.cnn_module = cnn_classifier

    def train(
        self,
        embeddings_train: np.ndarray = None,
        labels_train: np.ndarray = None,
        **kwargs,
    ) -> Tuple:
        data_dir = kwargs.get("data_dir", None)
        use_augmentation = kwargs.get("use_augmentation", True)
        return self.cnn_module.train_cnn_classifier(
            data_dir=data_dir,
            epochs=100,
            batch_size=32,
            use_augmentation=use_augmentation,
        )

    def identify_face(
        self,
        query_embedding: np.ndarray = None,
        image_path: str = None,
        image_array: np.ndarray = None,
    ) -> Tuple[str, float]:
        if image_path is not None:
            return self.cnn_module.identify_face_from_image(image_path)
        elif image_array is not None:
            return self.cnn_module.identify_face_from_array(image_array)
        else:
            raise ValueError("CNN classifier requires either image_path or image_array")

    def get_name(self) -> str:
        return "CNN"


class CNNEmbeddingClassifierStrategy(ClassifierStrategy):
    """Strategy for CNN-based embedding model trained with triplet loss"""
    
    def __init__(self):
        from . import cnn_embedding
        
        self.embedding_module = cnn_embedding
    
    def train(
        self,
        embeddings_train: np.ndarray = None,
        labels_train: np.ndarray = None,
        **kwargs,
    ) -> Tuple:
        # """Train the embedding model with triplet loss"""
        # data_dir = kwargs.get("data_dir", None)
        # epochs = kwargs.get("epochs", 100)
        # learning_rate = kwargs.get("learning_rate", 0.0001)
        # margin = kwargs.get("margin", 0.3)
        
        # # Train embedding model with triplet loss
        # model, history = self.embedding_module.train_embedding_model(
        #     data_dir=data_dir,
        #     epochs=epochs,
        #     learning_rate=learning_rate,
        #     margin=margin
        # )
        
        # Generate embedding database from already trained model
        database = self.embedding_module.save_embedding_database(model=None)
        
        return None, database
    
    def identify_face(
        self,
        query_embedding: np.ndarray = None,
        image_path: str = None,
        image_array: np.ndarray = None,
        expected_name: str = None,
    ) -> Tuple[str, float]:
        """Identify face using embedding similarity"""
        if image_path is not None:
            return self.embedding_module.identify_face_from_image(
                image_path, expected_name=expected_name
            )
        elif image_array is not None:
            return self.embedding_module.identify_face_from_array(
                image_array, expected_name=expected_name
            )
        else:
            raise ValueError(
                "CNN Embedding classifier requires either image_path or image_array"
            )
    
    def get_name(self) -> str:
        return "CNN Embedding (Triplet Loss)"


class FaceIdentifier:
    def __init__(self, classifier_strategy: ClassifierStrategy):
        self.classifier_strategy = classifier_strategy

    def set_strategy(self, classifier_strategy: ClassifierStrategy):
        self.classifier_strategy = classifier_strategy

    def train(
        self,
        embeddings_train: np.ndarray = None,
        labels_train: np.ndarray = None,
        **kwargs,
    ):
        return self.classifier_strategy.train(embeddings_train, labels_train, **kwargs)

    def identify_face(
        self, query_embedding: np.ndarray = None, **kwargs
    ) -> Tuple[str, float]:
        return self.classifier_strategy.identify_face(query_embedding, **kwargs)

    def get_classifier_name(self) -> str:
        return self.classifier_strategy.get_name()


def create_classifier_strategy(classifier_type: str) -> ClassifierStrategy:
    classifier_type_lower = classifier_type.lower()

    if classifier_type_lower in ["nn", "neural_network", "neuralnetwork"]:
        return NeuralNetworkClassifierStrategy()
    elif classifier_type_lower == "cnn":
        return CNNClassifierStrategy()
    elif classifier_type_lower in ["cnn_embedding", "embedding", "triplet"]:
        return CNNEmbeddingClassifierStrategy()
    else:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}. Use 'nn', 'cnn', or 'cnn_embedding'"
        )
