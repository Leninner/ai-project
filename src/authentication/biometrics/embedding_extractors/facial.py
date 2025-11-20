from typing import List
import numpy as np
import torch
from PIL import Image

from .base import EmbeddingExtractor
from ..face_detector import FaceDetector


class FacialEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self):
        self._resnet = None
        self._load_models()
    
    def _load_models(self) -> None:
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ValueError("FaceNet not available. Please install facenet-pytorch")
        
        if self._resnet is None:
            self._resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    def extract(self, image_path: str) -> List[float]:
        if self._resnet is None:
            raise ValueError("Facial models not initialized")
        
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        if img_array.shape == (160, 160, 3):
            img_array_normalized = img_array.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array_normalized).permute(2, 0, 1)
        else:
            img_cropped = FaceDetector.extract_face_from_image(img)
            if img_cropped is None:
                raise ValueError("No se detectó un rostro en la imagen. Por favor, asegúrate de que tu cara esté completamente visible y bien iluminada.")
            img_tensor = img_cropped
        
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            embedding = self._resnet(img_tensor)
        
        embedding_numpy = embedding.squeeze().numpy()
        return embedding_numpy.tolist() if hasattr(embedding_numpy, 'tolist') else embedding_numpy

