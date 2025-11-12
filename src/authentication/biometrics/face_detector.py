from typing import Optional
import numpy as np
from PIL import Image


class FaceDetector:
    _mtcnn = None
    _image_size = 160
    _margin = 0
    _min_face_size = 20
    
    @classmethod
    def get_mtcnn(cls):
        if cls._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                cls._mtcnn = MTCNN(
                    image_size=cls._image_size,
                    margin=cls._margin,
                    min_face_size=cls._min_face_size
                )
            except ImportError:
                raise ValueError("FaceNet not available. Please install facenet-pytorch")
        return cls._mtcnn
    
    @classmethod
    def extract_face_from_image(cls, image: Image.Image) -> Optional[Image.Image]:
        mtcnn = cls.get_mtcnn()
        img_cropped = mtcnn(image)
        return img_cropped
    
    @classmethod
    def extract_face_from_frame(cls, frame: np.ndarray) -> Optional[np.ndarray]:
        mtcnn = cls.get_mtcnn()
        import cv2
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        boxes, probs = mtcnn.detect(frame_pil)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        best_box_idx = np.argmax(probs)
        box = boxes[best_box_idx]
        
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        return face_crop

