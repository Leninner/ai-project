import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional
import logging
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)


class FacialPreprocessor:
    IMAGE_SIZE = 160
    MARGIN = 0
    MIN_FACE_SIZE = 20
    
    _mtcnn = None
    
    def __init__(self):
        self._initialize_mtcnn()
    
    @classmethod
    def _initialize_mtcnn(cls):
        if cls._mtcnn is None:
            cls._mtcnn = MTCNN(
                image_size=cls.IMAGE_SIZE,
                margin=cls.MARGIN,
                min_face_size=cls.MIN_FACE_SIZE
            )
    
    @classmethod
    def get_mtcnn(cls):
        if cls._mtcnn is None:
            cls._initialize_mtcnn()
        return cls._mtcnn
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.preprocess_from_pil(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    def preprocess_from_pil(self, image: Image.Image) -> np.ndarray:
        mtcnn = self.get_mtcnn()
        cropped_face_tensor = mtcnn(image)
        
        if cropped_face_tensor is None:
            raise ValueError("No faces detected in image")
        
        face_array = cropped_face_tensor.numpy()
        face_array = np.transpose(face_array, (1, 2, 0))
        face_array = (face_array * 255.0).astype(np.uint8)
        
        return face_array
    
    def resize_image_only(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            resized_image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.LANCZOS)
            return np.array(resized_image)
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {str(e)}")
            raise ValueError(f"Failed to resize image {image_path}: {str(e)}")
    
    def preprocess_cropped_face(self, face_array: np.ndarray) -> np.ndarray:
        try:
            if len(face_array.shape) == 2:
                image = Image.fromarray(face_array, mode='L').convert('RGB')
            elif face_array.shape[2] == 3:
                image = Image.fromarray(face_array, mode='RGB')
            elif face_array.shape[2] == 4:
                image = Image.fromarray(face_array, mode='RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported image array shape: {face_array.shape}")
            
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.LANCZOS)
            face_array = np.array(image)
            
            return face_array
        except Exception as e:
            logger.error(f"Error preprocessing cropped face: {str(e)}")
            raise ValueError(f"Failed to preprocess cropped face: {str(e)}")
    
    def preprocess_from_array(self, image_array: np.ndarray) -> np.ndarray:
        try:
            if len(image_array.shape) == 2:
                image = Image.fromarray(image_array, mode='L').convert('RGB')
            elif image_array.shape[2] == 3:
                image = Image.fromarray(image_array, mode='RGB')
            elif image_array.shape[2] == 4:
                image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported image array shape: {image_array.shape}")
            
            return self.preprocess_from_pil(image)
        except Exception as e:
            logger.error(f"Error preprocessing array: {str(e)}")
            raise ValueError(f"Failed to preprocess image array: {str(e)}")
    
    def preprocess_and_save(
        self,
        source_path: str,
        output_path: str,
        source_is_array: bool = False
    ) -> str:
        try:
            if source_is_array:
                import cv2
                image_array = cv2.imread(source_path)
                if image_array is None:
                    raise ValueError(f"Failed to read image array from {source_path}")
                preprocessed = self.preprocess_from_array(image_array)
            else:
                preprocessed = self.preprocess_image(source_path)
            
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            preprocessed_image = Image.fromarray(preprocessed)
            preprocessed_image.save(output_path)
            
            logger.info(f"Preprocessed and saved image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error preprocessing and saving {source_path} to {output_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess and save image: {str(e)}")
    
    def preprocess_batch(
        self,
        source_dir: str,
        output_dir: str,
        person_name: Optional[str] = None
    ) -> int:
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        if person_name:
            source_path = source_path / person_name
            output_path = output_path / person_name
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        for image_name in os.listdir(source_path):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            source_image_path = source_path / image_name
            output_image_path = output_path / image_name
            
            try:
                self.preprocess_and_save(str(source_image_path), str(output_image_path))
                processed_count += 1
            except Exception as e:
                logger.warning(f"Skipping {source_image_path}: {str(e)}")
                continue
        
        logger.info(f"Processed {processed_count} images from {source_path}")
        return processed_count
    
    def preprocess_all_in_place(self, data_dir: str) -> dict:
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        results = {
            "total_processed": 0,
            "total_failed": 0,
            "by_person": {}
        }
        
        for person_dir in data_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            processed_count = 0
            failed_count = 0
            
            for image_file in person_dir.iterdir():
                if not image_file.is_file():
                    continue
                
                if not image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    continue
                
                try:
                    resized_array = self.resize_image_only(str(image_file))
                    resized_image = Image.fromarray(resized_array)
                    resized_image.save(str(image_file))
                    processed_count += 1
                    logger.debug(f"Resized and replaced: {image_file}")
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to resize {image_file}: {str(e)}")
                    continue
            
            results["by_person"][person_name] = {
                "processed": processed_count,
                "failed": failed_count
            }
            results["total_processed"] += processed_count
            results["total_failed"] += failed_count
            
            logger.info(f"Person '{person_name}': {processed_count} processed, {failed_count} failed")
        
        logger.info(f"Total: {results['total_processed']} processed, {results['total_failed']} failed")
        return results

