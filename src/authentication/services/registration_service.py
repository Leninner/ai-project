from typing import Tuple, Optional, List
from django.core.files.uploadedfile import UploadedFile
from pathlib import Path
import shutil
from ..models import User
from .video_processor import VideoProcessor
from .training_service import TrainingService
from ..biometrics.embedding_extractors import FacialEmbeddingExtractor
import logging

logger = logging.getLogger(__name__)


class RegistrationService:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.facial_extractor = FacialEmbeddingExtractor()
    
    def _validate_frames_contain_faces(self, frame_paths: List[str]) -> Tuple[bool, str]:
        if not frame_paths:
            return False, "No se extrajeron frames del video"
        
        validation_sample_size = min(20, len(frame_paths))
        step = max(1, len(frame_paths) // validation_sample_size)
        sample_frames = [frame_paths[i] for i in range(0, len(frame_paths), step)][:validation_sample_size]
        
        failed_frames = []
        for frame_path in sample_frames:
            try:
                self.facial_extractor.extract(frame_path)
            except ValueError as e:
                failed_frames.append((frame_path, str(e)))
                logger.warning(f"Face validation failed for frame {frame_path}: {str(e)}")
        
        if failed_frames:
            error_msg = "No se detectaron rostros en algunas imágenes del video. Por favor, asegúrate de que tu cara esté completamente visible, bien iluminada y mirando hacia la cámara durante toda la grabación."
            return False, error_msg
        
        logger.info(f"Face validation passed for {len(sample_frames)} sample frames")
        return True, "Validation successful"
    
    def _cleanup_user_data(self, username: str) -> None:
        facial_dir = self.video_processor.FACIAL_DATA_DIR / username
        voice_dir = self.video_processor.VOICE_DATA_DIR / username
        
        if facial_dir.exists():
            shutil.rmtree(facial_dir)
            logger.info(f"Cleaned up facial data for {username}")
        
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
            logger.info(f"Cleaned up voice data for {username}")
    
    def register_user(
        self,
        username: str,
        video_file: UploadedFile
    ) -> Tuple[bool, Optional[User], str]:
        if User.objects.filter(username=username).exists():
            return False, None, "Username already registered"
        
        try:
            frame_count, audio_count = self.video_processor.process_registration_video(
                video_file,
                username
            )
            
            logger.info(f"Extracted {frame_count} frames and {audio_count} audio segments for user {username}")
            
            user_facial_dir = self.video_processor.FACIAL_DATA_DIR / username
            frame_paths = [str(p) for p in sorted(user_facial_dir.glob("*.png"))]
            
            is_valid, validation_message = self._validate_frames_contain_faces(frame_paths)
            
            if not is_valid:
                self._cleanup_user_data(username)
                logger.warning(f"Face validation failed for {username}: {validation_message}")
                return False, None, validation_message
            
            user = User.objects.create(username=username)
            
            TrainingService.trigger_async_training()
            
            return True, user, f"Registration successful. Extracted {frame_count} facial samples and {audio_count} voice samples."
        
        except Exception as e:
            self._cleanup_user_data(username)
            logger.error(f"Registration failed for {username}: {str(e)}")
            return False, None, f"Registration failed: {str(e)}"

