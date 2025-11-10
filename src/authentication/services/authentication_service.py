from typing import Tuple, Optional
from django.core.files.uploadedfile import UploadedFile
from ..models import User
from ..validators import BiometricValidator
from .video_processor import VideoProcessor
from .file_handler import FileHandler
import logging

logger = logging.getLogger(__name__)


class AuthenticationService:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.biometric_validator = BiometricValidator()
        self.file_handler = FileHandler()
    
    def authenticate(
        self,
        username: str,
        video_file: UploadedFile,
        request
    ) -> Tuple[bool, Optional[User], str]:
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return False, None, "Usuario no encontrado"
        
        image_temp_path = None
        voice_temp_path = None
        
        try:
            image_temp_path, voice_temp_path = self.video_processor.process_authentication_video(
                video_file
            )
            
            is_valid, error_message = self.biometric_validator.validate_all(
                voice_temp_path,
                image_temp_path,
                username
            )
            
            if not is_valid:
                logger.warning(f"Biometric validation failed for {username}: {error_message}")
                return False, None, error_message
            
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            logger.info(f"User {username} authenticated successfully")
            return True, user, "Autenticación exitosa"
        
        except Exception as e:
            logger.error(f"Authentication failed for {username}: {str(e)}")
            return False, None, f"Error al autenticar: {str(e)}"
        
        finally:
            self.file_handler.cleanup_files(image_temp_path, voice_temp_path)

