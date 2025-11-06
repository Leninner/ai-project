from typing import Tuple, Optional
from django.core.files.uploadedfile import UploadedFile
from django.contrib.auth import login as django_login
from ..models import User
from ..utils import extract_voice_embedding, extract_image_embedding
from ..validators import BiometricValidator
from .file_handler import FileHandler


class AuthenticationService:
    def __init__(self):
        self.file_handler = FileHandler()
        self.biometric_validator = BiometricValidator()
    
    def authenticate(
        self,
        email: str,
        password: str,
        voice_file: UploadedFile,
        image_file: UploadedFile,
        request
    ) -> Tuple[bool, Optional[User], str]:
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return False, None, "Invalid credentials"
        
        if not user.check_password(password):
            return False, None, "Invalid credentials"
        
        voice_temp_path = None
        image_temp_path = None
        
        try:
            voice_temp_path = self.file_handler.save_voice_file(voice_file)
            image_temp_path = self.file_handler.save_image_file(image_file)
            
            is_valid, error_message = self.biometric_validator.validate_all(
                voice_temp_path,
                image_temp_path,
                user.name
            )
            
            if not is_valid:
                return False, None, error_message
            
            current_voice_embedding = extract_voice_embedding(voice_temp_path)
            current_image_embedding = extract_image_embedding(image_temp_path)
            
            user.voice_embeddings.append(current_voice_embedding)
            user.image_embeddings.append(current_image_embedding)
            user.save()
            
            django_login(request, user)
            return True, user, "Authentication successful"
        
        except Exception as e:
            return False, None, f"Login failed: {str(e)}"
        
        finally:
            self.file_handler.cleanup_files(voice_temp_path, image_temp_path)

