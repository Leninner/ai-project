from typing import Tuple
from django.core.files.uploadedfile import UploadedFile
from ..models import User
from ..utils import extract_voice_embedding, extract_image_embedding
from .file_handler import FileHandler


class RegistrationService:
    def __init__(self):
        self.file_handler = FileHandler()
    
    def register_user(
        self,
        email: str,
        password: str,
        name: str,
        voice_file: UploadedFile,
        image_file: UploadedFile
    ) -> Tuple[bool, User, str]:
        if User.objects.filter(email=email).exists():
            return False, None, "Email already registered"
        
        voice_temp_path = None
        image_temp_path = None
        
        try:
            voice_temp_path = self.file_handler.save_voice_file(voice_file)
            image_temp_path = self.file_handler.save_image_file(image_file)
            
            voice_embedding = extract_voice_embedding(voice_temp_path)
            image_embedding = extract_image_embedding(image_temp_path)
            
            user = User.objects.create_user(
                email=email,
                password=password,
                name=name
            )
            user.voice_embeddings.append(voice_embedding)
            user.image_embeddings.append(image_embedding)
            user.save()
            
            return True, user, "Registration successful"
        
        except Exception as e:
            return False, None, f"Registration failed: {str(e)}"
        
        finally:
            self.file_handler.cleanup_files(voice_temp_path, image_temp_path)

