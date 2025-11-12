import os
import tempfile
from typing import Optional
from django.core.files.uploadedfile import UploadedFile


class VideoFileHandler:
    @staticmethod
    def save_uploaded_file(uploaded_file: UploadedFile, suffix: str = ".webm") -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
        
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def cleanup_file(file_path: Optional[str]) -> None:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    
    @staticmethod
    def cleanup_files(*file_paths: Optional[str]) -> None:
        for file_path in file_paths:
            VideoFileHandler.cleanup_file(file_path)

