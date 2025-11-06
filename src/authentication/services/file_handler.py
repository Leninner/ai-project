import os
import tempfile
from typing import Tuple, Optional
from contextlib import contextmanager


class FileHandler:
    @staticmethod
    @contextmanager
    def save_uploaded_file(uploaded_file, suffix: str = ''):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.close()
            yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    @staticmethod
    def save_voice_file(voice_file) -> str:
        original_filename = voice_file.name
        suffix = '.webm' if '.webm' in original_filename.lower() else '.wav'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in voice_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def save_image_file(image_file) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        for chunk in image_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def cleanup_file(file_path: str) -> None:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    
    @staticmethod
    def cleanup_files(*file_paths: str) -> None:
        for file_path in file_paths:
            FileHandler.cleanup_file(file_path)

