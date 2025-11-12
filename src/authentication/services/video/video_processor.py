import cv2
import os
import tempfile
from pathlib import Path
from typing import Tuple, List
import numpy as np

from ..extractors import VideoFrameExtractor, VideoAudioExtractor
from .video_file_handler import VideoFileHandler
from ..biometric_media_processor import BiometricMediaProcessor


class VideoProcessor:
    FACIAL_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "models" / "facial" / "data"
    VOICE_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "models" / "voice" / "data"
    
    def __init__(self):
        self.FACIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.frame_extractor = VideoFrameExtractor()
        self.audio_extractor = VideoAudioExtractor()
        self.file_handler = VideoFileHandler()
        self.media_processor = BiometricMediaProcessor()
    
    def process_registration_video(
        self,
        video_file,
        username: str
    ) -> Tuple[int, int]:
        temp_video_path = None
        
        try:
            temp_video_path = self.file_handler.save_uploaded_file(video_file)
            
            frame_paths, audio_paths = self.extract_frames_and_audio(
                temp_video_path,
                username
            )
            
            return len(frame_paths), len(audio_paths)
        
        finally:
            self.file_handler.cleanup_file(temp_video_path)
    
    def process_authentication_video(
        self,
        video_file
    ) -> Tuple[str, str]:
        temp_video_path = None
        temp_frame_path = None
        temp_audio_path = None
        
        try:
            temp_video_path = self.file_handler.save_uploaded_file(video_file)
            temp_frame_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            temp_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            
            face_crop, _ = self.media_processor.extract_facial_embedding_from_video(
                temp_video_path,
                min_duration_seconds=self.audio_extractor.MIN_AUDIO_DURATION_SECONDS
            )
            
            if face_crop is None:
                raise ValueError(
                    "No se detectó un rostro en el video. Por favor, asegúrate de que tu cara "
                    "esté completamente visible y bien iluminada."
                )
            
            cv2.imwrite(temp_frame_path, face_crop)
            
            _, _ = self.media_processor.extract_voice_embedding_from_video(
                temp_video_path,
                output_audio_path=temp_audio_path
            )
            
            return temp_frame_path, temp_audio_path
        
        except Exception as e:
            self.file_handler.cleanup_files(temp_frame_path, temp_audio_path)
            raise e
        
        finally:
            self.file_handler.cleanup_file(temp_video_path)
    
    def extract_frames_and_audio(
        self,
        video_path: str,
        username: str
    ) -> Tuple[List[str], List[str]]:
        user_facial_dir = self.FACIAL_DATA_DIR / username
        user_voice_dir = self.VOICE_DATA_DIR / username
        
        user_facial_dir.mkdir(parents=True, exist_ok=True)
        user_voice_dir.mkdir(parents=True, exist_ok=True)
        
        existing_frames = len(list(user_facial_dir.glob("*.png")))
        existing_audio = len(list(user_voice_dir.glob("*.wav")))
        
        frame_paths = self.frame_extractor.extract_frames(
            video_path,
            user_facial_dir,
            existing_frames
        )
        
        audio_paths = self.audio_extractor.extract_for_registration(
            video_path,
            user_voice_dir,
            existing_audio
        )
        
        return frame_paths, audio_paths

