from typing import Tuple, List, Optional
import numpy as np
import cv2
import os
import tempfile

from .extractors import VideoFrameExtractor, VideoAudioExtractor
from ..biometrics.embedding_extractors import FacialEmbeddingExtractor, VoiceEmbeddingExtractor


class BiometricMediaProcessor:
    def __init__(self):
        self.video_frame_extractor = VideoFrameExtractor()
        self.video_audio_extractor = VideoAudioExtractor()
        self.facial_embedding_extractor = FacialEmbeddingExtractor()
        self.voice_embedding_extractor = VoiceEmbeddingExtractor()
    
    def extract_facial_embedding_from_video(
        self,
        video_path: str,
        min_duration_seconds: float = 3.0
    ) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        face_crop = self.video_frame_extractor.extract_single_face(
            video_path,
            min_duration_seconds
        )
        
        if face_crop is None:
            return None, None
        
        temp_frame = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_frame.close()
        
        try:
            cv2.imwrite(temp_frame.name, face_crop)
            embedding = self.facial_embedding_extractor.extract(temp_frame.name)
            return face_crop, embedding
        finally:
            if os.path.exists(temp_frame.name):
                os.unlink(temp_frame.name)
    
    def extract_voice_embedding_from_video(
        self,
        video_path: str,
        output_audio_path: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[List[float]]]:
        temp_audio = None
        
        try:
            if output_audio_path is None:
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio.close()
                output_audio_path = temp_audio.name
            
            self.video_audio_extractor.extract_for_authentication(
                video_path,
                output_audio_path
            )
            
            embedding = self.voice_embedding_extractor.extract(output_audio_path)
            
            return output_audio_path, embedding
        
        except Exception as e:
            if temp_audio and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            raise e
    
    def extract_facial_embeddings_from_frames(
        self,
        frame_paths: List[str]
    ) -> List[List[float]]:
        embeddings = []
        for frame_path in frame_paths:
            try:
                embedding = self.facial_embedding_extractor.extract(frame_path)
                embeddings.append(embedding)
            except Exception:
                continue
        return embeddings
    
    def extract_voice_embeddings_from_audio_files(
        self,
        audio_paths: List[str]
    ) -> List[List[float]]:
        embeddings = []
        for audio_path in audio_paths:
            try:
                embedding = self.voice_embedding_extractor.extract(audio_path)
                embeddings.append(embedding)
            except Exception:
                continue
        return embeddings

