import cv2
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

from ...biometrics.face_detector import FaceDetector
from ...services.video.video_converter import VideoConverter


class VideoFrameExtractor:
    REQUIRED_FRAMES = 100
    MIN_VIDEO_DURATION_SECONDS = 30
    DURATION_TOLERANCE_SECONDS = 0.2
    
    def __init__(self):
        pass
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: Path,
        start_index: int
    ) -> List[str]:
        temp_mp4_path = None
        
        try:
            if video_path.endswith('.webm'):
                temp_mp4_path = VideoConverter.convert_webm_to_mp4(video_path)
                video_path = temp_mp4_path
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el archivo de video")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self._get_fps(cap)
            video_duration = total_frames / fps
            
            self._validate_video_duration(video_duration, total_frames)
            
            frame_paths = self._extract_faces_from_video(
                cap, total_frames, output_dir, start_index
            )
            
            cap.release()
            
            if len(frame_paths) != self.REQUIRED_FRAMES:
                self._cleanup_extracted_frames(frame_paths)
                raise ValueError(
                    f"No se pudieron extraer exactamente {self.REQUIRED_FRAMES} frames del video. "
                    f"Solo se extrajeron {len(frame_paths)} frames. Por favor, intenta grabar nuevamente."
                )
            
            return frame_paths
        
        finally:
            if temp_mp4_path and os.path.exists(temp_mp4_path):
                os.unlink(temp_mp4_path)
    
    def extract_single_face(
        self,
        video_path: str,
        min_duration_seconds: float = 3.0
    ) -> Optional[np.ndarray]:
        temp_mp4_path = None
        
        try:
            if video_path.endswith('.webm'):
                temp_mp4_path = VideoConverter.convert_webm_to_mp4(video_path)
                video_path = temp_mp4_path
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el archivo de video")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                raise ValueError("El video no contiene frames")
            
            fps = self._get_fps(cap)
            video_duration = frame_count / fps
            
            effective_min_duration = min_duration_seconds - self.DURATION_TOLERANCE_SECONDS
            
            if video_duration < effective_min_duration:
                cap.release()
                raise ValueError(
                    f"El video es demasiado corto para autenticación. Se requiere un video de al menos "
                    f"{min_duration_seconds} segundos de duración. Tu video tiene {video_duration:.1f} segundos. "
                    f"Por favor, graba un video más largo hablando continuamente."
                )
            
            face_crop = self._find_face_in_video(cap, frame_count)
            cap.release()
            
            return face_crop
        
        finally:
            if temp_mp4_path and os.path.exists(temp_mp4_path):
                os.unlink(temp_mp4_path)
    
    def _get_fps(self, cap: cv2.VideoCapture) -> float:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            return 30.0
        return fps
    
    def _validate_video_duration(self, video_duration: float, total_frames: int) -> None:
        if video_duration < self.MIN_VIDEO_DURATION_SECONDS:
            raise ValueError(
                f"El video es demasiado corto. Se requiere un video de al menos "
                f"{self.MIN_VIDEO_DURATION_SECONDS} segundos para extraer {self.REQUIRED_FRAMES} frames "
                f"y segmentos de audio. Tu video tiene {video_duration:.1f} segundos. "
                f"Por favor, graba un video de al menos {self.MIN_VIDEO_DURATION_SECONDS} segundos."
            )
        
        if total_frames < self.REQUIRED_FRAMES:
            raise ValueError(
                f"El video no tiene suficientes frames. Se requieren al menos "
                f"{self.REQUIRED_FRAMES} frames, pero el video solo tiene {total_frames} frames. "
                f"Por favor, graba un video más largo."
            )
    
    def _extract_faces_from_video(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        output_dir: Path,
        start_index: int
    ) -> List[str]:
        frame_interval = max(1, total_frames // self.REQUIRED_FRAMES)
        frame_paths = []
        saved_count = start_index
        frames_attempted = 0
        max_attempts = total_frames
        
        i = 0
        while len(frame_paths) < self.REQUIRED_FRAMES and frames_attempted < max_attempts:
            frame_index = i * frame_interval
            if frame_index >= total_frames:
                frame_index = frames_attempted % total_frames
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                frames_attempted += 1
                i += 1
                continue
            
            face_crop = FaceDetector.extract_face_from_frame(frame)
            
            if face_crop is not None:
                saved_count += 1
                frame_path = output_dir / f"{saved_count}.png"
                cv2.imwrite(str(frame_path), face_crop)
                frame_paths.append(str(frame_path))
            
            frames_attempted += 1
            i += 1
        
        return frame_paths
    
    def _find_face_in_video(
        self,
        cap: cv2.VideoCapture,
        frame_count: int
    ) -> Optional[np.ndarray]:
        max_attempts = min(20, frame_count)
        frame_indices = [
            int(frame_count * 0.25),
            int(frame_count * 0.5),
            int(frame_count * 0.75),
        ]
        
        for i in range(max_attempts):
            if i < len(frame_indices):
                frame_index = frame_indices[i]
            else:
                frame_index = int((i / max_attempts) * frame_count)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            face_crop = FaceDetector.extract_face_from_frame(frame)
            
            if face_crop is not None:
                return face_crop
        
        return None
    
    def _cleanup_extracted_frames(self, frame_paths: List[str]) -> None:
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.unlink(frame_path)

