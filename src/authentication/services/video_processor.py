import cv2
import os
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional
import subprocess
import numpy as np
from PIL import Image


class VideoProcessor:
    FACIAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "models" / "facial" / "data"
    VOICE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "models" / "voice" / "data"
    REQUIRED_FRAMES = 100
    AUDIO_SEGMENT_DURATION = 3
    MIN_AUDIO_SEGMENTS = 10
    MAX_AUDIO_SEGMENTS = 15
    MIN_VIDEO_DURATION_SECONDS = 30
    
    def __init__(self):
        self.FACIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._mtcnn = None
    
    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                self._mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
            except ImportError:
                raise ValueError("FaceNet not available. Please install facenet-pytorch")
        return self._mtcnn
    
    def _extract_face_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        mtcnn = self._get_mtcnn()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        boxes, probs = mtcnn.detect(frame_pil)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        best_box_idx = np.argmax(probs)
        box = boxes[best_box_idx]
        
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        return face_crop
    
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
        
        frame_paths = self._extract_frames(video_path, user_facial_dir, existing_frames)
        audio_paths = self._extract_audio(video_path, user_voice_dir, existing_audio)
        
        return frame_paths, audio_paths
    
    def _extract_frames(
        self, 
        video_path: str, 
        output_dir: Path, 
        start_index: int
    ) -> List[str]:
        temp_mp4 = None
        
        try:
            if video_path.endswith('.webm'):
                temp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_mp4.close()
                
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-y', temp_mp4.name
                ], check=True, capture_output=True)
                
                video_path = temp_mp4.name
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el archivo de video")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 30
            
            video_duration = total_frames / fps
            
            if video_duration < self.MIN_VIDEO_DURATION_SECONDS:
                cap.release()
                raise ValueError(f"El video es demasiado corto. Se requiere un video de al menos {self.MIN_VIDEO_DURATION_SECONDS} segundos para extraer {self.REQUIRED_FRAMES} frames y {self.MIN_AUDIO_SEGMENTS} segmentos de audio de {self.AUDIO_SEGMENT_DURATION} segundos cada uno. Tu video tiene {video_duration:.1f} segundos. Por favor, graba un video de al menos {self.MIN_VIDEO_DURATION_SECONDS} segundos.")
            
            if total_frames < self.REQUIRED_FRAMES:
                cap.release()
                raise ValueError(f"El video no tiene suficientes frames. Se requieren al menos {self.REQUIRED_FRAMES} frames, pero el video solo tiene {total_frames} frames. Por favor, graba un video más largo.")
            
            frame_interval = max(1, total_frames // self.REQUIRED_FRAMES)
            
            frame_paths = []
            saved_count = start_index
            frames_to_extract = self.REQUIRED_FRAMES
            frames_attempted = 0
            max_attempts = total_frames
            
            i = 0
            while len(frame_paths) < frames_to_extract and frames_attempted < max_attempts:
                frame_index = i * frame_interval
                if frame_index >= total_frames:
                    frame_index = frames_attempted % total_frames
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                ret, frame = cap.read()
                if not ret:
                    frames_attempted += 1
                    i += 1
                    continue
                
                face_crop = self._extract_face_from_frame(frame)
                
                if face_crop is not None:
                    saved_count += 1
                    frame_path = output_dir / f"{saved_count}.png"
                    cv2.imwrite(str(frame_path), face_crop)
                    frame_paths.append(str(frame_path))
                
                frames_attempted += 1
                i += 1
            
            cap.release()
            
            if len(frame_paths) != self.REQUIRED_FRAMES:
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.unlink(frame_path)
                
                raise ValueError(f"No se pudieron extraer exactamente {self.REQUIRED_FRAMES} frames del video. Solo se extrajeron {len(frame_paths)} frames. Por favor, intenta grabar nuevamente.")
            
            return frame_paths
        
        finally:
            if temp_mp4 and os.path.exists(temp_mp4.name):
                os.unlink(temp_mp4.name)
    
    def _extract_audio(
        self, 
        video_path: str, 
        output_dir: Path, 
        start_index: int
    ) -> List[str]:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        try:
            result = subprocess.run([
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', temp_audio.name
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise ValueError(f"Error al extraer el audio: {result.stderr}")
            
            if not os.path.exists(temp_audio.name) or os.path.getsize(temp_audio.name) == 0:
                raise ValueError("El archivo de audio extraído está vacío o no existe")
            
            audio_paths = self._split_audio(temp_audio.name, output_dir, start_index)
            
            return audio_paths
        
        except Exception as e:
            raise ValueError(f"Error al extraer el audio: {str(e)}")
        
        finally:
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
    
    def _split_audio(
        self, 
        audio_path: str, 
        output_dir: Path, 
        start_index: int
    ) -> List[str]:
        import soundfile as sf
        
        audio_data, sample_rate = sf.read(audio_path)
        
        segment_samples = int(self.AUDIO_SEGMENT_DURATION * sample_rate)
        total_samples = len(audio_data)
        audio_duration = total_samples / sample_rate
        
        audio_paths = []
        segment_count = start_index
        
        min_segment_length = int(sample_rate * self.AUDIO_SEGMENT_DURATION * 0.9)
        
        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            segment = audio_data[start:end]
            
            if len(segment) >= min_segment_length:
                segment_count += 1
                segment_path = output_dir / f"{segment_count}.wav"
                sf.write(str(segment_path), segment, sample_rate)
                audio_paths.append(str(segment_path))
        
        if not audio_paths or len(audio_paths) < self.MIN_AUDIO_SEGMENTS:
            raise ValueError(f"No se extrajeron suficientes segmentos de audio del video. Se necesitan al menos {self.MIN_AUDIO_SEGMENTS} segmentos de {self.AUDIO_SEGMENT_DURATION} segundos cada uno (mínimo {self.MIN_AUDIO_SEGMENTS * self.AUDIO_SEGMENT_DURATION} segundos de audio), pero solo se extrajeron {len(audio_paths)} segmentos de {audio_duration:.1f} segundos de audio. Por favor, graba un video más largo hablando continuamente.")
        
        if len(audio_paths) > self.MAX_AUDIO_SEGMENTS:
            raise ValueError(f"El video es demasiado largo. Se extrajeron {len(audio_paths)} segmentos, pero el máximo permitido es {self.MAX_AUDIO_SEGMENTS}. Por favor, graba un video de máximo {self.MAX_AUDIO_SEGMENTS * self.AUDIO_SEGMENT_DURATION} segundos.")
        
        return audio_paths
    
    def process_registration_video(
        self, 
        video_file, 
        username: str
    ) -> Tuple[int, int]:
        temp_video = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        
        try:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video.close()
            
            frame_paths, audio_paths = self.extract_frames_and_audio(
                temp_video.name, 
                username
            )
            
            return len(frame_paths), len(audio_paths)
        
        finally:
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
    
    def process_authentication_video(
        self, 
        video_file
    ) -> Tuple[str, str]:
        temp_video = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        temp_frame = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_mp4 = None
        
        try:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video.close()
            
            temp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_mp4.close()
            
            subprocess.run([
                'ffmpeg', '-i', temp_video.name,
                '-c:v', 'libx264', '-preset', 'fast',
                '-y', temp_mp4.name
            ], check=True, capture_output=True)
            
            cap = cv2.VideoCapture(temp_mp4.name)
            
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el archivo de video")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                raise ValueError("El video no contiene frames")
            
            face_crop = None
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
                
                face_crop = self._extract_face_from_frame(frame)
                
                if face_crop is not None:
                    break
            
            cap.release()
            
            if face_crop is None:
                raise ValueError("No se detectó un rostro en el video. Por favor, asegúrate de que tu cara esté completamente visible y bien iluminada.")
            
            cv2.imwrite(temp_frame.name, face_crop)
            
            result = subprocess.run([
                'ffmpeg', '-i', temp_video.name,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', temp_audio.name
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise ValueError(f"Error al extraer el audio: {result.stderr}")
            
            return temp_frame.name, temp_audio.name
        
        except Exception as e:
            if os.path.exists(temp_frame.name):
                os.unlink(temp_frame.name)
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            raise e
        
        finally:
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
            if temp_mp4 and os.path.exists(temp_mp4.name):
                os.unlink(temp_mp4.name)

