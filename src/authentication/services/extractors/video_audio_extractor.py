import os
import tempfile
from pathlib import Path
from typing import List
import soundfile as sf

from ...biometrics.audio_processor import AudioProcessor


class VideoAudioExtractor:
    AUDIO_SEGMENT_DURATION = 3
    MIN_AUDIO_SEGMENTS = 10
    MAX_AUDIO_SEGMENTS = 15
    MIN_AUDIO_DURATION_SECONDS = 3
    
    def __init__(self):
        pass
    
    def extract_for_registration(
        self,
        video_path: str,
        output_dir: Path,
        start_index: int
    ) -> List[str]:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        try:
            AudioProcessor.extract_audio_with_noise_reduction(video_path, temp_audio.name)
            audio_paths = self._split_audio(temp_audio.name, output_dir, start_index)
            return audio_paths
        
        except Exception as e:
            raise ValueError(f"Error al extraer el audio: {str(e)}")
        
        finally:
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
    
    def extract_for_authentication(
        self,
        video_path: str,
        output_path: str
    ) -> None:
        AudioProcessor.extract_audio_with_noise_reduction(video_path, output_path)
        self._validate_audio_duration(output_path)
    
    def _split_audio(
        self,
        audio_path: str,
        output_dir: Path,
        start_index: int
    ) -> List[str]:
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
        
        self._validate_audio_segments(audio_paths, audio_duration)
        
        return audio_paths
    
    def _validate_audio_segments(
        self,
        audio_paths: List[str],
        audio_duration: float
    ) -> None:
        if not audio_paths or len(audio_paths) < self.MIN_AUDIO_SEGMENTS:
            raise ValueError(
                f"No se extrajeron suficientes segmentos de audio del video. "
                f"Se necesitan al menos {self.MIN_AUDIO_SEGMENTS} segmentos de "
                f"{self.AUDIO_SEGMENT_DURATION} segundos cada uno "
                f"(mínimo {self.MIN_AUDIO_SEGMENTS * self.AUDIO_SEGMENT_DURATION} segundos de audio), "
                f"pero solo se extrajeron {len(audio_paths)} segmentos de {audio_duration:.1f} segundos de audio. "
                f"Por favor, graba un video más largo hablando continuamente."
            )
        
        if len(audio_paths) > self.MAX_AUDIO_SEGMENTS:
            raise ValueError(
                f"El video es demasiado largo. Se extrajeron {len(audio_paths)} segmentos, "
                f"pero el máximo permitido es {self.MAX_AUDIO_SEGMENTS}. "
                f"Por favor, graba un video de máximo {self.MAX_AUDIO_SEGMENTS * self.AUDIO_SEGMENT_DURATION} segundos."
            )
    
    def _validate_audio_duration(self, audio_path: str) -> None:
        audio_data, sample_rate = sf.read(audio_path)
        audio_duration = len(audio_data) / sample_rate
        
        if audio_duration < self.MIN_AUDIO_DURATION_SECONDS:
            raise ValueError(
                f"El audio extraído es demasiado corto. Se requiere al menos "
                f"{self.MIN_AUDIO_DURATION_SECONDS} segundos de audio para autenticación. "
                f"El audio extraído tiene {audio_duration:.1f} segundos. "
                f"Por favor, graba un video más largo hablando continuamente."
            )

