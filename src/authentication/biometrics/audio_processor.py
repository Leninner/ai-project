import os
import subprocess
import tempfile
from typing import Optional


class AudioProcessor:
    NOISE_REDUCTION_FILTER = 'highpass=f=80,lowpass=f=8000,afftdn=nr=15'
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CODEC = 'pcm_s16le'
    
    @classmethod
    def extract_audio_with_noise_reduction(
        cls,
        video_path: str,
        output_path: str
    ) -> None:
        result = subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vn', '-af', cls.NOISE_REDUCTION_FILTER,
            '-acodec', cls.CODEC,
            '-ar', str(cls.SAMPLE_RATE), '-ac', str(cls.CHANNELS),
            '-y', output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ValueError(f"Error al extraer el audio: {result.stderr}")
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("El archivo de audio extraído está vacío o no existe")
    
    @classmethod
    def extract_audio_without_noise_reduction(
        cls,
        video_path: str,
        output_path: str
    ) -> None:
        result = subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', cls.CODEC,
            '-ar', str(cls.SAMPLE_RATE), '-ac', str(cls.CHANNELS),
            '-y', output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ValueError(f"Error al extraer el audio: {result.stderr}")
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("El archivo de audio extraído está vacío o no existe")
    

