import subprocess
import tempfile
import os
from typing import Optional


class VideoConverter:
    @staticmethod
    def convert_webm_to_mp4(webm_path: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            temp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_mp4.close()
            output_path = temp_mp4.name
        
        subprocess.run([
            'ffmpeg', '-i', webm_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-y', output_path
        ], check=True, capture_output=True)
        
        return output_path

