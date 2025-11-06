from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch
import warnings
from PIL import Image

from .config import BiometricConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class EmbeddingExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> Union[List[float], np.ndarray]:
        pass


class VoiceEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self):
        self._encoder = None
        self._load_encoder()
    
    def _load_encoder(self) -> None:
        if self._encoder is None:
            try:
                from speechbrain.inference import EncoderClassifier
                self._encoder = EncoderClassifier.from_hparams(
                    source=BiometricConfig.VOICE_ENCODER_MODEL
                )
            except ImportError:
                raise ValueError("Voice encoder not available. Please install speechbrain.")
    
    def extract(self, audio_path: str) -> List[float]:
        import os
        import tempfile
        if self._encoder is None:
            raise ValueError("Voice encoder not initialized")
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        signal = None
        sample_rate = None
        error_messages = []
        converted_path = None
        
        audio_path_lower = audio_path.lower()
        is_webm = audio_path_lower.endswith('.webm') or audio_path_lower.endswith('.weba')
        
        if is_webm:
            try:
                import subprocess
                result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
                if result.returncode == 0:
                    converted_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                    conv_result = subprocess.run(
                        ['ffmpeg', '-i', audio_path, '-ac', '1', '-ar', '16000', '-f', 'wav', '-y', converted_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if conv_result.returncode == 0:
                        audio_path = converted_path
                    else:
                        error_messages.append(f"ffmpeg conversion failed: {conv_result.stderr[:200]}")
                else:
                    error_messages.append("ffmpeg not found - will try librosa directly")
            except subprocess.TimeoutExpired:
                error_messages.append("ffmpeg conversion timed out")
            except Exception as e:
                error_messages.append(f"ffmpeg conversion error: {str(e)}")
        
        try:
            import librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                    signal = torch.from_numpy(audio_data).float()
                except Exception as librosa_err:
                    error_messages.append(f"librosa error: {str(librosa_err)}")
                    raise
        except Exception as e:
            if "librosa error" not in str(e):
                error_messages.append(f"librosa import/load error: {str(e)}")
        
        if signal is None:
            try:
                import torchaudio
                try:
                    audio_data, sample_rate = torchaudio.load(audio_path)
                    if audio_data.dim() > 1 and audio_data.shape[0] > 1:
                        audio_data = torch.mean(audio_data, dim=0, keepdim=True)
                    signal = audio_data.float()
                except Exception as e:
                    error_messages.append(f"torchaudio load error: {str(e)}")
            except ImportError:
                error_messages.append("torchaudio not available")
        
        if signal is None:
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path, dtype='float32')
                signal = torch.from_numpy(audio_data).float()
            except Exception as e:
                error_messages.append(f"soundfile error: {str(e)}")
        
        if converted_path and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except Exception:
                pass
        
        if signal is None:
            raise ValueError(f"Could not read audio file '{audio_path}'. Tried multiple methods. Errors: {'; '.join(error_messages)}")
        
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        if signal.shape[1] == 0:
            raise ValueError(f"Audio file '{audio_path}' appears to be empty or corrupted (signal length: 0)")
        
        target_sample_rate = 16000
        if sample_rate and sample_rate != target_sample_rate:
            try:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                signal = resampler(signal)
                sample_rate = target_sample_rate
            except Exception:
                pass
        
        emb = self._encoder.encode_batch(signal)
        embedding = emb.squeeze().detach().numpy()
        
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding


class FacialEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self):
        self._mtcnn = None
        self._resnet = None
        self._load_models()
    
    def _load_models(self) -> None:
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError:
            raise ValueError("FaceNet not available. Please install facenet-pytorch")
        
        if self._mtcnn is None:
            self._mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
        
        if self._resnet is None:
            self._resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    def extract(self, image_path: str) -> List[float]:
        if self._mtcnn is None or self._resnet is None:
            raise ValueError("Facial models not initialized")
        
        img = Image.open(image_path).convert('RGB')
        img_cropped = self._mtcnn(img)
        
        if img_cropped is None:
            raise ValueError(f"No faces found in image {image_path}")
        
        img_cropped = img_cropped.unsqueeze(0)
        with torch.no_grad():
            embedding = self._resnet(img_cropped)
        
        embedding_numpy = embedding.squeeze().numpy()
        return embedding_numpy.tolist() if hasattr(embedding_numpy, 'tolist') else embedding_numpy

