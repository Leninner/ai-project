import subprocess
import threading
from pathlib import Path
from typing import Optional
import logging

from ..biometrics.config import BiometricConfig

logger = logging.getLogger(__name__)


class TrainingService:
    FACIAL_PIPELINE = Path(__file__).parent.parent.parent.parent / "models" / "facial" / "pipeline.py"
    VOICE_PIPELINE = Path(__file__).parent.parent.parent.parent / "models" / "voice" / "pipeline.py"
    
    _training_lock = threading.Lock()
    _is_training = False
    
    @classmethod
    def is_training(cls) -> bool:
        with cls._training_lock:
            return cls._is_training
    
    @classmethod
    def trigger_async_training(cls, facial_classifier_type: Optional[str] = None, voice_classifier_type: Optional[str] = None) -> None:
        if cls.is_training():
            logger.warning("Training already in progress, skipping new training request")
            return
        
        facial_type = facial_classifier_type or BiometricConfig.FACIAL_CLASSIFIER_TYPE
        voice_type = voice_classifier_type or BiometricConfig.VOICE_CLASSIFIER_TYPE
        
        thread = threading.Thread(target=cls._train_models, args=(facial_type, voice_type), daemon=True)
        thread.start()
    
    @classmethod
    def _train_models(cls, facial_classifier_type: str, voice_classifier_type: str) -> None:
        with cls._training_lock:
            if cls._is_training:
                return
            cls._is_training = True
        
        try:
            logger.info(f"Starting facial recognition model training with {facial_classifier_type.upper()} classifier...")
            cls._train_facial_model(facial_classifier_type)
            
            logger.info(f"Starting voice recognition model training with {voice_classifier_type.upper()} classifier...")
            cls._train_voice_model(voice_classifier_type)
            
            logger.info("Model training completed successfully")
        
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
        
        finally:
            with cls._training_lock:
                cls._is_training = False
    
    @classmethod
    def _train_facial_model(cls, classifier_type: str = None) -> None:
        if not cls.FACIAL_PIPELINE.exists():
            raise FileNotFoundError(f"Facial pipeline not found: {cls.FACIAL_PIPELINE}")
        
        classifier_type = classifier_type or BiometricConfig.FACIAL_CLASSIFIER_TYPE
        
        if classifier_type not in ['nn', 'cnn']:
            raise ValueError(f"Invalid classifier type: {classifier_type}. Must be 'nn', or 'cnn'")
        
        result = subprocess.run(
            ['python', str(cls.FACIAL_PIPELINE), '--classifier', classifier_type],
            cwd=cls.FACIAL_PIPELINE.parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.error(f"Facial training stderr: {result.stderr}")
            raise RuntimeError(f"Facial model training failed: {result.stderr}")
        
        logger.info(f"Facial training output: {result.stdout}")
    
    @classmethod
    def _train_voice_model(cls, classifier_type: str = None) -> None:
        if not cls.VOICE_PIPELINE.exists():
            raise FileNotFoundError(f"Voice pipeline not found: {cls.VOICE_PIPELINE}")
        
        classifier_type = classifier_type or BiometricConfig.VOICE_CLASSIFIER_TYPE
        
        if classifier_type not in ['nn']:
            raise ValueError(f"Invalid classifier type: {classifier_type}. Must be 'nn'")
        
        result = subprocess.run(
            ['python', str(cls.VOICE_PIPELINE), '--classifier', classifier_type],
            cwd=cls.VOICE_PIPELINE.parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.error(f"Voice training stderr: {result.stderr}")
            raise RuntimeError(f"Voice model training failed: {result.stderr}")
        
        logger.info(f"Voice training output: {result.stdout}")

