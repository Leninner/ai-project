import subprocess
import threading
from pathlib import Path
from typing import Optional
import logging

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
    def trigger_async_training(cls) -> None:
        if cls.is_training():
            logger.warning("Training already in progress, skipping new training request")
            return
        
        thread = threading.Thread(target=cls._train_models, daemon=True)
        thread.start()
    
    @classmethod
    def _train_models(cls) -> None:
        with cls._training_lock:
            if cls._is_training:
                return
            cls._is_training = True
        
        try:
            logger.info("Starting facial recognition model training...")
            cls._train_facial_model()
            
            logger.info("Starting voice recognition model training...")
            cls._train_voice_model()
            
            logger.info("Model training completed successfully")
        
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
        
        finally:
            with cls._training_lock:
                cls._is_training = False
    
    @classmethod
    def _train_facial_model(cls) -> None:
        if not cls.FACIAL_PIPELINE.exists():
            raise FileNotFoundError(f"Facial pipeline not found: {cls.FACIAL_PIPELINE}")
        
        result = subprocess.run(
            ['python', str(cls.FACIAL_PIPELINE)],
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
    def _train_voice_model(cls) -> None:
        if not cls.VOICE_PIPELINE.exists():
            raise FileNotFoundError(f"Voice pipeline not found: {cls.VOICE_PIPELINE}")
        
        result = subprocess.run(
            ['python', str(cls.VOICE_PIPELINE)],
            cwd=cls.VOICE_PIPELINE.parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.error(f"Voice training stderr: {result.stderr}")
            raise RuntimeError(f"Voice model training failed: {result.stderr}")
        
        logger.info(f"Voice training output: {result.stdout}")

