import logging
from typing import Tuple
from .voice_validator import VoiceValidator
from .facial_validator import FacialValidator

logger = logging.getLogger(__name__)


class BiometricValidator:
    def __init__(self):
        logger.debug("🔧 Initializing BiometricValidator")
        self.voice_validator = VoiceValidator()
        self.facial_validator = FacialValidator()
        logger.debug("   ✓ BiometricValidator initialized successfully")

    def validate_voice(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        logger.debug(f"   → Delegating voice validation to VoiceValidator")
        return self.voice_validator.validate(audio_path, expected_name)

    def validate_facial(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        logger.debug(f"   → Delegating facial validation to FacialValidator")
        return self.facial_validator.validate(image_path, expected_name)

    def validate_all(self, audio_path: str, image_path: str, expected_name: str) -> Tuple[bool, str]:
        logger.info("")
        logger.info("═" * 60)
        logger.info(f"🔐 Complete Biometric Validation")
        logger.info(f"   └─ User: {expected_name}")
        logger.info("═" * 60)

        logger.info("")
        logger.info("Step 1/2: Voice Validation")
        logger.info("─" * 40)
        voice_valid, voice_msg, voice_conf = self.validate_voice(audio_path, expected_name)
        
        if not voice_valid:
            logger.warning("")
            logger.warning("═" * 60)
            logger.warning(f"✗ Complete biometric validation FAILED")
            logger.warning(f"   └─ Reason: Voice validation did not pass")
            logger.warning(f"   └─ Confidence: {voice_conf:.2%}")
            logger.warning("═" * 60)
            return False, voice_msg

        logger.info("")
        logger.info("Step 2/2: Facial Validation")
        logger.info("─" * 40)
        facial_valid, facial_msg, facial_conf = self.validate_facial(image_path, expected_name)
        
        if not facial_valid:
            logger.warning("")
            logger.warning("═" * 60)
            logger.warning(f"✗ Complete biometric validation FAILED")
            logger.warning(f"   └─ Reason: Facial validation did not pass")
            logger.warning("═" * 60)
            return False, facial_msg

        logger.info("")
        logger.info("═" * 60)
        logger.info(f"✓ Complete biometric validation SUCCESSFUL")
        logger.info(f"   └─ User: {expected_name}")
        logger.info(f"   └─ Facial Confidence: {facial_conf:.2%}")
        logger.info("═" * 60)
        logger.info("")
        return True, "Verificación biométrica exitosa"

