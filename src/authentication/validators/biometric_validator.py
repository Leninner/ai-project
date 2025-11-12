import logging
from typing import Tuple
from .base_validator import BaseValidator
from ..utils import identify_speaker_from_audio, identify_face_from_image
from ..biometrics.config import BiometricConfig

logger = logging.getLogger(__name__)


class VoiceValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.75
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison: lowercase, strip, replace underscores/spaces"""
        return name.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    
    def validate(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        classifier_type = BiometricConfig.VOICE_CLASSIFIER_TYPE.upper()
        logger.info(f"🔊 Voice Validation")
        logger.info(f"   └─ Expected User: {expected_name}")
        logger.info(f"   └─ Classifier: {classifier_type}")

        try:
            identified_speaker, confidence = identify_speaker_from_audio(audio_path)
            logger.debug(f"   └─ Identification Result: {identified_speaker} (Confidence: {confidence:.2%})")
        except Exception as e:
            logger.error(f"   ✗ Voice verification failed - Error processing audio", exc_info=True)
            logger.error(f"   └─ Error: {str(e)}")
            return False, f"Error al procesar el audio: {str(e)}", 0.0

        if identified_speaker == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(f"   ✗ Voice verification failed - Speaker not recognized")
            logger.warning(f"   └─ Confidence: {confidence:.2%} (Threshold: {self.CONFIDENCE_THRESHOLD:.2%})")
            return False, f"No se pudo reconocer la voz (confianza: {confidence:.2f})", confidence

        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_speaker)
        logger.debug(f"   └─ Name Comparison:")
        logger.debug(f"      ├─ Expected: '{expected_normalized}'")
        logger.debug(f"      └─ Identified: '{identified_normalized}'")

        if expected_normalized != identified_normalized:
            logger.warning(f"   ✗ Voice verification failed - Name mismatch")
            logger.warning(f"   └─ Expected: '{expected_name}' → Identified: '{identified_speaker}'")
            logger.warning(f"   └─ Confidence: {confidence:.2%}")
            return False, f"No se pudo reconocer la voz (confianza: {confidence:.2f}). Por favor, verifica que el modelo de voz haya sido entrenado con los datos correctos.", confidence

        logger.info(f"   ✓ Voice validation successful")
        logger.info(f"   └─ User: {identified_speaker} | Confidence: {confidence:.2%}")
        return True, identified_speaker, confidence


class FacialValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.75
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison: lowercase, strip, replace underscores/spaces"""
        return name.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    
    def validate(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        classifier_type = BiometricConfig.FACIAL_CLASSIFIER_TYPE.upper()
        logger.info(f"👤 Facial Validation")
        logger.info(f"   └─ Expected User: {expected_name}")
        logger.info(f"   └─ Classifier: {classifier_type}")

        try:
            identified_person, confidence = identify_face_from_image(image_path)
            logger.debug(f"   └─ Identification Result: {identified_person} (Confidence: {confidence:.2%})")
        except Exception as e:
            logger.error(f"   ✗ Facial verification failed - Error processing image", exc_info=True)
            logger.error(f"   └─ Error: {str(e)}")
            return False, f"Error al procesar la imagen: {str(e)}", 0.0

        if identified_person == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(f"   ✗ Facial verification failed - Person not recognized")
            logger.warning(f"   └─ Confidence: {confidence:.2%} (Threshold: {self.CONFIDENCE_THRESHOLD:.2%})")
            return False, f"No se pudo reconocer la persona (confianza: {confidence:.2f})", confidence

        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_person)
        logger.debug(f"   └─ Name Comparison:")
        logger.debug(f"      ├─ Expected: '{expected_normalized}'")
        logger.debug(f"      └─ Identified: '{identified_normalized}'")

        if expected_normalized != identified_normalized:
            logger.warning(f"   ✗ Facial verification failed - Name mismatch")
            logger.warning(f"   └─ Expected: '{expected_name}' → Identified: '{identified_person}'")
            logger.warning(f"   └─ Confidence: {confidence:.2%}")
            return False, f"No se pudo reconocer la persona (confianza: {confidence:.2f}). Por favor, verifica que el modelo facial haya sido entrenado con los datos correctos.", confidence

        logger.info(f"   ✓ Facial validation successful")
        logger.info(f"   └─ User: {identified_person} | Confidence: {confidence:.2%}")
        return True, identified_person, confidence


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
        # logger.info(f"   ├─ Voice Confidence: {voice_conf:.2%}")
        logger.info(f"   └─ Facial Confidence: {facial_conf:.2%}")
        logger.info("═" * 60)
        logger.info("")
        return True, "Verificación biométrica exitosa"

