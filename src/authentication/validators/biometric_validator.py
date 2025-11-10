import logging
from typing import Tuple
from .base_validator import BaseValidator
from ..utils import identify_speaker_from_audio, identify_face_from_image

logger = logging.getLogger(__name__)


class VoiceValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.3
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison: lowercase, strip, replace underscores/spaces"""
        return name.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    
    def validate(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        logger.info(f"Starting voice validation for expected user: {expected_name}")
        logger.debug(f"Audio path: {audio_path}")

        try:
            identified_speaker, confidence = identify_speaker_from_audio(audio_path)
            logger.debug(f"Voice identification complete - Speaker: {identified_speaker}, Confidence: {confidence:.2f}")
        except Exception as e:
            logger.error(f"Voice verification failed - Error processing audio: {str(e)}", exc_info=True)
            return False, f"Error al procesar el audio: {str(e)}", 0.0

        if identified_speaker == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(f"Voice verification failed - Speaker not recognized (confidence: {confidence:.2f}, threshold: {self.CONFIDENCE_THRESHOLD})")
            return False, f"No se pudo reconocer la voz (confianza: {confidence:.2f})", confidence

        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_speaker)
        logger.debug(f"Name comparison - Expected: '{expected_normalized}', Identified: '{identified_normalized}'")

        if expected_normalized != identified_normalized:
            logger.warning(f"Voice verification failed - Name mismatch: Expected '{expected_name}', identified as '{identified_speaker}' (confidence: {confidence:.2f})")
            return False, f"No se pudo reconocer la voz (confianza: {confidence:.2f}). Por favor, verifica que el modelo de voz haya sido entrenado con los datos correctos.", confidence

        logger.info(f"Voice validation successful - User: {identified_speaker}, Confidence: {confidence:.2f}")
        return True, identified_speaker, confidence


class FacialValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.75
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison: lowercase, strip, replace underscores/spaces"""
        return name.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    
    def validate(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        logger.info(f"Starting facial validation for expected user: {expected_name}")
        logger.debug(f"Image path: {image_path}")

        try:
            identified_person, confidence = identify_face_from_image(image_path)
            logger.debug(f"Facial identification complete - Person: {identified_person}, Confidence: {confidence:.2f}")
        except Exception as e:
            logger.error(f"Facial verification failed - Error processing image: {str(e)}", exc_info=True)
            return False, f"Error al procesar la imagen: {str(e)}", 0.0

        if identified_person == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(f"Facial verification failed - Person not recognized (confidence: {confidence:.2f}, threshold: {self.CONFIDENCE_THRESHOLD})")
            return False, f"No se pudo reconocer la persona (confianza: {confidence:.2f})", confidence

        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_person)
        logger.debug(f"Name comparison - Expected: '{expected_normalized}', Identified: '{identified_normalized}'")

        if expected_normalized != identified_normalized:
            logger.warning(f"Facial verification failed - Name mismatch: Expected '{expected_name}', identified as '{identified_person}' (confidence: {confidence:.2f})")
            return False, f"No se pudo reconocer la persona (confianza: {confidence:.2f}). Por favor, verifica que el modelo facial haya sido entrenado con los datos correctos.", confidence

        logger.info(f"Facial validation successful - User: {identified_person}, Confidence: {confidence:.2f}")
        return True, identified_person, confidence


class BiometricValidator:
    def __init__(self):
        logger.debug("Initializing BiometricValidator")
        self.voice_validator = VoiceValidator()
        self.facial_validator = FacialValidator()
        logger.debug("BiometricValidator initialized successfully")

    def validate_voice(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        logger.debug(f"BiometricValidator delegating voice validation to VoiceValidator")
        return self.voice_validator.validate(audio_path, expected_name)

    def validate_facial(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        logger.debug(f"BiometricValidator delegating facial validation to FacialValidator")
        return self.facial_validator.validate(image_path, expected_name)

    def validate_all(self, audio_path: str, image_path: str, expected_name: str) -> Tuple[bool, str]:
        logger.info(f"Starting complete biometric validation for user: {expected_name}")

        # Validar voz primero
        logger.debug("Proceeding with voice validation")
        voice_valid, voice_msg, voice_conf = self.validate_voice(audio_path, expected_name)
        logger.debug(f"Voice validation result - Valid: {voice_valid}, Message: {voice_msg}, Confidence: {voice_conf}")

        if not voice_valid:
            logger.warning(f"Complete biometric validation failed - Voice validation did not pass")
            return False, voice_msg

        # Validar facial después
        logger.debug("Proceeding with facial validation")
        facial_valid, facial_msg, facial_conf = self.validate_facial(image_path, expected_name)
        logger.debug(f"Facial validation result - Valid: {facial_valid}, Message: {facial_msg}, Confidence: {facial_conf}")

        if not facial_valid:
            logger.warning(f"Complete biometric validation failed - Facial validation did not pass")
            return False, facial_msg

        logger.info(f"Complete biometric validation successful for user: {expected_name}")
        return True, "Verificación biométrica exitosa"

