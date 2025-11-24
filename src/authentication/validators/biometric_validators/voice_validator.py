import logging
from typing import Tuple
from ..base_validator import BaseValidator
from ...utils import identify_speaker_from_audio
from ...biometrics.config import BiometricConfig

logger = logging.getLogger(__name__)


class VoiceValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.75

    def _normalize_name(self, name: str) -> str:
        return name.lower().strip().replace("_", "").replace("-", "").replace(" ", "")

    def validate(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        classifier_type = BiometricConfig.VOICE_CLASSIFIER_TYPE.upper()
        logger.info("🔊 Voice Validation")
        logger.info(f"   └─ Expected User: {expected_name}")
        logger.info(f"   └─ Classifier: {classifier_type}")

        try:
            identified_speaker, confidence = identify_speaker_from_audio(audio_path)
            logger.debug(
                f"   └─ Identification Result: {identified_speaker} (Confidence: {confidence:.2%})"
            )
        except Exception as e:
            logger.error(
                "   ✗ Voice verification failed - Error processing audio",
                exc_info=True,
            )
            logger.error(f"   └─ Error: {str(e)}")
            return False, f"Error al procesar el audio: {str(e)}", 0.0

        if identified_speaker == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning("   ✗ Voice verification failed - Speaker not recognized")
            logger.warning(
                f"   └─ Confidence: {confidence:.2%} (Threshold: {self.CONFIDENCE_THRESHOLD:.2%})"
            )
            return (
                False,
                f"No se pudo reconocer la voz (confianza: {confidence:.2f})",
                confidence,
            )

        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_speaker)
        logger.debug("   └─ Name Comparison:")
        logger.debug(f"      ├─ Expected: '{expected_normalized}'")
        logger.debug(f"      └─ Identified: '{identified_normalized}'")

        if expected_normalized != identified_normalized:
            logger.warning("   ✗ Voice verification failed - Name mismatch")
            logger.warning(
                f"   └─ Expected: '{expected_name}' → Identified: '{identified_speaker}'"
            )
            logger.warning(f"   └─ Confidence: {confidence:.2%}")
            return (
                False,
                f"No se pudo reconocer la voz (confianza: {confidence:.2f}). Por favor, verifica que el modelo de voz haya sido entrenado con los datos correctos.",
                confidence,
            )

        logger.info("   ✓ Voice validation successful")
        logger.info(f"   └─ User: {identified_speaker} | Confidence: {confidence:.2%}")
        return True, identified_speaker, confidence
