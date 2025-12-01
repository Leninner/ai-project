import logging
from typing import Tuple
from ..base_validator import BaseValidator
from ...utils import identify_face_from_image
from ...biometrics.config import BiometricConfig

logger = logging.getLogger(__name__)


class FacialValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.75

    def _normalize_name(self, name: str) -> str:
        return name.lower().strip().replace("_", "").replace("-", "").replace(" ", "")

    def validate(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        classifier_type = BiometricConfig.FACIAL_CLASSIFIER_TYPE.upper()
        logger.info("👤 Facial Validation")
        logger.info(f"   └─ Expected User: {expected_name}")
        logger.info(f"   └─ Classifier: {classifier_type}")

        try:
            # Pass expected_name to only check against that specific user
            identified_person, confidence = identify_face_from_image(image_path, expected_name=expected_name)
            logger.debug(
                f"   └─ Identification Result: {identified_person} (Confidence: {confidence:.2%})"
            )
        except Exception as e:
            logger.error(
                "   ✗ Facial verification failed - Error processing image",
                exc_info=True,
            )
            logger.error(f"   └─ Error: {str(e)}")
            return False, f"Error al procesar la imagen: {str(e)}", 0.0

        if identified_person == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning("   ✗ Facial verification failed - Person not recognized")
            logger.warning(
                f"   └─ Confidence: {confidence:.2%} (Threshold: {self.CONFIDENCE_THRESHOLD:.2%})"
            )
            return (
                False,
                f"No se pudo reconocer la persona (confianza: {confidence:.2f})",
                confidence,
            )

        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_person)
        logger.debug("   └─ Name Comparison:")
        logger.debug(f"      ├─ Expected: '{expected_normalized}'")
        logger.debug(f"      └─ Identified: '{identified_normalized}'")

        if expected_normalized != identified_normalized:
            logger.warning("   ✗ Facial verification failed - Name mismatch")
            logger.warning(
                f"   └─ Expected: '{expected_name}' → Identified: '{identified_person}'"
            )
            logger.warning(f"   └─ Confidence: {confidence:.2%}")
            return (
                False,
                f"No se pudo reconocer la persona (confianza: {confidence:.2f}). Por favor, verifica que el modelo facial haya sido entrenado con los datos correctos.",
                confidence,
            )

        logger.info("   ✓ Facial validation successful")
        logger.info(f"   └─ User: {identified_person} | Confidence: {confidence:.2%}")
        return True, identified_person, confidence
