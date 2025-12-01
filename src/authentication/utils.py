from .biometrics import (
    VoiceEmbeddingExtractor,
    FacialEmbeddingExtractor,
    VoiceIdentifier,
    FacialIdentifier,
)
from .biometrics.config import BiometricConfig

_voice_extractor = VoiceEmbeddingExtractor()
_facial_extractor = FacialEmbeddingExtractor()
_voice_identifier = None
_facial_identifier = None


def _get_voice_identifier():
    global _voice_identifier
    if _voice_identifier is None:
        _voice_identifier = VoiceIdentifier()
    return _voice_identifier


def _get_facial_identifier():
    global _facial_identifier
    if _facial_identifier is None:
        _facial_identifier = FacialIdentifier()
    return _facial_identifier


def _reset_identifiers():
    global _voice_identifier, _facial_identifier
    _voice_identifier = None
    _facial_identifier = None


def identify_speaker_from_audio(audio_path: str):
    identifier = _get_voice_identifier()
    current_classifier_type = BiometricConfig.VOICE_CLASSIFIER_TYPE

    if identifier.model_loader.classifier_type != current_classifier_type:
        _reset_identifiers()
        identifier = _get_voice_identifier()

    return identifier.identify(audio_path)


def identify_face_from_image(image_path: str, expected_name: str = None):
    identifier = _get_facial_identifier()
    current_classifier_type = BiometricConfig.FACIAL_CLASSIFIER_TYPE

    if identifier.model_loader.classifier_type != current_classifier_type:
        _reset_identifiers()
        identifier = _get_facial_identifier()

    return identifier.identify(image_path, expected_name=expected_name)
