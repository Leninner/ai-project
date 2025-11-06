from .biometrics import (
    VoiceEmbeddingExtractor,
    FacialEmbeddingExtractor,
    VoiceIdentifier,
    FacialIdentifier,
    SimilarityCalculator,
)

_voice_extractor = VoiceEmbeddingExtractor()
_facial_extractor = FacialEmbeddingExtractor()
_voice_identifier = VoiceIdentifier()
_facial_identifier = FacialIdentifier()
_similarity_calc = SimilarityCalculator()


def extract_voice_embedding(audio_path: str):
    return _voice_extractor.extract(audio_path)


def extract_image_embedding(image_path: str):
    return _facial_extractor.extract(image_path)


def cosine_similarity(a, b):
    return _similarity_calc.cosine_similarity(a, b)


def calculate_mean_embedding(embeddings_list):
    result = _similarity_calc.calculate_mean_embedding(embeddings_list)
    return result if result else None


def identify_speaker_from_audio(audio_path: str):
    return _voice_identifier.identify(audio_path)


def identify_face_from_image(image_path: str):
    return _facial_identifier.identify(image_path)

