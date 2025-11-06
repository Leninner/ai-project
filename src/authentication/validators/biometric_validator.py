from typing import Tuple
from .base_validator import BaseValidator
from ..utils import identify_speaker_from_audio, identify_face_from_image


class VoiceValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.3
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison: lowercase, strip, replace underscores/spaces"""
        return name.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    
    def validate(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        try:
            identified_speaker, confidence = identify_speaker_from_audio(audio_path)
        except Exception as e:
            return False, f"Voice verification failed: Error processing audio - {str(e)}", 0.0
        
        if identified_speaker == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            return False, f"Voice verification failed: Speaker not recognized (confidence: {confidence:.2f})", confidence
        
        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_speaker)
        
        if expected_normalized != identified_normalized:
            return False, f"Voice verification failed: Expected {expected_name}, but identified as {identified_speaker} (confidence: {confidence:.2f}). Please verify the voice model was trained with the correct data.", confidence
        
        return True, identified_speaker, confidence


class FacialValidator(BaseValidator):
    CONFIDENCE_THRESHOLD = 0.75
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison: lowercase, strip, replace underscores/spaces"""
        return name.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
    
    def validate(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        try:
            identified_person, confidence = identify_face_from_image(image_path)
        except Exception as e:
            return False, f"Face verification failed: Error processing image - {str(e)}", 0.0
        
        if identified_person == "unknown" or confidence < self.CONFIDENCE_THRESHOLD:
            return False, f"Face verification failed: Person not recognized (confidence: {confidence:.2f})", confidence
        
        expected_normalized = self._normalize_name(expected_name)
        identified_normalized = self._normalize_name(identified_person)
        
        if expected_normalized != identified_normalized:
            return False, f"Face verification failed: Expected {expected_name}, but identified as {identified_person} (confidence: {confidence:.2f}). Please verify the facial model was trained with the correct data.", confidence
        
        return True, identified_person, confidence


class BiometricValidator:
    def __init__(self):
        self.voice_validator = VoiceValidator()
        self.facial_validator = FacialValidator()
    
    def validate_voice(self, audio_path: str, expected_name: str) -> Tuple[bool, str, float]:
        return self.voice_validator.validate(audio_path, expected_name)
    
    def validate_facial(self, image_path: str, expected_name: str) -> Tuple[bool, str, float]:
        return self.facial_validator.validate(image_path, expected_name)
    
    def validate_all(self, audio_path: str, image_path: str, expected_name: str) -> Tuple[bool, str]:
        # # Validar voz primero
        # voice_valid, voice_msg, voice_conf = self.validate_voice(audio_path, expected_name)
        # if not voice_valid:
        #     return False, voice_msg
        
        # Validar facial después
        facial_valid, facial_msg, facial_conf = self.validate_facial(image_path, expected_name)

        print(facial_valid, facial_msg, facial_conf)

        if not facial_valid:
            return False, facial_msg
        
        return True, "Biometric verification successful"

