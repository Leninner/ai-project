from typing import Tuple, Optional, List
from django.core.files.uploadedfile import UploadedFile
import shutil
from PIL import Image
from ..models import User
from .video import VideoProcessor
from .training_service import TrainingService
from ..biometrics.face_detector import FaceDetector
import logging

logger = logging.getLogger(__name__)


class RegistrationService:
    def __init__(self):
        self.video_processor = VideoProcessor()

    def _validate_frames_contain_faces(
        self, frame_paths: List[str]
    ) -> Tuple[bool, str]:
        if not frame_paths:
            return False, "No se extrajeron frames del video"

        validation_sample_size = min(40, len(frame_paths))
        step = max(1, len(frame_paths) // validation_sample_size)
        sample_frames = [frame_paths[i] for i in range(0, len(frame_paths), step)][
            :validation_sample_size
        ]

        faces_detected = 0
        for frame_path in sample_frames:
            try:
                image = Image.open(frame_path).convert("RGB")
                face_crop = FaceDetector.extract_face_from_image(image)
                if face_crop is not None:
                    faces_detected += 1
            except Exception as e:
                logger.debug(f"Error detecting face in {frame_path}: {str(e)}")
                continue

        if faces_detected < len(sample_frames) * 0.8:
            error_msg = "No se detectaron rostros en algunas imágenes del video. Por favor, asegúrate de que tu cara esté completamente visible, bien iluminada y mirando hacia la cámara durante toda la grabación."
            logger.warning(
                f"Face validation failed: only {faces_detected}/{len(sample_frames)} frames had valid faces"
            )
            return False, error_msg

        logger.info(
            f"Face validation passed for {faces_detected}/{len(sample_frames)} sample frames"
        )
        return True, "Validation successful"

    def _cleanup_user_data(self, username: str) -> None:
        facial_dir = self.video_processor.FACIAL_DATA_DIR / username
        voice_dir = self.video_processor.VOICE_DATA_DIR / username

        if facial_dir.exists():
            shutil.rmtree(facial_dir)
            logger.info(f"Cleaned up facial data for {username}")

        if voice_dir.exists():
            shutil.rmtree(voice_dir)
            logger.info(f"Cleaned up voice data for {username}")

    def register_user(
        self, username: str, video_file: UploadedFile
    ) -> Tuple[bool, Optional[User], str]:
        if User.objects.filter(username=username).exists():
            return False, None, "El nombre de usuario ya está registrado"

        try:
            frame_count, audio_count = self.video_processor.process_registration_video(
                video_file, username
            )

            logger.info(
                f"Extracted {frame_count} frames and {audio_count} audio segments for user {username}"
            )

            user_facial_dir = self.video_processor.FACIAL_DATA_DIR / username
            frame_paths = [str(p) for p in sorted(user_facial_dir.glob("*.png"))]

            is_valid, validation_message = self._validate_frames_contain_faces(
                frame_paths
            )

            if not is_valid:
                self._cleanup_user_data(username)
                logger.warning(
                    f"Face validation failed for {username}: {validation_message}"
                )
                return False, None, validation_message

            user = User.objects.create(username=username)

            TrainingService.trigger_async_training()

            return (
                True,
                user,
                f"Registro exitoso. Extracción de {frame_count} muestras faciales y {audio_count} muestras de voz.",
            )

        except Exception as e:
            self._cleanup_user_data(username)
            logger.error(f"Registration failed for {username}: {str(e)}")
            return False, None, f"Registration failed: {str(e)}"
