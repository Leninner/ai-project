"""
WebSocket Consumer for Real-time Registration

Handles real-time frame and audio capture during user registration.
Progressively saves frames and audio chunks to the correct directories.
"""

import json
import base64
import io
import logging
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import tempfile
import os

from ..biometrics.face_detector import FaceDetector
from ..models import User
from ..services.training_service import TrainingService

logger = logging.getLogger(__name__)


class RegistrationConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time registration with frame capture

    Protocol:
    Client → Server:
        {"type": "start", "username": "user123"}
        {"type": "frame", "image": "base64_jpeg", "timestamp": 1234}
        {"type": "audio", "audio": "base64_wav"}
        {"type": "complete"}

    Server → Client:
        {"type": "progress", "frames_saved": 50, "required": 200, "faces_detected": 45}
        {"type": "face_detected", "detected": true}
        {"type": "error", "message": "..."}
        {"type": "registration_complete", "success": true, "redirect": "/login/"}
    """

    # Constants
    REQUIRED_FRAMES = 200
    MIN_FACE_DETECTION_RATE = 0.8  # 80% of frames must have faces
    FACIAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "models" / "facial" / "data"
    VOICE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "models" / "voice" / "data"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_detector = FaceDetector()
        self.username = None
        self.frame_count = 0
        self.frames_saved = 0
        self.faces_detected = 0
        self.audio_count = 0
        self.user_facial_dir = None
        self.user_voice_dir = None

    async def connect(self):
        """Accept WebSocket connection"""
        await self.accept()
        logger.info("Registration WebSocket connection established")

        await self.send(
            text_data=json.dumps(
                {
                    "type": "connected",
                    "message": "WebSocket connected. Ready for registration.",
                }
            )
        )

    async def disconnect(self, close_code):
        """Clean up on disconnect"""
        logger.info(f"Registration WebSocket disconnected: {close_code}")

    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get("type")

            if message_type == "start":
                await self.handle_start(data)
            elif message_type == "frame":
                await self.handle_frame(data)
            elif message_type == "audio":
                await self.handle_audio(data)
            elif message_type == "complete":
                await self.handle_complete(data)
            else:
                await self.send_error(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error in receive: {e}", exc_info=True)
            await self.send_error(f"Server error: {str(e)}")

    async def handle_start(self, data: Dict):
        """Initialize registration session"""
        self.username = data.get("username")
        if not self.username:
            await self.send_error("Username required")
            return

        # Check if user already exists
        user_exists = await sync_to_async(
            User.objects.filter(username=self.username).exists
        )()
        if user_exists:
            await self.send_error("Username already exists")
            return

        # Create user directories
        self.user_facial_dir = self.FACIAL_DATA_DIR / self.username
        self.user_voice_dir = self.VOICE_DATA_DIR / self.username

        await sync_to_async(self.user_facial_dir.mkdir)(parents=True, exist_ok=True)
        await sync_to_async(self.user_voice_dir.mkdir)(parents=True, exist_ok=True)

        # Reset counters
        self.frame_count = 0
        self.frames_saved = 0
        self.faces_detected = 0
        self.audio_count = 0

        await self.send(
            text_data=json.dumps(
                {
                    "type": "session_started",
                    "username": self.username,
                    "message": "Registration session started. Begin recording.",
                    "required_frames": self.REQUIRED_FRAMES,
                }
            )
        )
        logger.info(f"Registration session started for user: {self.username}")

    async def handle_frame(self, data: Dict):
        """Process and save incoming video frame"""
        try:
            if not self.username:
                await self.send_error("Session not started")
                return

            # Check if we already have enough frames
            if self.frames_saved >= self.REQUIRED_FRAMES:
                return

            # Decode image
            image_data = data.get("image")
            if not image_data:
                await self.send_error("No image data")
                return

            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            self.frame_count += 1

            # Process frame and detect face
            face_detected, face_crop = await self.process_and_save_frame(image)

            if face_detected:
                self.faces_detected += 1

            # Send progress update
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "progress",
                        "frames_saved": self.frames_saved,
                        "required": self.REQUIRED_FRAMES,
                        "faces_detected": self.faces_detected,
                        "face_detected": face_detected,
                        "progress_percent": int((self.frames_saved / self.REQUIRED_FRAMES) * 100),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            await self.send_error(f"Frame processing error: {str(e)}")

    async def process_and_save_frame(self, image: Image.Image) -> tuple[bool, Optional[np.ndarray]]:
        """Detect face and save frame if valid"""
        try:
            # Convert PIL Image to numpy array
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect and extract face
            face_crop = await sync_to_async(FaceDetector.extract_face_from_frame)(frame_bgr)

            if face_crop is not None:
                # Save the face crop
                self.frames_saved += 1
                frame_path = self.user_facial_dir / f"{self.frames_saved}.png"

                # Convert to RGB and preprocess
                rgb_array = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Import and use preprocessor
                import sys
                facial_models_path = Path(__file__).parent.parent.parent.parent / "models" / "facial"
                if str(facial_models_path) not in sys.path:
                    sys.path.insert(0, str(facial_models_path))
                from facial_preprocessor import FacialPreprocessor

                
                preprocessor = FacialPreprocessor()
                preprocessed = preprocessor.preprocess_cropped_face(rgb_array)

                # Save preprocessed image
                preprocessed_image = Image.fromarray(preprocessed)
                await sync_to_async(preprocessed_image.save)(str(frame_path))

                logger.debug(f"Saved frame {self.frames_saved} for user {self.username}")
                return True, face_crop
            else:
                return False, None

        except Exception as e:
            logger.error(f"Error in process_and_save_frame: {e}", exc_info=True)
            return False, None

    async def handle_audio(self, data: Dict):
        """Process and save incoming audio chunk"""
        try:
            if not self.username:
                await self.send_error("Session not started")
                return

            audio_data = data.get("audio")
            if not audio_data:
                await self.send_error("No audio data")
                return

            if "," in audio_data:
                audio_data = audio_data.split(",")[1]

            audio_bytes = base64.b64decode(audio_data)

            # Save audio chunk - convert from webm to wav if needed
            self.audio_count += 1
            
            # Save temporary webm file
            temp_webm = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
            temp_webm.write(audio_bytes)
            temp_webm.close()
            
            try:
                # Convert webm to wav using ffmpeg
                audio_path = self.user_voice_dir / f"{self.audio_count}.wav"
                
                import subprocess
                result = await sync_to_async(subprocess.run)([
                    'ffmpeg', '-i', temp_webm.name,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    str(audio_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    # If ffmpeg fails, save as webm and log warning
                    audio_path = self.user_voice_dir / f"{self.audio_count}.webm"
                    await sync_to_async(audio_path.write_bytes)(audio_bytes)
                    logger.warning(f"Saved audio as webm (ffmpeg failed) for chunk {self.audio_count}")
                else:
                    logger.debug(f"Converted and saved audio chunk {self.audio_count} for user {self.username}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_webm.name):
                    os.unlink(temp_webm.name)

            await self.send(
                text_data=json.dumps(
                    {
                        "type": "audio_saved",
                        "audio_count": self.audio_count,
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await self.send_error(f"Audio processing error: {str(e)}")

    async def handle_complete(self, data: Dict):
        """Finalize registration and create user account"""
        try:
            if not self.username:
                await self.send_error("Session not started")
                return

            # Validate we have enough frames
            if self.frames_saved < self.REQUIRED_FRAMES:
                await self.send_error(
                    f"Insufficient frames captured. Got {self.frames_saved}, need {self.REQUIRED_FRAMES}"
                )
                return

            # Validate face detection rate
            face_detection_rate = self.faces_detected / self.frame_count if self.frame_count > 0 else 0
            if face_detection_rate < self.MIN_FACE_DETECTION_RATE:
                await self.send_error(
                    f"Insufficient face detection. Only {face_detection_rate:.0%} of frames had faces. "
                    f"Please ensure your face is visible and well-lit during recording."
                )
                # Clean up data
                await self.cleanup_user_data()
                return

            # Validate we have audio
            if self.audio_count == 0:
                await self.send_error("No audio data received")
                await self.cleanup_user_data()
                return

            # Create user account
            user = await sync_to_async(User.objects.create)(username=self.username)
            logger.info(f"Created user account: {self.username}")

            # Trigger async training
            await sync_to_async(TrainingService.trigger_async_training)()

            # Send success response
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "registration_complete",
                        "success": True,
                        "username": self.username,
                        "frames_saved": self.frames_saved,
                        "audio_chunks": self.audio_count,
                        "message": f"Registration successful! Captured {self.frames_saved} facial samples and {self.audio_count} voice samples.",
                        "redirect": "/login/",
                    }
                )
            )

            logger.info(
                f"Registration completed for {self.username}: "
                f"{self.frames_saved} frames, {self.audio_count} audio chunks"
            )

            # Close connection
            await self.close()

        except Exception as e:
            logger.error(f"Error completing registration: {e}", exc_info=True)
            await self.send_error(f"Registration completion error: {str(e)}")
            await self.cleanup_user_data()

    async def cleanup_user_data(self):
        """Remove user data directories on failure"""
        try:
            if self.user_facial_dir and await sync_to_async(self.user_facial_dir.exists)():
                import shutil
                await sync_to_async(shutil.rmtree)(self.user_facial_dir)
                logger.info(f"Cleaned up facial data for {self.username}")

            if self.user_voice_dir and await sync_to_async(self.user_voice_dir.exists)():
                import shutil
                await sync_to_async(shutil.rmtree)(self.user_voice_dir)
                logger.info(f"Cleaned up voice data for {self.username}")
        except Exception as e:
            logger.error(f"Error cleaning up user data: {e}", exc_info=True)

    async def send_error(self, message: str):
        """Send error message"""
        await self.send(text_data=json.dumps({"type": "error", "message": message}))
        logger.warning(f"Sent error to client: {message}")
