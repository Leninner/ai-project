"""
WebSocket Consumer for Real-time Biometric Authentication

Simplified version - uses CNN facial recognition + voice authentication
No liveness detection required
"""

import json
import base64
import io
import logging
import numpy as np
from PIL import Image
from typing import Optional, Dict
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import tempfile
import os
import secrets
from django.core.cache import cache

from ..biometrics.face_detector import FaceDetector
from ..biometrics.identifiers import FacialIdentifier, VoiceIdentifier
from ..models import User

logger = logging.getLogger(__name__)


class LivenessConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time biometric authentication

    Protocol:
    Client → Server:
        {"type": "start", "username": "user123"}
        {"type": "frame", "username": "user123", "image": "base64_jpeg"}
        {"type": "audio", "username": "user123", "audio": "base64_wav"}

    Server → Client:
        {"type": "face_result", "recognized": true, "username": "user123", "confidence": 0.95}
        {"type": "voice_result", "recognized": true, "confidence": 0.85}
        {"type": "authentication_complete", "success": true, "redirect": "/dashboard/"}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_detector = FaceDetector()
        self.facial_identifier = FacialIdentifier()
        self.voice_identifier = VoiceIdentifier()

        self.username = None
        self.frame_count = 0
        self.face_recognitions = []  # Store recognition results
        self.voice_retries = 0

        # Recognition thresholds
        self.MIN_FACE_CONFIRMATIONS = 3  # Need 3 consistent recognitions
        self.PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
        self.MAX_VOICE_RETRIES = 3  # Maximum voice retries before failure

        self.face_authenticated = False
        self.voice_authenticated = False

    async def connect(self):
        """Accept WebSocket connection"""
        await self.accept()
        logger.info("WebSocket connection established")

        await self.send(
            text_data=json.dumps(
                {
                    "type": "connected",
                    "message": "WebSocket connected. Ready for authentication.",
                }
            )
        )

    async def disconnect(self, close_code):
        """Clean up on disconnect"""
        logger.info(f"WebSocket disconnected: {close_code}")

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
            else:
                await self.send_error(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error in receive: {e}", exc_info=True)
            await self.send_error(f"Server error: {str(e)}")

    async def handle_start(self, data: Dict):
        """Handle session start"""
        self.username = data.get("username")
        if not self.username:
            await self.send_error("Username required")
            return

        # Verify user exists
        user_exists = await sync_to_async(
            User.objects.filter(username=self.username).exists
        )()
        if not user_exists:
            await self.send_error("User not found")
            return

        # Reset state
        self.frame_count = 0
        self.face_recognitions = []
        self.face_authenticated = False
        self.voice_authenticated = False

        await self.send(
            text_data=json.dumps(
                {
                    "type": "session_started",
                    "username": self.username,
                    "message": "Session started. Show your face to the camera.",
                }
            )
        )
        logger.info(f"Session started for user: {self.username}")

    async def handle_frame(self, data: Dict):
        """Process incoming video frame"""
        try:
            if not self.username:
                await self.send_error("Session not started")
                return

            if self.face_authenticated:
                return  # Already authenticated, skip processing

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

            # Process every Nth frame to reduce load
            if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                result = await self.process_face_recognition(image)
                if result:
                    await self.send(text_data=json.dumps(result))
                    await self.check_face_authentication()

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            await self.send_error(f"Frame processing error: {str(e)}")

    async def process_face_recognition(self, image: Image.Image) -> Optional[Dict]:
        """Run CNN facial recognition"""
        try:
            # Detect face first
            frame = np.array(image)
            boxes, probs = await sync_to_async(self.face_detector.get_mtcnn().detect)(
                image
            )

            if boxes is None or len(boxes) == 0:
                return {
                    "type": "face_result",
                    "face_detected": False,
                    "message": "No face detected",
                }

            # Save to temp file for CNN
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image.save(tmp_file, format="JPEG")
                tmp_path = tmp_file.name

            try:
                # Run CNN identification - pass expected username for targeted validation
                identified_person, confidence = await sync_to_async(
                    self.facial_identifier.identify
                )(tmp_path, expected_name=self.username)

                # Store result
                matches_claimed = identified_person == self.username
                if matches_claimed and confidence > 0.65:
                    self.face_recognitions.append(
                        {"username": identified_person, "confidence": confidence}
                    )

                return {
                    "type": "face_result",
                    "face_detected": True,
                    "recognized": identified_person != "unknown",
                    "username": identified_person,
                    "confidence": float(confidence),
                    "matches_claim": matches_claimed,
                    "confirmations": len(self.face_recognitions),
                    "required": self.MIN_FACE_CONFIRMATIONS,
                    "message": f"Recognized: {identified_person} ({confidence:.0%})"
                    if identified_person != "unknown"
                    else "Not recognized",
                }

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Face recognition error: {e}", exc_info=True)
            return {"type": "face_result", "recognized": False, "error": str(e)}

    async def check_face_authentication(self):
        """Check if face authentication complete"""
        if len(self.face_recognitions) >= self.MIN_FACE_CONFIRMATIONS:
            self.face_authenticated = True
            avg_confidence = sum(r["confidence"] for r in self.face_recognitions) / len(
                self.face_recognitions
            )

            logger.info(f"Face authenticated: {self.username} ({avg_confidence:.0%})")

            await self.send(
                text_data=json.dumps(
                    {
                        "type": "face_authenticated",
                        "username": self.username,
                        "confidence": float(avg_confidence),
                        "message": "✅ Rostro reconocido! Ahora habla para verificar tu voz.",
                    }
                )
            )

            await self.check_complete_authentication()

    async def handle_audio(self, data: Dict):
        """Process audio for voice authentication"""
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

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_audio_path = tmp_file.name

            try:
                # Run voice identification
                identified_person, confidence = await sync_to_async(
                    self.voice_identifier.identify
                )(tmp_audio_path)

                matches_claimed = identified_person == self.username

                if matches_claimed and confidence > 0.5:
                    self.voice_authenticated = True
                    logger.info(
                        f"Voice authenticated: {self.username} ({confidence:.0%})"
                    )

                    await self.send(
                        text_data=json.dumps(
                            {
                                "type": "voice_result",
                                "recognized": True,
                                "username": identified_person,
                                "confidence": float(confidence),
                                "message": f"✅ Voz reconocida! ({confidence:.0%})",
                            }
                        )
                    )
                else:
                    self.voice_retries += 1
                    if self.voice_retries >= self.MAX_VOICE_RETRIES:
                        logger.warning(
                            f"Voice authentication failed after {self.MAX_VOICE_RETRIES} attempts for user {self.username}"
                        )
                        await self.send(
                            text_data=json.dumps(
                                {
                                    "type": "voice_result",
                                    "recognized": False,
                                    "error": True,
                                    "message": "❌ Autenticación de voz fallida después de varios intentos. Por favor, inténtelo de nuevo más tarde.",
                                }
                            )
                        )
                        self.voice_authenticated = False
                        await self.close()  # Close connection on final failure
                        return  # Exit handle_audio
                    else:
                        logger.info(
                            f"Voice not recognized, prompting retry: {identified_person} ({confidence:.0%})"
                        )
                        await self.send(
                            text_data=json.dumps(
                                {
                                    "type": "voice_result",
                                    "recognized": False,
                                    "retry": True,
                                    "username": identified_person,
                                    "confidence": float(confidence),
                                    "message": f"Voz no reconocida. Por favor, inténtelo de nuevo. Intentos restantes: {self.MAX_VOICE_RETRIES - self.voice_retries}",
                                }
                            )
                        )

                await self.check_complete_authentication()

            finally:
                if os.path.exists(tmp_audio_path):
                    os.unlink(tmp_audio_path)

        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)
            await self.send_error(f"Audio error: {str(e)}")

    async def check_complete_authentication(self):
        """Check if both face and voice authenticated"""
        if self.face_authenticated and self.voice_authenticated:
            logger.info(f"Complete authentication successful: {self.username}")

            # Verify user exists before getting
            user_exists = await sync_to_async(
                User.objects.filter(username=self.username).exists
            )()
            
            if not user_exists:
                await self.send_error(f"User '{self.username}' not found in database")
                logger.error(f"Authentication failed: User '{self.username}' does not exist")
                return

            # Get user ID
            user = await sync_to_async(User.objects.get)(username=self.username)

            # Generate one-time session token
            session_token = secrets.token_urlsafe(32)
            cache_key = f"auth_token:{session_token}"

            # Store user_id in cache with 30 second expiration
            await sync_to_async(cache.set)(cache_key, user.id, timeout=30)

            logger.info(
                f"Generated session token for {self.username}: {session_token[:10]}..."
            )

            await self.send(
                text_data=json.dumps(
                    {
                        "type": "authentication_complete",
                        "success": True,
                        "username": self.username,
                        "message": "Autenticación exitosa!",
                        "redirect": "/dashboard/",
                        "session_token": session_token,
                    }
                )
            )

            # Close connection
            await self.close()

    async def send_error(self, message: str):
        """Send error message"""
        await self.send(text_data=json.dumps({"type": "error", "message": message}))
