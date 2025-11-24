from django.urls import path
from .consumers.liveness_consumer import LivenessConsumer
from .consumers.registration_consumer import RegistrationConsumer

websocket_urlpatterns = [
    path("ws/liveness/", LivenessConsumer.as_asgi()),
    path("ws/registration/", RegistrationConsumer.as_asgi()),
]
