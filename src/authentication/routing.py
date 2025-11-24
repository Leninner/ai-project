from django.urls import path
from .consumers.liveness_consumer import LivenessConsumer

websocket_urlpatterns = [
    path("ws/liveness/", LivenessConsumer.as_asgi()),
]
