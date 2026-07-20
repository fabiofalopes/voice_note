"""API clients for voice transcription services"""

from .base_client import BaseSTTClient, Segment, ChunkResult, TranscriptionResult
from .groq_client import GroqWhisperClient
from .modelos_client import ModelosSTTClient

__all__ = [
    "BaseSTTClient",
    "Segment",
    "ChunkResult",
    "TranscriptionResult",
    "GroqWhisperClient",
    "ModelosSTTClient",
]
