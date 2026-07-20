"""
Simple configuration for voice transcriber
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODELOS_AI_KEY = os.getenv("MODELOS_AI_KEY")

# Audio Configuration
DEFAULT_AUDIO_CONFIG = {
    "chunk_size": 4096,
    "format": "paInt16",  # Will be converted to pyaudio constant
    "channels": 1,
    "rate": 44100,
    "output_directory": "recordings",
}


def get_groq_api_key():
    """Get Groq API key from environment"""
    return GROQ_API_KEY


def get_modelos_ai_key():
    """Get Modelos AI API key from environment"""
    return MODELOS_AI_KEY


def get_audio_config():
    """Get audio configuration"""
    return DEFAULT_AUDIO_CONFIG.copy()
