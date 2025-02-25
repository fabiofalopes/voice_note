from typing import Dict, Any, Optional
from .groq_whisper import GroqWhisperAPI
import platform
import warnings
import os

# No need to import the old LocalWhisperAPI since we're not using it
# Instead, just initialize the APIs properly

# Suppress specific warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

class APIRouter:
    def __init__(self):
        self.default_api = "groq"
        # Initialize APIs properly using our new method
        self.initialize_apis()
        
        # Add Groq API if not already added
        if "groq" not in self.apis:
            self.apis["groq"] = GroqWhisperAPI()

    def initialize_apis(self):
        """Initialize API clients based on environment variables and available packages."""
        # Initialize APIs dict
        self.apis = {}
        
        # Try to load faster-whisper API as the local implementation
        try:
            from api_integrations.faster_whisper_api import FasterWhisperAPI
            self.apis['local'] = FasterWhisperAPI(
                compute_type="int8" if platform.system() == "Darwin" else "float16",
                device="auto",
                cpu_threads=None,  # Auto-select based on platform
                num_workers=4
            )
            print("Initialized faster-whisper as local transcription API")
        except ImportError:
            print("faster-whisper is not installed, local transcription will not be available")
        
        # Initialize other APIs here...

    def list_available_models(self):
        """List available models for each API."""
        models = {}
        
        if 'local' in self.apis:
            # Get models from FasterWhisperAPI
            models['local (faster-whisper)'] = self.apis['local'].models
        
        # Add other API models here...
        
        return models

    def transcribe_audio(self, file_path, api_name='groq', model_id=None, **kwargs):
        """
        Transcribe audio using the specified API.
        
        Args:
            file_path: Path to the audio file
            api_name: Name of the API to use (default: 'groq')
            model_id: Model ID to use for transcription
            **kwargs: Additional arguments for the API
            
        Returns:
            Transcription result
        """
        if api_name not in self.apis:
            return {"error": f"API '{api_name}' not found or not initialized"}
        
        api = self.apis[api_name]
        api_method = getattr(api, 'transcribe_audio')
        
        # Filter kwargs based on the API being used
        if api_name == 'local':
            # All parameters are valid for local faster-whisper API
            return api_method(file_path, model_id=model_id, **kwargs)
        else:
            # For non-local APIs, filter out faster-whisper specific parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['vad_filter', 'beam_size', 'word_timestamps']}
            return api_method(file_path, model_id=model_id, **filtered_kwargs)

    def translate_audio(self, file_path, api_name='groq', model_id=None, **kwargs):
        """
        Translate audio using the specified API.
        
        Args:
            file_path: Path to the audio file
            api_name: Name of the API to use (default: 'groq')
            model_id: Model ID to use for translation
            **kwargs: Additional arguments for the API
            
        Returns:
            Translation result
        """
        if api_name not in self.apis:
            return {"error": f"API '{api_name}' not found or not initialized"}
        
        api = self.apis[api_name]
        api_method = getattr(api, 'translate_audio')
        
        # Filter kwargs based on the API being used
        if api_name == 'local':
            # All parameters are valid for local faster-whisper API
            return api_method(file_path, model_id=model_id, **kwargs)
        else:
            # For non-local APIs, filter out faster-whisper specific parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['vad_filter', 'beam_size', 'word_timestamps']}
            return api_method(file_path, model_id=model_id, **filtered_kwargs)
