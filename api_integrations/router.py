from typing import Dict, Any, Optional
from .groq_whisper import GroqWhisperAPI

# Import LocalWhisperAPI with error handling
try:
    from .local_whisper import LocalWhisperAPI
except ImportError:
    print("Warning: LocalWhisperAPI could not be imported. Make sure whisper is installed.")
    LocalWhisperAPI = None

class APIRouter:
    def __init__(self):
        self.default_api = "groq"
        self.apis = {
            "groq": GroqWhisperAPI(),
        }
        if LocalWhisperAPI is not None:
            self.apis["local"] = LocalWhisperAPI()

    def list_available_models(self) -> Dict[str, list]:
        available_models = {}
        for api_name, api_instance in self.apis.items():
            if hasattr(api_instance, 'models'):
                available_models[api_name] = api_instance.models
        return available_models

    def transcribe_audio(
        self,
        file_path: str,
        api_name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        api_to_use = api_name or self.default_api
        if api_to_use not in self.apis:
            raise ValueError(f"API '{api_to_use}' is not available.")

        api_instance = self.apis[api_to_use]
        if model_id is None and hasattr(api_instance, 'default_model'):
            model_id = api_instance.default_model
        return api_instance.transcribe_audio(file_path, model_id=model_id, **kwargs)

    def translate_audio(
        self,
        file_path: str,
        api_name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        api_to_use = api_name or self.default_api
        if api_to_use not in self.apis:
            raise ValueError(f"API '{api_to_use}' is not available.")

        api_instance = self.apis[api_to_use]
        if model_id is None and hasattr(api_instance, 'default_model'):
            model_id = api_instance.default_model
        return api_instance.translate_audio(file_path, model_id=model_id, **kwargs)
