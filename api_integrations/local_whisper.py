import whisper
from typing import Dict, Any, Optional, List
import os

class LocalWhisperAPI:
    models = ["tiny", "base", "small", "medium", "large"]
    default_model = "base"

    def __init__(self):
        self.model = None

    def transcribe_audio(self, file_path: str, model_id: str = None, **kwargs):
        model_id = model_id or self.default_model
        if self.model is None or self.model.name != model_id:
            self.model = whisper.load_model(model_id)
        
        result = self.model.transcribe(file_path)
        return {
            "text": result["text"],
            "model": model_id,
            "summary": "Summary not available for local transcription",
            "sentiment_analysis": "Sentiment analysis not available for local transcription",
            "task_analysis": "Task analysis not available for local transcription"
        }

    def translate_audio(self, file_path: str, model_id: str = None, **kwargs):
        model_id = model_id or self.default_model
        if self.model is None or self.model.name != model_id:
            self.model = whisper.load_model(model_id)
        
        result = self.model.transcribe(file_path, task="translate")
        return {
            "text": result["text"],
            "model": model_id
        }
