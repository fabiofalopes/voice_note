#!/usr/bin/env python3
"""
faster_whisper_daemon/api.py - API for faster-whisper models

This module provides a simple API for using faster-whisper models directly,
without the daemon architecture. It's useful for benchmarking and simple use cases.
"""

import os
import torch
import platform
import time
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from pydub import AudioSegment
import gc

# Import our service
from .service import FasterWhisperService

class FasterWhisperAPI:
    """API for faster-whisper models using CTranslate2 backend"""
    
    models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "distil-large-v3"]
    default_model = "large-v3"
    
    def __init__(self, 
                 compute_type="float16",
                 device="auto", 
                 cpu_threads=None,
                 num_workers=4,
                 download_root=None,
                 local_files_only=False):
        """
        Initialize the FasterWhisperAPI.
        
        Args:
            compute_type: Compute type for the model. Options: "float32", "float16", "int8_float16", "int8"
            device: Device to use. Options: "auto", "cpu", "cuda", "mps"
            cpu_threads: Number of CPU threads to use
            num_workers: Number of workers for parallel processing
            download_root: Directory to download models to
            local_files_only: Only use local files, don't download
        """
        self.service = FasterWhisperService(
            compute_type=compute_type,
            device=device,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
            verbose=True
        )
    
    def transcribe_audio(self, audio_path, model_id=None, **options):
        """
        Transcribe audio using the specified model.
        
        Args:
            audio_path: Path to audio file
            model_id: Model ID to use. If None, uses the default model.
            **options: Options to pass to the model's transcribe method
            
        Returns:
            Transcription result
        """
        # Load model if needed
        if model_id is None:
            model_id = self.default_model
            
        self.service.load_model(model_id)
        
        # Transcribe
        return self.service.transcribe(audio_path, **options)
    
    def translate_audio(self, audio_path, model_id=None, **options):
        """
        Translate audio to English using the specified model.
        
        Args:
            audio_path: Path to audio file
            model_id: Model ID to use. If None, uses the default model.
            **options: Options to pass to the model's transcribe method
            
        Returns:
            Translation result
        """
        # Set task to translate
        options["task"] = "translate"
        
        # Transcribe with translation task
        return self.transcribe_audio(audio_path, model_id, **options)
    
    def get_audio_duration(self, audio_path):
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return None
    
    def get_info(self):
        """Get information about the service and loaded model"""
        return self.service.get_info() 