#!/usr/bin/env python3
"""
faster_whisper_daemon/service.py - Core service for faster-whisper models

This module provides the FasterWhisperService class, which manages the loading
and execution of faster-whisper models using the CTranslate2 backend.
"""

import os
import torch
import platform
import time
import threading
import gc
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import psutil

class FasterWhisperService:
    """Singleton service for faster-whisper models using CTranslate2 backend"""
    
    _instance = None
    _lock = threading.Lock()
    
    models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "distil-large-v3"]
    default_model = "large-v3"
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FasterWhisperService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 compute_type="float16",
                 device="auto", 
                 cpu_threads=None,
                 num_workers=4,
                 download_root=None,
                 local_files_only=False,
                 model_id=None,
                 verbose=False):
        """
        Initialize the FasterWhisperService.
        
        Args:
            compute_type: Compute type for the model. Options: "float32", "float16", "int8_float16", "int8"
            device: Device to use. Options: "auto", "cpu", "cuda", "mps"
            cpu_threads: Number of CPU threads to use
            num_workers: Number of workers for parallel processing
            download_root: Directory to download models to
            local_files_only: Only use local files, don't download
            model_id: Model ID to load. If None, no model is loaded initially.
            verbose: Whether to show verbose output
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self.compute_type = compute_type
        self.device = device
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        self.local_files_only = local_files_only
        self.verbose = verbose
        
        # Initialize model-related attributes
        self.model = None
        self.model_id = None
        self.model_loading = False
        self.model_load_time = None
        self.model_load_error = None
        self._model_lock = threading.Lock()
        
        # System info
        self.system_info = self._get_system_info()
        
        # Load model if specified
        if model_id:
            self.load_model(model_id)
            
        self._initialized = True
    
    def _get_system_info(self):
        """Get system information"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
        else:
            info["cuda_available"] = False
            
        # Add MPS info if available (Apple Silicon)
        if hasattr(torch, 'mps') and torch.mps.is_available():
            info["mps_available"] = True
        else:
            info["mps_available"] = False
            
        return info
    
    def load_model(self, model_id):
        """
        Load a faster-whisper model.
        
        Args:
            model_id: Model ID to load. Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Check if model is already loaded
        if self.model_id == model_id and self.model is not None:
            return True
            
        # Check if model is currently loading
        if self.model_loading:
            return False
            
        # Set loading flag
        self.model_loading = True
        self.model_load_error = None
        
        try:
            # Import here to avoid loading unnecessary dependencies
            from faster_whisper import WhisperModel
            
            # Determine device
            device = self.device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch, 'mps') and torch.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            # Adjust compute_type for Apple Silicon (MPS)
            compute_type = self.compute_type
            if device == "mps" and compute_type == "int8":
                compute_type = "float16"
                if self.verbose:
                    print(f"Apple Silicon (MPS) detected. Switching compute type from int8 to float16 for compatibility.")
                    
            # Log loading info
            if self.verbose:
                print(f"Loading model {model_id} on {device} with compute type {compute_type}")
                
            # Unload existing model if any
            if self.model is not None:
                self.model = None
                # Force garbage collection
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            # Measure loading time
            start_time = time.time()
            
            # Load the model
            with self._model_lock:
                self.model = WhisperModel(
                    model_id,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                    download_root=self.download_root,
                    local_files_only=self.local_files_only
                )
                
            # Update model info
            self.model_id = model_id
            self.model_load_time = time.time() - start_time
            
            if self.verbose:
                print(f"Model loaded in {self.model_load_time:.2f} seconds")
                
            return True
            
        except Exception as e:
            self.model = None
            self.model_id = None
            error_msg = str(e)
            
            # Provide more helpful error messages
            if "CUDA" in error_msg and not torch.cuda.is_available():
                error_msg = "CUDA requested but not available. Try using 'cpu' or 'auto' for device."
            elif "MPS" in error_msg:
                error_msg = f"Error with Apple Silicon (MPS): {error_msg}. Try using 'cpu' for device or 'float16' for compute_type."
            elif "int8" in error_msg and device == "mps":
                error_msg = "int8 compute type is not compatible with Apple Silicon (MPS). Use float16 instead."
            elif "out of memory" in error_msg.lower():
                error_msg = f"Out of memory error: {error_msg}. Try using a smaller model or different compute_type."
            
            self.model_load_error = error_msg
            if self.verbose:
                print(f"Error loading model: {error_msg}")
            return False
            
        finally:
            self.model_loading = False
    
    def transcribe(self, audio_path, **options):
        """
        Transcribe audio using the loaded model.
        
        Args:
            audio_path: Path to audio file
            **options: Options to pass to the model's transcribe method
            
        Returns:
            Transcription result
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        # Default options
        default_options = {
            "beam_size": 5,
            "language": None,
            "task": "transcribe",
            "temperature": 0,
            "initial_prompt": None,
            "word_timestamps": False,
            "vad_filter": True,
            "vad_parameters": None
        }
        
        # Update with user options
        for k, v in options.items():
            if v is not None:
                default_options[k] = v
                
        # Map option names to the ones expected by faster-whisper
        option_mapping = {
            "prompt": "initial_prompt",
            "word_timestamps": "word_timestamps",
            "beam_size": "beam_size",
            "language": "language",
            "task": "task",
            "temperature": "temperature",
            "vad_filter": "vad_filter",
            "vad_parameters": "vad_parameters"
        }
        
        # Create options dict for faster-whisper
        whisper_options = {}
        for k, v in option_mapping.items():
            if k in options and options[k] is not None:
                whisper_options[v] = options[k]
            elif v in default_options:
                whisper_options[v] = default_options[v]
                
        # Transcribe
        with self._model_lock:
            segments, info = self.model.transcribe(audio_path, **whisper_options)
            
            # Process segments
            result = {
                "text": "",
                "segments": [],
                "language": info.language,
                "language_probability": info.language_probability
            }
            
            # Collect segments
            for segment in segments:
                result["text"] += segment.text + " "
                segment_data = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                }
                
                # Add word timestamps if available
                if hasattr(segment, "words") and segment.words:
                    segment_data["words"] = [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                    
                result["segments"].append(segment_data)
                
            result["text"] = result["text"].strip()
            return result
            
    def get_info(self):
        """Get information about the service and loaded model"""
        info = {
            "system": self.system_info,
            "service": {
                "compute_type": self.compute_type,
                "device": self.device,
                "cpu_threads": self.cpu_threads,
                "num_workers": self.num_workers
            },
            "model": {
                "loaded": self.model is not None,
                "id": self.model_id,
                "loading": self.model_loading,
                "load_time": self.model_load_time,
                "load_error": self.model_load_error
            }
        }
        return info 