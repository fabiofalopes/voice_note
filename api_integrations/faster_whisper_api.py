import os
import torch
import platform
import time
from typing import Dict, Any, Optional, List
from tqdm import tqdm

class FasterWhisperAPI:
    """API for faster-whisper models using CTranslate2 backend"""
    
    models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "distil-large-v3"]
    default_model = "large-v3"
    #default_model = "tiny"
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
            device: Device to use ("cpu", "cuda", "auto")
            cpu_threads: Number of CPU threads to use (None = auto)
            num_workers: Number of workers for parallel processing
            download_root: Path to download and cache the model
            local_files_only: If True, only use local files for model loading
        """
        self.model = None
        self.metrics = {"load_times": {}, "transcription_times": {}}
        
        # Mac-specific optimizations
        is_mac = platform.system() == "Darwin"
        if is_mac:
            # On Mac, auto-set the number of CPU threads if not specified
            if cpu_threads is None:
                import multiprocessing
                cpu_threads = max(4, multiprocessing.cpu_count() - 2)  # Leave some cores free
            
            # Int8 is generally faster on Mac CPUs
            if compute_type == "float16" and device != "cuda":
                compute_type = "int8"
                print("Using int8 for better performance on Mac")
        
        # Device selection
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        self.local_files_only = local_files_only
    
    def load_model(self, model_id=None):
        """Load a model - with PROPER caching to avoid repeated downloads"""
        from faster_whisper import WhisperModel
        import os
        
        model_id = model_id or self.default_model
        
        # Start timing model loading
        start_time = time.time()
        
        # Set up a PERMANENT cache directory - properly created and managed
        if self.download_root is None:
            self.download_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper-models")
            os.makedirs(self.download_root, exist_ok=True)
            print(f"📂 Using cache directory: {self.download_root}")
        
        # Check if model is already loaded
        if self.model is not None and getattr(self.model, 'model_size', None) == model_id:
            print(f"🔄 Model {model_id} already loaded in memory - reusing")
            return self.model
        
        # Check if model files already exist in cache to avoid download
        model_dir = os.path.join(self.download_root, model_id)
        model_exists = os.path.exists(os.path.join(model_dir, "model.bin"))
        
        if model_exists:
            print(f"📦 Found existing model files for {model_id} in cache")
        else:
            print(f"⬇️  Model {model_id} not in cache - downloading...")
            # Create a fake progress bar for download as we can't hook into the download process directly
            with tqdm(total=100, desc=f"Downloading {model_id} model", unit="%") as pbar:
                for i in range(10):
                    time.sleep(0.1)  # Just to show the progress bar
                    pbar.update(10)
        
        # Mac-specific optimizations
        is_mac = platform.system() == "Darwin"
        compute_type = self.compute_type
        
        if is_mac:
            # For all models on Mac, int8 is much faster with Apple Silicon
            if compute_type != "int8":
                compute_type = "int8"
                print(f"🍎 Using int8 for {model_id} model on Apple Silicon")
            
            # Ensure we're using enough CPU threads on Mac
            if self.cpu_threads is None or self.cpu_threads < 4:
                import multiprocessing
                self.cpu_threads = max(4, multiprocessing.cpu_count() - 1)
                print(f"⚙️  Using {self.cpu_threads} CPU threads on Mac")
        
        # Initialize with optimized parameters
        try:
            print(f"🔄 Loading model {model_id}...")
            with tqdm(total=100, desc="Loading model", unit="%") as pbar:
                # Create a progress bar for model loading
                for i in range(5):
                    pbar.update(10)
                    time.sleep(0.1)
                
                self.model = WhisperModel(
                    model_size_or_path=model_id,
                    device=self.device,
                    compute_type=compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                    download_root=self.download_root,
                    local_files_only=False  # Allow download if needed the first time
                )
                
                # Update progress bar to completion
                pbar.update(50)
            
            load_time = time.time() - start_time
            self.metrics["load_times"][model_id] = load_time
            
            print(f"✅ Model {model_id} loaded successfully in {load_time:.2f}s")
            return self.model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try with more conservative settings if initial load failed
            try:
                print("⚠️ Retrying with conservative settings...")
                self.model = WhisperModel(
                    model_size_or_path=model_id,
                    device="cpu",  # Fall back to CPU
                    compute_type="int8",  # Use int8 for reliability
                    cpu_threads=2,  # Minimum threads
                    download_root=self.download_root
                )
                load_time = time.time() - start_time
                self.metrics["load_times"][model_id] = load_time
                print(f"✅ Model loaded with conservative settings in {load_time:.2f}s")
                return self.model
            except Exception as e:
                print(f"❌ Failed to load model even with conservative settings: {e}")
                raise
    
    def transcribe_audio(self, file_path: str, model_id: str = None, **kwargs):
        """
        Transcribe audio using faster-whisper.
        
        Args:
            file_path: Path to the audio file
            model_id: Model ID to use for transcription 
            **kwargs: Additional arguments for transcription
        """
        model_id = model_id or self.default_model
        model = self.load_model(model_id)
        
        # Start timing transcription
        start_time = time.time()
        
        # Extract known parameters but remove them from kwargs
        beam_size = kwargs.pop('beam_size', 5)
        language = kwargs.pop('language', None)
        task = kwargs.pop('task', 'transcribe')
        vad_filter = kwargs.pop('vad_filter', True)
        word_timestamps = kwargs.pop('word_timestamps', False)
        
        # Handle parameters that faster-whisper doesn't support
        prompt = kwargs.pop('prompt', None)  # Store it but don't pass to faster-whisper
        temperature = kwargs.pop('temperature', None)  # Store but don't pass
        
        # If prompt was provided, log it but explain it's not supported
        if prompt:
            print(f"ℹ️  Note: Prompt '{prompt}' will be ignored - faster-whisper doesn't support prompting")
        
        # Mac-specific optimizations
        is_mac = platform.system() == "Darwin"
        if is_mac:
            # These parameters have been found to work well on Mac
            vad_parameters = kwargs.pop('vad_parameters', {
                "min_silence_duration_ms": 500,  # Shorter silence detection
                "threshold": 0.5  # Default threshold
            })
        else:
            vad_parameters = kwargs.pop('vad_parameters', None)
        
        # Nice formatted parameters display
        print("\n" + "="*50)
        print("🎯 TRANSCRIPTION PARAMETERS")
        print("="*50)
        print(f"🔊 Model: {model_id}")
        print(f"🔍 Beam size: {beam_size}")
        if language:
            print(f"🌐 Language: {language}")
        print(f"🔉 VAD filter: {'Enabled' if vad_filter else 'Disabled'}")
        print(f"⏱️  Word timestamps: {'Enabled' if word_timestamps else 'Disabled'}")
        print(f"📋 Task: {task}")
        print("="*50 + "\n")
        
        print(f"⏳ Processing audio with {model_id} model...")
        
        # Run transcription with explicit parameters
        segments, info = model.transcribe(
            file_path,
            beam_size=beam_size,
            language=language,
            task=task,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None,
            word_timestamps=word_timestamps,
            **kwargs  # Any remaining kwargs
        )
        
        # Create a progress bar for transcription completion
        print("⏳ Assembling transcription results...")
        with tqdm(desc="Processing", unit="segment") as pbar:
            # Convert generator to list with progress updates
            segments_list = []
            for segment in segments:
                segments_list.append(segment)
                pbar.update(1)
        
        # Calculate and store transcription time
        transcription_time = time.time() - start_time
        self.metrics["transcription_times"][model_id] = transcription_time
        
        # Format response to match other APIs
        full_text = " ".join(segment.text for segment in segments_list)
        
        # Format segments into a standard format if needed
        formatted_segments = []
        for segment in segments_list:
            formatted_segment = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            
            # Add word-level info if requested
            if word_timestamps and hasattr(segment, 'words'):
                formatted_segment["words"] = [
                    {"start": word.start, "end": word.end, "word": word.word}
                    for word in segment.words
                ]
                
            formatted_segments.append(formatted_segment)
        
        # Calculate real-time factor
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            audio_duration = len(audio) / 1000  # in seconds
            real_time_factor = transcription_time / audio_duration
        except Exception:
            real_time_factor = None
            audio_duration = None
        
        print(f"✅ Transcription completed in {transcription_time:.2f}s")
        if real_time_factor:
            print(f"⏱️  Real-time factor: {real_time_factor:.2f}x (lower is better)")
            print(f"📊 Audio duration: {audio_duration:.2f}s, Processing time: {transcription_time:.2f}s")
        
        return {
            "text": full_text,
            "model": model_id,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": formatted_segments,
            "summary": "Summary not available for local transcription",
            "sentiment_analysis": "Sentiment analysis not available for local transcription",
            "task_analysis": "Task analysis not available for local transcription",
            "metrics": {
                "transcription_time": transcription_time,
                "real_time_factor": real_time_factor,
                "audio_duration": audio_duration
            }
        }
    
    def transcribe_with_vad(self, file_path: str, model_id: str = None, **kwargs):
        """Transcribe with VAD enabled (just a convenience method)"""
        kwargs['vad_filter'] = True
        return self.transcribe_audio(file_path, model_id, **kwargs)
    
    def transcribe_with_word_timestamps(self, file_path: str, model_id: str = None, **kwargs):
        """Transcribe with word timestamps enabled (just a convenience method)"""
        kwargs['word_timestamps'] = True
        return self.transcribe_audio(file_path, model_id, **kwargs)
        
    def translate_audio(self, file_path: str, model_id: str = None, **kwargs):
        """Translate audio to English (or target language if specified)"""
        kwargs['task'] = 'translate'
        return self.transcribe_audio(file_path, model_id, **kwargs) 