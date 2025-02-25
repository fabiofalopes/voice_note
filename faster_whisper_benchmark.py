#!/usr/bin/env python3
"""
faster_whisper_benchmark.py - A script to benchmark faster-whisper models

Usage:
    python faster_whisper_benchmark.py --audio_file <path_to_audio_file> [--output_dir results]
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import psutil
import torch
import platform
from pydub import AudioSegment

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from api_integrations.faster_whisper_api import FasterWhisperAPI
from audio_processing.recorder import AudioRecorder

def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:.2f}s"

def format_memory(bytes):
    """Format bytes into a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} GB"

def benchmark_models(audio_path, output_dir):
    """Benchmark different faster-whisper models on the given audio file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio file size and duration
    audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000  # in seconds
    print(f"Audio file: {audio_path} ({audio_size_mb:.2f} MB, {audio_duration:.2f} seconds)")
    
    # Initialize results dictionary
    results = {}
    model_data = []
    
    # Test each faster-whisper model size
    models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    
    # Mac-specific model selection
    is_mac = platform.system() == "Darwin"
    if is_mac:
        print("Running on Mac - optimizing model selection")
    
    # Track the true process baseline before any models are loaded
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(2)  # Wait for memory to stabilize
    true_baseline = psutil.Process(os.getpid()).memory_info().rss
    print(f"Initial process memory: {format_memory(true_baseline)}")
        
    for base_model in models:
        model_name = f"faster-{base_model}"
        print(f"\n{'='*50}")
        print(f"Testing model: {model_name}")
        print(f"{'='*50}")
        
        try:
            # Force garbage collection and restart Python's memory tracking
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            time.sleep(2)  # Give system time to stabilize
            
            # Get baseline memory before we do anything with this specific model
            pre_model_memory = psutil.Process(os.getpid()).memory_info().rss
            print(f"Pre-model memory: {format_memory(pre_model_memory)}")
            
            # Create API instance with optimized parameters
            load_start = time.time()
            whisper_api = FasterWhisperAPI(
                compute_type="int8" if is_mac else "float16",
                device="auto",
                cpu_threads=8 if is_mac else None,  # Use more cores on Mac
                num_workers=4   # Parallel processing
            )
            
            # Load the model
            model = whisper_api.load_model(base_model)
            load_end = time.time()
            load_time = load_end - load_start
            print(f"Model loading time: {format_time(load_time)}")
            
            # Take multiple measurements after loading for stability
            time.sleep(1)  # Allow memory to stabilize
            post_load_memory = 0
            for _ in range(3):  # Take 3 measurements and average
                current = psutil.Process(os.getpid()).memory_info().rss
                post_load_memory += current
                time.sleep(0.5)
            post_load_memory = post_load_memory / 3  # Average
            
            memory_used_load = post_load_memory - pre_model_memory
            
            # Force positive memory usage - models MUST use memory
            if memory_used_load <= 0:
                print(f"WARNING: Detected impossible negative memory usage. Using absolute measurement.")
                memory_used_load = post_load_memory - true_baseline
            
            print(f"Memory used for model loading: {format_memory(memory_used_load)}")
            
            # Now run transcription
            print("Transcribing audio...")
            transcribe_start = time.time()
            
            result = whisper_api.transcribe_audio(
                audio_path,
                vad_filter=True,
                beam_size=5
            )
            
            transcribe_end = time.time()
            transcribe_time = transcribe_end - transcribe_start
            print(f"Transcription time: {format_time(transcribe_time)}")
            
            # Take final memory measurements
            time.sleep(1)  # Allow memory to stabilize after transcription
            final_memory = 0
            for _ in range(3):  # Take 3 measurements and average
                current = psutil.Process(os.getpid()).memory_info().rss
                final_memory += current
                time.sleep(0.5)
            final_memory = final_memory / 3  # Average
            
            # Calculate against true baseline for consistent comparison
            memory_footprint = final_memory - true_baseline
            
            # Reliable metrics
            rtf = transcribe_time / audio_duration
            text = result.get("text", "")
            word_count = len(text.split())
            words_per_second = word_count / transcribe_time if transcribe_time > 0 else 0
            
            print(f"Total memory used: {format_memory(memory_footprint)}")
            print(f"Real-time factor: {rtf:.2f}x (lower is better)")
            print(f"Processing speed: {words_per_second:.2f} words/second")
            print(f"Word count: {word_count}")
            
            # Store results
            results[model_name] = {
                "model_name": model_name,
                "load_time_seconds": load_time,
                "transcribe_time_seconds": transcribe_time,
                "total_time_seconds": load_time + transcribe_time,
                "memory_usage_bytes": memory_footprint,
                "peak_memory_bytes": max(memory_used_load, memory_footprint),
                "words_per_second": words_per_second,
                "word_count": word_count,
                "sample_text": text[:150],
                "real_time_factor": rtf
            }
            
            model_data.append(results[model_name])
            
            # Clean up
            whisper_api.model = None
            del model
            del whisper_api
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            time.sleep(1)  # Give time for memory to be released
            
        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
            results[model_name] = {
                "model_name": model_name,
                "error": str(e)
            }
    
    # Save benchmark results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n\n" + "="*80)
    print(" FASTER-WHISPER MODEL COMPARISON ".center(80, "="))
    print("="*80)
    print(f"{'Model':<10} | {'Time':<15} | {'Memory':<12} | {'Words/sec':<10} | {'Word Count':<10}")
    print("-"*80)
    
    for model in model_data:
        print(
            f"{model['model_name']:<10} | "
            f"{format_time(model['total_time_seconds']):<15} | "
            f"{format_memory(model['memory_usage_bytes']):<12} | "
            f"{model['words_per_second']:<10.2f} | "
            f"{model['word_count']:<10}"
        )
    
    print("="*80)
    print(f"Results saved to {results_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper models")
    parser.add_argument("--audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--record", action="store_true", help="Record audio for a fixed duration")
    parser.add_argument("--record-until-q", action="store_true", help="Record audio until Ctrl+C is pressed")
    parser.add_argument("--duration", type=int, default=5, help="Duration to record in seconds (default: 5)")
    parser.add_argument("--output_dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--input-device", type=int, help="Input device index to use for recording")
    
    args = parser.parse_args()
    
    # Handle recording or using an existing file
    if args.record_until_q:
        print("Recording audio. Press Ctrl+C to stop recording when finished...")
        recorder = AudioRecorder()
        audio_path = recorder.record_until_q("benchmark_input.wav", args.input_device)
        if not audio_path:
            print("Error: Recording failed. Please check your microphone.")
            sys.exit(1)
        print(f"Recording saved to {audio_path}")
    elif args.record:
        print(f"Recording audio for {args.duration} seconds...")
        recorder = AudioRecorder()
        audio_path = recorder.record(args.duration, "benchmark_input.wav", args.input_device)
        if not audio_path:
            print("Error: Recording failed. Please check your microphone.")
            sys.exit(1)
        print(f"Recording saved to {audio_path}")
    elif args.audio_file:
        audio_path = args.audio_file
        if not os.path.exists(audio_path):
            print(f"Error: Audio file '{audio_path}' not found")
            sys.exit(1)
    else:
        print("Error: Either --audio_file, --record, or --record-until-q must be specified")
        parser.print_help()
        sys.exit(1)
    
    benchmark_models(audio_path, args.output_dir) 