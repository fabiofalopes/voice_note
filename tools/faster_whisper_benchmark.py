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

from faster_whisper_daemon.api import FasterWhisperAPI
from audio_processing.recorder import AudioRecorder

def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)} minutes {int(seconds)} seconds"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"

def format_memory(bytes):
    """Format bytes into a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"

def benchmark_models(audio_path, output_dir):
    """Benchmark different faster-whisper models on the specified audio file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio file info
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000.0  # in seconds
    audio_size = os.path.getsize(audio_path)
    
    print(f"🎵 Audio file: {audio_path}")
    print(f"   Duration: {format_time(audio_duration)}")
    print(f"   Size: {format_memory(audio_size)}")
    print()
    
    # System info
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        system_info["cuda_available"] = True
        system_info["cuda_version"] = torch.version.cuda
        system_info["cuda_device_count"] = torch.cuda.device_count()
        system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        system_info["cuda_available"] = False
        
    # Add MPS info if available (Apple Silicon)
    if hasattr(torch, 'mps') and torch.mps.is_available():
        system_info["mps_available"] = True
    else:
        system_info["mps_available"] = False
    
    print("💻 System information:")
    print(f"   Platform: {system_info['platform']}")
    print(f"   Python: {system_info['python_version']}")
    print(f"   CPU: {system_info['physical_cpu_count']} cores ({system_info['cpu_count']} threads)")
    print(f"   Memory: {format_memory(system_info['memory_total'])}")
    
    if system_info.get("cuda_available"):
        print(f"   CUDA: {system_info['cuda_version']}")
        print(f"   GPU: {system_info['cuda_device_name']}")
    elif system_info.get("mps_available"):
        print(f"   MPS: Available (Apple Silicon)")
    else:
        print(f"   GPU: Not available")
    print()
    
    # Models to benchmark
    models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    
    # Compute types to test
    if system_info.get("cuda_available"):
        compute_types = ["float16", "int8"]
    elif system_info.get("mps_available"):
        compute_types = ["float16"]
    else:
        compute_types = ["int8"]
    
    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "audio": {
            "path": audio_path,
            "duration": audio_duration,
            "size": audio_size
        },
        "system": system_info,
        "benchmarks": []
    }
    
    # Run benchmarks
    for model in models:
        for compute_type in compute_types:
            print(f"🔍 Benchmarking model: {model} (compute_type: {compute_type})")
            
            # Initialize API
            api = FasterWhisperAPI(compute_type=compute_type)
            
            # Measure memory before loading model
            memory_before = psutil.Process(os.getpid()).memory_info().rss
            
            # Measure model loading time
            load_start = time.time()
            api.service.load_model(model)
            load_time = time.time() - load_start
            
            # Measure memory after loading model
            memory_after = psutil.Process(os.getpid()).memory_info().rss
            memory_used = memory_after - memory_before
            
            print(f"   Model loaded in {load_time:.2f} seconds")
            print(f"   Memory used: {format_memory(memory_used)}")
            
            # Measure transcription time
            transcribe_start = time.time()
            result = api.transcribe_audio(audio_path, model_id=model)
            transcribe_time = time.time() - transcribe_start
            
            # Calculate real-time factor
            rtf = transcribe_time / audio_duration
            
            print(f"   Transcription time: {transcribe_time:.2f} seconds")
            print(f"   Real-time factor: {rtf:.2f}x")
            print(f"   Text length: {len(result.get('text', ''))}")
            print()
            
            # Add to results
            benchmark = {
                "model": model,
                "compute_type": compute_type,
                "load_time": load_time,
                "transcribe_time": transcribe_time,
                "memory_used": memory_used,
                "rtf": rtf,
                "text_length": len(result.get('text', '')),
                "text": result.get('text', '')
            }
            results["benchmarks"].append(benchmark)
            
            # Force garbage collection
            api = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Wait a bit to let memory settle
            time.sleep(1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Benchmark results saved to {output_file}")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"{'Model':<10} {'Compute':<8} {'Load Time':<12} {'Transcribe':<12} {'RTF':<8} {'Memory':<12}")
    print("-" * 70)
    
    for benchmark in results["benchmarks"]:
        print(f"{benchmark['model']:<10} {benchmark['compute_type']:<8} "
              f"{benchmark['load_time']:.2f}s{'':<6} "
              f"{benchmark['transcribe_time']:.2f}s{'':<4} "
              f"{benchmark['rtf']:.2f}x{'':<3} "
              f"{format_memory(benchmark['memory_used'])}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper models")
    parser.add_argument("--audio_file", required=True, help="Path to audio file to transcribe")
    parser.add_argument("--output_dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--record", action="store_true", help="Record audio instead of using a file")
    parser.add_argument("--duration", type=int, default=10, help="Duration to record in seconds")
    
    args = parser.parse_args()
    
    # Record audio if requested
    if args.record:
        print(f"🎤 Recording {args.duration} seconds of audio for benchmarking...")
        recorder = AudioRecorder()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        audio_file = f"benchmark_recording_{timestamp}.wav"
        recorder.record(args.duration, audio_file)
        print(f"✅ Recorded audio: {audio_file}")
        args.audio_file = audio_file
    
    # Run benchmark
    benchmark_models(args.audio_file, args.output_dir)

if __name__ == "__main__":
    main() 