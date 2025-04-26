"""
audio_processing/utils.py - Audio processing utilities

This module provides utility functions for audio processing, such as format conversion,
audio analysis, and other common operations.
"""

import os
import subprocess
import tempfile
import shutil
import wave

def convert_audio_to_wav(input_file, output_file=None):
    """
    Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str, optional): Path to save the output WAV file. 
                                    If None, creates a file with the same name but .wav extension.
    
    Returns:
        str: Path to the converted WAV file
    
    Raises:
        RuntimeError: If ffmpeg is not installed or conversion fails
    """
    if output_file is None:
        # Create output filename with same name but .wav extension
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.wav"
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg to convert audio files.")
    
    # Create a temporary file for the conversion
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, "temp_output.wav")
    
    try:
        # Run ffmpeg to convert the file
        cmd = [
            "ffmpeg",
            "-i", input_file,
            "-ar", "44100",  # Sample rate
            "-ac", "1",      # Mono channel
            "-c:a", "pcm_s16le",  # 16-bit PCM
            "-y",            # Overwrite output file if it exists
            temp_output
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Move the temporary file to the final location
        shutil.move(temp_output, output_file)
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error converting audio file: {e.stderr.decode('utf-8')}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def get_audio_duration(audio_file):
    """
    Get the duration of a WAV audio file in seconds.
    
    Args:
        audio_file (str): Path to the WAV audio file
    
    Returns:
        float: Duration of the audio file in seconds
    
    Raises:
        ValueError: If the file is not a valid WAV file
    """
    try:
        with wave.open(audio_file, 'rb') as wf:
            # Get the number of frames and the framerate
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        # If it's not a WAV file, try using ffmpeg
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                audio_file
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except:
            raise ValueError(f"Could not determine duration of audio file: {audio_file}")

def is_silent(audio_data, threshold=500):
    """
    Check if audio data is silent based on a threshold.
    
    Args:
        audio_data (bytes): Audio data as bytes
        threshold (int): Silence threshold
    
    Returns:
        bool: True if the audio is silent, False otherwise
    """
    # Convert bytes to integers
    import struct
    import numpy as np
    
    # Assuming 16-bit audio (2 bytes per sample)
    format_str = f"<{len(audio_data)//2}h"
    try:
        samples = struct.unpack(format_str, audio_data)
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.array(samples)**2))
        return rms < threshold
    except:
        # If unpacking fails, assume it's silent
        return True 