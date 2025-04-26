#!/usr/bin/env python3
"""
faster_whisper_daemon/client.py - Client for the FasterWhisperDaemon

This module provides a command-line interface for testing and interacting with 
a running FasterWhisperDaemon. It can connect to the daemon, check its status,
and perform transcription tasks.

Usage:
    python -m faster_whisper_daemon.client [--audio_file FILE] [--socket_path PATH] [--use_tcp]
"""

import os
import sys
import time
import argparse
import tempfile
import glob

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .daemon import FasterWhisperClient
from audio_processing.recorder import AudioRecorder

# Re-export FasterWhisperClient for use by other modules
__all__ = ['FasterWhisperClient', 'find_latest_socket']

def find_latest_socket():
    """Find the most recently created whisper daemon socket"""
    socket_pattern = os.path.join(tempfile.gettempdir(), "whisper_daemon_*.sock")
    sockets = glob.glob(socket_pattern)
    
    if not sockets:
        return None
        
    # Sort by creation time (newest first)
    sockets.sort(key=os.path.getctime, reverse=True)
    return sockets[0]

def main():
    """Main function for testing the daemon client"""
    parser = argparse.ArgumentParser(description='Test the FasterWhisperDaemon client')
    
    # Socket options
    parser.add_argument('--socket-path', type=str, help='Path to the daemon socket file')
    parser.add_argument('--host', type=str, default='localhost', help='Daemon host (if using TCP)')
    parser.add_argument('--port', type=int, default=9876, help='Daemon port (if using TCP)')
    parser.add_argument('--use-tcp', action='store_true', help='Use TCP instead of Unix socket')
    
    # Test options
    parser.add_argument('--audio-file', type=str, help='Audio file to transcribe')
    parser.add_argument('--record', action='store_true', help='Record audio before transcribing')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds')
    parser.add_argument('--model', type=str, help='Model to use for transcription')
    
    args = parser.parse_args()
    
    # Find socket path if not provided
    socket_path = args.socket_path
    if not socket_path and not args.use_tcp:
        socket_path = find_latest_socket()
        if not socket_path:
            print("❌ No daemon socket found. Is the daemon running?")
            print("   Start it with: python -m faster_whisper_daemon.cli start")
            return
            
    # Create client
    client = FasterWhisperClient(
        socket_path=socket_path,
        host=args.host,
        port=args.port,
        use_tcp=args.use_tcp
    )
    
    # Check daemon status
    try:
        status = client.get_status()
        print(f"✅ Connected to daemon. Status: {status.get('status', 'unknown')}")
        
        # Print model info
        info = status.get('info', {})
        model_info = info.get('model', {})
        if model_info.get('loaded'):
            print(f"   Model: {model_info.get('id')}")
        else:
            print("   No model loaded")
            
    except Exception as e:
        print(f"❌ Error connecting to daemon: {e}")
        print("   Is the daemon running? Start it with: python -m faster_whisper_daemon.cli start")
        return
        
    # Record audio if requested
    audio_file = args.audio_file
    if args.record:
        print(f"🎤 Recording for {args.duration} seconds...")
        recorder = AudioRecorder()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        audio_file = f"recording_{timestamp}.wav"
        recorder.record(args.duration, audio_file)
        print(f"✅ Recording saved to {audio_file}")
        
    # Transcribe audio if available
    if audio_file:
        print(f"🔊 Transcribing {audio_file}...")
        
        try:
            # Submit transcription job
            response = client.transcribe(audio_file, model_id=args.model)
            
            if 'error' in response:
                print(f"❌ Error: {response['error']}")
                return
                
            job_id = response.get('job_id')
            print(f"✅ Job submitted. ID: {job_id}")
            
            # Wait for job to complete
            print("⏳ Waiting for transcription to complete...")
            start_time = time.time()
            
            while True:
                job_status = client.get_job_status(job_id)
                
                if 'error' in job_status:
                    print(f"❌ Error: {job_status['error']}")
                    break
                    
                status = job_status.get('status')
                progress = job_status.get('progress', 0)
                
                print(f"   Status: {status}, Progress: {progress}%", end="\r")
                
                if status == 'completed':
                    result = job_status.get('result', {})
                    elapsed = time.time() - start_time
                    print(f"\n✅ Transcription completed in {elapsed:.2f} seconds")
                    
                    # Display the result
                    print(f"\n🔤 Transcription result:")
                    print(f"\n{result.get('text', 'No text returned')}")
                    
                    # Clean up job
                    client.cleanup_job(job_id)
                    break
                    
                elif status == 'failed':
                    print(f"\n❌ Transcription failed: {job_status.get('error')}")
                    break
                    
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
if __name__ == "__main__":
    main() 