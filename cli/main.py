#!/usr/bin/env python3
"""
cli/main.py - Main CLI for Voice Note

This module provides the main command-line interface for the Voice Note application,
allowing users to record, transcribe, and analyze audio using various APIs.
"""

import argparse
import os
import sys
import json
import pyaudio
import time
from tqdm import tqdm
import pyperclip  # Add this import for clipboard functionality

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from api_integrations.groq_whisper import GroqWhisperAPI
from faster_whisper_daemon.client import FasterWhisperClient, find_latest_socket
from audio_processing.recorder import AudioRecorder, list_audio_devices
from audio_processing.utils import convert_audio_to_wav

def print_header(title):
    """Print a nicely formatted header"""
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Voice Note - Record and transcribe audio')
    
    # Input/output arguments
    parser.add_argument('audio_file', nargs='?', help='Audio file to transcribe (if not recording)')
    parser.add_argument('--output', '-o', type=str, help='Output file for transcription results (JSON)')
    
    # Recording options
    recording_group = parser.add_argument_group('Recording options')
    recording_group.add_argument('--record', '-r', action='store_true', help='Record audio before transcribing')
    recording_group.add_argument('--record-until-q', action='store_true', help='Record until q is pressed')
    recording_group.add_argument('--duration', '-d', type=int, default=10, help='Recording duration in seconds')
    recording_group.add_argument('--input-device', '-i', type=int, help='Input device index')
    recording_group.add_argument('--list-devices', action='store_true', help='List available audio input devices')
    
    # Transcription options
    transcription_group = parser.add_argument_group('Transcription options')
    transcription_group.add_argument('--transcribe', '-t', action='store_true', help='Transcribe the audio')
    transcription_group.add_argument('--translate', action='store_true', help='Translate the audio to English')
    transcription_group.add_argument('--api', choices=['local', 'groq', 'auto'], default='auto', 
                                    help='API to use for transcription (auto will try local first, then groq)')
    transcription_group.add_argument('--model', type=str, help='Model to use for transcription')
    transcription_group.add_argument('--language', type=str, help='Language code (e.g., "en", "fr")')
    transcription_group.add_argument('--prompt', type=str, help='Initial prompt for transcription')
    transcription_group.add_argument('--temperature', type=float, default=0.0, 
                                    help='Temperature for sampling (0.0-1.0)')
    transcription_group.add_argument('--word-timestamps', action='store_true', 
                                    help='Include timestamps for each word')
    transcription_group.add_argument('--vad', action='store_true', 
                                    help='Use Voice Activity Detection')
    transcription_group.add_argument('--beam-size', type=int, 
                                    help='Beam size for faster-whisper')
    transcription_group.add_argument('--raw-transcription', action='store_true',
                                    help='Return only the raw transcription without additional analysis')
    
    # Daemon options
    daemon_group = parser.add_argument_group('Daemon options (for local API)')
    daemon_group.add_argument('--daemon-socket', type=str, help='Path to the daemon socket file')
    daemon_group.add_argument('--daemon-host', type=str, default='localhost', help='Daemon host (if using TCP)')
    daemon_group.add_argument('--daemon-port', type=int, default=9876, help='Daemon port (if using TCP)')
    daemon_group.add_argument('--daemon-tcp', action='store_true', help='Use TCP to connect to daemon instead of Unix socket')
    daemon_group.add_argument('--no-fallback', action='store_true', help='Disable fallback to Groq API if local transcription fails')
    
    args = parser.parse_args()
    
    # List audio devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Record audio if requested
    if args.record or args.record_until_q:
        if not args.audio_file:
            # Generate a default filename if none provided
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args.audio_file = f"recording_{timestamp}.wav"
        
        print(f"🎤 Recording to {args.audio_file}...")
        recorder = AudioRecorder(device_index=args.input_device)
        
        if args.record_until_q:
            print("Press 'q' to stop recording...")
            args.audio_file = recorder.record_until_q(args.audio_file)
        else:
            print(f"Recording for {args.duration} seconds...")
            args.audio_file = recorder.record(args.duration, args.audio_file)
        
        print(f"✅ Recording saved to {args.audio_file}")
    
    # Check if we have an audio file to work with
    if not args.audio_file:
        parser.print_help()
        print("\n❌ Error: No audio file specified. Use --record or provide an audio file.")
        return
    
    # Convert audio to WAV if needed
    if not args.audio_file.lower().endswith('.wav'):
        print(f"🔄 Converting {args.audio_file} to WAV format...")
        wav_file = convert_audio_to_wav(args.audio_file)
        print(f"✅ Converted to {wav_file}")
        args.audio_file = wav_file
    
    # Transcribe audio if requested
    if args.transcribe or args.translate:
        if not args.raw_transcription:
            print(f"🔊 Processing audio: {args.audio_file}")
        
        # Prepare transcription options
        options = {
            'language': args.language,
            'prompt': args.prompt,
            'temperature': args.temperature,
            'word_timestamps': args.word_timestamps,
            'beam_size': args.beam_size,
            'vad_filter': args.vad,
            'task': 'translate' if args.translate else 'transcribe'
        }
        
        # Filter out None values
        options = {k: v for k, v in options.items() if v is not None}
        
        # Determine which API to use
        use_local = args.api in ['local', 'auto']
        use_groq = args.api in ['groq']
        allow_fallback = args.api == 'auto' and not args.no_fallback
        
        result = None
        local_error = None
        
        # Try local API first if requested
        if use_local:
            try:
                # Find socket path if not provided
                socket_path = args.daemon_socket
                if not socket_path and not args.daemon_tcp:
                    socket_path = find_latest_socket()
                
                client = FasterWhisperClient(
                    socket_path=socket_path,
                    host=args.daemon_host,
                    port=args.daemon_port,
                    use_tcp=args.daemon_tcp
                )
                
                # Check daemon status
                try:
                    status = client.get_status()
                    if not args.raw_transcription:
                        print(f"✅ Connected to daemon. Status: {status.get('status', 'unknown')}")
                except Exception as e:
                    error_msg = f"Error connecting to daemon: {e}"
                    if not args.raw_transcription:
                        print(f"❌ {error_msg}")
                        print("   Is the daemon running? Start it with: python -m faster_whisper_daemon.cli start")
                    local_error = error_msg
                    if allow_fallback:
                        if not args.raw_transcription:
                            print("   Falling back to Groq API...")
                        use_groq = True
                    else:
                        return
                
                if not local_error:
                    # Display transcription settings
                    if not args.raw_transcription:
                        print(f"🔍 Transcribing with local API (faster-whisper)")
                        model_display = args.model if args.model else "Default (daemon's default model)"
                        print(f"   Model: {model_display}")
                    
                    try:
                        # Transcribe using the daemon
                        start_time = time.time()
                        
                        # Submit the job to the daemon
                        if not args.raw_transcription:
                            print("📤 Submitting transcription job to daemon...")
                        response = client.transcribe(
                            args.audio_file,
                            model_id=args.model,
                            **options
                        )
                        
                        if 'error' in response:
                            error_msg = f"Error submitting job: {response['error']}"
                            if not args.raw_transcription:
                                print(f"❌ {error_msg}")
                            local_error = error_msg
                            if allow_fallback:
                                if not args.raw_transcription:
                                    print("   Falling back to Groq API...")
                                use_groq = True
                            else:
                                return
                        else:
                            job_id = response.get('job_id')
                            if not args.raw_transcription:
                                print(f"✅ Job submitted. ID: {job_id}")
                            
                            # Wait for job to complete
                            if not args.raw_transcription:
                                print("⏳ Waiting for transcription to complete...")
                            while True:
                                job_status = client.get_job_status(job_id)
                                if not job_status:
                                    error_msg = "Job not found"
                                    if not args.raw_transcription:
                                        print(f"❌ {error_msg}")
                                    local_error = error_msg
                                    break
                                    
                                status = job_status.get('status')
                                progress = job_status.get('progress', 0)
                                
                                if not args.raw_transcription:
                                    print(f"   Status: {status}, Progress: {progress}%", end="\r")
                                
                                if status == 'completed':
                                    result = job_status.get('result', {})
                                    elapsed = time.time() - start_time
                                    if not args.raw_transcription:
                                        print(f"\n✅ Transcription completed in {elapsed:.2f} seconds")
                                    
                                    # Clean up job
                                    client.cleanup_job(job_id)
                                    break
                                    
                                elif status == 'failed':
                                    error_msg = f"Transcription failed: {job_status.get('error')}"
                                    if not args.raw_transcription:
                                        print(f"\n❌ {error_msg}")
                                    local_error = error_msg
                                    if allow_fallback:
                                        if not args.raw_transcription:
                                            print("   Falling back to Groq API...")
                                        use_groq = True
                                    break
                                    
                                time.sleep(0.5)
                            
                    except Exception as e:
                        error_msg = f"Error using daemon: {e}"
                        if not args.raw_transcription:
                            print(f"❌ {error_msg}")
                        local_error = error_msg
                        if allow_fallback:
                            if not args.raw_transcription:
                                print("   Falling back to Groq API...")
                            use_groq = True
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                if not args.raw_transcription:
                    print(f"❌ {error_msg}")
                local_error = error_msg
                if allow_fallback:
                    if not args.raw_transcription:
                        print("   Falling back to Groq API...")
                    use_groq = True
        
        # Use Groq API if requested or as fallback
        if use_groq and not result:
            try:
                client = GroqWhisperAPI()
                
                if not args.raw_transcription:
                    print(f"🔍 Transcribing with Groq API")
                    print(f"   Model: {args.model or 'whisper-large-v3'}")
                
                start_time = time.time()
                
                # Prepare Groq-specific options
                groq_options = {
                    'model_id': args.model or "whisper-large-v3",
                    'prompt': args.prompt,
                    'language': args.language,
                    'temperature': args.temperature,
                    'raw_transcription': args.raw_transcription
                }
                
                # Filter out None values
                groq_options = {k: v for k, v in groq_options.items() if v is not None}
                
                if args.translate:
                    result = client.translate_audio(args.audio_file, **groq_options)
                else:
                    result = client.transcribe_audio(args.audio_file, **groq_options)
                
                elapsed = time.time() - start_time
                if not args.raw_transcription:
                    print(f"✅ Transcription completed in {elapsed:.2f} seconds")
                
            except Exception as e:
                if not args.raw_transcription:
                    print(f"❌ Error using Groq API: {e}")
                    if local_error:
                        print(f"   Local API also failed: {local_error}")
                return
        
        # Display and save results if we have them
        if result:
            # For raw transcription, just show the text and copy to clipboard
            if args.raw_transcription:
                text = result if isinstance(result, str) else result.get('text', '')
                print(text)  # Just print the text without any formatting
                try:
                    pyperclip.copy(text)
                    print("\n📋 Copied to clipboard!")
                except Exception as e:
                    print(f"\n⚠️ Could not copy to clipboard: {e}")
                return
            
            # Display the result with full formatting
            print(f"\n🔤 Transcription result:")
            
            if isinstance(result, dict):
                text = result.get('text', '')
                print(f"\n{text}")
                
                # Show additional info if available
                if 'summary' in result:
                    print(f"\n📝 Summary: {result['summary']}")
                if 'sentiment_analysis' in result:
                    print(f"\n😊 Sentiment: {result['sentiment_analysis']}")
                if 'task_analysis' in result:
                    print(f"\n🎯 Task Analysis: {result['task_analysis']}")
                if 'thinking_tags' in result:
                    print(f"\n🤔 Thinking Tags: {result['thinking_tags']}")
                
                # Save to file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n💾 Results saved to {args.output}")
            else:
                print(f"\n{result}")
                
                # Save to file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(result)
                    print(f"\n💾 Results saved to {args.output}")
        else:
            print("\n❌ No transcription result obtained.")

if __name__ == "__main__":
    main()