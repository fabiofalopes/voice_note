#!/usr/bin/env python3
"""
Voice Transcriber CLI

Main command-line interface for the voice transcriber.
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Voice Transcriber - Record and transcribe audio')
    parser.add_argument('file', nargs='?', help='Audio file to transcribe (optional - if not provided, will record)')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices')
    parser.add_argument('--test-mic', action='store_true', help='Test microphone input levels')
    parser.add_argument('--device', '-d', type=int, help='Audio input device index')
    parser.add_argument('--model', '-m', default='whisper-large-v3', 
                       choices=['whisper-large-v3', 'whisper-large-v3-turbo', 'distil-whisper-large-v3-en'],
                       help='Whisper model to use')
    parser.add_argument('--output', '-o', default='recording.wav', help='Output audio filename')
    parser.add_argument('--no-clipboard', action='store_true', help='Don\'t copy to clipboard')
    parser.add_argument('--translate', action='store_true', help='Translate to English instead of transcribe')
    
    args = parser.parse_args()
    
    # Import dependencies only when needed (after help is shown)
    try:
        import pyperclip
        from audio_processing.recorder import AudioRecorder, list_audio_devices
        from api.groq_client import GroqWhisperClient
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install dependencies with: pip install -r requirements.txt")
        return 1
    
    # List audio devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0
    
    # Test microphone if requested
    if args.test_mic:
        recorder = AudioRecorder(device_index=args.device)
        if recorder.test_microphone(device_id=args.device):
            return 0
        else:
            return 1
    
    # Check if we're transcribing an existing file or recording new audio
    if args.file:
        # Transcribe existing file
        if not os.path.exists(args.file):
            print(f"❌ Audio file not found: {args.file}")
            return 1
        audio_file = args.file
        print(f"📁 Transcribing existing file: {audio_file}")
    else:
        # Record new audio
        # Initialize components
        recorder = AudioRecorder(device_index=args.device)
        
        # Show recording configuration
        if not recorder.recording_method:
            print("❌ No working audio recording method found.")
            print("💡 Try: python3 transcribe.py --list-devices")
            print("   Or: python3 transcribe.py --test-mic")
            return 1
        
        # Show what we're using (brief)
        print(f"🎤 {recorder.get_recording_info()}")
        
        # Record audio
        audio_file = recorder.record_until_q(args.output)
        if not audio_file:
            print("❌ Recording failed.")
            return 1
    
    # Transcribe or translate
    print(f"🔄 {'Translating' if args.translate else 'Transcribing'} with {args.model}...")
    
    try:
        client = GroqWhisperClient()
        if args.translate:
            text = client.translate(audio_file, model=args.model)
        else:
            text = client.transcribe(audio_file, model=args.model)
    except Exception as e:
        print(f"❌ API client error: {e}")
        return 1
    
    if text:
        print("\n" + "="*50)
        print("TRANSCRIPTION:" if not args.translate else "TRANSLATION:")
        print("="*50)
        print(text)
        print("="*50)
        
        # Copy to clipboard
        if not args.no_clipboard:
            try:
                pyperclip.copy(text)
                print("📋 Copied to clipboard!")
            except Exception as e:
                print(f"⚠️ Could not copy to clipboard: {e}")
        
        return 0
    else:
        print(f"❌ {'Translation' if args.translate else 'Transcription'} failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())