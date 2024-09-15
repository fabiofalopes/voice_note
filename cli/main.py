import argparse
import os
import sys
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from api_integrations.groq_whisper import GroqWhisperAPI
from audio_processing.preprocess import preprocess_audio
from audio_processing.recorder import AudioRecorder

def main():
    parser = argparse.ArgumentParser(description="Record, transcribe, and analyze audio files.")
    parser.add_argument('--record', action='store_true', help='Record audio instead of using an existing file')
    parser.add_argument('--duration', type=int, default=10, help='Duration of recording in seconds (ignored if --record-until-q is used)')
    parser.add_argument('--record-until-q', action='store_true', help='Record audio until Ctrl+C is pressed')
    parser.add_argument('--list-devices', action='store_true', help='List available input devices')
    parser.add_argument('--input-device', type=int, help='Input device index to use for recording')
    parser.add_argument('--model', type=str, default='whisper-large-v3', help='Transcription model ID.')
    parser.add_argument('--prompt', type=str, help='Prompt for transcription')
    parser.add_argument('--language', type=str, help='Language of the audio')
    parser.add_argument('--temperature', type=float, help='Temperature for transcription')
    parser.add_argument('--output', type=str, help='Path to save the transcription output')
    parser.add_argument('file_path', type=str, nargs='?', help='Path to save the audio file or existing audio file.')

    args = parser.parse_args()

    recorder = AudioRecorder()

    if args.list_devices:
        recorder.list_input_devices()
        return

    if args.file_path is None and not args.list_devices:
        parser.error("file_path is required unless --list-devices is used")

    if args.record or args.record_until_q:
        try:
            if args.record_until_q:
                input_file = recorder.record_until_q(os.path.basename(args.file_path), args.input_device)
            else:
                input_file = recorder.record(args.duration, os.path.basename(args.file_path), args.input_device)
            if input_file is None:
                print("Recording failed. Please check your audio device settings.")
                return
            print(f"Audio recorded and saved to: {input_file}")
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            return
    else:
        input_file = args.file_path

    preprocessed_file = "preprocessed_" + os.path.basename(input_file)

    # Preprocess the audio file
    success = preprocess_audio(input_file, preprocessed_file)
    if not success:
        print("Audio preprocessing failed.")
        return

    # Initialize API
    api = GroqWhisperAPI()

    # Transcribe the audio
    result = api.transcribe_audio(
        preprocessed_file,
        model_id=args.model,
        prompt=args.prompt,
        language=args.language,
        temperature=args.temperature
    )

    # Handle transcription output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Transcription saved to {args.output}")
    else:
        print("Transcription Output:")
        print(json.dumps(result, indent=2))

    # Clean up preprocessed file
    os.remove(preprocessed_file)

if __name__ == "__main__":
    main()