import argparse
import os
import sys
import json
import pyaudio
import time
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from api_integrations.router import APIRouter
from api_integrations.groq_whisper import GroqWhisperAPI
from audio_processing.preprocess import preprocess_audio
from audio_processing.recorder import AudioRecorder

def print_header(title):
    """Print a nicely formatted header"""
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)

def main():
    # Start timing the whole process
    start_time = time.time()
    
    router = APIRouter()
    parser = argparse.ArgumentParser(description="Record, transcribe, and analyze audio files.")
    parser.add_argument('--record', action='store_true', help='Record audio instead of using an existing file')
    parser.add_argument('--duration', type=int, default=10, help='Duration of recording in seconds (ignored if --record-until-q is used)')
    parser.add_argument('--record-until-q', action='store_true', help='Record audio until Ctrl+C is pressed')
    parser.add_argument('--list-devices', action='store_true', help='List available input devices')
    parser.add_argument('--input-device', type=int, help='Input device index to use for recording')
    parser.add_argument('--prompt', type=str, help='Prompt for transcription')
    parser.add_argument('--language', type=str, help='Language of the audio')
    parser.add_argument('--temperature', type=float, help='Temperature for transcription')
    parser.add_argument('--output', type=str, help='Path to save the transcription output')
    parser.add_argument('--list-models', action='store_true', help='List available models for each API')
    parser.add_argument('--transcribe', action='store_true', help='Transcribe the audio')
    parser.add_argument('--translate', action='store_true', help='Translate the audio')
    parser.add_argument('--api', type=str, default='groq', help='Specify the API to use')
    parser.add_argument('--model', type=str, help='Transcription model ID (if not specified, API default will be used)')
    parser.add_argument('file_path', type=str, nargs='?', help='Path to save the audio file or existing audio file.')
    parser.add_argument("--vad", action="store_true", help="Use Voice Activity Detection with faster-whisper")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps in output")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for faster-whisper decoding (default: 5)")

    args = parser.parse_args()

    recorder = AudioRecorder()
    print_header("🎤 AUDIO DEVICE SETUP")
    print("Available input devices:")
    recorder.list_input_devices()

    # Validate input device before proceeding
    if args.input_device is not None:
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        available_devices = []
        for i in range(info.get('deviceCount')):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                available_devices.append(i)
        p.terminate()

        if args.input_device not in available_devices:
            print(f"Error: Input device {args.input_device} not available.")
            print(f"Available input devices: {available_devices}")
            return

    if args.list_devices:
        return

    # Check if a file path was provided
    if not args.file_path and not args.list_models:
        print("Error: You must provide a file path. Use --help for more information.")
        return

    # If neither --record nor --record-until-q are provided, assume existing file
    if args.record_until_q or args.record:
        # Generate a timestamp for unique file naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create a unique filename with timestamp
        if os.path.basename(args.file_path) == args.file_path:
            # If only a filename was given, add a timestamp
            base, ext = os.path.splitext(args.file_path)
            unique_filename = f"{base}_{timestamp}{ext}"
        else:
            # If a path with directories was given
            dir_path = os.path.dirname(args.file_path)
            base, ext = os.path.splitext(os.path.basename(args.file_path))
            unique_filename = os.path.join(dir_path, f"{base}_{timestamp}{ext}")
        
        # Use the unique filename for recording
        print(f"Generated unique filename: {unique_filename}")
        
        # Record audio
        recording_start = time.time()
        
        print_header("🎙️ RECORDING SETUP")
        print(f"Recording mode: {'Timed ({args.duration} seconds)' if args.record else 'Continuous'}")
        print(f"Output file: {unique_filename}")
        print(f"Input device: {args.input_device if args.input_device is not None else 'Default'}")
        
        if args.record_until_q:
            recorded_path = recorder.record_until_q(unique_filename, input_device=args.input_device)
        else:
            recorded_path = recorder.record(args.duration, unique_filename, input_device=args.input_device)
        
        if not recorded_path:
            print("Error during recording. Please check your microphone and try again.")
            return
        
        print(f"\n✅ Audio recorded successfully in {time.time() - recording_start:.2f}s")
        
        # Use recordings dir if the path doesn't include a directory
        if not os.path.dirname(recorded_path):
            os.makedirs("recordings", exist_ok=True)
            recorded_path = os.path.join("recordings", recorded_path)
        
        # Assign the recorded path to input_file
        input_file = recorded_path
    else:
        # Use the provided file path directly for existing files
        if not os.path.exists(args.file_path):
            print(f"Error: File '{args.file_path}' not found.")
            return
        input_file = args.file_path

    # Preprocess audio for transcription
    print("\n⏳ Preprocessing audio...")
    preprocess_start = time.time()
    with tqdm(total=100, desc="Preprocessing audio", unit="%") as pbar:
        for i in range(5):
            time.sleep(0.1)  # Simulate preprocessing steps
            pbar.update(20)
        
        # Create output path for preprocessed file
        preprocessed_file = f"preprocessed_{os.path.basename(input_file)}"
        preprocessed_path = os.path.join(os.path.dirname(input_file), preprocessed_file)
        
        # Call preprocess_audio with both input and output paths
        input_file = preprocess_audio(input_file, preprocessed_path)

    preprocess_time = time.time() - preprocess_start
    print(f"✅ Audio preprocessing completed in {preprocess_time:.2f}s")

    # Prepare for transcription
    result = None
    if args.transcribe:
        try:
            print_header("🔍 TRANSCRIPTION SETUP")
            print(f"Input file: {input_file}")
            print(f"API: {args.api}")
            model_display = args.model if args.model else f"Default ({router.apis[args.api].default_model if hasattr(router.apis[args.api], 'default_model') else 'API default'})"
            print(f"Using {'faster-whisper' if args.api == 'local' else args.api} model: {model_display}")

            # Transcribe audio
            transcription_start = time.time()
            result = router.transcribe_audio(
                input_file,
                api_name=args.api,
                model_id=args.model,
                prompt=args.prompt,
                language=args.language,
                temperature=args.temperature,
                vad_filter=args.vad,
                word_timestamps=args.word_timestamps,
                beam_size=args.beam_size
            )
            transcription_time = time.time() - transcription_start

            # Save transcription if output path is provided
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n📄 Transcription saved to {args.output}")

            print_header("🔤 TRANSCRIPTION RESULTS")
            print(f"\n{result.get('text', 'N/A')}\n")
            
            print_header("📊 STATS")
            print(f"🤖 Model: {result.get('model', 'N/A')}")
            if 'language' in result:
                print(f"🌐 Detected language: {result.get('language', 'N/A')} (confidence: {result.get('language_probability', 0):.2f})")
            
            # Display metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                if 'transcription_time' in metrics:
                    print(f"⏱️  Processing time: {metrics['transcription_time']:.2f}s")
                if 'real_time_factor' in metrics and metrics['real_time_factor']:
                    print(f"⚡ Real-time factor: {metrics['real_time_factor']:.2f}x (lower is better)")
                if 'audio_duration' in metrics and metrics['audio_duration']:
                    print(f"🔊 Audio duration: {metrics['audio_duration']:.2f}s")

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return

    if args.list_models:
        print_header("📚 AVAILABLE MODELS")
        available_models = router.list_available_models()
        for api, models in available_models.items():
            print(f"{api.capitalize()} models:")
            for model in models:
                print(f"  - {model}")
        return

    if args.translate:
        print_header("🌎 TRANSLATION")
        result = router.translate_audio(
            input_file,
            api_name=args.api,
            model_id=args.model,
            prompt=args.prompt,
            language=args.language,
            temperature=args.temperature
        )
        
        # Display translation results
        print(f"\n{result.get('text', 'N/A')}\n")

    # Clean up preprocessed file if it's different from the original
    if input_file != args.file_path:
        os.remove(input_file)

    if args.api == 'local' and 'local' not in router.apis:
        print("❌ Error: Local API is not available. Make sure faster-whisper is installed.")
        return
    
    # Display total execution time
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"✨ Process completed in {total_time:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()