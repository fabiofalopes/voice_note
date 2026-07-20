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


def _build_client(provider: str):
    """Instantiate the right STT client for the given provider name."""
    if provider == "groq":
        from api.groq_client import GroqWhisperClient

        return GroqWhisperClient()
    elif provider == "modelos":
        from api.modelos_client import ModelosSTTClient

        return ModelosSTTClient()
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'groq' or 'modelos'.")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Transcriber — record and transcribe audio"
    )

    # Positional
    parser.add_argument(
        "file", nargs="?", help="Audio file to transcribe (omit to record)"
    )

    # Provider & model
    parser.add_argument(
        "--provider",
        "-p",
        choices=["groq", "modelos"],
        default="groq",
        help="STT provider to use (default: groq)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Model name. Defaults to provider default. "
            "Groq: whisper-large-v3[-turbo], distil-whisper-large-v3-en. "
            "Modelos: stt-large-v3-turbo."
        ),
    )

    # Transcription options
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        metavar="LANG",
        help="ISO-639-1 language hint, e.g. 'pt', 'en' (improves accuracy + speed)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Save .srt and .json in addition to .txt (uses verbose_json mode)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate to English instead of transcribe",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Base output path (without extension). Derived from input file if omitted.",
    )

    # Recording options (only used when no file is given)
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--test-mic", action="store_true", help="Test microphone input levels"
    )
    parser.add_argument("--device", "-d", type=int, help="Audio input device index")
    parser.add_argument(
        "--record-output",
        default="recording.wav",
        help="Output WAV filename when recording (default: recording.wav)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use robust recorder (auto-chunking, device failure recovery)",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=5,
        help="Minutes per chunk for robust recording (default: 5)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        help="Maximum recording duration in minutes",
    )

    # Misc
    parser.add_argument(
        "--no-clipboard", action="store_true", help="Don't copy text to clipboard"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Device listing / mic test (no API needed)
    # -----------------------------------------------------------------------
    if args.list_devices:
        try:
            from audio_processing.recorder import list_audio_devices

            list_audio_devices()
        except ImportError as e:
            print(f"Missing dependency: {e}")
            return 1
        return 0

    if args.test_mic:
        try:
            from audio_processing.recorder import AudioRecorder

            recorder = AudioRecorder(device_index=args.device)
            return 0 if recorder.test_microphone(device_id=args.device) else 1
        except ImportError as e:
            print(f"Missing dependency: {e}")
            return 1

    # -----------------------------------------------------------------------
    # Determine audio source
    # -----------------------------------------------------------------------
    if args.file:
        if not os.path.exists(args.file):
            print(f"Audio file not found: {args.file}")
            return 1
        audio_file = args.file
        print(f"Transcribing existing file: {audio_file}")
    else:
        # Record new audio
        if args.robust:
            # Use robust recorder with auto-chunking and error recovery
            try:
                from audio_processing.robust_recorder import record_robust
            except ImportError as e:
                print(f"Missing dependency: {e}")
                return 1

            chunk_files = record_robust(
                output_dir=os.path.dirname(args.record_output) or "recordings",
                chunk_minutes=args.chunk_minutes,
                max_duration_minutes=args.max_duration,
                device_index=args.device,
                merge_after=True,  # Merge chunks into single file
            )

            if not chunk_files:
                print("Recording failed.")
                return 1

            audio_file = str(chunk_files[0])  # Use merged file
            print(f"\n✅ Final recording: {audio_file}")

        else:
            # Use standard recorder
            try:
                from audio_processing.recorder import AudioRecorder
            except ImportError as e:
                print(f"Missing dependency: {e}")
                return 1

            recorder = AudioRecorder(device_index=args.device)
            if not recorder.recording_method:
                print("No working audio recording method found.")
                print("Try: python3 transcribe.py --list-devices")
                return 1

            print(recorder.get_recording_info())
            audio_file = recorder.record_until_q(args.record_output)
            if not audio_file:
                print("Recording failed.")
                return 1

    # -----------------------------------------------------------------------
    # Build STT client
    # -----------------------------------------------------------------------
    try:
        client = _build_client(args.provider)
    except (ValueError, ImportError) as e:
        print(f"Could not initialise provider: {e}")
        return 1

    # -----------------------------------------------------------------------
    # Transcribe / translate
    # -----------------------------------------------------------------------
    action = "Translating" if args.translate else "Transcribing"
    print(f"{action} with provider={args.provider} ...")

    try:
        if args.translate:
            result = client.translate(
                audio_file,
                model=args.model,
                output_file=args.output,
                verbose=args.timestamps,
            )
        else:
            result = client.transcribe(
                audio_file,
                model=args.model,
                output_file=args.output,
                language=args.language,
                verbose=args.timestamps,
            )
    except Exception as e:
        print(f"API error: {e}")
        return 1

    if not result:
        print(f"{'Translation' if args.translate else 'Transcription'} failed.")
        return 1

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    label = "TRANSLATION" if args.translate else "TRANSCRIPTION"
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    # Print first 500 chars so the terminal isn't overwhelmed for long files
    preview = result.text[:500]
    if len(result.text) > 500:
        preview += (
            f"\n... [{len(result.text) - 500} more chars — see {result.output_file}]"
        )
    print(preview)
    print("=" * 60)
    print(f"Saved: {result.output_file}")
    if result.srt_file:
        print(f"  SRT: {result.srt_file}")
    if result.json_file:
        print(f" JSON: {result.json_file}")
    if result.detected_language:
        print(f" Lang: {result.detected_language}")

    # -----------------------------------------------------------------------
    # Clipboard
    # -----------------------------------------------------------------------
    if not args.no_clipboard:
        try:
            import pyperclip

            pyperclip.copy(result.text)
            print("Copied to clipboard.")
        except Exception as e:
            print(f"Could not copy to clipboard: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
