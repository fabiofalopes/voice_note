## Installation

To set up this project, follow these steps:

1. Clone the repository and navigate to the project directory.

2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Generate a requirements file with all dependencies:
   ```
   pip freeze > requirements.txt
   ```

### Troubleshooting PyAudio Installation

If you encounter issues installing PyAudio, you may need to install some additional system dependencies. On Ubuntu or Debian-based systems, try the following:

1. Install PortAudio and other required system packages:
   ```
   sudo apt-get update
   sudo apt-get install python3-dev build-essential libasound2-dev portaudio19-dev
   ```

2. After installing these dependencies, try installing the requirements again:
   ```
   pip install -r requirements.txt
   ```

3. If you're still having issues, you can try installing PyAudio separately:
   ```
   pip install pyaudio
   ```

If you continue to face problems, please open an issue in the repository with details about your system and the error messages you're seeing.

## Usage

To use this application, you can run it from the command line with various options:

1. List available audio input devices:
   ```
   python -m cli.main --list-devices
   ```

2. Record audio until interrupted (Ctrl+C) and transcribe:
   ```
   python -m cli.main --record-until-q output.wav --model whisper-large-v3 --output transcription.json --transcribe
   ```

3. Record audio for a specific duration and transcribe:
   ```
   python -m cli.main --record --duration 30 output.wav --model whisper-large-v3 --output transcription.json --transcribe
   ```

4. Transcribe an existing audio file:
   ```
   python -m cli.main existing_audio.wav --model whisper-large-v3 --output transcription.json --transcribe
   ```

5. Translate an audio file:
   ```
   python -m cli.main existing_audio.wav --model whisper-large-v3 --output translation.json --translate
   ```

Additional options:
- `--input-device`: Specify the input device index for recording
- `--prompt`: Provide a prompt for transcription
- `--language`: Specify the language of the audio
- `--temperature`: Set the temperature for transcription

## Audio Input Setup

To identify and select the correct audio input source:

1. List available devices:
   ```
   python -m cli.main --list-devices
   ```

2. Use the `--input-device` flag with the desired device ID:
   ```
   python -m cli.main --record-until-q my_recording.wav --input-device <DEVICE_ID> --model whisper-large-v3 --output transcription_result.json --transcribe
   ```

   Replace `<DEVICE_ID>` with the ID of your chosen input device.

## TODO: Future Improvements

1. Implement a graphical user interface (GUI) for easier interaction.
2. Add support for batch processing of multiple audio files.
3. Implement real-time transcription for live audio input.
4. Add more post-processing options for the transcribed text (e.g., summarization, translation).
5. Improve error handling and provide more detailed error messages.
6. Add support for more transcription models and APIs.
7. Implement a progress bar for long transcriptions.
8. Add an option to save the preprocessed audio file.
9. Implement automatic language detection.
10. Add support for custom vocabulary or domain-specific language models.
11. Implement a system to request and handle audio input permissions on different operating systems.
12. Add a troubleshooting guide for common audio input issues.
13. Implement a testing suite for audio input functionality across different environments.