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

### Troubleshooting PyAudio Installation

If you encounter issues installing PyAudio, you may need to install some additional system dependencies. On Ubuntu or Debian-based systems, try the following:

1. Install PortAudio and other required system packages:
   ```
   sudo apt-get update
   sudo apt-get install python3-dev
   sudo apt-get install build-essential
   sudo apt-get install libasound2-dev
   sudo apt-get install portaudio19-dev
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
   python -m cli.main --record-until-q output.wav --model whisper-large-v3 --output transcription.json
   ```

3. Record audio for a specific duration and transcribe:
   ```
   python -m cli.main --record --duration 30 output.wav --model whisper-large-v3 --output transcription.json
   ```

4. Transcribe an existing audio file:
   ```
   python -m cli.main existing_audio.wav --model whisper-large-v3 --output transcription.json
   ```

Additional options:
- `--input-device`: Specify the input device index for recording
- `--prompt`: Provide a prompt for transcription
- `--language`: Specify the language of the audio
- `--temperature`: Set the temperature for transcription

## Known Issues

We are currently experiencing issues with audio recording:
- The application may have difficulty properly opening and recording audio.
- Audio files seem to be captured, but may not contain the expected content.
- Users may need to manually specify the input source or microphone.

## TODO: Future Improvements

1. Fix audio recording issues:
   - Implement automatic detection and use of the system default audio input.
   - Add proper error handling for audio device permissions.
   - Improve robustness of audio stream handling.

2. Implement a graphical user interface (GUI) for easier interaction.
3. Add support for batch processing of multiple audio files.
4. Implement real-time transcription for live audio input.
5. Add more post-processing options for the transcribed text (e.g., summarization, translation).
6. Improve error handling and provide more detailed error messages.
7. Add support for more transcription models and APIs.
8. Implement a progress bar for long transcriptions.
9. Add an option to save the preprocessed audio file.
10. Implement automatic language detection.
11. Add support for custom vocabulary or domain-specific language models.
12. Implement a system to request and handle audio input permissions on different operating systems.
13. Add a troubleshooting guide for common audio input issues.
14. Implement a testing suite for audio input functionality across different environments.

---

## Command Usage for `cli.main` (for now)

## Recording and Transcription Commands

1. **Record Audio and Transcribe:**
    ```bash
    python -m cli.main --record-until-q my_recording.wav --model whisper-large-v3 --output transcription_result.json
    ```

2. **Record Audio with Superuser Privileges:**
    ```bash
    sudo python -m cli.main --record-until-q my_recording.wav --model whisper-large-v3 --output transcription_result.json
    ```

## Listing Audio Devices

3. **List Available Audio Input Devices:**
    ```bash
    python -m cli.main --list-devices
    ```

## Recording with Specific Input Device

4. **Record Using Specific Input Device:**
    ```bash
    python -m cli.main --record-until-q my_recording.wav --input-device <DEVICE_ID> --model whisper-large-v3 --output transcription_result.json
    ```

    Replace `<DEVICE_ID>` with the desired input device ID (e.g., `0`, `4`, `14`, etc.).

## Example Commands with Different Device IDs

- Record using device ID `14`:
    ```bash
    python -m cli.main --record-until-q my_recording.wav --input-device 14 --model whisper-large-v3 --output transcription_result.json
    ```



This format provides a simple and direct list of commands without much explanation or extra formatting.