# Voice Notes 

A command-line tool for recording, transcribing, and analyzing audio using both local models and API services.

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Local transcription using CTranslate2
- [Whisper](https://github.com/openai/whisper) - Original OpenAI model
- [Groq](https://console.groq.com/docs/speech-text) - API transcription service

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

This application offers two main transcription methods:
- Local transcription using faster-whisper (high-quality, offline, requires more resources)
- API transcription using Groq (requires API key, internet connection)

### Audio Input Setup

To identify and select the correct audio input source:

1. List available devices:
   ```
   python -m cli.main --list-devices
   ```

2. Use the `--input-device` flag with the desired device ID when recording.

### Basic Commands

#### Local Transcription (using faster-whisper)

1. Record audio until interrupted (Ctrl+C) and transcribe with local model:
   ```
   python -m cli.main --api local --record-until-q recording.wav --transcribe
   ```

2. Record audio for a specific duration (30 seconds) and transcribe with local model:
   ```
   python -m cli.main --api local --record --duration 30 recording.wav --transcribe
   ```

3. Transcribe an existing audio file with local model:
   ```
   python -m cli.main --api local existing_audio.wav --transcribe
   ```

#### API Transcription (using Groq)

1. Record audio until interrupted (Ctrl+C) and transcribe with Groq API:
   ```
   python -m cli.main --api groq --record-until-q recording.wav --model whisper-large-v3 --transcribe
   ```

2. Record audio for a specific duration and transcribe with Groq API:
   ```
   python -m cli.main --api groq --record --duration 30 recording.wav --model whisper-large-v3 --transcribe
   ```

3. Transcribe an existing audio file with Groq API:
   ```
   python -m cli.main --api groq existing_audio.wav --model whisper-large-v3 --transcribe
   ```

4. Translate an audio file:
   ```
   python -m cli.main --api groq existing_audio.wav --model whisper-large-v3 --translate
   ```

### Advanced Options

- `--input-device <ID>`: Specify the input device index for recording
- `--prompt <TEXT>`: Provide a prompt for transcription
- `--language <CODE>`: Specify the language of the audio (e.g., "en", "fr")
- `--temperature <VALUE>`: Set the temperature for transcription (0.0-1.0)
- `--output <FILE>`: Save the transcription output to a JSON file
- `--vad`: Use Voice Activity Detection (only with faster-whisper)
- `--word-timestamps`: Include word-level timestamps in output
- `--beam-size <NUMBER>`: Set beam size for faster-whisper decoding (default: 5)

### Benchmarking faster-whisper Models

To compare the performance of different faster-whisper models:
