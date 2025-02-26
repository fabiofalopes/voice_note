# Voice Note

A command-line tool for recording, transcribing, and analyzing audio using both local whisper models and API services.

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Local transcription using CTranslate2
- [Whisper](https://github.com/openai/whisper) - Original OpenAI model
- [Groq](https://console.groq.com/docs/speech-text) - API transcription service

## Installation

To set up this project, follow these steps:

1. Clone the repository and navigate to the project directory.

2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

3. Ensure pip is up to date:
   ```
   pip install --upgrade pip setuptools wheel
   ```

4. Choose your installation approach:

   a. For a complete installation (all features):
   ```
   pip install -r requirements.txt
   ```

   b. For minimal installation with API transcription only:
   ```
   pip install -r requirements-core.txt
   pip install -r requirements-api.txt
   ```

   c. For local transcription only:
   ```
   pip install -r requirements-core.txt
   pip install -r requirements-local.txt
   ```

5. Set up your environment variables:
   - Create a `.env` file with your API keys if using Groq
   ```
   GROQ_API_KEY=your_api_key_here
   ```

### Platform-Specific Installation Notes

#### macOS

For macOS users (especially Apple Silicon):

1. Install PyAudio dependencies:
   ```
   brew install portaudio
   ```

2. For optimal performance with faster-whisper on Apple Silicon:
   ```
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. If using local transcription, install int8 compute optimizations:
   ```
   pip install ctranslate2
   ```

#### Windows

For Windows users:

1. If you encounter issues with PyAudio, you may need to install it from a wheel:
   ```
   pip install pipwin
   pipwin install pyaudio
   ```

2. Install ffmpeg:
   - Download from https://ffmpeg.org/download.html and add to your PATH
   - Or install via Chocolatey: `choco install ffmpeg`

3. For CUDA support (if you have an NVIDIA GPU):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### Linux

For Linux (Ubuntu/Debian) users:

1. Install system dependencies:
   ```
   sudo apt update
   sudo apt install python3-dev portaudio19-dev ffmpeg libsndfile1
   ```

2. If using NVIDIA GPU for acceleration:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Troubleshooting Dependencies

If you encounter issues with installing dependencies, try the following:

1. For PyAudio issues:
   - On Ubuntu/Debian:
     ```
     sudo apt-get install python3-dev build-essential libasound2-dev portaudio19-dev
     pip install pyaudio
     ```
   - On macOS:
     ```
     brew install portaudio
     pip install pyaudio
     ```
   - On Windows:
     ```
     pip install pipwin
     pipwin install pyaudio
     ```

2. For faster-whisper issues:
   - Make sure you have appropriate C++ build tools installed
   - On Windows, you may need Visual C++ Build Tools
   - Try installing prerequisites separately:
     ```
     pip install ctranslate2
     pip install faster-whisper
     ```

3. For torch/CUDA issues:
   - Check CUDA compatibility with your GPU drivers
   - Install the appropriate torch version for your setup:
     ```
     # CPU only
     pip install torch torchvision torchaudio
     
     # CUDA 11.8
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

4. If you encounter "No module named 'pip._vendor.packaging.version'" error:
   ```
   python -m pip install --upgrade pip
   pip install --upgrade setuptools
   ```

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
- `--vad`: Use Voice Activity Detection to filter out silence and background noise, improving accuracy and processing speed
- `--word-timestamps`: Generate precise timing information for each word, useful for subtitles or audio-text alignment
- `--beam-size <NUMBER>`: Control the search breadth during transcription (higher values = more accurate but slower, lower values = faster but potentially less accurate)

### Benchmarking faster-whisper Models

To compare the performance of different faster-whisper models:
