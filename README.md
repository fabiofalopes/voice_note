# Voice Note

A powerful command-line tool for recording, transcribing, and analyzing audio using both local whisper models and API services.

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Local transcription using CTranslate2
- [Whisper](https://github.com/openai/whisper) - Original OpenAI model
- [Groq](https://console.groq.com/docs/speech-text) - API transcription service

## Features

- **High-Performance Daemon Architecture**: Load whisper models once and keep them in memory for faster transcription
- **Multiple Transcription Options**: Use local models or cloud APIs based on your needs
- **Audio Recording**: Record audio directly from your microphone with customizable settings
- **Flexible CLI**: Comprehensive command-line interface for all operations
- **Automatic Fallback**: Automatically falls back to Groq API if local transcription fails

## Project Structure

The project has been simplified to focus on core functionality:

- `faster_whisper_daemon/`: Daemon service for running faster-whisper models in the background
- `api_integrations/`: Contains only the Groq API integration
- `cli/`: Command-line interface for interacting with the transcription services
- `audio_processing/`: Utilities for recording and processing audio

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

#### macOS (especially Apple Silicon)

For macOS users (especially Apple Silicon):

1. Install PyAudio dependencies:
   ```
   brew install portaudio
   ```

2. For optimal performance with faster-whisper on Apple Silicon:
   ```
   pip install torch torchvision torchaudio
   ```

3. If using local transcription, install CTranslate2:
   ```
   pip install ctranslate2
   ```

4. **Important for Apple Silicon (M1/M2/M3)**: When starting the daemon, use `float16` compute type:
   ```
   python -m faster_whisper_daemon.cli start --compute-type float16
   ```
   The daemon will automatically detect Apple Silicon and use the appropriate settings, but explicitly setting `float16` is recommended.

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

## Usage

Voice Note offers two main transcription methods:
- Local transcription using faster-whisper (high-quality, offline, requires more resources)
- API transcription using Groq (requires API key, internet connection)

### Daemon Architecture

Voice Note uses a client-server architecture for local transcription, which significantly improves performance by:
1. Loading the model only once and keeping it in memory
2. Processing multiple transcription requests without reloading the model
3. Reducing memory usage and startup time

#### Step 1: Start the Daemon

First, start the daemon in a terminal window:

```bash
# Start with default settings (large-v3 model)
python -m faster_whisper_daemon.cli start

# For Apple Silicon (M1/M2/M3), use float16 compute type
python -m faster_whisper_daemon.cli start --compute-type float16

# For lower-end hardware, use a smaller model
python -m faster_whisper_daemon.cli start --model medium --compute-type float16

# For more options
python -m faster_whisper_daemon.cli start --help
```

Keep this terminal window open while using the application.

#### Step 2: Use the CLI with the Daemon

In a new terminal window, use the CLI:

```bash
# Transcribe an existing file (auto mode will try local daemon first, then Groq API)
python -m cli.main existing_audio.wav --transcribe

# Force using local daemon only (no fallback)
python -m cli.main --api local existing_audio.wav --transcribe

# Force using Groq API only
python -m cli.main --api groq existing_audio.wav --transcribe

# Record and transcribe
python -m cli.main --record --duration 30 recording.wav --transcribe
```

If you need to specify custom daemon connection parameters:

```bash
python -m cli.main existing_audio.wav --transcribe --daemon-socket /path/to/socket
```

#### Testing the Daemon Connection

You can test if the daemon is running correctly:

```bash
# Check daemon status
python -m faster_whisper_daemon.cli status

# Test transcription with a recording
python -m faster_whisper_daemon.cli test --record

# Test with an existing audio file
python -m faster_whisper_daemon.cli test --audio-file path/to/audio.wav
```

### Audio Input Setup

To identify and select the correct audio input source:

1. List available devices:
   ```
   python -m cli.main --list-devices
   ```

2. Use the `--input-device` flag with the desired device ID when recording.

### Basic Commands

#### Core Transcription Commands using Groq

```bash
# Record audio until interrupted (Ctrl+C) and transcribe with full analysis
python -m cli.main --api groq --record-until-q recording.wav --transcribe

# Record audio until interrupted (Ctrl+C) and transcribe with raw output only
python -m cli.main --api groq --record-until-q recording.wav --transcribe --raw-transcription
```

The first command provides full analysis including:
- Raw transcription text
- Summary
- Sentiment analysis
- Task analysis
- Thinking tags

The second command with `--raw-transcription` flag returns only the raw transcription text, skipping all additional analysis.

#### Auto Mode (Default)

The CLI will automatically try local transcription first, then fall back to Groq API if local fails:

```bash
# Record audio until interrupted (Ctrl+C) and transcribe
python -m cli.main --record-until-q recording.wav --transcribe

# Record audio for a specific duration (30 seconds) and transcribe
python -m cli.main --record --duration 30 recording.wav --transcribe

# Transcribe an existing audio file
python -m cli.main existing_audio.wav --transcribe
```

#### Local Transcription (using faster-whisper)

Force using only the local daemon:

```bash
# Record and transcribe with local model only
python -m cli.main --api local --record --duration 30 recording.wav --transcribe

# Transcribe an existing file with local model only
python -m cli.main --api local existing_audio.wav --transcribe
```

#### Groq API Transcription

Force using only the Groq API:

```bash
# Record and transcribe with Groq API
python -m cli.main --api groq --record --duration 30 recording.wav --transcribe

# Transcribe an existing file with Groq API
python -m cli.main --api groq existing_audio.wav --transcribe
```

For convenience, you can add this alias to your shell's rc file (e.g., `.bashrc`, `.zshrc`):

```bash
alias voice_note='cd ~/Documents/projetos/hub/voice_note && source .venv/bin/activate && python -m cli.main --api groq --record-until-q recording.wav --transcribe --raw-transcription && deactivate'
```

This alias will:
1. Navigate to the project directory
2. Activate the virtual environment
3. Run the transcription command
4. Deactivate the virtual environment

After adding the alias, you can simply run `voice_note` from anywhere to start recording and transcribing.

## Troubleshooting

### Daemon Issues

If you encounter issues with the daemon:

1. Check if the daemon is running:
   ```
   python -m faster_whisper_daemon.cli status
   ```

2. If the daemon fails to load models on Apple Silicon:
   ```
   # Use float16 compute type instead of int8
   python -m faster_whisper_daemon.cli start --compute-type float16
   ```

3. If you're low on memory, try a smaller model:
   ```
   python -m faster_whisper_daemon.cli start --model medium --compute-type float16
   ```

4. Check the daemon log for errors:
   ```
   cat daemon.log
   ```

### API Issues

If you encounter issues with the Groq API:

1. Check that your API key is correctly set in the `.env` file
2. Verify your internet connection
3. Check the Groq API status at https://status.groq.com/

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the original model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for the optimized implementation
- [Groq](https://groq.com/) for their API services
