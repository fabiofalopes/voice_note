# Voice Transcriber

Cross-platform voice recording and transcription using Groq's Whisper API. Optimized for modern Linux systems with PipeWire, while maintaining full compatibility with macOS and Windows.

## Quick Start

```bash
# Install
./scripts/setup.sh

# Add your GROQ_API_KEY to .env
cp .env.example .env

# Use immediately
python3 transcribe.py
```

The system auto-configures in ~0.03 seconds and shows you exactly what audio method will be used.

## Usage

```bash
python3 transcribe.py                    # Record and transcribe
python3 transcribe.py --list-devices     # List audio devices  
python3 transcribe.py --test-mic         # Test microphone levels
python3 transcribe.py --translate        # Translate to English
python3 transcribe.py --model whisper-large-v3-turbo  # Fast model
python3 transcribe.py --device 7         # Use specific device
```

## Audio System Support

✅ **Linux**: PipeWire, PulseAudio, ALSA (auto-detected)  
✅ **macOS**: Core Audio  
✅ **Windows**: DirectSound/WASAPI  

The system automatically uses the best recording method for your platform:
- **Linux + PipeWire**: Uses `parecord` (native support)
- **Linux + PulseAudio**: Uses `parecord` or PyAudio fallback
- **macOS/Windows**: Uses PyAudio (cross-platform)

## Troubleshooting

```bash
# Quick diagnostics
python3 transcribe.py --list-devices

# Test microphone
python3 transcribe.py --test-mic
```

For detailed troubleshooting, see [docs/troubleshooting.md](docs/troubleshooting.md).

## Models

- `whisper-large-v3` - Best accuracy (default)
- `whisper-large-v3-turbo` - Faster processing
- `distil-whisper-large-v3-en` - English only, fastest

## Requirements

- Python 3.7+
- Groq API key (free at https://console.groq.com/)
- Microphone (built-in laptop mic works great)

## Project Structure

```
transcribe.py           # Main entry point
src/
├── cli.py             # Command-line interface
├── api/               # API clients
├── audio_processing/  # Recording logic
└── config.py         # Configuration
scripts/               # Setup and utility scripts
future_features/       # Preserved LLM assets
```

## Documentation

- **[Audio System Architecture](docs/audio-system.md)** - How the cross-platform audio system works
- **[PipeWire Integration](docs/pipewire-integration.md)** - Modern Linux audio support
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions
- **[Development Notes](docs/development-notes.md)** - Technical implementation details

## Aliases

Run `./scripts/create_alias.sh` for convenient shortcuts:
- `transcribe` - Basic usage
- `transcribe-quiet` - No ALSA warnings (Linux)
- `transcribe-fast` - Fast model
- `translate` - Translate to English
- `voice-devices` - List audio devices