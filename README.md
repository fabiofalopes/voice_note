# Voice Transcriber

Contract-first, cross-platform voice recording and transcription using Groq or
modelos. JSON and NDJSON modes provide a stable machine-readable interface;
human output remains the default in a terminal.

## Quick Start

```bash
# Install
./scripts/setup.sh

# Add your GROQ_API_KEY to .env
cp .env.example .env

# Use immediately
./venv/bin/python transcribe.py
```

## Programmatic Use

Machine-readable output follows schema version `1.0`:

```bash
# One JSON document after completion
./venv/bin/python transcribe.py recording.wav --json | jq '.status, .code'

# Streaming events; the final line is the commit point
./venv/bin/python transcribe.py recording.wav --ndjson | jq -c 'select(.type == "segment")'

# Read the final NDJSON status
./venv/bin/python transcribe.py recording.wav --ndjson | jq -s 'last | .data.code'
```

Machine modes keep stdout JSON-only. Human diagnostics use stderr. Public exit
codes are `0` (success), `1` (generic error), `2` (usage error), and `130`
(interrupt); use the JSON `code` field for granular semantics.

The system auto-configures in ~0.03 seconds and shows you exactly what audio method will be used.

## Usage

```bash
./venv/bin/python transcribe.py                         # Record and transcribe
./venv/bin/python transcribe.py recording.wav           # Transcribe a file
./venv/bin/python transcribe.py recording.wav --provider modelos
./venv/bin/python transcribe.py --list-devices          # List audio devices
./venv/bin/python transcribe.py --test-mic              # Test microphone levels
./venv/bin/python transcribe.py recording.wav --translate
./venv/bin/python transcribe.py recording.wav --timestamps
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
./venv/bin/python transcribe.py --list-devices

# Test microphone
./venv/bin/python transcribe.py --test-mic
```

For detailed troubleshooting, see [docs/troubleshooting.md](docs/troubleshooting.md).

## Models

- Groq: `whisper-large-v3` (default), `whisper-large-v3-turbo`, `distil-whisper-large-v3-en`
- modelos: `stt-large-v3-turbo`

## Requirements

- Python 3.14 in `./venv`
- `GROQ_API_KEY` and/or `MODELOS_AI_KEY` in `.env`
- Microphone (built-in laptop mic works great)

## Project Structure

```
transcribe.py           # Main entry point
src/
├── cli.py              # Command-line interface
├── contract.py         # Pydantic v1.0 contract models
├── emitter.py          # Human, JSON, and NDJSON renderers
├── i18n.py             # Language normalization
├── api/                # Groq and modelos clients
└── audio_processing/   # Recording logic
schema/                 # Committed JSON Schema artifacts
tests/                  # Executable contract specification
scripts/               # Setup and utility scripts
```

## Documentation

- **[Audio System Architecture](docs/audio-system.md)** - How the cross-platform audio system works
- **[PipeWire Integration](docs/pipewire-integration.md)** - Modern Linux audio support
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions
- **[Output Contract](docs/CONTRACT.md)** - voice_note v1.0 contract spec (Stream A reference)
- **[Agent Rules](AGENTS.md)** - Operating rules for any agent touching voice_note code
- **[Project Memory](MEMORY.md)** - State, history, decisions, validation log

## Aliases

Run `./scripts/create_alias.sh` for convenient shortcuts:
- `transcribe` - Basic usage
- `transcribe-quiet` - No ALSA warnings (Linux)
- `transcribe-fast` - Fast model
- `translate` - Translate to English
- `voice-devices` - List audio devices
