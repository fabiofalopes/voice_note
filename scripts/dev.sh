#!/bin/bash

# Development helper script

echo "🔧 Voice Transcriber Development Tools"
echo "====================================="

case "$1" in
    "test")
        echo "🧪 Testing installation..."
        python3 transcribe.py --help
        ;;
    "devices")
        echo "🎤 Listing audio devices..."
        python3 transcribe.py --list-devices
        ;;
    "clean")
        echo "🧹 Cleaning up recordings..."
        rm -f *.wav recordings/*.wav 2>/dev/null
        echo "✅ Cleaned up audio files"
        ;;
    "structure")
        echo "📁 Project structure:"
        tree -I 'venv|__pycache__|*.pyc|.git' .
        ;;
    *)
        echo "Available commands:"
        echo "  ./scripts/dev.sh test      - Test installation"
        echo "  ./scripts/dev.sh devices   - List audio devices"
        echo "  ./scripts/dev.sh clean     - Clean up recordings"
        echo "  ./scripts/dev.sh structure - Show project structure"
        ;;
esac