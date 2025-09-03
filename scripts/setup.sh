#!/bin/bash

echo "🎤 Voice Transcriber Setup"
echo "=========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env and add your GROQ_API_KEY"
    echo "   Get your free API key at: https://console.groq.com/"
else
    echo "✅ .env file already exists"
fi

# Test the installation
echo ""
echo "🧪 Testing installation..."
if python3 transcribe.py --help > /dev/null 2>&1; then
    echo "✅ Installation successful!"
else
    echo "❌ Installation test failed. Check dependencies."
    exit 1
fi

echo ""
echo "🚀 Setup complete! Usage examples:"
echo "   python3 transcribe.py                    # Basic usage"
echo "   python3 transcribe.py --list-devices     # List audio devices"
echo "   python3 transcribe.py --translate        # Translate to English"
echo ""
echo "💡 Want convenient aliases? Run: ./scripts/create_alias.sh"