#!/bin/bash

# Create convenient aliases for voice transcriber
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRANSCRIBER_PATH="$PROJECT_DIR/transcribe.py"

echo "🎤 Setting up Voice Transcriber aliases..."

# Detect shell
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
else
    echo "⚠️  Unsupported shell. Please add aliases manually."
    exit 1
fi

# Add aliases to shell config
echo "" >> "$SHELL_RC"
echo "# Voice Transcriber aliases" >> "$SHELL_RC"
echo "alias transcribe='python3 $TRANSCRIBER_PATH'" >> "$SHELL_RC"
echo "alias transcribe-quiet='$PROJECT_DIR/scripts/transcribe-quiet.sh'" >> "$SHELL_RC"
echo "alias transcribe-fast='python3 $TRANSCRIBER_PATH --model whisper-large-v3-turbo'" >> "$SHELL_RC"
echo "alias translate='python3 $TRANSCRIBER_PATH --translate'" >> "$SHELL_RC"
echo "alias voice-devices='python3 $TRANSCRIBER_PATH --list-devices'" >> "$SHELL_RC"

echo "✅ Aliases added to $SHELL_RC"
echo ""
echo "🔄 Reload your shell or run: source $SHELL_RC"
echo ""
echo "📝 Available aliases:"
echo "   transcribe          - Record and transcribe"
echo "   transcribe-quiet    - Record and transcribe (no ALSA warnings)"
echo "   transcribe-fast     - Record and transcribe (fast model)"
echo "   translate           - Record and translate to English"
echo "   voice-devices       - List audio devices"