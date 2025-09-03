# Troubleshooting Guide

## Quick Diagnostics

### Check System Status
```bash
# See what recording method will be used
python3 transcribe.py --list-devices

# Test microphone levels
python3 transcribe.py --test-mic

# List all available devices
python3 transcribe.py --list-devices
```

## Common Issues

### 1. No Audio Recorded (Silent Files)

**Symptoms**: Recording completes but file contains no audio

**Linux Solutions**:
```bash
# Check if microphone is muted
pactl list sources | grep -A 10 "Name.*input"

# Unmute default microphone
pactl set-source-mute @DEFAULT_SOURCE@ false

# Test with parecord directly
parecord --format=s16le --rate=44100 --channels=1 test.wav
# Speak for a few seconds, then Ctrl+C
```

**All Platforms**:
- Check system microphone permissions
- Ensure microphone is not used by another application
- Try a different microphone if available

### 2. Permission Denied Errors

**Linux**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Check audio device permissions
ls -la /dev/snd/

# Restart audio system (if needed)
systemctl --user restart pipewire
```

**macOS**:
- Go to System Preferences > Security & Privacy > Privacy > Microphone
- Enable microphone access for Terminal or your Python environment

**Windows**:
- Go to Settings > Privacy > Microphone
- Enable microphone access for apps

### 3. Device Not Found

**Check Available Devices**:
```bash
python3 transcribe.py --list-devices
```

**Try Specific Device**:
```bash
# Use device ID from list above
python3 transcribe.py --device 8
```

### 4. PyAudio Installation Issues

**Linux**:
```bash
# Install development packages
sudo apt install portaudio19-dev python3-pyaudio

# Or install via pip
pip install pyaudio
```

**macOS**:
```bash
# Install with Homebrew
brew install portaudio
pip install pyaudio
```

**Windows**:
```bash
# Usually works with pip
pip install pyaudio
```

### 5. PipeWire-Specific Issues

**Install PipeWire Support**:
```bash
# Debian/Ubuntu
sudo apt install pipewire-audio-client-libraries pipewire-pulse

# Check PipeWire status
systemctl --user status pipewire
```

**Reset PipeWire**:
```bash
# Restart PipeWire services
systemctl --user restart pipewire pipewire-pulse
```

### 6. ALSA Warnings/Errors

**Symptoms**: Lots of "ALSA lib" error messages

**Solution**: These are usually harmless warnings. The system automatically suppresses them, but if you see them:

```bash
# Use the quiet script
./scripts/transcribe-quiet.sh

# Or redirect stderr
python3 transcribe.py 2>/dev/null
```

## Advanced Troubleshooting

### Debug Audio System
```bash
# Check what audio system is running
ps aux | grep -E "(pipewire|pulseaudio|alsa)"

# List all audio processes
lsof /dev/snd/*

# Check audio hardware
cat /proc/asound/cards
```

### Test Audio Pipeline
```bash
# Test system audio recording
arecord -f cd -t wav -d 5 test_system.wav

# Test PipeWire recording
parecord --format=s16le --rate=44100 --channels=1 test_pipewire.wav

# Test with specific device
arecord -D hw:0,0 -f cd -t wav -d 5 test_hw.wav
```

### Python Environment Issues
```bash
# Check Python audio modules
python3 -c "import pyaudio; print('PyAudio OK')"
python3 -c "import wave; print('Wave OK')"

# Check if running in virtual environment
which python3
pip list | grep -E "(pyaudio|groq)"
```

## Error Messages Reference

### "No working audio input device found"
- **Cause**: No microphone detected or accessible
- **Solution**: Check device connections, permissions, and system audio settings

### "Recording produced empty file"
- **Cause**: Microphone is muted, suspended, or not receiving input
- **Solution**: Check microphone settings, test with system tools

### "parecord: command not found"
- **Cause**: PipeWire client tools not installed
- **Solution**: Install `pipewire-audio-client-libraries` or equivalent

### "Permission denied: /dev/snd/"
- **Cause**: User not in audio group or insufficient permissions
- **Solution**: Add user to audio group, check device permissions

### "Device busy or not available"
- **Cause**: Another application is using the microphone
- **Solution**: Close other audio applications, check for background processes

## Getting Help

### Collect System Information
```bash
# System info
uname -a
cat /etc/os-release

# Audio system info
python3 transcribe.py --list-devices > audio_info.txt

# Process info
ps aux | grep -E "(pipewire|pulse|alsa)" > process_info.txt
```

### Test Commands
```bash
# Quick test
python3 transcribe.py --test-mic

# Full device list
python3 transcribe.py --list-devices

# Try specific device
python3 transcribe.py --device X --test-mic
```

When reporting issues, include:
1. Operating system and version
2. Output of `--list-devices`
3. Any error messages
4. What you were trying to do
5. Whether it worked before