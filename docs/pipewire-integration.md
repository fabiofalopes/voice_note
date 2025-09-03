# PipeWire Integration Guide

## The Challenge

Modern Linux distributions (Debian 13, Ubuntu 22.04+, Fedora 34+) have adopted PipeWire as their default audio system, replacing PulseAudio and ALSA in many cases. This created compatibility issues with PyAudio, which was designed primarily for older audio systems.

## Problem Analysis

### PyAudio + PipeWire Issues
- **Device Mapping**: PyAudio couldn't properly map to PipeWire sources
- **Silent Recordings**: Audio streams opened but captured no data
- **ALSA Warnings**: Excessive error messages cluttering output
- **Device Detection**: Incorrect or non-functional device enumeration

### Investigation Results
```bash
# PipeWire sources (what actually works)
$ pactl list sources short
alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source

# PyAudio devices (what PyAudio sees)
$ python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"
# Often shows devices that don't actually capture audio
```

## Our Solution

### Dual-Method Architecture
1. **Primary**: Use `parecord` for native PipeWire support
2. **Fallback**: Use PyAudio for cross-platform compatibility
3. **Auto-Detection**: System automatically chooses the best method

### Implementation Details

#### PipeWire Detection
```python
def get_system_info():
    # Check if PipeWire is running
    result = subprocess.run(['pgrep', '-f', 'pipewire'], capture_output=True)
    if result.returncode == 0:
        info['audio_backend'] = 'pipewire'
        
    # Get actual PipeWire sources
    result = subprocess.run(['pactl', 'list', 'sources', 'short'], capture_output=True)
    # Parse and store available sources
```

#### Native Recording with parecord
```python
def record_with_parecord(output_file):
    cmd = ['parecord', '--format=s16le', '--rate=44100', '--channels=1', output_file]
    process = subprocess.Popen(cmd)
    # Handle user interruption and cleanup
```

#### Smart Method Selection
```python
def _auto_configure(self):
    if system == 'Linux' and backend in ['pipewire', 'pulseaudio']:
        if parecord_available():
            self.recording_method = 'parecord'  # Best for PipeWire
            return
    
    if pyaudio_device_works():
        self.recording_method = 'pyaudio'  # Cross-platform fallback
        return
```

## Benefits of This Approach

### For Users
- **Zero Configuration**: Works out of the box on PipeWire systems
- **No Failed Recordings**: System knows what will work before recording
- **Clear Feedback**: Shows exactly which audio source is being used
- **Fast Startup**: Auto-configuration happens in ~0.03 seconds

### For Developers
- **Future-Proof**: Ready for PipeWire adoption across Linux distros
- **Maintainable**: Clean separation between recording methods
- **Debuggable**: Clear logging of which method is being used
- **Extensible**: Easy to add new audio backends

## PipeWire-Specific Optimizations

### Audio Settings
- **Sample Rate**: 44.1kHz (matches PipeWire defaults)
- **Format**: s16le (16-bit little-endian, widely supported)
- **Channels**: 1 (mono, sufficient for speech transcription)
- **Buffering**: Let PipeWire handle buffering (more reliable)

### Source Selection
- **Default Source**: Use system default microphone
- **Active Sources**: Prefer non-suspended sources
- **Built-in Priority**: Favor built-in laptop microphones over monitors

### Error Handling
- **Graceful Fallback**: If parecord fails, try PyAudio
- **Clear Messages**: Show which PipeWire source is being used
- **Troubleshooting**: Provide PipeWire-specific debugging steps

## Testing on Different Systems

### Debian 13 (PipeWire)
```bash
$ python3 transcribe.py --list-devices
System: Linux | Audio Backend: pipewire
🎤 parecord → alsa_input...Mic1__source
✅ Ready to record immediately!
```

### Ubuntu 20.04 (PulseAudio)
```bash
$ python3 transcribe.py --list-devices
System: Linux | Audio Backend: pulseaudio
🎤 parecord → alsa_input...Mic__source
✅ Ready to record immediately!
```

### Older Systems (ALSA)
```bash
$ python3 transcribe.py --list-devices
System: Linux | Audio Backend: alsa
🎤 PyAudio → Device 8
✅ Ready to record immediately!
```

## Troubleshooting PipeWire Issues

### Common Problems
1. **No Audio Captured**: Check if microphone is suspended
2. **Permission Denied**: Add user to audio group
3. **parecord Not Found**: Install pipewire-audio-client-libraries

### Diagnostic Commands
```bash
# Check PipeWire status
systemctl --user status pipewire

# List audio sources
pactl list sources short

# Test recording directly
parecord --format=s16le --rate=44100 --channels=1 test.wav

# Check permissions
ls -la /dev/snd/
```

### Installation Commands
```bash
# Debian/Ubuntu
sudo apt install pipewire-audio-client-libraries pipewire-pulse

# Fedora
sudo dnf install pipewire-pulseaudio pipewire-utils

# Arch
sudo pacman -S pipewire-pulse pipewire-alsa
```

## Future Considerations

As PipeWire continues to evolve:
- **JACK Compatibility**: PipeWire's JACK implementation may require specific handling
- **Bluetooth Audio**: PipeWire's Bluetooth support is improving rapidly
- **Low-Latency**: PipeWire offers better low-latency audio than PulseAudio
- **Professional Audio**: PipeWire may enable more advanced audio features

Our dual-method architecture is designed to adapt to these changes while maintaining backward compatibility.