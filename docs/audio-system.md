# Audio System Architecture

## Overview

The Voice Transcriber uses a sophisticated dual-method audio recording system designed for maximum compatibility across different operating systems and audio backends.

## Architecture

```
AudioRecorder
├── Auto-Configuration (on init)
│   ├── System Detection (Linux/macOS/Windows)
│   ├── Audio Backend Detection (PipeWire/PulseAudio/ALSA/CoreAudio)
│   └── Method Selection (parecord/PyAudio)
│
├── Recording Methods
│   ├── parecord (Linux native - PipeWire/PulseAudio)
│   └── PyAudio (Cross-platform fallback)
│
└── Smart Fallback Chain
    ├── Linux: parecord → PyAudio
    └── Others: PyAudio → System-specific
```

## Method Selection Logic

### Linux Systems
1. **PipeWire/PulseAudio Detected**: Use `parecord` (native support)
2. **PyAudio Available**: Use PyAudio with optimized device detection
3. **Fallback**: Clear error messages with troubleshooting steps

### macOS Systems
1. **Core Audio**: Use PyAudio with Core Audio backend
2. **Device Priority**: Built-in microphones → USB → Bluetooth

### Windows Systems
1. **DirectSound/WASAPI**: Use PyAudio with Windows audio APIs
2. **Device Priority**: Default → Built-in → USB → Bluetooth

## Performance Characteristics

- **Initialization Time**: ~0.03 seconds
- **Device Detection**: Silent background testing
- **Memory Usage**: Minimal (streaming audio processing)
- **CPU Usage**: Low (optimized chunk sizes)

## Audio Configuration

### Optimal Settings
- **Sample Rate**: 44.1kHz (CD quality)
- **Bit Depth**: 16-bit (good quality/size balance)
- **Channels**: Mono (sufficient for speech)
- **Chunk Size**: 512-1024 samples (low latency)

### PipeWire Optimizations
- Smaller chunk sizes (512 samples)
- Direct parecord integration
- Automatic source selection
- Buffer overflow handling

## Error Handling

The system provides graceful degradation:

1. **Method Fails**: Automatic fallback to next method
2. **Device Unavailable**: Clear error with device list
3. **Permission Issues**: Specific troubleshooting steps
4. **Audio Overflows**: Graceful recovery with user notification

## Cross-Platform Compatibility

| Platform | Primary Method | Fallback | Status |
|----------|---------------|----------|---------|
| Linux (PipeWire) | parecord | PyAudio | ✅ Optimized |
| Linux (PulseAudio) | parecord | PyAudio | ✅ Tested |
| Linux (ALSA) | PyAudio | parecord | ✅ Compatible |
| macOS | PyAudio | - | ✅ Native |
| Windows | PyAudio | - | ✅ Native |

## Future Enhancements

- **JACK Support**: For professional audio setups
- **ASIO Support**: For low-latency Windows audio
- **Bluetooth Optimization**: Better wireless microphone support
- **Multi-channel Recording**: For advanced use cases