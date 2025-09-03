# Development Notes

## Project Evolution

### Initial Implementation
- **Goal**: Simple voice transcription using Groq's Whisper API
- **Audio**: Basic PyAudio implementation
- **Scope**: Cross-platform compatibility

### PipeWire Challenge
- **Problem**: Modern Linux distributions adopted PipeWire
- **Impact**: PyAudio compatibility issues on Debian 13, Ubuntu 22.04+
- **Symptoms**: Silent recordings, device detection failures

### Solution Architecture
- **Approach**: Dual-method recording system
- **Primary**: Native system tools (parecord for Linux)
- **Fallback**: PyAudio for cross-platform compatibility
- **Auto-config**: Smart method selection during initialization

## Technical Decisions

### Why parecord over PyAudio for PipeWire?
1. **Native Support**: parecord is designed for PipeWire/PulseAudio
2. **Reliability**: Direct integration with audio daemon
3. **Simplicity**: No complex device mapping required
4. **Future-proof**: Will work as PipeWire evolves

### Why Keep PyAudio?
1. **Cross-platform**: Works on macOS and Windows
2. **Fallback**: Useful when parecord isn't available
3. **Device Control**: Allows specific device selection
4. **Ecosystem**: Well-established Python audio library

### Auto-Configuration Strategy
- **Fast**: Complete in ~0.03 seconds
- **Silent**: No verbose output during initialization
- **Smart**: Tests methods without user interaction
- **Informative**: Shows user what will be used

## Code Organization

### Project Structure
```
voice_transcriber/
├── src/                    # Main source code
│   ├── cli.py             # Command-line interface
│   ├── api/               # API clients (Groq)
│   └── audio_processing/  # Recording system
├── scripts/               # Setup and utility scripts
├── docs/                  # Documentation
├── future_features/       # Preserved development assets
└── recordings/           # Default output directory
```

### Key Classes

#### AudioRecorder
- **Purpose**: Main recording interface
- **Features**: Auto-configuration, dual-method support
- **Methods**: 
  - `_auto_configure()`: Smart method selection
  - `record_until_q()`: Main recording function
  - `get_recording_info()`: User-friendly status

#### GroqWhisperClient
- **Purpose**: API integration for transcription
- **Features**: Error handling, model selection
- **Models**: whisper-large-v3, whisper-large-v3-turbo, distil-whisper

### Design Patterns

#### Strategy Pattern
- **Context**: AudioRecorder
- **Strategies**: parecord, PyAudio
- **Selection**: Based on system capabilities

#### Factory Pattern
- **Purpose**: Audio method creation
- **Logic**: System detection → Method selection → Configuration

#### Graceful Degradation
- **Primary**: Try best method first
- **Fallback**: Automatic fallback to alternatives
- **Error**: Clear messages when all methods fail

## Performance Optimizations

### Initialization Speed
- **Before**: 2-5 seconds (trial-and-error approach)
- **After**: ~0.03 seconds (smart pre-configuration)
- **Method**: Silent device testing during init

### Memory Usage
- **Streaming**: Process audio in chunks, not full files
- **Cleanup**: Automatic temporary file removal
- **Efficiency**: Minimal memory footprint

### CPU Usage
- **Chunk Size**: Optimized for different systems
- **PipeWire**: 512 samples (low latency)
- **Others**: 1024 samples (balanced)

## Testing Strategy

### Development Testing
During development, we created numerous test files to validate:
- System detection accuracy
- Method selection logic
- Cross-platform compatibility
- Error handling robustness

### Cleanup Decision
All temporary test files were removed because:
- **Purpose-built**: Created for specific development phases
- **Outdated**: Don't reflect current architecture
- **Clutter**: Confuse the codebase purpose
- **Maintenance**: Would require constant updates

### Production Testing
Users can test the system with:
```bash
python3 transcribe.py --list-devices  # System info
python3 transcribe.py --test-mic      # Microphone test
python3 transcribe.py                 # Full workflow
```

## Future Enhancements

### Short-term
- **Bluetooth**: Better wireless microphone support
- **USB**: Enhanced USB microphone detection
- **Quality**: Adaptive quality based on use case

### Medium-term
- **JACK**: Professional audio system support
- **Multi-channel**: Support for stereo/multi-mic setups
- **Noise Reduction**: Basic audio preprocessing

### Long-term
- **Real-time**: Live transcription capabilities
- **Voice Activity**: Smart recording start/stop
- **Speaker ID**: Multiple speaker identification

## Lessons Learned

### Audio System Complexity
- **Assumption**: PyAudio would work everywhere
- **Reality**: Modern Linux audio is more complex
- **Solution**: Embrace native tools where available

### User Experience Priority
- **Problem**: Technical failures frustrate users
- **Solution**: Front-load configuration, provide clear feedback
- **Result**: Users know immediately if system will work

### Cross-platform Challenges
- **Challenge**: Different audio systems per platform
- **Approach**: Abstraction layer with smart defaults
- **Outcome**: Single codebase works everywhere

### Documentation Importance
- **Need**: Complex audio systems require explanation
- **Solution**: Comprehensive docs with troubleshooting
- **Benefit**: Users can self-diagnose issues

## Code Quality Principles

### Simplicity
- **User Interface**: Simple commands, clear output
- **Internal Logic**: Complex but well-organized
- **Error Messages**: Actionable and specific

### Reliability
- **Fallbacks**: Multiple methods for robustness
- **Testing**: Validate methods before use
- **Recovery**: Graceful handling of failures

### Maintainability
- **Structure**: Clear separation of concerns
- **Documentation**: Comprehensive inline and external docs
- **Extensibility**: Easy to add new audio backends

### Performance
- **Startup**: Fast initialization
- **Runtime**: Efficient audio processing
- **Resources**: Minimal system impact