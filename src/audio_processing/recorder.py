import pyaudio
import wave
import os
import sys
import time
import platform
import subprocess
import contextlib
import tempfile

# Suppress ALSA warnings on Linux
@contextlib.contextmanager
def suppress_alsa_warnings():
    """Context manager to suppress ALSA warnings on Linux"""
    if platform.system() == 'Linux':
        # Redirect stderr to devnull to suppress ALSA warnings
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    else:
        yield

def get_system_info():
    """Get system information for audio troubleshooting"""
    system = platform.system()
    info = {
        'system': system,
        'audio_backend': 'unknown',
        'pipewire_sources': [],
        'default_source': None
    }
    
    if system == 'Linux':
        # Check for PipeWire
        try:
            result = subprocess.run(['pgrep', '-f', 'pipewire'], capture_output=True, text=True)
            if result.returncode == 0:
                info['audio_backend'] = 'pipewire'
                
                # Get PipeWire sources
                try:
                    result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if 'source' in line and 'monitor' not in line:
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    info['pipewire_sources'].append({
                                        'id': parts[0],
                                        'name': parts[1],
                                        'status': parts[4] if len(parts) > 4 else 'UNKNOWN'
                                    })
                except:
                    pass
                
                # Get default source
                try:
                    result = subprocess.run(['pactl', 'info'], capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'Default Source:' in line:
                                info['default_source'] = line.split(':', 1)[1].strip()
                                break
                except:
                    pass
            else:
                # Check for PulseAudio
                result = subprocess.run(['pgrep', '-f', 'pulseaudio'], capture_output=True, text=True)
                if result.returncode == 0:
                    info['audio_backend'] = 'pulseaudio'
                else:
                    info['audio_backend'] = 'alsa'
        except:
            info['audio_backend'] = 'unknown'
    
    return info

def record_with_parecord(output_file, duration=None):
    """Record audio using parecord (PipeWire/PulseAudio) as fallback"""
    try:
        cmd = ['parecord', '--format=s16le', '--rate=44100', '--channels=1']
        if duration:
            # For fixed duration, we'll handle it differently
            pass
        cmd.append(output_file)
        
        print("🎤 Recording with parecord... Press Ctrl+C to stop.")
        process = subprocess.Popen(cmd)
        
        if duration:
            time.sleep(duration)
            process.terminate()
        else:
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n✅ Recording stopped by user.")
                process.terminate()
                process.wait()
        
        return output_file if os.path.exists(output_file) and os.path.getsize(output_file) > 0 else None
        
    except Exception as e:
        print(f"❌ parecord failed: {e}")
        return None

def list_audio_devices():
    """
    List all available audio input devices with system information and recommendations.
    This function can be called directly without creating an AudioRecorder instance.
    """
    system_info = get_system_info()
    print(f"System: {system_info['system']} | Audio Backend: {system_info['audio_backend']}")
    print("=" * 60)
    
    # Show PipeWire/PulseAudio sources first (most relevant for Linux)
    if system_info['system'] == 'Linux' and system_info['pipewire_sources']:
        print("🎤 PipeWire/PulseAudio Sources (Recommended):")
        for source in system_info['pipewire_sources']:
            status_icon = "🎤" if source['status'] != 'SUSPENDED' else "💤"
            default_marker = " [DEFAULT]" if source['name'] == system_info['default_source'] else ""
            print(f"   {status_icon} {source['name']}{default_marker}")
        print()
    
    # Show PyAudio devices
    try:
        with suppress_alsa_warnings():
            p = pyaudio.PyAudio()
        
        print("🔧 PyAudio Devices:")
        print("-" * 60)
        
        # Check all devices across all host APIs
        device_count = p.get_device_count()
        input_devices = []
        
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    host_api = p.get_host_api_info_by_index(device_info['hostApi'])
                    input_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'rate': device_info['defaultSampleRate'],
                        'host_api': host_api['name']
                    })
            except Exception as e:
                print(f"  Device {i}: Error reading device info - {e}")
        
        # Sort devices by preference (default first, then by name)
        try:
            default_device = p.get_default_input_device_info()
            default_index = default_device['index']
        except:
            default_index = -1
        
        # Show default device first
        for device in input_devices:
            if device['index'] == default_index:
                print(f"  {device['index']}: {device['name']} [DEFAULT]")
                print(f"      Channels: {device['channels']}, Rate: {device['rate']:.0f}Hz, API: {device['host_api']}")
        
        # Show other devices
        for device in input_devices:
            if device['index'] != default_index:
                print(f"  {device['index']}: {device['name']}")
                print(f"      Channels: {device['channels']}, Rate: {device['rate']:.0f}Hz, API: {device['host_api']}")
        
        print("-" * 60)
        print(f"Total PyAudio devices found: {len(input_devices)}")
        
        p.terminate()
        
    except Exception as e:
        print(f"Error listing PyAudio devices: {e}")
    
    # Show recommendations
    print("\n💡 Recommendations:")
    if system_info['system'] == 'Linux' and system_info['audio_backend'] == 'pipewire':
        print("  - For best results on PipeWire, use default auto-detection")
        print("  - System will automatically use parecord for optimal compatibility")
    print("  - Built-in laptop microphones typically work best")
    print("  - For external mics, ensure they're set as system default")
    
    # Show current auto-configuration
    print("\n🚀 Current Auto-Configuration:")
    try:
        from audio_processing.recorder import AudioRecorder
        recorder = AudioRecorder()
        print(f"   {recorder.get_recording_info()}")
        if recorder.recording_method:
            print("   ✅ Ready to record immediately!")
        else:
            print("   ❌ No working method - check troubleshooting below")
    except Exception as e:
        print(f"   Error: {e}")
    
    if system_info['system'] == 'Linux':
        print("\n🔧 Linux Troubleshooting:")
        print("  - Install: sudo apt install pipewire-audio-client-libraries")
        print("  - Check permissions: ls -la /dev/snd/")
        print("  - Add to audio group: sudo usermod -a -G audio $USER")
        print("  - Test with: parecord --format=s16le test.wav")

class AudioRecorder:
    def __init__(self, output_directory='recordings', device_index=None):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.output_directory = output_directory
        self.input_device_index = device_index
        self.recording_method = None  # Will be determined automatically
        self.system_info = None
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Auto-configure the best recording method on initialization
        self._auto_configure()

    def list_input_devices(self):
        """
        List all available audio input devices.
        This is an instance method that does the same as the standalone function.
        """
        list_audio_devices()

    def save_wav(self, frames, file_path):
        """
        Save recorded frames as a WAV file with timestamp to prevent overwriting.
        
        Args:
            frames: List of audio data frames
            file_path: Path to save the file
            
        Returns:
            Updated file path with timestamp
        """
        # Add timestamp to filename if not already present
        if "_20" not in file_path:  # Check if timestamp is already in filename
            dir_path = os.path.dirname(file_path)
            base, ext = os.path.splitext(os.path.basename(file_path))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(dir_path, f"{base}_{timestamp}{ext}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Open and write the WAV file
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"\nSaved audio to {file_path}")
        
        return file_path  # Return the updated file path

    def _find_working_device(self):
        """Find a working audio input device with cross-platform compatibility (silent)"""
        with suppress_alsa_warnings():
            p = pyaudio.PyAudio()
        
        # Get all input devices
        input_devices = []
        device_count = p.get_device_count()
        
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    host_api = p.get_host_api_info_by_index(device_info['hostApi'])
                    input_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'rate': device_info['defaultSampleRate'],
                        'host_api': host_api['name'],
                        'info': device_info
                    })
            except Exception:
                continue
        
        if not input_devices:
            p.terminate()
            print("❌ No input devices found")
            return None
        
        # Try devices in order of preference based on system
        device_priority = self._get_device_priority(input_devices, self.system_info or get_system_info())
        
        for device in device_priority:
            device_id = device['index']
            
            # Try different configurations for better compatibility
            configs = self._get_audio_configs(device, self.system_info or get_system_info())
            
            for config in configs:
                try:
                    test_stream = p.open(
                        format=config['format'],
                        channels=config['channels'],
                        rate=config['rate'],
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=config['chunk_size']
                    )
                    
                    # Test reading some data
                    test_data = test_stream.read(config['chunk_size'], exception_on_overflow=False)
                    test_stream.close()
                    
                    # Update our settings to match working config
                    self.format = config['format']
                    self.channels = config['channels']
                    self.rate = config['rate']
                    self.chunk = config['chunk_size']
                    
                    p.terminate()
                    return device_id
                    
                except Exception as e:
                    continue
        
        p.terminate()
        return None
    
    def _get_device_priority(self, input_devices, system_info):
        """Get device priority order based on system and audio backend"""
        # Try to get default device first
        default_device = None
        try:
            p = pyaudio.PyAudio()
            default_info = p.get_default_input_device_info()
            default_index = default_info['index']
            p.terminate()
            
            for device in input_devices:
                if device['index'] == default_index:
                    default_device = device
                    break
        except:
            pass
        
        # Create priority list
        priority_devices = []
        
        # Add default device first if found
        if default_device:
            priority_devices.append(default_device)
        
        # System-specific priorities
        if system_info['system'] == 'Linux':
            if system_info['audio_backend'] == 'pipewire':
                # PipeWire preferences
                api_priority = ['ALSA', 'pulse', 'jack']
                name_keywords = ['default', 'pipewire', 'pulse', 'built-in', 'internal']
            else:
                # PulseAudio/ALSA preferences
                api_priority = ['pulse', 'ALSA', 'jack']
                name_keywords = ['default', 'pulse', 'built-in', 'internal', 'usb']
        elif system_info['system'] == 'Darwin':  # macOS
            api_priority = ['Core Audio']
            name_keywords = ['built-in', 'internal', 'macbook', 'imac']
        else:  # Windows
            api_priority = ['MME', 'DirectSound', 'WASAPI']
            name_keywords = ['default', 'microphone', 'built-in', 'internal']
        
        # Add devices by API priority
        for api in api_priority:
            for device in input_devices:
                if (device not in priority_devices and 
                    api.lower() in device['host_api'].lower()):
                    priority_devices.append(device)
        
        # Add remaining devices with preferred names
        for keyword in name_keywords:
            for device in input_devices:
                if (device not in priority_devices and 
                    keyword.lower() in device['name'].lower()):
                    priority_devices.append(device)
        
        # Add any remaining devices
        for device in input_devices:
            if device not in priority_devices:
                priority_devices.append(device)
        
        return priority_devices
    
    def _get_audio_configs(self, device, system_info):
        """Get audio configurations to try for a device"""
        configs = []
        
        # Base configuration
        base_config = {
            'format': pyaudio.paInt16,
            'channels': min(device['channels'], 2),  # Prefer mono/stereo
            'rate': int(device['rate']),
            'chunk_size': 1024
        }
        
        # System-specific optimizations
        if system_info['system'] == 'Linux':
            if system_info['audio_backend'] == 'pipewire':
                # PipeWire works better with specific settings
                configs.extend([
                    {**base_config, 'channels': 1, 'rate': 44100, 'chunk_size': 512},
                    {**base_config, 'channels': 1, 'rate': 48000, 'chunk_size': 1024},
                    {**base_config, 'channels': 2, 'rate': 44100, 'chunk_size': 1024},
                ])
            else:
                # PulseAudio/ALSA configurations
                configs.extend([
                    {**base_config, 'channels': 1, 'rate': 44100, 'chunk_size': 1024},
                    {**base_config, 'channels': 2, 'rate': 44100, 'chunk_size': 2048},
                    {**base_config, 'channels': 1, 'rate': 48000, 'chunk_size': 1024},
                ])
        else:
            # Windows/macOS configurations
            configs.extend([
                {**base_config, 'channels': 1, 'rate': 44100, 'chunk_size': 1024},
                {**base_config, 'channels': 2, 'rate': 44100, 'chunk_size': 1024},
                {**base_config, 'channels': 1, 'rate': 48000, 'chunk_size': 1024},
            ])
        
        # Add fallback configurations
        configs.extend([
            {**base_config, 'channels': 1, 'rate': 22050, 'chunk_size': 512},
            {**base_config, 'channels': 1, 'rate': 16000, 'chunk_size': 512},
        ])
        
        return configs
    
    def _auto_configure(self):
        """Automatically configure the best recording method and settings"""
        self.system_info = get_system_info()
        
        # Quick and silent configuration
        if self.system_info['system'] == 'Linux' and self.system_info['audio_backend'] in ['pipewire', 'pulseaudio']:
            # For Linux with PipeWire/PulseAudio, prefer parecord if available
            if self._test_parecord_availability():
                self.recording_method = 'parecord'
                return
        
        # Try PyAudio configuration
        if self._configure_pyaudio():
            self.recording_method = 'pyaudio'
            return
        
        # Fallback to parecord for Linux
        if self.system_info['system'] == 'Linux' and self._test_parecord_availability():
            self.recording_method = 'parecord'
            return
        
        # No working method found
        self.recording_method = None
    
    def _test_parecord_availability(self):
        """Quick test if parecord is available and working"""
        try:
            result = subprocess.run(['which', 'parecord'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def _configure_pyaudio(self):
        """Try to configure PyAudio with a working device"""
        try:
            device_id = self._find_working_device()
            if device_id is not None:
                self.input_device_index = device_id
                return True
        except:
            pass
        return False
    
    def get_recording_info(self):
        """Get information about the configured recording method"""
        if not self.recording_method:
            return "❌ No working recording method found"
        
        if self.recording_method == 'parecord':
            default_source = self.system_info.get('default_source', 'Unknown')
            return f"🎤 parecord → {default_source}"
        elif self.recording_method == 'pyaudio':
            return f"🎤 PyAudio → Device {self.input_device_index}"
        
        return "❓ Unknown recording method"

    def record_until_q(self, file_path, input_device=None):
        """
        Record audio until Ctrl+C is pressed, using pre-configured method.
        
        Args:
            file_path: Path to save the recorded audio file
            input_device: Input device index to use for recording (overrides auto-config)
            
        Returns:
            Path to the recorded audio file or None if recording failed
        """
        # Ensure file_path includes output directory if it doesn't have a directory part
        if os.path.dirname(file_path) == '':
            file_path = os.path.join(self.output_directory, file_path)
        
        # If user specified a device, try PyAudio with that device
        if input_device is not None:
            print(f"🎤 Using specified device {input_device} with PyAudio...")
            return self._try_pyaudio_recording(file_path, input_device, self.system_info)
        
        # Use pre-configured recording method
        if not self.recording_method:
            print("❌ No working recording method available.")
            self._print_troubleshooting_tips()
            return None
        
        # Show what we're using
        print(self.get_recording_info())
        
        if self.recording_method == 'parecord':
            return self._try_parecord_recording(file_path, self.system_info)
        elif self.recording_method == 'pyaudio':
            return self._try_pyaudio_recording(file_path, self.input_device_index, self.system_info)
        
        print("❌ Unknown recording method.")
        return None
    
    def _try_pyaudio_recording(self, file_path, input_device, system_info):
        """Try recording with PyAudio"""
        try:
            # Use the device from constructor if not specified
            if input_device is None:
                input_device = self.input_device_index
            
            # If no device specified, try to find a working one
            if input_device is None:
                print("🔍 Auto-detecting working audio device...")
                input_device = self._find_working_device()
                if input_device is None:
                    print("❌ No working PyAudio device found.")
                    return None
            
            # Verify device works before starting recording
            if not self._test_device_quickly(input_device):
                print(f"❌ PyAudio device {input_device} is not working properly.")
                return None
            
            print(f"🎤 Using PyAudio device: {input_device}")
            print(f"   Format: {self.channels}ch, {self.rate}Hz, chunk={self.chunk}")
            
            # Initialize PyAudio
            with suppress_alsa_warnings():
                p = pyaudio.PyAudio()
            
            # Configure the stream with current settings
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.chunk
            )
            
            print("Recording with PyAudio... Press Ctrl+C to stop.")
            frames = []
            overflow_count = 0
            
            try:
                # Keep recording until Ctrl+C is pressed
                while True:
                    try:
                        data = stream.read(self.chunk, exception_on_overflow=False)
                        frames.append(data)
                        
                        # Visual feedback for recording in progress
                        if len(frames) % 10 == 0:  # Less frequent updates
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.01)
                        
                    except OSError as e:
                        # Handle overflow errors gracefully
                        if "[Errno -9981] Input overflowed" in str(e) or "Input overflowed" in str(e):
                            overflow_count += 1
                            if overflow_count % 10 == 0:
                                sys.stdout.write('O')  # Indicate overflow
                                sys.stdout.flush()
                            time.sleep(0.05)  # Give time to recover
                            continue
                        else:
                            # For other OSErrors, re-raise
                            raise
            
            except KeyboardInterrupt:
                print(f"\n✅ Recording stopped by user. ({len(frames)} frames recorded)")
                if overflow_count > 0:
                    print(f"   Note: {overflow_count} buffer overflows occurred (this is usually OK)")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Make sure we actually recorded something
            if len(frames) == 0:
                print("❌ No audio data was recorded with PyAudio.")
                return None
            
            # Check if we got actual audio data (not just silence)
            audio_detected = self._check_audio_content(frames)
            if not audio_detected:
                print("⚠️  PyAudio recorded only silence - microphone may not be connected properly")
                return None
            
            # Save the audio file and get the updated path
            updated_file_path = self.save_wav(frames, file_path)
            return updated_file_path
            
        except Exception as e:
            print(f"❌ PyAudio recording failed: {e}")
            return None
    
    def _try_parecord_recording(self, file_path, system_info):
        """Try recording with parecord (PipeWire/PulseAudio)"""
        try:
            # Create temporary WAV file
            temp_file = file_path + '.temp.wav'
            
            print("Recording... Press Ctrl+C to stop.")
            
            result = record_with_parecord(temp_file)
            if result and os.path.exists(temp_file):
                # Check if we got actual audio
                if os.path.getsize(temp_file) > 1000:  # More than just header
                    # Move temp file to final location
                    final_path = self.save_wav_from_file(temp_file, file_path)
                    os.remove(temp_file)
                    return final_path
                else:
                    print("❌ Recording produced empty file - check microphone")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            return None
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
    
    def _check_audio_content(self, frames):
        """Check if frames contain actual audio data (not just silence)"""
        if not frames:
            return False
        
        # Sample a few frames to check for non-zero audio
        sample_frames = frames[::max(1, len(frames)//10)]  # Sample every 10th frame
        
        for frame in sample_frames:
            # Convert bytes to integers and check for significant audio levels
            for i in range(0, len(frame) - 1, 2):
                sample = int.from_bytes(frame[i:i+2], 'little', signed=True)
                if abs(sample) > 100:  # Threshold for detecting actual audio
                    return True
        
        return False
    
    def save_wav_from_file(self, source_file, target_path):
        """Copy and rename a WAV file with timestamp"""
        # Add timestamp to filename if not already present
        if "_20" not in target_path:  # Check if timestamp is already in filename
            dir_path = os.path.dirname(target_path)
            base, ext = os.path.splitext(os.path.basename(target_path))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            target_path = os.path.join(dir_path, f"{base}_{timestamp}{ext}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(target_path)) if os.path.dirname(target_path) else '.', exist_ok=True)
        
        # Copy the file
        import shutil
        shutil.copy2(source_file, target_path)
        print(f"\nSaved audio to {target_path}")
        
        return target_path
    
    def _test_device_quickly(self, device_id):
        """Quick test to see if a device works"""
        try:
            with suppress_alsa_warnings():
                p = pyaudio.PyAudio()
                stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.chunk
            )
            
            # Try to read a small amount of data
            data = stream.read(self.chunk, exception_on_overflow=False)
            
            stream.close()
            p.terminate()
            return True
            
        except Exception:
            return False
    
    def _print_troubleshooting_tips(self):
        """Print system-specific troubleshooting tips"""
        system_info = get_system_info()
        
        print("\n🔧 Troubleshooting steps:")
        print("1. List devices: python3 transcribe.py --list-devices")
        print("2. Try specific device: python3 transcribe.py --device <device_id>")
        print("3. Test microphone: python3 transcribe.py --test-mic")
        
        if system_info['system'] == 'Linux':
            print("\n🐧 Linux-specific tips:")
            print("4. Check microphone permissions: ls -la /dev/snd/")
            print("5. Add user to audio group: sudo usermod -a -G audio $USER")
            
            if system_info['audio_backend'] == 'pipewire':
                print("6. Check PipeWire sources: pactl list sources short")
                print("7. Test with parecord: parecord --format=s16le test.wav")
                print("8. Install PipeWire support: sudo apt install pipewire-audio-client-libraries")
                print("9. Restart PipeWire: systemctl --user restart pipewire")
                
                if system_info['default_source']:
                    print(f"10. Default source: {system_info['default_source']}")
                
                if system_info['pipewire_sources']:
                    print("11. Available sources:")
                    for source in system_info['pipewire_sources']:
                        status = "ACTIVE" if source['status'] != 'SUSPENDED' else "SUSPENDED"
                        print(f"    - {source['name']} ({status})")
            else:
                print("6. Install audio dev packages: sudo apt install portaudio19-dev python3-pyaudio")
                print("7. Check PulseAudio: pulseaudio --check -v")
        
        elif system_info['system'] == 'Darwin':
            print("\n🍎 macOS-specific tips:")
            print("4. Check microphone permissions in System Preferences > Security & Privacy")
            print("5. Try restarting Core Audio: sudo killall coreaudiod")
        
        elif system_info['system'] == 'Windows':
            print("\n🪟 Windows-specific tips:")
            print("4. Check microphone permissions in Windows Settings > Privacy")
            print("5. Update audio drivers")
            print("6. Try running as administrator")
        
        print("\n12. Close other audio applications (Zoom, Discord, etc.)")
        print("13. Try a different USB port (for USB microphones)")
        print("14. Check if microphone is muted in system settings")
        print("15. Restart the application")

    def record(self, duration, filename, input_device=None):
        """
        Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds
            filename: Name of the output file
            input_device: Input device index to use for recording
            
        Returns:
            Path to the recorded audio file or None if recording failed
        """
        # Verify device before recording
        if not self.verify_input_device(input_device):
            print("Failed to verify input device. Please check your microphone settings.")
            return None

        p = pyaudio.PyAudio()

        if input_device is None:
            input_device = p.get_default_input_device_info()['index']

        try:
            stream = p.open(format=self.format,
                           channels=self.channels,
                           rate=self.rate,
                           input=True,
                           input_device_index=input_device,
                           frames_per_buffer=self.chunk)

            print(f"* Recording from device {input_device} for {duration} seconds.")
            print("* Recording level: ", end='', flush=True)

            frames = []
            # Make sure to use exception_on_overflow=False to prevent crashes
            for i in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                # Print a simple level indicator
                if i % 8 == 0:
                    print(".", end='', flush=True)

            print("\n* Done recording")

            stream.stop_stream()
            stream.close()
            p.terminate()

            file_path = os.path.join(self.output_directory, filename)
            wf = wave.open(file_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            return file_path

        except Exception as e:
            print(f"\nError during recording: {e}")
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            p.terminate()
            return None

    def record_and_transcribe(self, duration, filename, transcription_api):
        file_path = self.record(duration, filename)
        return transcription_api.transcribe_audio(file_path)

    def verify_input_device(self, input_device=None):
        """
        Verify that the selected input device is working properly.
        Returns True if device is working, False otherwise.
        """
        try:
            p = pyaudio.PyAudio()
            
            # List available devices for debugging
            info = p.get_host_api_info_by_index(0)
            available_devices = []
            for i in range(info.get('deviceCount')):
                if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                    available_devices.append(i)
            
            if input_device is None:
                input_device = p.get_default_input_device_info()['index']
            
            # Check if device index exists
            if input_device not in available_devices:
                print(f"Error: Device index {input_device} not found. Available devices are: {available_devices}")
                return False
            
            # Get device info
            device_info = p.get_device_info_by_host_api_device_index(0, input_device)
            
            # Configure stream based on device capabilities
            channels = min(device_info.get('maxInputChannels'), self.channels)
            
            # Try to open a short test stream
            stream = p.open(format=self.format,
                           channels=channels,  # Use device's supported channels
                           rate=self.rate,
                           input=True,
                           input_device_index=input_device,
                           frames_per_buffer=self.chunk)
            
            # Read a small amount of data to verify the stream works
            data = stream.read(self.chunk, exception_on_overflow=False)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return True
        except Exception as e:
            print(f"Error verifying input device {input_device}: {e}")
            if p:
                p.terminate()
            return False

    def test_microphone(self, device_id=None, duration=5):
        """Test microphone input levels to verify it's working with cross-platform support"""
        if device_id is None:
            print("🔍 Auto-detecting working microphone...")
            device_id = self._find_working_device()
            if device_id is None:
                print("❌ No working microphone found")
                self._print_troubleshooting_tips()
                return False
        
        try:
            with suppress_alsa_warnings():
                p = pyaudio.PyAudio()
                device_info = p.get_device_info_by_index(device_id)
            
            print(f"🎤 Testing microphone on device {device_id}: {device_info.get('name')}")
            print(f"   Format: {self.channels}ch, {self.rate}Hz")
            print("   Speak into your microphone for the next few seconds...")
            print()
            
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.chunk
            )
            
            max_level = 0
            avg_level = 0
            sample_count = 0
            frames_per_second = self.rate // self.chunk
            
            for i in range(frames_per_second * duration):
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    
                    # Calculate RMS audio level
                    if len(data) >= 2:
                        # Convert bytes to integers and calculate RMS
                        samples = []
                        for j in range(0, len(data) - 1, 2):
                            sample = int.from_bytes(data[j:j+2], 'little', signed=True)
                            samples.append(sample)
                        
                        if samples:
                            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                            audio_level = int(rms)
                            max_level = max(max_level, audio_level)
                            avg_level = (avg_level * sample_count + audio_level) / (sample_count + 1)
                            sample_count += 1
                            
                            # Visual level indicator (0-30 bars)
                            level_bars = min(int(audio_level / 200), 30)
                            bar_display = "█" * level_bars + "░" * (30 - level_bars)
                            
                            # Show level with percentage
                            percentage = min(int(audio_level / 100), 100)
                            print(f'\r   Level: {bar_display} {percentage:3d}%', end='', flush=True)
                    
                    time.sleep(0.05)  # Small delay for smoother display
                    
                except Exception as e:
                    if "overflow" in str(e).lower():
                        print('\r   Level: [OVERFLOW - reduce input volume]', end='', flush=True)
                        time.sleep(0.1)
                        continue
                    else:
                        raise
            
            stream.close()
            p.terminate()
            
            print(f"\n")
            print(f"   Max level detected: {max_level}")
            print(f"   Average level: {avg_level:.0f}")
            
            # Determine if microphone is working
            if max_level > 500:
                print("✅ Microphone is working well!")
                return True
            elif max_level > 100:
                print("⚠️  Microphone detected but signal is weak")
                print("   Try speaking louder or moving closer to the microphone")
                return True
            else:
                print("❌ Very low or no microphone input detected")
                print("   Check if microphone is muted or disconnected")
                self._print_troubleshooting_tips()
                return False
                
        except Exception as e:
            print(f"\n❌ Error testing microphone: {e}")
            self._print_troubleshooting_tips()
            return False

    def check_audio_levels(self, duration=3, input_device_index=None):
        """
        Monitor audio levels for a few seconds to help users verify their microphone is working.
        """
        try:
            p = pyaudio.PyAudio()
            if input_device_index is None:
                input_device_index = p.get_default_input_device_info()['index']

            stream = p.open(format=self.format,
                           channels=self.channels,
                           rate=self.rate,
                           input=True,
                           input_device_index=input_device_index,
                           frames_per_buffer=self.chunk)

            print(f"Monitoring audio levels for {duration} seconds...")
            print("Level: ", end='', flush=True)

            for i in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk, exception_on_overflow=False)
                # Calculate RMS value
                rms = sum(int.from_bytes(data[i:i+2], 'little', signed=True)**2 
                         for i in range(0, len(data), 2)) / (len(data)/2)
                level = min(int(rms**0.5/100), 8)  # Scale to 0-8
                
                if i % 4 == 0:
                    print('\r' + 'Level: ' + '█' * level + '░' * (8-level), end='', flush=True)

            print("\nDone monitoring audio levels")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return True

        except Exception as e:
            print(f"\nError monitoring audio levels: {e}")
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            p.terminate()
            return False