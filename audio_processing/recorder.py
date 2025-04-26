import pyaudio
import wave
import os
import threading
from config.config import AUDIO_CONFIG
import sys
import time

def list_audio_devices():
    """
    List all available audio input devices.
    This function can be called directly without creating an AudioRecorder instance.
    """
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("Available audio input devices:")
    print("-" * 40)
    
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
    
    print("-" * 40)
    p.terminate()

class AudioRecorder:
    def __init__(self, output_directory='recordings', device_index=None):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.output_directory = output_directory
        self.input_device_index = device_index if device_index is not None else AUDIO_CONFIG.get('input_device_index')
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

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

    def record_until_q(self, file_path, input_device=None):
        """
        Record audio until Ctrl+C is pressed, with improved error handling.
        
        Args:
            file_path: Path to save the recorded audio file
            input_device: Input device index to use for recording
            
        Returns:
            Path to the recorded audio file or None if recording failed
        """
        # Ensure file_path includes output directory if it doesn't have a directory part
        if os.path.dirname(file_path) == '':
            file_path = os.path.join(self.output_directory, file_path)
        
        print(f"Attempting to open stream with device {input_device if input_device is not None else 0}")
        
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Use a larger chunk size to prevent overflow
            chunk_size = 4096  # Increased from 1024
            
            # Configure the stream with overflow tolerance
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=chunk_size
            )
            
            print("Recording... Press Ctrl+C to stop.")
            frames = []
            
            try:
                # Keep recording until Ctrl+C is pressed
                while True:
                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        frames.append(data)
                        # Visual feedback for recording in progress
                        sys.stdout.write('.')
                        sys.stdout.flush()
                        time.sleep(0.1)  # Small delay to reduce terminal output
                    except OSError as e:
                        # Log but continue if we get an overflow error
                        if "[Errno -9981] Input overflowed" in str(e):
                            sys.stdout.write('O')  # Indicate overflow
                            sys.stdout.flush()
                            time.sleep(0.2)  # Give more time to recover
                            continue
                        else:
                            # For other OSErrors, re-raise
                            raise
            
            except KeyboardInterrupt:
                print("\nRecording stopped by user.")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Make sure we actually recorded something
            if len(frames) == 0:
                print("No audio data was recorded. Check your microphone.")
                return None
            
            # Save the audio file and get the updated path
            updated_file_path = self.save_wav(frames, file_path)
            return updated_file_path
            
        except Exception as e:
            print(f"Error during recording: {e}")
            print("Try one of these troubleshooting steps:")
            print("1. Try recording with fixed duration instead: --record --duration 5")
            print("2. Restart your terminal/application")
            print("3. Check if another application is using the microphone")
            return None

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