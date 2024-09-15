import pyaudio
import wave
import os
import threading

class AudioRecorder:
    def __init__(self, output_directory='recordings'):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.output_directory = output_directory
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def list_input_devices(self):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print(f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
        
        p.terminate()

    def record_until_q(self, filename, input_device_index=None):
        p = pyaudio.PyAudio()

        if input_device_index is None:
            input_device_index = p.get_default_input_device_info()['index']

        print(f"Attempting to open stream with device {input_device_index}")
        try:
            stream = p.open(format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=self.chunk)
            print("Stream opened successfully")

            print(f"* Recording from device {input_device_index}. Press Ctrl+C to stop.")

            frames = []
            try:
                while True:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
            except KeyboardInterrupt:
                print("* Recording stopped.")
            except Exception as e:
                print(f"* Error during recording: {e}")
            finally:
                if stream.is_active():
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
            print(f"Error opening stream: {e}")
            p.terminate()
            return None

    def record(self, duration, filename, input_device_index=None):
        p = pyaudio.PyAudio()

        if input_device_index is None:
            input_device_index = p.get_default_input_device_info()['index']

        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=self.chunk)

        print(f"* Recording from device {input_device_index} for {duration} seconds.")

        frames = []

        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("* Done recording")

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

    def record_and_transcribe(self, duration, filename, transcription_api):
        file_path = self.record(duration, filename)
        return transcription_api.transcribe_audio(file_path)