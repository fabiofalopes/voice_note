import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from audio_processing.recorder import AudioRecorder
from api_integrations.groq_whisper import GroqWhisperAPI

class TestAudioRecording(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test recordings
        self.temp_dir = tempfile.mkdtemp()
        self.recorder = AudioRecorder(output_directory=self.temp_dir)
        
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    @patch('pyaudio.PyAudio')
    def test_verify_input_device(self, mock_pyaudio):
        # Mock the PyAudio setup
        mock_instance = MagicMock()
        mock_pyaudio.return_value = mock_instance
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        
        # Mock the stream
        mock_stream = MagicMock()
        mock_instance.open.return_value = mock_stream
        mock_stream.read.return_value = b'0' * 1024  # Simulate some audio data
        
        # Test device verification
        result = self.recorder.verify_input_device()
        self.assertTrue(result)
        
    def test_end_to_end_recording_and_transcription(self):
        """
        Test actual recording and transcription with Groq API.
        This test requires a working microphone and API key.
        """
        # Skip if no API key is set
        if not os.getenv('GROQ_API_KEY'):
            self.skipTest("GROQ_API_KEY not set")
            
        # Record a short 3-second clip
        filename = "test_recording.wav"
        
        # First check audio levels
        self.assertTrue(self.recorder.check_audio_levels(duration=2))
        
        # Record audio
        file_path = self.recorder.record(duration=3, filename=filename)
        self.assertIsNotNone(file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Try to transcribe
        api = GroqWhisperAPI()
        result = api.transcribe_audio(file_path)
        
        # Check transcription result
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertNotEqual(result['text'], '')
        print(f"\nTranscription result: {result['text']}")

    @patch('pyaudio.PyAudio')
    @patch('api_integrations.groq_whisper.GroqWhisperAPI.transcribe_audio')
    def test_mocked_recording_and_transcription(self, mock_transcribe, mock_pyaudio):
        # Mock PyAudio setup
        mock_instance = MagicMock()
        mock_pyaudio.return_value = mock_instance
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        
        # Mock audio stream
        mock_stream = MagicMock()
        mock_instance.open.return_value = mock_stream
        mock_stream.read.return_value = b'0' * 1024
        
        # Mock transcription response
        mock_transcribe.return_value = {
            "text": "This is a test transcription",
            "model": "whisper-large-v3"
        }
        
        # Test recording and transcription
        api = GroqWhisperAPI()
        result = self.recorder.record_and_transcribe(
            duration=1,
            filename="test_mock.wav",
            transcription_api=api
        )
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertEqual(result["text"], "This is a test transcription")

if __name__ == '__main__':
    unittest.main() 