import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api_integrations.groq_whisper import GroqWhisperAPI

class TestTranscription(unittest.TestCase):
    @patch('api_integrations.groq_whisper.Groq')
    def test_transcribe_audio(self, mock_groq):
        # Mock the Groq client
        mock_client = MagicMock()
        mock_groq.return_value = mock_client

        # Mock the transcription response
        mock_transcription = MagicMock()
        mock_transcription.text = "This is a test transcription."
        mock_transcription.to_dict.return_value = {"text": "This is a test transcription."}
        mock_client.audio.transcriptions.create.return_value = mock_transcription

        # Initialize GroqWhisperAPI
        api = GroqWhisperAPI()

        # Test file path
        test_file_path = "UsingOllamatoRunLocalLLMsontheRaspberryPi5.mp3"

        # Mock file opening
        with patch('builtins.open', mock_open(read_data=b'dummy audio content')):
            # Test transcription
            result = api.transcribe_audio(
                file_path=test_file_path,
                model_id="distil-whisper-large-v3-en",
                prompt="Video by Ian Wootten: Using Ollama to Run Local LLMs on the Raspberry Pi 5",
                response_format="json",
                language="en",
                temperature=0.0
            )

        # Assert the result
        self.assertEqual(result, {"text": "This is a test transcription."})

        # Verify that the Groq client was called with the correct parameters
        mock_client.audio.transcriptions.create.assert_called_once_with(
            file=(os.path.basename(test_file_path), b'dummy audio content'),
            model="distil-whisper-large-v3-en",
            prompt="Video by Ian Wootten: Using Ollama to Run Local LLMs on the Raspberry Pi 5",
            response_format="json",
            language="en",
            temperature=0.0
        )

    @patch('api_integrations.groq_whisper.Groq')
    def test_transcribe_audio_failure(self, mock_groq):
        # Mock the Groq client to raise an exception
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("API Error")

        # Initialize GroqWhisperAPI
        api = GroqWhisperAPI()

        # Test file path
        test_file_path = "UsingOllamatoRunLocalLLMsontheRaspberryPi5.mp3"

        # Mock file opening
        with patch('builtins.open', mock_open(read_data=b'dummy audio content')):
            # Test transcription with expected failure
            result = api.transcribe_audio(test_file_path)

        # Assert that the result contains an error message
        self.assertIn('error', result)
        self.assertEqual(result['error'], "API Error")

if __name__ == '__main__':
    unittest.main()
