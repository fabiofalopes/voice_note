import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import unittest
from unittest.mock import patch, MagicMock
from api_integrations.groq_whisper import GroqWhisperAPI

filename = "UsingOllamatoRunLocalLLMsontheRaspberryPi5.mp3"

class TestGroqWhisperAPI(unittest.TestCase):
    @patch('api_integrations.groq_whisper.Groq')
    def test_transcribe_audio_success(self, mock_groq):
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_transcription = MagicMock()
        mock_transcription.to_dict.return_value = {'text': 'Transcription successful.'}
        mock_client.audio.transcriptions.create.return_value = mock_transcription

        api = GroqWhisperAPI()
        result = api.transcribe_audio(filename)
        self.assertEqual(result, {'text': 'Transcription successful.'})

    @patch('api_integrations.groq_whisper.Groq')
    def test_transcribe_audio_failure(self, mock_groq):
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception('API Error')

        api = GroqWhisperAPI()
        result = api.transcribe_audio(filename)
        self.assertIn('error', result)

    @patch('api_integrations.groq_whisper.Groq')
    def test_translate_audio_success(self, mock_groq):
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_translation = MagicMock()
        mock_translation.to_dict.return_value = {'text': 'Translation successful.'}
        mock_client.audio.translations.create.return_value = mock_translation

        api = GroqWhisperAPI()
        result = api.translate_audio(filename)
        self.assertEqual(result, {'text': 'Translation successful.'})

    @patch('api_integrations.groq_whisper.Groq')
    def test_translate_audio_failure(self, mock_groq):
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_client.audio.translations.create.side_effect = Exception('API Error')

        api = GroqWhisperAPI()
        result = api.translate_audio(filename)
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()
