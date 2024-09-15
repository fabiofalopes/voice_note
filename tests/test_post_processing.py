import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from post_processing.analyzer import TextAnalyzer

class TestTextAnalyzer(unittest.TestCase):
    def setUp(self):
        os.environ['GROQ_API_KEY'] = 'test_api_key'
        self.analyzer = TextAnalyzer()

    @patch('post_processing.analyzer.Groq.chat.completions.create')  # Updated patch target
    def test_summarize_text(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a summary."
        mock_create.return_value = mock_response

        summary = self.analyzer.summarize_text("This is a long text that needs summarization.")
        self.assertEqual(summary, "This is a summary.")

    @patch('post_processing.analyzer.Groq.chat.completions.create')  # Updated patch target
    def test_analyze_sentiment(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"sentiment_analysis": {"sentiment": "positive", "confidence_score": 0.95}}'
        mock_create.return_value = mock_response

        sentiment = self.analyzer.analyze_sentiment("I love this product!")
        expected = {
            "sentiment_analysis": {
                "sentiment": "positive",
                "confidence_score": 0.95
            }
        }
        self.assertEqual(sentiment, expected)

    def tearDown(self):
        del os.environ['GROQ_API_KEY']

if __name__ == '__main__':
    unittest.main()