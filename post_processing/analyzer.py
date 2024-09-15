import os
from typing import Any, Dict, Optional
from groq import Groq
from config.config import GROQ_API_KEY
import json  # Add this import at the top of the file

models = [
    "gemma-7b-it",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-8b-8192",
    "llama3-70b-8192",
]

MODEL = models[3]
    
class TextAnalyzer:
    """
    A class to perform post-processing analyses on transcribed text
    using Groq's Chat Completions API.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        self.client = Groq()

    def summarize_text(self, text: str, summary_length: Optional[int] = 150) -> str:
        """
        Summarize the provided text.

        :param text: The text to summarize.
        :param summary_length: Desired length of the summary.
        :return: Summarized text.
        """
        try:
            messages = [
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": f"Summarize the following text in {summary_length} words:\n\n{text}"}
            ]

            response = self.client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=0.3,
                max_tokens=summary_length * 2,
                top_p=1,
                stop=None,
                stream=False
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            print(f"An error occurred during summarization: {e}")
            return ""

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the provided text.

        :param text: The text to analyze.
        :return: A dictionary containing sentiment analysis results.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a data analyst API capable of sentiment analysis that responds in JSON. The JSON schema should include {\"sentiment_analysis\": {\"sentiment\": \"string (positive, negative, neutral)\", \"confidence_score\": \"number (0-1)\"}}."
                },
                {"role": "user", "content": text}
            ]

            response = self.client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=0.0,
                max_tokens=100,
                top_p=1,
                stop=None,
                stream=False,
                response_format={"type": "json_object"}
            )

            # Parse the JSON content from the response
            sentiment_data = json.loads(response.choices[0].message.content)
            return sentiment_data

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {"error": "Failed to parse sentiment analysis result"}
        except Exception as e:
            print(f"An error occurred during sentiment analysis: {e}")
            return {"error": str(e)}