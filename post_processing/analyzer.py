import os
from typing import Any, Dict, Optional
from groq import Groq
from config.config import GROQ_API_KEY
import json  # Add this import at the top of the file

models = [
    "gemma2-9b-it",  # 0 - Google, 8,192 context window
    "llama-3.3-70b-versatile",  # 1 - Meta, 128K context, 32,768 max completion
    "llama-3.1-8b-instant",  # 2 - Meta, 128K context, 8,192 max completion
    "llama-3.3-70b-specdec",  # 3 - Meta, 8,192 context window
    "llama3-8b-8192",  # 4 - Meta, 8,192 context window
    "mixtral-8x7b-32768",  # 5 - Mistral, 32,768 context window
    "deepseek-r1-distill-llama-70b",  # 6 - DeepSeek, 128K context, 16,384 max completion
    "deepseek-r1-distill-llama-70b-specdec",  # 7 - DeepSeek, 128K context, 16,384 max completion
    "llama-3.2-1b-preview",  # 8 - Meta, 128K context, 8,192 max completion
    "llama-3.2-3b-preview",  # 9 - Meta, 128K context, 8,192 max completion
    "llama-3.2-11b-vision-preview",  # 10 - Meta, 128K context, 8,192 max completion
    "llama-3.2-90b-vision-preview",  # 11 - Meta, 128K context, 8,192 max completion
]

MODEL = models[3]
    
# Add the following dictionary with the models information

GROQ_MODELS = {
    "Production": {
        "distil-whisper-large-v3-en": {
            "developer": "HuggingFace",
            "context_window": None,
            "max_completion_tokens": None,
            "max_file_size": "25 MB",
            "card_link": "https://huggingface.co/distil-whisper/distil-large-v3",
        },
        "gemma2-9b-it": {
            "developer": "Google",
            "context_window": 8192,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://huggingface.co/google/gemma-2-9b-it",
        },
        "llama-3.3-70b-versatile": {
            "developer": "Meta",
            "context_window": 128000,
            "max_completion_tokens": 32768,
            "max_file_size": None,
            "card_link": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md",
        },
        "llama-3.1-8b-instant": {
            "developer": "Meta",
            "context_window": 128000,
            "max_completion_tokens": 8192,
            "max_file_size": None,
            "card_link": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
        },
        "llama-guard-3-8b": {
            "developer": "Meta",
            "context_window": 8192,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://console.groq.com/docs/model/llama-guard-3-8b",
        },
        "llama3-70b-8192": {
            "developer": "Meta",
            "context_window": 8192,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
        },
        "llama3-8b-8192": {
            "developer": "Meta",
            "context_window": 8192,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
        },
        "mixtral-8x7b-32768": {
            "developer": "Mistral",
            "context_window": 32768,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        },
        "whisper-large-v3": {
            "developer": "OpenAI",
            "context_window": None,
            "max_completion_tokens": None,
            "max_file_size": "25 MB",
            "card_link": "https://huggingface.co/openai/whisper-large-v3",
        },
        "whisper-large-v3-turbo": {
            "developer": "OpenAI",
            "context_window": None,
            "max_completion_tokens": None,
            "max_file_size": "25 MB",
            "card_link": "https://huggingface.co/openai/whisper-large-v3-turbo",
        },
    },
    "Preview": {
        "qwen-2.5-coder-32b": {
            "developer": "Alibaba Cloud",
            "context_window": 128000,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
        },
        "qwen-2.5-32b": {
            "developer": "Alibaba Cloud",
            "context_window": 128000,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
        },
        "deepseek-r1-distill-qwen-32b": {
            "developer": "DeepSeek",
            "context_window": 128000,
            "max_completion_tokens": 16384,
            "max_file_size": None,
            "card_link": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        },
        "deepseek-r1-distill-llama-70b-specdec": {
            "developer": "DeepSeek",
            "context_window": 128000,
            "max_completion_tokens": 16384,
            "max_file_size": None,
            "card_link": "https://console.groq.com/docs/model/deepseek-r1-distill-llama-70b",
        },
        "deepseek-r1-distill-llama-70b": {
            "developer": "DeepSeek",
            "context_window": 128000,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://console.groq.com/docs/model/deepseek-r1-distill-llama-70b",
        },
        "llama-3.3-70b-specdec": {
            "developer": "Meta",
            "context_window": 8192,
            "max_completion_tokens": None,
            "max_file_size": None,
            "card_link": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md",
        },
        "llama-3.2-1b-preview": {
            "developer": "Meta",
            "context_window": 128000,
            "max_completion_tokens": 8192,
            "max_file_size": None,
            "card_link": "https://huggingface.co/meta-llama/Llama-3.2-1B",
        },
        "llama-3.2-3b-preview": {
            "developer": "Meta",
            "context_window": 128000,
            "max_completion_tokens": 8192,
            "max_file_size": None,
            "card_link": "https://huggingface.co/meta-llama/Llama-3.2-3B",
        },
        "llama-3.2-11b-vision-preview": {
            "developer": "Meta",
            "context_window": 128000,
            "max_completion_tokens": 8192,
            "max_file_size": None,
            "card_link": "https://huggingface.co/meta-llama/Llama-3.2-11B-Vision",
        },
        "llama-3.2-90b-vision-preview": {
            "developer": "Meta",
            "context_window": 128000,
            "max_completion_tokens": 8192,
            "max_file_size": None,
            "card_link": "https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct",
        },
    },
}

# You can now use this dictionary to access model information.  For example:
# print(GROQ_MODELS["Production"]["llama3-70b-8192"]["card_link"])

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

    def extract_task_requirements(self, text: str) -> Dict[str, Any]:
        """
        Extract tasks, requirements, and steps from the provided text.

        :param text: The text to analyze.
        :return: A dictionary containing structured task information.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert task analyzer. Given an input, extract the main task, requirements, steps, and any additional components needed to achieve the objective. Provide a structured response in JSON format."
                },
                {"role": "user", "content": f"Analyze the following text and extract task information:\n\n{text}"}
            ]

            response = self.client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=0.2,
                max_tokens=1000,
                top_p=1,
                stop=None,
                stream=False,
                response_format={"type": "json_object"}
            )

            task_data = json.loads(response.choices[0].message.content)
            return task_data

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {"error": "Failed to parse task analysis result"}
        except Exception as e:
            print(f"An error occurred during task analysis: {e}")
            return {"error": str(e)}

    def get_thinking_tags(self, text: str) -> Dict[str, Any]:
        """
        Extract thinking tags from the provided text, using a DeepSeek model.

        :param text: The text to analyze.
        :return: A dictionary containing structured thinking tag information.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert tag extractor. Given an input, extract key concepts and represent them as concise tags. Provide a structured response in JSON format with a 'thinking_tags' key."
                },
                {"role": "user", "content": f"Analyze the following text and extract thinking tags:\n\n{text}"}
            ]

            response = self.client.chat.completions.create(
                messages=messages,
                model=models[6],  # You might want a separate DeepSeek model here if needed.
                temperature=0.2,
                max_tokens=500, # Adjust as needed
                top_p=1,
                stop=None,
                stream=False,
                response_format={"type": "json_object"}
            )

            tag_data = json.loads(response.choices[0].message.content)
            return tag_data

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {"error": "Failed to parse thinking tag analysis result"}
        except Exception as e:
            print(f"An error occurred during thinking tag analysis: {e}")
            return {"error": str(e)}