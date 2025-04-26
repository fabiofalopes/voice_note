import os
from typing import Any, Dict, Optional, List
from groq import Groq, APIError
from config.config import GROQ_API_KEY
import json
import time

# Define the preferred order of LLMs to use for analysis
PREFERRED_LLM_IDS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-8b-instant",
    "mistral-saba-24b",
    "gemma2-9b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "qwen-qwq-32b",
    "allam-2-7b",
]

class NoAvailableLLMError(Exception):
    """Custom exception raised when no suitable LLM is available or responds."""
    pass

class TextAnalyzer:
    """
    A class to perform post-processing analyses on transcribed text
    using Groq's Chat Completions API, with dynamic model selection.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        self.client = Groq()
        self.available_llms: Dict[str, Dict[str, Any]] = {}
        self.prioritized_llms: List[str] = []
        self._fetch_and_filter_models()

        if not self.prioritized_llms:
            raise NoAvailableLLMError("Initialization failed: No suitable LLMs found available via Groq API.")
        print(f"Initialized TextAnalyzer. Prioritized models: {self.prioritized_llms}")

    def _fetch_and_filter_models(self):
        """Fetches models from Groq API, filters for suitable LLMs, and prioritizes them."""
        try:
            models_response = self.client.models.list()
            all_models = models_response.data
        except APIError as e:
            print(f"Error fetching models from Groq API: {e}")
            # Optionally, load from a cached list or raise error
            return

        self.available_llms = {}
        excluded_keywords = ["whisper", "tts", "guard", "compound"]

        for model in all_models:
            if model.active:
                model_id = model.id
                # Filter out non-LLM models based on keywords
                if not any(keyword in model_id for keyword in excluded_keywords):
                    self.available_llms[model_id] = {
                        "context_window": model.context_window,
                        "max_completion_tokens": getattr(model, 'max_completion_tokens', None) # Use getattr for safety
                    }
        
        print(f"Found available LLMs: {list(self.available_llms.keys())}")

        # Create prioritized list
        self.prioritized_llms = []
        for pref_id in PREFERRED_LLM_IDS:
            if pref_id in self.available_llms:
                self.prioritized_llms.append(pref_id)

        # Add any remaining available LLMs not in the preferred list as fallback
        for available_id in self.available_llms:
            if available_id not in self.prioritized_llms:
                 self.prioritized_llms.append(available_id)

    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Makes the API call, iterating through prioritized models on failure."""
        last_error = None
        for model_id in self.prioritized_llms:
            try:
                print(f"Attempting API call with model: {model_id}")
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model_id,
                    **kwargs
                )
                print(f"API call successful with model: {model_id}")
                return response.choices[0].message.content # Return content directly
            
            except APIError as e:
                # Check for specific error codes indicating model unavailability
                # Adjust codes based on actual Groq SDK behavior if different
                if e.status_code == 400 and e.body and ("model_decommissioned" in e.body.get("code", "") or "model_not_found" in e.body.get("code", "")):
                    print(f"Model {model_id} is unavailable ({e.body.get('code', 'Unknown code')}). Trying next model.")
                    last_error = e
                    continue # Try the next model
                elif e.status_code == 429: # Rate limiting
                    print(f"Rate limit hit with model {model_id}. Waiting and retrying... ({e})")
                    time.sleep(5) # Simple backoff
                    # Consider re-trying the same model or moving to the next
                    last_error = e
                    continue # Re-try or try next model
                else:
                    print(f"An unexpected API error occurred with model {model_id}: {e}")
                    last_error = e
                    # Depending on the error, might want to break or continue
                    continue # Or raise e directly
            except Exception as e:
                print(f"A non-API error occurred during the call with model {model_id}: {e}")
                last_error = e
                continue # Try the next model

        # If loop finishes without success
        if last_error:
             print(f"All attempts failed. Last error: {last_error}")
             # Propagate the last known error or raise custom error
             raise NoAvailableLLMError(f"Failed to get completion from any model. Last error: {last_error}") from last_error
        else:
            # Should not happen if prioritized_llms was not empty initially, but good practice
            raise NoAvailableLLMError("Failed to get completion. No models were attempted.")

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
            
            # Use the helper method
            summary = self._make_api_call(
                messages=messages,
                temperature=0.3,
                max_tokens=summary_length * 2, # Allow ample tokens for summary
                top_p=1,
                stop=None,
                stream=False
            )
            return summary.strip() # .strip() the final result

        except NoAvailableLLMError as e:
            print(f"Summarization failed: {e}")
            return "" # Or raise
        except Exception as e:
            print(f"An unexpected error occurred during summarization: {e}")
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
                    "content": "You are a data analyst API capable of sentiment analysis that responds in JSON. The JSON schema should include {\"sentiment_analysis\": {\"sentiment\": \"string (positive, negative, neutral)\", \"confidence_score\": \"number (0-1)\"}}"
                },
                {"role": "user", "content": text}
            ]
            
            # Use the helper method
            response_content = self._make_api_call(
                messages=messages,
                temperature=0.0,
                max_tokens=100,
                top_p=1,
                stop=None,
                stream=False,
                response_format={"type": "json_object"}
            )

            sentiment_data = json.loads(response_content)
            return sentiment_data

        except NoAvailableLLMError as e:
            print(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from API response: {e}")
            return {"error": "Failed to parse sentiment analysis result"}
        except Exception as e:
            print(f"An unexpected error occurred during sentiment analysis: {e}")
            return {"error": str(e)}

    def analyze_key_components(self, text: str) -> Dict[str, Any]:
        """
        Analyze the provided text to extract key components, considerations, and important aspects.
        This method helps identify critical elements, potential areas of focus, and important considerations
        in any given text, regardless of whether it's task-oriented or not.
        :param text: The text to analyze.
        :return: A dictionary containing structured analysis of key components and considerations.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert task analyzer. Given an input, extract the main task, requirements, steps, and any additional components needed to achieve the objective. Provide a structured response in JSON format."
                    #"content": "You are an expert text analyzer. Given an input, extract key components, important considerations, potential areas of focus, and any additional elements that deserve attention. Provide a structured response in JSON format."
                },
                {"role": "user", "content": f"Analyze the following text and extract task information:\n\n{text}"}            
                #{"role": "user", "content": f"Analyze the following text and extract key components and considerations:\n\n{text}"}
            ]
            
            # Use the helper method
            response_content = self._make_api_call(
                messages=messages,
                temperature=0.2,
                max_tokens=1000, # Ensure enough tokens for potentially complex analysis
                top_p=1,
                stop=None,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            analysis_data = json.loads(response_content)
            return analysis_data

        except NoAvailableLLMError as e:
            print(f"Key component analysis failed: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from API response: {e}")
            return {"error": "Failed to parse analysis result"}
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}")
            return {"error": str(e)}

    def get_thinking_tags(self, text: str) -> Dict[str, Any]:
        """
        Extract thinking tags from the provided text.
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
            
            # Use the helper method
            response_content = self._make_api_call(
                messages=messages,
                temperature=0.2,
                max_tokens=500, # Adjust as needed
                top_p=1,
                stop=None,
                stream=False,
                response_format={"type": "json_object"}
            )

            tag_data = json.loads(response_content)
            return tag_data
            
        except NoAvailableLLMError as e:
            print(f"Thinking tag extraction failed: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from API response: {e}")
            return {"error": "Failed to parse thinking tag analysis result"}
        except Exception as e:
            print(f"An unexpected error occurred during thinking tag analysis: {e}")
            return {"error": str(e)}