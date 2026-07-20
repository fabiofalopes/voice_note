---
created: 2024-09-15T01:54:13 (UTC +01:00)
tags: []
source: https://console.groq.com/docs/speech-text
author: 
---

# GroqCloud

> ## Excerpt
> Experience the fastest inference in the world

---
## Speech

Groq API is the fastest speech-to-text solution available, offering OpenAI-compatible endpoints that enable real-time transcriptions and translations. With Groq API, you can integrate high-quality audio processing into your applications at speeds that rival human interaction.

### [API Endpoints](https://console.groq.com/docs/speech-text#api-endpoints)

We support two endpoints:

-   **Transcriptions:**
    
    -   **Usage:** Convert audio to text.
    -   **API Endpoint:** `https://api.groq.com/openai/v1/audio/transcriptions`
-   **Translations:**
    
    -   **Usage:** Translate audio to English text.
    -   **API Endpoint:** `https://api.groq.com/openai/v1/audio/translations`

### [Supported Models](https://console.groq.com/docs/speech-text#supported-models)

**Distil-Whisper English**

-   **Model ID:** `distil-whisper-large-v3-en`
-   **Description:** Distil-Whisper English is a distilled, or compressed, version of OpenAI's Whisper model, designed to provide faster, lower cost English speech recognition while maintaining comparable accuracy.
-   **Supported Language(s):** English-only

**Whisper large-v3**

-   **Model ID:** `whisper-large-v3`
-   **Description:** Whisper large-v3 provides state-of-the-art performance with high accuracy for multilingual transcription and translation tasks.
-   **Supported Language(s):** Multilingual

  

#### Performance Comparisons

Compared to Whisper large-v3, Distil-Whisper English is 2 times faster and 49% smaller, with only 756 million parameters versus Whisper large-v3's 1.55 billion. Despite this reduction in size, Distil-Whisper English performs remarkably well, achieving a Word Error Rate (WER) within 1% of Whisper large-v3 on both short-form and long-form transcriptions. The distilled model excels in robustness to noise and shows reduced hallucination, with 1.3 times fewer instances of repeated 5-gram word duplicates and a 2.1% reduction in insertion error rate compared to Whisper large-v3.

### [Audio File Limitations](https://console.groq.com/docs/speech-text#audio-file-limitations)

-   **Max File Size:** 25 MB
-   **Minimum File Length:** .01 seconds
-   **Supported File Types:** `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`
-   **Single Audio Track:** Only the first track will be transcribed for files with multiple audio tracks. (e.g. dubbed video)
-   **Supported Response Formats:** `json`, `verbose_json`, `text`

### [Preprocessing Audio Files](https://console.groq.com/docs/speech-text#preprocessing-audio-files)

Our speech-to-text models will downsample audio to 16,000 Hz mono before transcribing. This preprocessing can be performed client-side to reduce file size and allow longer files to be uploaded to Groq.

The following `ffmpeg` command can be used to reduce file size:

  

```
ffmpeg \
  -i <your file> \
  -ar 16000 \
  -ac 1 \
  -map 0:a: \
  <output file name>

```

### [Transcription Endpoint Usage](https://console.groq.com/docs/speech-text#transcription-endpoint-usage)

The transcription endpoint allows you to transcribe spoken words in audio or video files. You can provide optional request parameters to customize the transcription output.

  

#### Optional Request Parameters

-   `prompt`:
    -   **Description:** Provide context or specify how to spell unfamiliar words (limited to 224 tokens).
    -   **Type:** `string`
    -   **Default:** `None`
-   `response_format`:
    -   **Description:** Define the output response format.
    -   **Type:** `string`
    -   **Default:** `json`
        -   Set to `verbose_json` to receive timestamps for audio segments.
        -   Set to `text` to return a text response.
-   `temperature`:
    -   **Description:** Specify a value between 0 and 1 to control the translation output.
    -   **Type:** `float`
    -   **Default:** `None`
-   `language`:
    -   **whisper-large-v3 only!**
    -   **Description:** Specify the language for transcription.
    -   **Type:** `string`
    -   **Default:** `None` (Models will auto-detect if not specified)
        -   Use [ISO 639-1 language codes](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) (e.g. "en" for English, "fr" for French, etc.).
        -   Specifying a language may improve transcription accuracy and speed.

#### Example Usage

The Groq SDK package can be installed using the following command:

```
pip install groq
```

The following code snippet demonstrates how to use Groq API to transcribe an audio file in Python:

```python
import os
from groq import Groq

# Initialize the Groq client
client = Groq()

# Specify the path to the audio file
filename = os.path.dirname(__file__) + "/sample_audio.m4a" # Replace with your audio file!

# Open the audio file
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
      file=(filename, file.read()), # Required audio file
      model="distil-whisper-large-v3-en", # Required model to use for transcription
      prompt="Specify context or spelling",  # Optional
      response_format="json",  # Optional
      language="en",  # Optional
      temperature=0.0  # Optional
    )
    # Print the transcription text
    print(transcription.text)
```

### [Translation Endpoint Usage](https://console.groq.com/docs/speech-text#translation-endpoint-usage)

The translation endpoint allows yout to translate spoken words in audio or video files to English. You can provide optional request parameters to customize the translation output.

  

#### Optional Request Parameters:

-   `prompt`:
    -   **Description:** Provide context or specify how to spell unfamiliar words (limited to 224 tokens).
    -   **Type:** `string`
    -   **Default:** `None`
-   `response_format`:
    -   **Description:** Define the output response format.
    -   **Type:** `string`
    -   **Default:** `json`
        -   Set to `verbose_json` to receive timestamps for audio segments.
        -   Set to `text` to return a text response.
-   `temperature`:
    -   **Description:** Specify a value between 0 and 1 to control the translation output.
    -   **Type:** `float`
    -   **Default:** `None`

#### Example Usage

The Groq SDK package can be installed using the following command:
```
pip install groq
```
The following code snippet demonstrates how to use Groq API to translate an audio file in Python:

```python
import os
from groq import Groq

# Initialize the Groq client
client = Groq()

# Specify the path to the audio file
filename = os.path.dirname(__file__) + "/sample_audio.m4a" # Replace with your audio file!

# Open the audio file
with open(filename, "rb") as file:
    # Create a translation of the audio file
    translation = client.audio.translations.create(
      file=(filename, file.read()), # Required audio file
      model="whisper-large-v3", # Required model to use for translation
      prompt="Specify context or spelling",  # Optional
      response_format="json",  # Optional
      temperature=0.0  # Optional
    )
    # Print the translation text
    print(translation.text)```

### [Prompting Guidelines](https://console.groq.com/docs/speech-text#prompting-guidelines)

The `prompt` parameter is an optional input of max 224 tokens that allows you to provide contextual information to the model, helping it maintain a consistent writing style.

**How It Works**

When you provide a `prompt` parameter, the speech-to-text model treats it as a prior transcript and follows its style, rather than adhering to the actual content of the audio segment. This means that the model will not:

-   Attempt to execute commands contained within the prompt
-   Follow instructions present in the prompt

In contrast to chat completion prompts, the `prompt` parameter is designed solely to provide stylistic guidance and contextual information to the model, rather than triggering specific actions or responses.

**Best Practices**

-   Provide contextual information about the audio segment, such as the type of conversation, topic, or speakers involved.
-   Use the same language as the language of the audio file.
-   Steer the model's output by denoting proper spellings or emulate a specific writing style or tone.
-   Keep the prompt concise and focused on stylistic guidance.

### [Use Cases](https://console.groq.com/docs/speech-text#use-cases)

Groq API offers low latency and fast inference for speech recognition and transcription, enabling developers to build a wide range of highly accurate, real-time applications, such as:

-   Real-Time Audio Translation: Translate audio files or real-time audio streams to break language barriers and facilitate global communication.
-   Customer Service: Create real-time, AI-powered customer service solutions that use speech recognition to route calls, transcribe conversations, and respond to customer inquiries.
-   Automated Speech-to-Text Systems: Implement automated speech-to-text systems in industries like healthcare, finance, and education, where accurate transcription is critical for compliance, record-keeping, and decision-making.
-   Voice-Controlled Interfaces: Develop voice-controlled interfaces for smart homes, cars, and other devices, where fast and accurate speech recognition is essential for user experience and safety.

We can't wait to see what you build! 🚀
