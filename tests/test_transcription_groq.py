import os
from groq import Groq

def transcribe():
    # Initialize the Groq client
    client = Groq()

    # Specify the path to the audio file
    filename = "UsingOllamatoRunLocalLLMsontheRaspberryPi5.mp3"  # Replace with your audio file path

    # Ensure the file exists
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return

    # Open the audio file
    with open(filename, "rb") as file:
        # Create a transcription of the audio file
        try:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(filename), file.read()),  # Required audio file
                model="distil-whisper-large-v3-en",  # Required model to use for transcription
                prompt="Video by Ian Wootten: Using Ollama to Run Local LLMs on the Raspberry Pi 5 – My favorite local LLM tool, Ollama, is simple to set up and works on a Raspberry Pi 5. I check it out and compare it to some benchmarks from more powerful machines. 00:00 Introduction, 00:41 Installation, 02:12 Model Runs, 09:01 Conclusion.",  # Optional
                response_format="json",  # Optional
                language="en",  # Optional
                temperature=0.0  # Optional
            )
            # Print the transcription text
            print(transcription.text)
        except Exception as e:
            print(f"An error occurred during transcription: {e}")

if __name__ == "__main__":
    transcribe()
