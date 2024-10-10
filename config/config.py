from dotenv import load_dotenv
import os
import json
#import litellm

#litellm.set_verbose(True)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the path for the config file
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'audio_config.json')

def load_config():
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)
    return {}

# Load the audio configuration
AUDIO_CONFIG = load_config()