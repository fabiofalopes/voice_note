import os
import sys
import pyaudio
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import CONFIG_FILE_PATH

def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    devices = []
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if (device_info.get('maxInputChannels')) > 0:
            # Filter out non-physical devices
            if 'hw:' in device_info.get('name') or 'USB' in device_info.get('name'):
                devices.append({
                    'index': i,
                    'name': device_info.get('name')
                })
    
    p.terminate()
    return devices

def setup_audio_device():
    devices = list_input_devices()
    
    if not devices:
        print("No physical audio input devices found.")
        return

    print("Available input devices:")
    for i, device in enumerate(devices, 1):
        print(f"{i}: {device['name']}")
    
    while True:
        try:
            choice = int(input("Enter the number of the device you want to use: "))
            if 1 <= choice <= len(devices):
                selected_device = devices[choice - 1]
                save_config({'input_device_index': selected_device['index']})
                print(f"Selected device: {selected_device['name']}")
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def save_config(config):
    with open(CONFIG_FILE_PATH, 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    setup_audio_device()
