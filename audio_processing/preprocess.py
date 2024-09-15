import ffmpeg
import os

def preprocess_audio(input_path: str, output_path: str) -> bool:
    """
    Preprocess the audio file to downsample to 16,000 Hz mono.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"Error preprocessing audio: {e}")
        return False