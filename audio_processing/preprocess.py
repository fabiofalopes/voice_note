import ffmpeg
import os
import shutil

def preprocess_audio(input_path, output_path=None):
    """
    Preprocess audio file for better transcription quality.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the preprocessed audio (optional, defaults to modified input path)
        
    Returns:
        Path to the preprocessed audio file
    """
    # If no output path is provided, create one based on the input file
    if output_path is None:
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        filename, ext = os.path.splitext(basename)
        output_path = os.path.join(dirname, f"{filename}_preprocessed{ext}")
    
    # For now, a simple copy of the file (placeholder for actual preprocessing)
    # In a real implementation, you would add noise reduction, normalization, etc.
    try:
        shutil.copy2(input_path, output_path)
        return output_path
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # Return the original file if preprocessing fails
        return input_path