"""
Groq Whisper API Client

Smart chunking client for Groq's Whisper transcription API.
Handles large audio files by intelligently splitting them into chunks.
"""

import os
import tempfile
import subprocess
from typing import Optional, List
from groq import Groq
from dotenv import load_dotenv
from audio_processing.utils import get_audio_duration

load_dotenv()


class GroqWhisperClient:
    """Smart chunking Groq Whisper API client"""
    
    AVAILABLE_MODELS = [
        "whisper-large-v3",
        "whisper-large-v3-turbo", 
        "distil-whisper-large-v3-en"
    ]
    
    # Groq API limits: 25MB file size limit (free tier)
    MAX_FILE_SIZE_MB = 20  # Conservative limit for free tier
    MAX_CHUNK_SIZE_MB = 20  # Each chunk must be under 25MB
    OVERLAP_SECONDS = 2  # Small overlap for continuity
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required. "
                "Get your free API key at https://console.groq.com/"
            )
        self.client = Groq(api_key=api_key)
    
    def transcribe(self, audio_file: str, model: str = "whisper-large-v3") -> Optional[str]:
        """
        Transcribe audio file using Groq Whisper API with smart chunking
        
        Args:
            audio_file: Path to audio file
            model: Whisper model to use
            
        Returns:
            Transcribed text or None if failed
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model} not available. Choose from: {self.AVAILABLE_MODELS}")
        
        try:
            # Check if file needs chunking
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            duration = get_audio_duration(audio_file)
            
            print(f"🔍 Audio file: {file_size_mb:.1f}MB, {duration:.1f}s")
            
            if file_size_mb <= self.MAX_FILE_SIZE_MB:
                # Small file - transcribe directly
                print("📝 Transcribing directly (fits in free tier limit)")
                return self._transcribe_single(audio_file, model)
            else:
                # Large file - use chunking
                print(f"🔄 Large file detected ({file_size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB) - using smart chunking")
                return self._transcribe_chunked(audio_file, model, duration)
                
        except FileNotFoundError:
            print(f"❌ Audio file not found: {audio_file}")
            return None
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def _transcribe_single(self, audio_file: str, model: str) -> Optional[str]:
        """Transcribe a single audio file"""
        try:
            with open(audio_file, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file), file.read()),
                    model=model,
                    response_format="text"
                )
            return transcription.strip()
        except Exception as e:
            print(f"❌ Single file transcription error: {e}")
            return None
    
    def _transcribe_chunked(self, audio_file: str, model: str, duration: float) -> Optional[str]:
        """Transcribe large audio file by splitting into chunks"""
        try:
            # Calculate optimal chunk duration based on file size
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            chunk_duration = self._calculate_optimal_chunk_duration(file_size_mb, duration)
            
            chunks = self._split_audio_into_chunks(audio_file, duration, chunk_duration)
            
            if not chunks:
                print("❌ Failed to create audio chunks")
                return None
            
            print(f"🔄 Processing {len(chunks)} chunks for transcription...")
            transcriptions = []
            
            for i, chunk_file in enumerate(chunks, 1):
                print(f"📝 Transcribing chunk {i}/{len(chunks)} ({i/len(chunks)*100:.0f}%)...")
                chunk_text = self._transcribe_single(chunk_file, model)
                if chunk_text:
                    transcriptions.append(chunk_text)
                    print(f"✅ Chunk {i} transcribed: {len(chunk_text)} characters")
                else:
                    print(f"⚠️ Warning: Chunk {i} failed to transcribe")
            
            # Clean up temporary chunk files
            for chunk_file in chunks:
                try:
                    os.unlink(chunk_file)
                except:
                    pass
            
            if not transcriptions:
                print("❌ No chunks were successfully transcribed")
                return None
            
            # Join transcriptions with proper spacing
            full_text = " ".join(transcriptions)
            print(f"✅ Successfully transcribed {len(chunks)} chunks")
            return full_text
            
        except Exception as e:
            print(f"❌ Chunked transcription error: {e}")
            return None
    
    def _calculate_optimal_chunk_duration(self, file_size_mb: float, total_duration: float) -> float:
        """Calculate optimal chunk duration to stay under 25MB limit"""
        # Calculate MB per second
        mb_per_second = file_size_mb / total_duration
        
        # Calculate max duration for 20MB chunks (conservative)
        max_duration_for_size = self.MAX_CHUNK_SIZE_MB / mb_per_second
        
        # Use a reasonable minimum (30 seconds) and maximum (10 minutes)
        chunk_duration = max(30, min(max_duration_for_size * 0.9, 600))  # 90% of calculated max for safety
        
        print(f"📊 Calculated chunk duration: {chunk_duration:.1f}s (based on {mb_per_second:.3f}MB/s)")
        return chunk_duration
    
    def _split_audio_into_chunks(self, audio_file: str, duration: float, chunk_duration: float) -> List[str]:
        """Split audio file into chunks using ffmpeg with size-aware chunking"""
        chunks = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Calculate number of chunks
            num_chunks = max(1, int((duration + chunk_duration - 1) // chunk_duration))
            print(f"🔪 Splitting into {num_chunks} chunks of ~{chunk_duration:.1f}s each...")
            
            chunk_start = 0
            chunk_num = 0
            
            while chunk_start < duration:
                chunk_num += 1
                chunk_end = min(chunk_start + chunk_duration, duration)
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_num:03d}.wav")
                
                print(f"🔪 Creating chunk {chunk_num}/{num_chunks} ({chunk_start:.1f}s-{chunk_end:.1f}s)...")
                
                # Use ffmpeg with Groq's recommended settings
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",  # Suppress verbose output
                    "-i", audio_file,
                    "-ss", str(chunk_start),
                    "-t", str(chunk_end - chunk_start),
                    "-ar", "16000",  # Groq downsamples to 16kHz anyway
                    "-ac", "1",      # Mono (Groq converts to mono)
                    "-c:a", "pcm_s16le",  # WAV format for lower latency
                    chunk_file
                ]
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                    chunk_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
                    
                    # Verify chunk is under size limit
                    if chunk_size_mb > self.MAX_CHUNK_SIZE_MB:
                        print(f"⚠️ Warning: Chunk {chunk_num} is {chunk_size_mb:.1f}MB (over {self.MAX_CHUNK_SIZE_MB}MB limit)")
                        # Still add it, but warn user
                    
                    chunks.append(chunk_file)
                    print(f"✅ Chunk {chunk_num} created: {chunk_size_mb:.1f}MB")
                else:
                    print(f"⚠️ Warning: Chunk {chunk_num} is empty or failed")
                
                # Move to next chunk with small overlap for continuity
                chunk_start = chunk_end - self.OVERLAP_SECONDS
                
                # Prevent infinite loop at the end
                if chunk_end >= duration:
                    break
            
            print(f"✅ Successfully created {len(chunks)} audio chunks")
            return chunks
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            print(f"❌ Error splitting audio: {error_msg}")
            return []
        except Exception as e:
            print(f"❌ Error creating chunks: {e}")
            return []
    
    def translate(self, audio_file: str, model: str = "whisper-large-v3") -> Optional[str]:
        """
        Translate audio file to English using Groq Whisper API with smart chunking
        
        Args:
            audio_file: Path to audio file
            model: Whisper model to use
            
        Returns:
            Translated text or None if failed
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model} not available. Choose from: {self.AVAILABLE_MODELS}")
        
        # Note: whisper-large-v3-turbo doesn't support translation
        if model == "whisper-large-v3-turbo":
            print("⚠️ whisper-large-v3-turbo doesn't support translation, switching to whisper-large-v3")
            model = "whisper-large-v3"
        
        try:
            # Check if file needs chunking
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            duration = get_audio_duration(audio_file)
            
            print(f"🔍 Audio file: {file_size_mb:.1f}MB, {duration:.1f}s")
            
            if file_size_mb <= self.MAX_FILE_SIZE_MB:
                # Small file - translate directly
                print("🌍 Translating directly (fits in free tier limit)")
                return self._translate_single(audio_file, model)
            else:
                # Large file - use chunking
                print(f"🔄 Large file detected ({file_size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB) - using smart chunking for translation")
                return self._translate_chunked(audio_file, model, duration)
                
        except FileNotFoundError:
            print(f"❌ Audio file not found: {audio_file}")
            return None
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return None
    
    def _translate_single(self, audio_file: str, model: str) -> Optional[str]:
        """Translate a single audio file"""
        try:
            with open(audio_file, "rb") as file:
                translation = self.client.audio.translations.create(
                    file=(os.path.basename(audio_file), file.read()),
                    model=model,
                    response_format="text"
                )
            return translation.strip()
        except Exception as e:
            print(f"❌ Single file translation error: {e}")
            return None
    
    def _translate_chunked(self, audio_file: str, model: str, duration: float) -> Optional[str]:
        """Translate large audio file by splitting into chunks"""
        try:
            # Calculate optimal chunk duration based on file size
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            chunk_duration = self._calculate_optimal_chunk_duration(file_size_mb, duration)
            
            chunks = self._split_audio_into_chunks(audio_file, duration, chunk_duration)
            
            if not chunks:
                print("❌ Failed to create audio chunks")
                return None
            
            print(f"🔄 Processing {len(chunks)} chunks for translation...")
            translations = []
            
            for i, chunk_file in enumerate(chunks, 1):
                print(f"🌍 Translating chunk {i}/{len(chunks)} ({i/len(chunks)*100:.0f}%)...")
                chunk_text = self._translate_single(chunk_file, model)
                if chunk_text:
                    translations.append(chunk_text)
                    print(f"✅ Chunk {i} translated: {len(chunk_text)} characters")
                else:
                    print(f"⚠️ Warning: Chunk {i} failed to translate")
            
            # Clean up temporary chunk files
            for chunk_file in chunks:
                try:
                    os.unlink(chunk_file)
                except:
                    pass
            
            if not translations:
                print("❌ No chunks were successfully translated")
                return None
            
            # Join translations with proper spacing
            full_text = " ".join(translations)
            print(f"✅ Successfully translated {len(chunks)} chunks")
            return full_text
            
        except Exception as e:
            print(f"❌ Chunked translation error: {e}")
            return None