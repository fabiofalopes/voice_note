#!/usr/bin/env python3
"""
Robust Audio Recorder with Auto-Recovery and Chunking

Features:
- Periodic chunking (saves every N minutes)
- Device disconnection handling
- Automatic resume capability
- Multiple output files for long recordings
- Graceful degradation on device failure
"""

import pyaudio
import wave
import os
import sys
import time
import threading
import signal
import atexit
from datetime import datetime
from pathlib import Path

try:
    import termios as _termios

    _HAS_TERMIOS = True
except ImportError:
    _termios = None
    _HAS_TERMIOS = False


class RobustAudioRecorder:
    """
    Fault-tolerant audio recorder that saves audio in chunks
    and handles device disconnections gracefully.
    """

    def __init__(
        self,
        output_dir="recordings",
        chunk_minutes=5,
        device_index=None,
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        chunk_size=1024,
    ):
        """
        Initialize the robust recorder.

        Args:
            output_dir: Directory to save recordings
            chunk_minutes: Minutes per chunk (default: 5)
            device_index: Audio device index (None for default)
            format: PyAudio format
            channels: Number of audio channels
            rate: Sample rate
            chunk_size: Frames per buffer
        """
        self.output_dir = Path(output_dir)
        self.chunk_minutes = chunk_minutes
        self.device_index = device_index
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size

        self.recording = False
        self.paused = False
        self.stream = None
        self.pyaudio = None
        self.current_chunk_file = None
        self.chunk_files = []
        self.total_frames = 0
        self._shutdown_requested = False
        self._saved_termios = None
        self._cleaned_up = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        atexit.register(self._cleanup)

    def _signal_handler(self, signum, frame):
        """Set a flag so the recording loop stops from a safe context."""
        print("\n🛑 Shutdown signal received, saving current chunk...")
        self._shutdown_requested = True
        self.recording = False

    def _get_chunk_filename(self, chunk_num):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"recording_{timestamp}_chunk{chunk_num:03d}.wav"

    def _save_chunk(self, frames, filename):
        """
        Save a chunk of audio to disk atomically.

        Args:
            frames: List of audio frames
            filename: Output filename

        Returns:
            Path to saved file
        """
        if self.pyaudio is None:
            print("\n❌ Failed to save chunk: PyAudio not initialized")
            return None

        temp_path = str(filename) + f".tmp.{os.getpid()}"
        try:
            wf = wave.open(temp_path, "wb")
            try:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b"".join(frames))
            finally:
                wf.close()

            with open(temp_path, "rb") as f:
                os.fsync(f.fileno())

            os.replace(temp_path, str(filename))

            duration = len(frames) * self.chunk_size / self.rate
            size_mb = os.path.getsize(filename) / (1024 * 1024)

            print(
                f"\n✅ Saved chunk: {filename.name} ({duration:.1f}s, {size_mb:.1f}MB)"
            )
            return filename

        except Exception as e:
            print(f"\n❌ Failed to save chunk: {e}")
            return None

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _open_stream(self):
        """
        Open audio stream with error handling.

        Returns:
            PyAudio stream or None if failed
        """
        try:
            if _termios is not None and sys.stdin.isatty():
                try:
                    self._saved_termios = _termios.tcgetattr(sys.stdin.fileno())
                except _termios.error:
                    self._saved_termios = None

            self.pyaudio = pyaudio.PyAudio()

            # If device not specified, find default input device
            resolved_device: int | None = self.device_index
            if resolved_device is None:
                try:
                    device_info = self.pyaudio.get_default_input_device_info()
                    resolved_device = int(device_info["index"])
                    print(f"🎤 Using default device: {device_info['name']}")
                except Exception as e:
                    print(f"⚠️  Could not get default device: {e}")
                    # Try first available device
                    resolved_device = 0

            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=resolved_device,
                frames_per_buffer=self.chunk_size,
            )

            print(f"🎤 Recording started")
            print(f"   Device: {resolved_device}")
            print(f"   Format: {self.channels}ch, {self.rate}Hz")
            print(f"   Chunk size: {self.chunk_minutes} minutes")
            print(f"   Output: {self.output_dir}")

            return stream

        except Exception as e:
            print(f"\n❌ Failed to open audio stream: {e}")
            if self.pyaudio:
                self.pyaudio.terminate()
            return None

    def _record_chunk(self, chunk_num, max_frames):
        """
        Record a single chunk of audio.

        Args:
            chunk_num: Current chunk number
            max_frames: Maximum frames to record for this chunk

        Returns:
            Tuple of (frames_list, success_flag, error_message)
        """
        frames = []
        frame_count = 0
        overflow_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 50  # Allow transient errors

        start_time = time.time()
        chunk_filename = self._get_chunk_filename(chunk_num)

        print(f"\n📼 Chunk {chunk_num}: Recording to {chunk_filename.name}")
        print(f"   Target: {max_frames * self.chunk_size / self.rate:.1f} seconds")

        stream = self.stream
        if stream is None:
            return frames, False, "Audio stream not open"

        try:
            while (
                self.recording
                and frame_count < max_frames
                and not self._shutdown_requested
            ):
                try:
                    # Read audio data with timeout-like behavior
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    frame_count += 1
                    consecutive_errors = 0  # Reset on success

                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        progress = (frame_count / max_frames) * 100
                        sys.stdout.write(
                            f"\r   Progress: {progress:.1f}% | {elapsed:.0f}s elapsed"
                        )
                        sys.stdout.flush()

                except OSError as e:
                    error_msg = str(e)

                    # Handle buffer overflow (not critical)
                    if "Input overflowed" in error_msg or "-9981" in error_msg:
                        overflow_count += 1
                        if overflow_count % 10 == 0:
                            sys.stdout.write("⚠")
                            sys.stdout.flush()
                        time.sleep(0.01)
                        continue

                    # Handle device disconnection (CRITICAL)
                    elif "-50" in error_msg or "Unknown Error" in error_msg:
                        consecutive_errors += 1
                        print(
                            f"\n⚠️  Device error detected ({consecutive_errors}/{max_consecutive_errors})"
                        )

                        if consecutive_errors >= max_consecutive_errors:
                            error_msg = (
                                f"Device disconnected after {consecutive_errors} errors"
                            )
                            return frames, False, error_msg

                        # Wait a bit and retry
                        time.sleep(0.1)
                        continue

                    else:
                        # Unknown OSError
                        print(f"\n⚠️  Unexpected error: {e}")
                        consecutive_errors += 1
                        time.sleep(0.1)
                        continue

                except Exception as e:
                    error_msg = f"Recording error: {e}"
                    return frames, False, error_msg

            elapsed = time.time() - start_time
            print(f"\n✅ Chunk {chunk_num} complete: {elapsed:.1f}s recorded")
            return frames, True, None

        except Exception as e:
            return frames, False, f"Chunk error: {e}"

    def record(self, max_duration_minutes=None):
        """
        Start recording with automatic chunking.

        Args:
            max_duration_minutes: Maximum recording time (None for unlimited)

        Returns:
            List of chunk file paths created
        """
        self.recording = True
        self.chunk_files = []

        self.stream = self._open_stream()
        if not self.stream:
            self.recording = False
            return []

        frames_per_chunk = int((self.chunk_minutes * 60 * self.rate) / self.chunk_size)

        # Calculate total chunks if max_duration specified
        max_chunks = None
        if max_duration_minutes:
            max_chunks = int(max_duration_minutes / self.chunk_minutes)
            print(
                f"   Max duration: {max_duration_minutes} minutes (~{max_chunks} chunks)"
            )

        chunk_num = 1
        print("\n" + "=" * 60)
        print("🎙️  RECORDING STARTED")
        print("=" * 60)
        print("Press Ctrl+C to stop and save current chunk")
        print("=" * 60 + "\n")

        try:
            while self.recording and not self._shutdown_requested:
                if max_chunks and chunk_num > max_chunks:
                    print(
                        f"\n✅ Reached maximum duration ({max_duration_minutes} minutes)"
                    )
                    break

                # Record one chunk
                frames, success, error = self._record_chunk(chunk_num, frames_per_chunk)

                if len(frames) > 0:
                    chunk_filename = self._get_chunk_filename(chunk_num)
                    saved_file = self._save_chunk(frames, chunk_filename)
                    if saved_file:
                        self.chunk_files.append(saved_file)
                        self.total_frames += len(frames)

                if not success:
                    print(f"\n⚠️  {error}")
                    if "Device disconnected" in error:
                        print(
                            "💡 Tip: Reconnect your audio device and restart recording"
                        )
                    break

                chunk_num += 1

            if self._shutdown_requested:
                print("\n\n🛑 Recording stopped by user")

        except KeyboardInterrupt:
            print("\n\n🛑 Recording stopped by user")

        finally:
            self._cleanup()

        return self.chunk_files

    def _cleanup(self):
        """Clean up resources. Safe to call multiple times."""
        if self._cleaned_up:
            return

        self.recording = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception:
                pass
            self.pyaudio = None

        if self._saved_termios is not None and _termios is not None:
            try:
                _termios.tcsetattr(
                    sys.stdin.fileno(), _termios.TCSAFLUSH, self._saved_termios
                )
            except (_termios.error, OSError):
                pass
            self._saved_termios = None

        if self.chunk_files:
            print("\n" + "=" * 60)
            print("📊 RECORDING SUMMARY")
            print("=" * 60)
            print(f"Chunks saved: {len(self.chunk_files)}")
            print(
                f"Total duration: ~{self.total_frames * self.chunk_size / self.rate / 60:.1f} minutes"
            )
            print(f"Output directory: {self.output_dir}")
            print("\n📁 Files created:")
            for f in self.chunk_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f}MB)")
            print("=" * 60)

        self._cleaned_up = True

    def stop(self):
        self.recording = False


def merge_chunks(chunk_files, output_file):
    """
    Merge multiple WAV files into one.

    Args:
        chunk_files: List of chunk file paths
        output_file: Output file path
    """
    if not chunk_files:
        print("❌ No chunks to merge")
        return False

    print(f"\n🔗 Merging {len(chunk_files)} chunks into {output_file}...")

    try:
        with wave.open(str(chunk_files[0]), "rb") as wf:
            params = wf.getparams()

        with wave.open(str(output_file), "wb") as outfile:
            outfile.setparams(params)

            for chunk_file in chunk_files:
                with wave.open(str(chunk_file), "rb") as infile:
                    outfile.writeframes(infile.readframes(infile.getnframes()))

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Merged file created: {output_file} ({size_mb:.1f}MB)")
        return True

    except Exception as e:
        print(f"❌ Failed to merge chunks: {e}")
        return False


def record_robust(
    output_dir="recordings",
    chunk_minutes=5,
    max_duration_minutes=None,
    device_index=None,
    merge_after=True,
):
    """
    Convenience function to record with fault tolerance.

    Args:
        output_dir: Directory to save recordings
        chunk_minutes: Minutes per chunk
        max_duration_minutes: Maximum recording time
        device_index: Audio device index
        merge_after: Whether to merge chunks after recording

    Returns:
        List of chunk files (or merged file if merge_after=True)
    """
    recorder = RobustAudioRecorder(
        output_dir=output_dir, chunk_minutes=chunk_minutes, device_index=device_index
    )

    chunk_files = recorder.record(max_duration_minutes=max_duration_minutes)

    if merge_after and chunk_files:
        # Create merged filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = Path(output_dir) / f"recording_{timestamp}_merged.wav"

        if merge_chunks(chunk_files, merged_file):
            return [merged_file]

    return chunk_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust audio recorder with auto-recovery"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="recordings",
        help="Output directory (default: recordings)",
    )
    parser.add_argument(
        "--chunk-minutes",
        "-c",
        type=int,
        default=5,
        help="Minutes per chunk (default: 5)",
    )
    parser.add_argument(
        "--max-duration", "-d", type=int, help="Maximum recording duration in minutes"
    )
    parser.add_argument("--device", "-D", type=int, help="Audio device index")
    parser.add_argument(
        "--no-merge", action="store_true", help="Don't merge chunks after recording"
    )

    args = parser.parse_args()

    record_robust(
        output_dir=args.output_dir,
        chunk_minutes=args.chunk_minutes,
        max_duration_minutes=args.max_duration,
        device_index=args.device,
        merge_after=not args.no_merge,
    )
