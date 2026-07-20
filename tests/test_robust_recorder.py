"""Tests for robust_recorder fault-recovery and chunk-merge behaviour.

These tests exercise pure-logic paths that don't require real audio hardware:
- merge_chunks() WAV concatenation
- merge_chunks() empty-list guard
- _signal_handler() sets recording=False for graceful shutdown
"""

import signal
import struct
import wave

from audio_processing.robust_recorder import RobustAudioRecorder, merge_chunks


def _make_wav(path, n_frames=100, rate=44100, channels=1, sampwidth=2):
    total_samples = n_frames * channels
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{total_samples}h", *([0] * total_samples)))
    return path


def test_merge_chunks_concatenates_wav_files(tmp_path):
    """merge_chunks must produce a WAV with all frames from input chunks."""
    chunk1 = _make_wav(tmp_path / "chunk1.wav", n_frames=100)
    chunk2 = _make_wav(tmp_path / "chunk2.wav", n_frames=200)
    output = tmp_path / "merged.wav"

    result = merge_chunks([chunk1, chunk2], output)

    assert result is True
    import wave

    with wave.open(str(output), "rb") as wf:
        assert wf.getnframes() == 300


def test_merge_chunks_empty_list_returns_false(tmp_path):
    """merge_chunks must return False for empty input, not crash."""
    output = tmp_path / "merged.wav"
    result = merge_chunks([], output)
    assert result is False


def test_signal_handler_sets_recording_false():
    """SIGINT/SIGTERM handler must set recording=False for graceful shutdown."""
    recorder = object.__new__(RobustAudioRecorder)
    recorder.recording = True

    recorder._signal_handler(signal.SIGINT, None)

    assert recorder.recording is False
