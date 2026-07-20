"""Shared pytest fixtures and path setup for voice_note tests."""

import sys
import json
import wave
import struct
from types import SimpleNamespace
from pathlib import Path

# Add src/ to sys.path so `from api.base_client import ...` works
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for attribute-style access.

    This simulates the Groq/OpenAI SDK response objects that _parse_response
    consumes via getattr().
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d


def make_wav(path, n_frames=100, rate=44100, channels=1, sampwidth=2):
    """Create a minimal valid WAV file for testing."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(
            struct.pack(
                f"<{n_frames * channels * sampwidth}h",
                *([0] * n_frames * channels * sampwidth),
            )
        )
    return path


import pytest


@pytest.fixture
def groq_verbose_response():
    """Groq verbose_json response with all fields populated.

    Based on MEMORY.md §11 empirical probe (2026-07-18):
    - language: "English" (full word — needs normalisation to "en")
    - all 10 per-segment fields populated
    - x_groq.id extension present
    """
    fixture_path = Path(__file__).parent / "fixtures" / "groq_verbose_response.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return dict_to_namespace(data)


@pytest.fixture
def modelos_verbose_response():
    """modelos verbose_json response with null quality fields + end hallucination.

    Based on MEMORY.md §11 empirical probe (2026-07-18):
    - language: "en" (ISO 639-1 directly)
    - id, seek, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob: ALL null
    - end hallucinated as 30.19 on 2.28s audio
    - words: null, usage: null
    """
    fixture_path = Path(__file__).parent / "fixtures" / "modelos_verbose_response.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return dict_to_namespace(data)
