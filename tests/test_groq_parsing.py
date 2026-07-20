"""Tests for Groq verbose_json response parsing.

Validates that GroqWhisperClient._parse_response() correctly extracts
all fields from a representative Groq verbose_json response.

Empirical basis: MEMORY.md §11 — Groq returns all 10 per-segment fields
populated, language as full word "English", x_groq.id extension present.
"""

import pytest
from api.groq_client import GroqWhisperClient
from api.base_client import ChunkResult, Segment


def _make_client():
    """Create a GroqWhisperClient without requiring an API key."""
    return object.__new__(GroqWhisperClient)


def test_groq_parse_all_fields_populated(groq_verbose_response):
    """All 10 per-segment fields must populate cleanly from Groq's verbose_json."""
    client = _make_client()
    result = client._parse_response(groq_verbose_response, verbose=True)

    assert isinstance(result, ChunkResult)
    assert result.text == "Okay. This is a test transcription."
    assert result.detected_language == "English"
    assert result.duration == 2.27
    assert len(result.segments) == 2

    seg = result.segments[0]
    assert isinstance(seg, Segment)
    assert seg.id == 0
    assert seg.start == 0.0
    assert seg.end == 2.06
    assert seg.text == "Okay."
    assert seg.avg_logprob == -0.42
    assert seg.no_speech_prob == 0.03
    assert seg.compression_ratio == 1.6
    assert seg.tokens == [50364, 543]


def test_groq_language_returns_full_word(groq_verbose_response):
    """Groq returns 'English' (full word), not 'en' — normaliser must handle this."""
    client = _make_client()
    result = client._parse_response(groq_verbose_response, verbose=True)

    assert result.detected_language == "English"
    assert result.provider_meta == {"x_groq": {"id": "req_abc123"}}


def test_groq_text_mode_parse():
    """When verbose=False, raw is a plain string — _parse_response returns text only."""
    client = _make_client()
    result = client._parse_response("Plain text transcription result.", verbose=False)

    assert isinstance(result, ChunkResult)
    assert result.text == "Plain text transcription result."
    assert result.segments == []
    assert result.detected_language is None
    assert result.duration is None
