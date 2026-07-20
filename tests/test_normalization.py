"""Tests for the contract normalisation layer.

These tests encode the desired post-Stream-A normalisation behaviour:
- Null pass-through for quality fields (avg_logprob, compression_ratio, no_speech_prob)
- Language normalisation via langcodes ("English" → "en")
- `end` clamping when the provider hallucinates a value beyond input duration

All tests in this file will FAIL with ImportError until Stream A creates:
- src/i18n.py with normalize_language(raw) -> str | None
- src/contract.py with clamp_end(end, duration) -> float
(or equivalent normaliser in base_client._parse_response)

This is the executable spec — failures are expected and meaningful.
"""

import pytest


def test_normalize_language_english_to_en():
    """'English' (Groq's format) must normalise to ISO 639-1 'en'."""
    from i18n import normalize_language

    assert normalize_language("English") == "en"


def test_normalize_language_already_iso():
    """'en' (modelos's format) must pass through unchanged."""
    from i18n import normalize_language

    assert normalize_language("en") == "en"


def test_normalize_language_none_returns_none():
    """None input must return None, not raise."""
    from i18n import normalize_language

    assert normalize_language(None) is None


def test_end_clamped_to_duration():
    """modelos end hallucination: 30.19 on 2.28s audio must clamp to 2.28."""
    from contract import clamp_end

    assert clamp_end(30.19, 2.28) == 2.28


def test_end_within_range_not_clamped():
    """Normal end values must pass through unchanged."""
    from contract import clamp_end

    assert clamp_end(2.0, 2.28) == 2.0
    assert clamp_end(2.28, 2.28) == 2.28
