"""Tests for modelos verbose_json response parsing.

Validates that ModelosSTTClient._parse_response() handles the modelos-specific
quirks documented in MEMORY.md §11 and §2.4:
- All quality fields (avg_logprob, compression_ratio, no_speech_prob) are null
- `end` hallucinated as 30.19 on 2.28s audio
- Language returned as ISO 639-1 directly ("en")

The contract (CONTRACT.md §2.3) requires null quality fields to be PRESERVED
as null, not defaulted. The current parser uses float(getattr(seg, "field", 0.0))
which raises TypeError on None — this is the known §2.2 gap that Stream A must fix.
"""

import pytest
from api.modelos_client import ModelosSTTClient
from api.base_client import ChunkResult, Segment


def _make_client():
    """Create a ModelosSTTClient without requiring an API key."""
    return object.__new__(ModelosSTTClient)


def test_modelos_parse_preserves_null_quality_fields(modelos_verbose_response):
    """modelos returns null for avg_logprob, compression_ratio, no_speech_prob.

    CONTRACT.md §2.3: these must be PRESERVED as null ("unknown"), not
    defaulted to 0.0/1.6. This is the known §2.2 gap — the current parser
    uses float(getattr(seg, "field", 0.0)) which raises TypeError on None.
    Stream A must implement a normaliser with null pass-through.
    """
    client = _make_client()

    try:
        result = client._parse_response(modelos_verbose_response, verbose=True)
    except TypeError as e:
        pytest.fail(
            f"Parser crashed on null quality field (known §2.2 gap). "
            f"Stream A normaliser must implement null pass-through. Error: {e}"
        )

    assert isinstance(result, ChunkResult)
    assert len(result.segments) == 1

    seg = result.segments[0]
    assert seg.avg_logprob is None, "avg_logprob must be null (preserved), not 0.0"
    assert seg.compression_ratio is None, "compression_ratio must be null, not 1.6"
    assert seg.no_speech_prob is None, "no_speech_prob must be null, not 0.0"


def test_modelos_end_hallucination(modelos_verbose_response):
    """modelos hallucinates end=30.19 on 2.28s audio (MEMORY.md §11).

    The raw fixture carries the hallucinated value. The contract normaliser
    (Stream A) must clamp end to input.duration_seconds and emit
    TIMESTAMP_CLAMPED warning. This test documents the hallucination in
    the raw fixture; clamping is tested in test_normalization.py.
    """
    client = _make_client()

    try:
        result = client._parse_response(modelos_verbose_response, verbose=True)
    except TypeError:
        pytest.skip(
            "Parser crashes on null fields (§2.2 gap) — end value not reachable yet"
        )

    seg = result.segments[0]
    assert seg.end == 30.19, "Fixture must carry the hallucinated end value"


def test_modelos_language_returns_iso_code(modelos_verbose_response):
    """modelos returns 'en' (ISO 639-1) directly — no normalisation needed."""
    client = _make_client()

    try:
        result = client._parse_response(modelos_verbose_response, verbose=True)
    except TypeError:
        pytest.skip(
            "Parser crashes on null fields (§2.2 gap) — language not reachable yet"
        )

    assert result.detected_language == "en"
