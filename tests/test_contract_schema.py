"""Tests for the future Pydantic v2 contract models in src/contract.py.

These tests define the expected contract schema shape. They will FAIL
with ImportError until Stream A creates src/contract.py with:
- Envelope (top-level JSON document, extra="forbid")
- Segment (per-segment data, null-preserving for quality fields)
- Capabilities (provider capability declaration)

This is the executable spec — failures are expected and meaningful.
"""

import pytest


def test_segment_model_preserves_null_quality_fields():
    """CONTRACT.md §2.3: avg_logprob, compression_ratio, no_speech_prob must accept None.

    null = "unknown," not a fabricated default. This is the core normalisation
    rule that the modelos fixture exercises.
    """
    from contract import Segment

    seg = Segment(
        start=0.0,
        end=2.28,
        text="test",
        avg_logprob=None,
        compression_ratio=None,
        no_speech_prob=None,
    )
    assert seg.avg_logprob is None
    assert seg.compression_ratio is None
    assert seg.no_speech_prob is None


def test_envelope_model_has_required_fields():
    """CONTRACT.md §2: Envelope must carry all mandatory top-level fields."""
    from contract import Envelope

    fields = set(Envelope.model_fields.keys())
    required = {
        "schema_version",
        "tool_version",
        "request_id",
        "mode",
        "status",
        "code",
        "message",
        "provider",
        "model",
        "capabilities",
        "input",
        "outputs",
        "warnings",
        "timing",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"


def test_capabilities_model_has_all_declaration_fields():
    """CONTRACT.md §3: Capabilities must declare all 5 capability booleans."""
    from contract import Capabilities

    fields = set(Capabilities.model_fields.keys())
    required = {
        "word_timestamps",
        "segment_timestamps",
        "language_detection",
        "quality_metrics",
        "speaker_diarization",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"
