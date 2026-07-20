"""Golden-shape tests for JSON and NDJSON contract renderers."""

import io
import json
from pathlib import Path
from uuid import UUID

import pytest

from contract import (
    AudioInput,
    Capabilities,
    Envelope,
    Outputs,
    Result,
    Status,
    Timing,
)
from emitter import HumanEmitter, JSONEmitter, NDJSONEmitter
from cli import _write_envelope_json


GOLDEN_DIR = Path(__file__).parent / "golden"
REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")


def _envelope() -> Envelope:
    return Envelope(
        request_id=REQUEST_ID,
        mode="transcribe",
        status=Status.OK,
        code="OK",
        message="Transcribed 0 segments",
        provider="groq",
        model="whisper-large-v3",
        capabilities=Capabilities(
            word_timestamps=True,
            segment_timestamps=True,
            language_detection=True,
            quality_metrics=True,
            speaker_diarization=False,
        ),
        input=AudioInput(
            audio_file="/tmp/audio.wav",
            duration_seconds=2.28,
            size_bytes=100,
        ),
        result=Result(kind="transcription", text="Okay.", language_detected="en"),
        outputs=Outputs(txt="/tmp/audio_transcription.txt"),
        timing=Timing(transcribe_secs=0.3, total_secs=0.31),
    )


def test_json_emitter_matches_golden_envelope():
    """JSON mode must emit the exact v1.0 envelope shape."""
    output = io.StringIO()
    JSONEmitter(request_id=REQUEST_ID, stdout=output).finalize(_envelope())

    actual = json.loads(output.getvalue())
    expected = json.loads((GOLDEN_DIR / "json-envelope.json").read_text())
    assert actual == expected


def test_ndjson_emitter_matches_golden_event_shape():
    """NDJSON lines must carry stable metadata and typed event payloads."""
    output = io.StringIO()
    emitter = NDJSONEmitter(request_id=REQUEST_ID, stdout=output)
    emitter.start(
        {
            "command": "transcribe",
            "mode": "transcribe",
            "provider": "groq",
            "audio": "/tmp/audio.wav",
            "duration_seconds": 2.28,
        }
    )
    emitter.warning("PROVIDER_FIELD_NULL", "quality fields unavailable", 0)
    emitter.end(
        {
            "status": "ok",
            "code": "OK",
            "mode": "transcribe",
            "segments_total": 0,
            "chars_total": 0,
            "outputs": {},
            "timing": {"record_secs": 0.0, "transcribe_secs": 0.3, "total_secs": 0.31},
            "warnings": [
                warning.model_dump(exclude_none=True) for warning in emitter.warnings
            ],
        }
    )

    actual = [json.loads(line) for line in output.getvalue().splitlines()]
    for event in actual:
        event["event_id"] = "<uuid>"
        event["time"] = "<time>"
    expected = json.loads((GOLDEN_DIR / "ndjson-events.json").read_text())
    assert actual == expected


def test_disk_json_matches_stdout_json(tmp_path):
    """On-disk JSON and JSONEmitter stdout must be byte-identical."""
    envelope = _envelope()
    output = io.StringIO()
    JSONEmitter(request_id=REQUEST_ID, stdout=output).finalize(envelope)
    disk_path = tmp_path / "result.json"

    _write_envelope_json(str(disk_path), envelope)

    assert disk_path.read_bytes() == output.getvalue().encode()
    assert not list(tmp_path.glob("*.tmp.*"))


def test_quiet_human_emitter_preserves_final_stdout():
    """Quiet mode suppresses diagnostics, not the final human result."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    emitter = HumanEmitter(
        request_id=REQUEST_ID,
        stdout=stdout,
        stderr=stderr,
        quiet=True,
    )

    emitter.warning("TEST_WARNING", "hidden warning")
    emitter.finalize(_envelope())

    assert "Okay." in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_events_after_end_are_rejected():
    """The end event is the immutable NDJSON commit point."""
    emitter = NDJSONEmitter(request_id=REQUEST_ID, stdout=io.StringIO())
    emitter.end({"status": "ok", "code": "OK"})

    with pytest.raises(RuntimeError, match="Events after end are forbidden"):
        emitter.warning("TOO_LATE", "must not render")
