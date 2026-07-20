"""Black-box contract tests for validation failures at the CLI boundary."""

import json
import subprocess
import sys
import wave
from pathlib import Path

import cli
from api.base_client import Segment, TranscriptionResult


ROOT = Path(__file__).parents[2]
ENTRYPOINT = ROOT / "transcribe.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_file_ndjson_exits_66_with_commit_event():
    """Missing input emits pure NDJSON ending in FILE_NOT_FOUND."""
    completed = _run("/definitely/missing.wav", "--ndjson")
    events = [json.loads(line) for line in completed.stdout.splitlines()]

    assert completed.returncode == 66
    assert [event["type"] for event in events] == ["error", "end"]
    assert events[-1]["data"]["code"] == "FILE_NOT_FOUND"
    assert completed.stderr == ""


def test_modelos_capability_check_precedes_auth(tmp_path):
    """Unsupported word timestamps fail before client construction or API use."""
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0\0" * 160)

    completed = _run(
        str(audio),
        "--provider",
        "modelos",
        "--word-timestamps",
        "--ndjson",
    )
    events = [json.loads(line) for line in completed.stdout.splitlines()]

    assert completed.returncode == 64
    assert events[-1]["data"]["code"] == "CAPABILITY_UNSUPPORTED"
    assert "does not support word timestamps" in events[0]["data"]["message"]


class SuccessfulClient:
    """CLI fixture returning one deterministic successful segment."""

    CAPABILITIES = {
        "word_timestamps": True,
        "segment_timestamps": True,
        "language_detection": True,
        "quality_metrics": True,
        "speaker_diarization": False,
    }

    def __init__(self, emitter):
        self.emitter = emitter

    def _resolve_model(self, model, for_translation=False):
        if model not in (None, "fixture-model"):
            raise ValueError(f"Unknown model '{model}'")
        return model or "fixture-model"

    def transcribe(self, audio_file, **kwargs):
        segment = Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Okay.",
            avg_logprob=-0.1,
            compression_ratio=1.0,
            no_speech_prob=0.0,
        )
        self.emitter.segment(
            {
                "chunk_index": 0,
                "offset_seconds": 0.0,
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Okay.",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
            }
        )
        return TranscriptionResult(
            text="Okay.", segments=[segment], detected_language="en"
        )


class InterruptingClient(SuccessfulClient):
    """CLI fixture interrupted after one emitted segment."""

    def transcribe(self, audio_file, **kwargs):
        self.emitter.segment(
            {
                "chunk_index": 0,
                "offset_seconds": 0.0,
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "partial",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
            }
        )
        raise KeyboardInterrupt


def _patch_cli(monkeypatch, client_type, audio):
    monkeypatch.setattr(
        cli, "_build_client", lambda provider, emitter: client_type(emitter)
    )
    monkeypatch.setattr(
        cli, "_provider_capabilities", lambda provider: client_type.CAPABILITIES
    )
    monkeypatch.setattr("audio_processing.utils.get_audio_duration", lambda _: 1.0)
    monkeypatch.setattr(
        sys,
        "argv",
        ["transcribe.py", str(audio), "--no-clipboard"],
    )


def test_json_success_envelope(monkeypatch, tmp_path, capsys):
    """Successful JSON mode exposes status, code, mode, and result."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    _patch_cli(monkeypatch, SuccessfulClient, audio)
    sys.argv.append("--json")

    return_code = cli.main()
    document = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert document["status"] == "ok"
    assert document["code"] == "OK"
    assert document["mode"] == "transcribe"
    assert document["result"]["text"] == "Okay."


def test_interrupt_after_segment_is_partial(monkeypatch, tmp_path, capsys):
    """Ctrl+C after emitted data exits 130 with partial end status."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    _patch_cli(monkeypatch, InterruptingClient, audio)
    sys.argv.append("--ndjson")

    return_code = cli.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert return_code == 130
    assert events[-1]["data"]["code"] == "USER_INTERRUPT"
    assert events[-1]["data"]["status"] == "partial"


def test_invalid_model_is_structured_usage_error(monkeypatch, tmp_path, capsys):
    """Invalid model input must not escape as a Python traceback."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    _patch_cli(monkeypatch, SuccessfulClient, audio)
    sys.argv.extend(["--model", "missing-model", "--ndjson"])

    return_code = cli.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert return_code == 64
    assert events[-1]["data"]["code"] == "MODEL_NOT_FOUND"
    assert events[0]["type"] == "error"
