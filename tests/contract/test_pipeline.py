"""Contract-boundary integration tests using a deterministic fake provider."""

import io
import json

from api.base_client import BaseSTTClient, ChunkResult, Segment
from emitter import NDJSONEmitter


class NullFieldClient(BaseSTTClient):
    """Provider fixture with modelos-like nulls and timestamp hallucination."""

    PROVIDER_NAME = "fixture"
    AVAILABLE_MODELS = ["fixture-model"]
    CHUNK_SECONDS = 500

    def _extract_chunk(self, audio_file, start, duration, output_file):
        with open(output_file, "wb") as file:
            file.write(b"audio")

    def _send_chunk(
        self,
        chunk_file,
        model,
        mode,
        chunk_num,
        total_chunks,
        language=None,
        prompt=None,
        verbose=False,
        max_retries=10,
    ):
        return ChunkResult(
            text="Okay.",
            detected_language="English",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=30.19,
                    text="Okay.",
                    avg_logprob=None,
                    compression_ratio=None,
                    no_speech_prob=None,
                )
            ],
        )


class SilentClient(NullFieldClient):
    """Provider fixture whose only segment is confidently silent."""

    def _send_chunk(self, *args, **kwargs):
        return ChunkResult(
            text="noise",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=2.0,
                    text="noise",
                    avg_logprob=-0.1,
                    compression_ratio=1.0,
                    no_speech_prob=0.99,
                )
            ],
        )


def test_pipeline_normalizes_provider_output(monkeypatch, tmp_path):
    """Pipeline must normalize provider data before emitting or persisting it."""
    monkeypatch.setattr("api.base_client.get_audio_duration", lambda _: 2.28)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = io.StringIO()
    emitter = NDJSONEmitter(stdout=output)
    client = NullFieldClient(emitter)

    result = client.transcribe(str(audio), output_file=str(tmp_path / "result"))

    assert result is not None
    assert result.detected_language == "en"
    assert result.segments[0].end == 2.28
    assert result.segments[0].avg_logprob is None
    assert (tmp_path / "result.txt").read_text() == "Okay."
    assert not list(tmp_path.glob("*.tmp.*"))

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    warning_codes = {
        event["data"]["code"] for event in events if event["type"] == "warning"
    }
    assert warning_codes == {"PROVIDER_FIELD_NULL", "TIMESTAMP_CLAMPED"}
    segment = next(event["data"] for event in events if event["type"] == "segment")
    assert segment["end"] == 2.28
    assert segment["avg_logprob"] is None


def test_pipeline_marks_all_chunks_silent(monkeypatch, tmp_path):
    """All-silent audio must produce the aggregate contract warning."""
    monkeypatch.setattr("api.base_client.get_audio_duration", lambda _: 2.0)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    emitter = NDJSONEmitter(stdout=io.StringIO())
    client = SilentClient(emitter)

    result = client.transcribe(str(audio), output_file=str(tmp_path / "result"))

    assert result is None
    assert {warning.code for warning in emitter.warnings} == {
        "SILENT_CHUNK_SKIPPED",
        "ALL_CHUNKS_SILENT",
    }
