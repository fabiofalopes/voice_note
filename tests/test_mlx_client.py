"""MLX client tests — backend selection, response parsing, error handling."""

from unittest.mock import MagicMock, patch

import pytest

from providers.mlx_client import MLXClient, MLX_BACKENDS, DEFAULT_BACKEND


@pytest.fixture
def emitter():
    mock = MagicMock()
    mock.warnings = []
    return mock


@pytest.fixture
def client(emitter):
    return MLXClient(emitter, backend="whisper")


def test_default_backend():
    assert DEFAULT_BACKEND == "whisper"


def test_all_backends_have_required_keys():
    for name, cfg in MLX_BACKENDS.items():
        assert "model" in cfg, f"{name} missing 'model'"
        assert "package" in cfg, f"{name} missing 'package'"


def test_unknown_backend_raises(emitter):
    with pytest.raises(ValueError, match="Unknown MLX backend"):
        MLXClient(emitter, backend="nonexistent")


def test_capabilities():
    assert MLXClient.CAPABILITIES["segment_timestamps"] is True
    assert MLXClient.CAPABILITIES["language_detection"] is True
    assert MLXClient.CAPABILITIES["word_timestamps"] is False


def test_provider_name():
    assert MLXClient.PROVIDER_NAME == "mlx"


def test_parse_mlx_result_dict(client):
    raw = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.5,
                "text": "Hello world",
                "tokens": [1, 2, 3],
                "avg_logprob": -0.25,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.01,
            }
        ],
    }
    result = client._parse_mlx_result(raw)
    assert result.text == "Hello world"
    assert result.detected_language == "en"
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.start == 0.0
    assert seg.end == 1.5
    assert seg.avg_logprob == -0.25
    assert seg.no_speech_prob == 0.01


def test_parse_mlx_result_string(client):
    result = client._parse_mlx_result("Just plain text")
    assert result.text == "Just plain text"
    assert result.segments == []


def test_parse_mlx_result_empty_segments(client):
    raw = {"text": "Hello", "language": "pt", "segments": []}
    result = client._parse_mlx_result(raw)
    assert result.text == "Hello"
    assert result.detected_language == "pt"
    assert result.segments == []


def test_parse_mlx_result_null_fields(client):
    raw = {
        "text": "Test",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Test",
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
            }
        ],
    }
    result = client._parse_mlx_result(raw)
    seg = result.segments[0]
    assert seg.avg_logprob is None
    assert seg.compression_ratio is None
    assert seg.no_speech_prob is None


def test_send_chunk_loads_model_and_calls_transcribe(client, emitter):
    mock_transcribe = MagicMock(
        return_value={"text": "transcribed", "language": "en", "segments": []}
    )
    client._transcribe_fn = mock_transcribe

    result = client._send_chunk(
        "/tmp/fake.opus", verbose=True, language="en", prompt=None
    )
    assert result is not None
    assert result.text == "transcribed"
    mock_transcribe.assert_called_once_with("/tmp/fake.opus", language="en")


def test_send_chunk_inference_failure_returns_none(client, emitter):
    client._transcribe_fn = MagicMock(side_effect=RuntimeError("GPU OOM"))
    result = client._send_chunk(
        "/tmp/fake.opus", verbose=True, language=None, prompt=None
    )
    assert result is None
    assert len(emitter.warnings) == 1 or emitter.warning.called


def test_mlx_registered_in_registry():
    from providers.registry import get_registry, register_builtins

    register_builtins()
    reg = get_registry()
    providers = reg.list_providers()
    assert "mlx" in providers
    assert providers["mlx"].pattern == "custom"
    assert "segment_timestamps" in providers["mlx"].capabilities
