"""Local MLX inference provider for Apple Silicon.

Supports multiple backends:
  - whisper: mlx-community/whisper-large-v3-turbo (widest language coverage)
  - qwen3-asr: Qwen3-ASR 0.6B (best accuracy, 52 languages incl. PT-BR)
  - parakeet: Parakeet TDT 0.6B v3 (fastest, ANE-optimized)

All backends run locally via MLX — no API keys, no network after model download.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from api.base_client import BaseSTTClient, ChunkResult, Segment

MLX_BACKENDS = {
    "whisper": {
        "model": "mlx-community/whisper-large-v3-turbo",
        "package": "mlx_whisper",
    },
    "qwen3-asr": {
        "model": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "package": "mlx_audio",
    },
    "parakeet": {
        "model": "mlx-community/parakeet-tdt-0.6b-v3",
        "package": "mlx_audio",
    },
}

DEFAULT_BACKEND = "whisper"


class MLXClient(BaseSTTClient):
    """Local STT inference on Apple Silicon via MLX."""

    PROVIDER_NAME = "mlx"
    AVAILABLE_MODELS = [
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/Qwen3-ASR-0.6B-8bit",
        "mlx-community/parakeet-tdt-0.6b-v3",
    ]
    CAPABILITIES = {
        "word_timestamps": False,
        "segment_timestamps": True,
        "language_detection": True,
        "quality_metrics": False,
        "speaker_diarization": False,
    }

    def __init__(self, emitter: Any = None, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(emitter)
        if backend not in MLX_BACKENDS:
            raise ValueError(
                f"Unknown MLX backend {backend!r}. Choose from: {list(MLX_BACKENDS)}"
            )
        self._backend = backend
        self._model_repo = MLX_BACKENDS[backend]["model"]
        self._transcribe_fn: Any = None

    def _load_model(self) -> None:
        if self._transcribe_fn is not None:
            return

        pkg_name = MLX_BACKENDS[self._backend]["package"]

        if pkg_name == "mlx_whisper":
            import mlx_whisper

            self._transcribe_fn = lambda path, **kw: mlx_whisper.transcribe(
                path, path_or_hf_repo=self._model_repo, **kw
            )
        elif pkg_name == "mlx_audio":
            try:
                from mlx_audio import transcribe as mlx_audio_transcribe

                self._transcribe_fn = lambda path, **kw: mlx_audio_transcribe(
                    path, model=self._model_repo, **kw
                )
            except ImportError:
                raise ImportError(
                    f"Backend {self._backend!r} requires the mlx-audio package. "
                    f"Install it with: pip install mlx-audio"
                )

    def _send_chunk(
        self,
        audio_path: str,
        verbose: bool,
        language: Optional[str],
        prompt: Optional[str],
    ) -> Optional[ChunkResult]:
        self._load_model()

        kwargs: dict[str, Any] = {}
        if language:
            kwargs["language"] = language

        try:
            raw = self._transcribe_fn(audio_path, **kwargs)
        except Exception as e:
            self.emitter.warning(
                "MLX_INFERENCE_FAILED",
                f"Local MLX inference failed: {e}",
            )
            return None

        return self._parse_mlx_result(raw)

    def _parse_mlx_result(self, raw: Any) -> ChunkResult:
        if isinstance(raw, str):
            return ChunkResult(text=raw.strip())

        text = (
            raw.get("text", "") if isinstance(raw, dict) else getattr(raw, "text", "")
        )
        language = (
            raw.get("language")
            if isinstance(raw, dict)
            else getattr(raw, "language", None)
        )

        segments: list[Segment] = []
        raw_segments = (
            raw.get("segments", [])
            if isinstance(raw, dict)
            else getattr(raw, "segments", [])
        ) or []

        for i, seg in enumerate(raw_segments):
            if isinstance(seg, dict):
                seg_id = seg.get("id", i)
                start = float(seg.get("start", 0.0) or 0.0)
                end = float(seg.get("end", 0.0) or 0.0)
                seg_text = seg.get("text", "") or ""
                tokens = list(seg.get("tokens", []) or [])
                avg_logprob = _to_number(seg.get("avg_logprob"))
                compression_ratio = _to_number(seg.get("compression_ratio"))
                no_speech_prob = _to_number(seg.get("no_speech_prob"))
            else:
                seg_id = getattr(seg, "id", i)
                start = float(getattr(seg, "start", 0.0) or 0.0)
                end = float(getattr(seg, "end", 0.0) or 0.0)
                seg_text = getattr(seg, "text", "") or ""
                tokens = list(getattr(seg, "tokens", []) or [])
                avg_logprob = _to_number(getattr(seg, "avg_logprob", None))
                compression_ratio = _to_number(getattr(seg, "compression_ratio", None))
                no_speech_prob = _to_number(getattr(seg, "no_speech_prob", None))

            segments.append(
                Segment(
                    id=seg_id if isinstance(seg_id, int) else i,
                    start=start,
                    end=end,
                    text=seg_text,
                    tokens=tokens,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=no_speech_prob,
                )
            )

        return ChunkResult(
            text=(text or "").strip(),
            detected_language=language,
            segments=segments,
        )


def _to_number(value: Any) -> Optional[float]:
    return None if value is None else float(value)
