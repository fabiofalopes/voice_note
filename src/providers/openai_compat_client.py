"""Dynamic OpenAI-compatible STT client builder (Pattern 2).

Creates a BaseSTTClient subclass from config alone — no provider-specific
code needed. Used by the registry for register_openai_compat() entries.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from api.base_client import BaseSTTClient, ChunkResult, Segment


def make_openai_compat_client(
    *,
    provider_name: str,
    base_url: str,
    default_model: str,
    api_key_env: str,
    capabilities: dict[str, bool],
) -> type[BaseSTTClient]:
    """Build a BaseSTTClient subclass configured for an OpenAI-compatible endpoint."""

    class _OpenAICompatClient(BaseSTTClient):
        PROVIDER_NAME = provider_name
        AVAILABLE_MODELS = [default_model]
        CAPABILITIES = capabilities
        DEFAULT_MODEL = default_model

        def __init__(self, emitter: Any = None) -> None:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Missing API key: set the {api_key_env} environment variable"
                )
            super().__init__(emitter)

            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=base_url)

        def _send_chunk(
            self,
            audio_path: str,
            verbose: bool,
            language: Optional[str],
            prompt: Optional[str],
        ) -> Optional[ChunkResult]:
            kwargs: dict[str, Any] = {
                "model": self.DEFAULT_MODEL,
                "file": open(audio_path, "rb"),
                "response_format": "verbose_json" if verbose else "text",
            }
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt

            try:
                raw = self._client.audio.transcriptions.create(**kwargs)
            except Exception as e:
                self.emitter.warning(
                    "PROVIDER_REQUEST_FAILED",
                    f"{provider_name} request failed: {e}",
                )
                return None
            finally:
                kwargs["file"].close()

            return self._parse_response(raw, verbose)

        def _parse_response(self, raw: Any, verbose: bool) -> ChunkResult:
            if not verbose:
                return ChunkResult(text=str(raw).strip())

            text = getattr(raw, "text", "") or ""
            language = getattr(raw, "language", None)
            duration = getattr(raw, "duration", None)

            segments: list[Segment] = []
            raw_segments = getattr(raw, "segments", []) or []
            for i, seg in enumerate(raw_segments):
                segments.append(
                    Segment(
                        id=getattr(seg, "id", None)
                        if isinstance(getattr(seg, "id", None), int)
                        else i,
                        start=float(getattr(seg, "start", 0.0) or 0.0),
                        end=float(getattr(seg, "end", 0.0) or 0.0),
                        text=getattr(seg, "text", "") or "",
                        tokens=list(getattr(seg, "tokens", None) or []),
                        avg_logprob=_to_number(getattr(seg, "avg_logprob", None)),
                        compression_ratio=_to_number(
                            getattr(seg, "compression_ratio", None)
                        ),
                        no_speech_prob=_to_number(getattr(seg, "no_speech_prob", None)),
                    )
                )

            return ChunkResult(
                text=text.strip(),
                language=language,
                duration=duration,
                segments=segments,
            )

    _OpenAICompatClient.__name__ = f"{provider_name.title()}STTClient"
    _OpenAICompatClient.__qualname__ = _OpenAICompatClient.__name__
    return _OpenAICompatClient


def _to_number(value: Any) -> Optional[float]:
    return None if value is None else float(value)
