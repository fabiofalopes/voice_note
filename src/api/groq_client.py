"""
Groq Whisper STT client.

Implements BaseSTTClient._send_chunk() for the Groq API.

Rate limits (free tier):
  - 7,200 audio-seconds / hour  (rolling window)
  - 28,800 audio-seconds / day
  - 20 RPM / 2,000 RPD

Models:
  - whisper-large-v3        : highest accuracy, supports translation
  - whisper-large-v3-turbo  : ~14% faster, transcription only
  - distil-whisper-large-v3-en : English-only distilled, fastest

The 429 error message from Groq includes the exact wait time
("Please try again in 1m7.5s") which is parsed and used directly.
"""

import os
import re
import time
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from api.base_client import BaseSTTClient, ChunkResult, Segment
from emitter import Emitter

load_dotenv()


class GroqWhisperClient(BaseSTTClient):
    """
    Groq Whisper API client.

    Rate-limit behaviour:
      CHUNK_SECONDS = 500  → 14 full chunks before hitting 7200s/hr limit.
      On 429: Groq's error message contains the exact wait duration — we parse
      it and sleep that amount +1s, then retry the same chunk file (no re-extract).
    """

    PROVIDER_NAME = "groq"

    AVAILABLE_MODELS = [
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        "distil-whisper-large-v3-en",
    ]

    # Translation is only supported on the large-v3 model
    TRANSLATION_MODEL = "whisper-large-v3"
    CAPABILITIES = {
        "word_timestamps": True,
        "segment_timestamps": True,
        "language_detection": True,
        "quality_metrics": True,
        "speaker_diarization": False,
    }

    # Free-tier: 7200 audio-sec/hr rolling → 500s × 14 chunks ≈ full window
    CHUNK_SECONDS = 500

    def __init__(self, emitter: Optional[Emitter] = None) -> None:
        super().__init__(emitter)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/"
            )
        self.client = Groq(api_key=api_key)

    # -----------------------------------------------------------------------
    # Resolve model (translation override)
    # -----------------------------------------------------------------------

    def _resolve_model(
        self, model: Optional[str], for_translation: bool = False
    ) -> str:
        resolved = super()._resolve_model(model, for_translation)
        if for_translation and resolved != self.TRANSLATION_MODEL:
            return self.TRANSLATION_MODEL
        return resolved

    # -----------------------------------------------------------------------
    # Provider hook
    # -----------------------------------------------------------------------

    def _send_chunk(
        self,
        chunk_file: str,
        model: str,
        mode: str,
        chunk_num: int,
        total_chunks: int,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        verbose: bool = False,
        max_retries: int = 10,
    ) -> Optional[ChunkResult]:
        """
        Send one opus chunk to Groq.

        - Uses response_format="verbose_json" when verbose=True, "text" otherwise.
        - Passes language hint and prompt when provided.
        - Retries on 429 using the exact wait time from Groq's error message.
        - Returns ChunkResult or None on permanent failure.
        """
        response_format = "verbose_json" if verbose else "text"

        for attempt in range(1, max_retries + 1):
            try:
                with open(chunk_file, "rb") as f:
                    data = f.read()

                kwargs: dict = {
                    "file": (os.path.basename(chunk_file), data),
                    "model": model,
                    "response_format": response_format,
                }
                if language and mode == "transcribe":
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt

                if mode == "transcribe":
                    raw = self.client.audio.transcriptions.create(**kwargs)
                else:
                    # Translation: no language param, prompt must be English
                    kwargs.pop("language", None)
                    raw = self.client.audio.translations.create(**kwargs)

                return self._parse_response(raw, verbose)

            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit_exceeded" in err:
                    wait = self._parse_retry_wait(err)
                    self.emitter.warning(
                        "PROVIDER_RATE_LIMIT",
                        f"Groq rate limit on chunk {chunk_num}/{total_chunks}; "
                        f"retry {attempt}/{max_retries} in {wait + 1:.0f}s",
                        chunk_num - 1,
                    )
                    time.sleep(wait + 1)
                    continue
                self.emitter.warning(
                    "PROVIDER_CHUNK_ERROR",
                    f"Groq failed chunk {chunk_num}",
                    chunk_num - 1,
                )
                return None

        self.emitter.warning(
            "PROVIDER_RETRIES_EXHAUSTED",
            f"Groq chunk {chunk_num} failed after {max_retries} retries",
            chunk_num - 1,
        )
        return None

    # -----------------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------------

    def _parse_response(self, raw, verbose: bool) -> ChunkResult:
        """Convert raw Groq API response to ChunkResult."""
        if not verbose:
            return ChunkResult(text=str(raw).strip())

        text = getattr(raw, "text", "") or ""
        detected_language = getattr(raw, "language", None)
        duration = getattr(raw, "duration", None)

        segments: list[Segment] = []
        for i, seg in enumerate(getattr(raw, "segments", []) or []):
            segments.append(
                Segment(
                    id=_segment_id(seg, i),
                    start=_field_float(seg, "start") or 0.0,
                    end=_field_float(seg, "end") or 0.0,
                    text=getattr(seg, "text", ""),
                    avg_logprob=_field_float(seg, "avg_logprob"),
                    no_speech_prob=_field_float(seg, "no_speech_prob"),
                    compression_ratio=_field_float(seg, "compression_ratio"),
                    tokens=list(getattr(seg, "tokens", None) or []),
                )
            )

        return ChunkResult(
            text=text.strip(),
            segments=segments,
            detected_language=detected_language,
            duration=_to_number(duration),
            provider_meta=_provider_meta(raw),
        )

    # -----------------------------------------------------------------------
    # Rate-limit wait parsing
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_retry_wait(error_str: str) -> float:
        """
        Extract wait duration from Groq 429 message.
        Example: "Please try again in 1m7.5s" → 67.5
        Falls back to 60s if not parseable.
        """
        match = re.search(r"in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", error_str)
        if match:
            return float(match.group(1) or 0) * 60 + float(match.group(2))
        return 60.0


def _to_number(value) -> Optional[float]:
    return None if value is None else float(value)


def _field_float(value, field_name: str) -> Optional[float]:
    return _to_number(getattr(value, field_name, None))


def _segment_id(value, fallback: int) -> int:
    """Read a provider segment id or use accumulation order."""
    segment_id = getattr(value, "id", None)
    return segment_id if isinstance(segment_id, int) else fallback


def _provider_meta(raw) -> dict:
    """Extract Groq extension metadata without exposing SDK objects."""
    value = getattr(raw, "x_groq", None)
    if value is None:
        return {}
    if isinstance(value, dict):
        return {"x_groq": value}
    if hasattr(value, "model_dump"):
        return {"x_groq": value.model_dump()}
    request_id = getattr(value, "id", None)
    return {"x_groq": {"id": request_id}} if request_id is not None else {}
