"""
Modelos AI STT client.

Implements BaseSTTClient._send_chunk() for the Modelos AI API.

Modelos AI exposes an OpenAI-compatible Whisper API at:
  https://modelos.ai.ulusofona.pt/

It accepts the standard OpenAI audio.transcriptions.create() / translations.create()
interface — so we use the openai SDK with a custom base_url.

Models:
  - stt-large-v3-turbo  : fast, high accuracy

Setup:
  1. Get API key from Modelos AI
  2. Set MODELOS_AI_KEY in .env

Usage:
  client = ModelosSTTClient()
  result = client.transcribe("recording.wav", language="pt")
"""

import os
import time
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from api.base_client import BaseSTTClient, ChunkResult, Segment
from emitter import Emitter

load_dotenv()

MODELOS_AI_BASE_URL = "https://modelos.ai.ulusofona.pt/"


class ModelosSTTClient(BaseSTTClient):
    """Modelos AI Whisper API client using the OpenAI-compatible endpoint."""

    PROVIDER_NAME = "modelos"

    AVAILABLE_MODELS = [
        "stt-large-v3-turbo",
    ]

    CHUNK_SECONDS = 500
    CAPABILITIES = {
        "word_timestamps": False,
        "segment_timestamps": True,
        "language_detection": True,
        "quality_metrics": False,
        "speaker_diarization": False,
    }

    def __init__(self, emitter: Optional[Emitter] = None) -> None:
        super().__init__(emitter)
        api_key = os.getenv("MODELOS_AI_KEY")
        if not api_key:
            raise ValueError("MODELOS_AI_KEY not set. Get your key from Modelos AI.")
        self.client = OpenAI(api_key=api_key, base_url=MODELOS_AI_BASE_URL)

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
        Send one opus chunk to Modelos AI.

        - Uses response_format="verbose_json" when verbose=True, "text" otherwise.
        - Passes language hint and prompt when provided.
        - Retries on 429 with exponential backoff.
        - Returns ChunkResult or None on permanent failure.
        """
        response_format = "verbose_json" if verbose else "text"
        backoff = 30.0  # initial backoff for rate limit retries

        for attempt in range(1, max_retries + 1):
            try:
                with open(chunk_file, "rb") as f:
                    audio_data = f.read()

                kwargs: dict = {
                    "file": (os.path.basename(chunk_file), audio_data),
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
                    # Translation: drop language param
                    kwargs.pop("language", None)
                    raw = self.client.audio.translations.create(**kwargs)

                return self._parse_response(raw, verbose)

            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait = backoff
                    self.emitter.warning(
                        "PROVIDER_RATE_LIMIT",
                        f"modelos rate limit on chunk {chunk_num}/{total_chunks}; "
                        f"retry {attempt}/{max_retries} in {wait:.0f}s",
                        chunk_num - 1,
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, 300)  # cap at 5 minutes
                    continue
                self.emitter.warning(
                    "PROVIDER_CHUNK_ERROR",
                    f"modelos failed chunk {chunk_num}",
                    chunk_num - 1,
                )
                return None

        self.emitter.warning(
            "PROVIDER_RETRIES_EXHAUSTED",
            f"modelos chunk {chunk_num} failed after {max_retries} retries",
            chunk_num - 1,
        )
        return None

    # -----------------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------------

    def _parse_response(self, raw, verbose: bool) -> ChunkResult:
        if not verbose:
            # response_format="text" → raw is a plain string (or object with .text)
            if isinstance(raw, str):
                return ChunkResult(text=raw.strip())
            return ChunkResult(text=str(getattr(raw, "text", raw)).strip())

        # response_format="verbose_json" → object with .text, .language, .segments
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
        )


def _to_number(value) -> Optional[float]:
    return None if value is None else float(value)


def _field_float(value, field_name: str) -> Optional[float]:
    return _to_number(getattr(value, field_name, None))


def _segment_id(value, fallback: int) -> int:
    """Read a provider segment id or use accumulation order."""
    segment_id = getattr(value, "id", None)
    return segment_id if isinstance(segment_id, int) else fallback
