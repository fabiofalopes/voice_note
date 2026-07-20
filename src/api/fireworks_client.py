"""
Fireworks AI STT client.

Implements BaseSTTClient._send_chunk() for the Fireworks AI API.

Fireworks AI exposes a Whisper-compatible API at:
  https://api.fireworks.ai/inference/v1

It accepts the standard OpenAI audio.transcriptions.create() / translations.create()
interface — so we use the openai SDK with a custom base_url.

Rate limits differ from Groq:
  - RPM-based, NOT audio-seconds-based  →  natural fallback when Groq hits hourly limit
  - See https://fireworks.ai/pricing for current limits

Models:
  - accounts/fireworks/models/whisper-v3        : highest accuracy
  - accounts/fireworks/models/whisper-v3-turbo  : faster, same quality

Setup:
  1. Create account at https://fireworks.ai
  2. Generate API key in the dashboard
  3. Set FIREWORKS_API_KEY in .env

Usage:
  client = FireworksSTTClient()
  result = client.transcribe("recording.wav", language="pt")
"""

import os
import time
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from api.base_client import BaseSTTClient, ChunkResult, Segment

load_dotenv()

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksSTTClient(BaseSTTClient):
    """
    Fireworks AI Whisper API client.

    Uses the openai SDK pointed at Fireworks' OpenAI-compatible endpoint.
    Rate limits are RPM-based (not audio-seconds), so this is a natural
    complement to Groq for long-form transcription jobs.
    """

    PROVIDER_NAME = "fireworks"

    AVAILABLE_MODELS = [
        "accounts/fireworks/models/whisper-v3-turbo",
        "accounts/fireworks/models/whisper-v3",
    ]

    # Fireworks RPM limits are more generous per-request, so 600s chunks work well
    CHUNK_SECONDS = 600

    def __init__(self) -> None:
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError(
                "FIREWORKS_API_KEY not set. "
                "Get a free key at https://fireworks.ai (no CC required)."
            )
        self.client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)

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
        Send one opus chunk to Fireworks AI.

        - Uses response_format="verbose_json" when verbose=True, "text" otherwise.
        - Passes language hint and prompt when provided.
        - Retries on 429 with exponential backoff (Fireworks doesn't include
          a precise wait time in its error message, unlike Groq).
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
                    print(
                        f"  [fireworks] rate limit  chunk {chunk_num}/{total_chunks}  "
                        f"attempt {attempt}/{max_retries}  "
                        f"→ sleeping {wait:.0f}s"
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, 300)  # cap at 5 minutes
                    continue
                print(f"  [fireworks] error chunk {chunk_num}: {e}")
                return None

        print(f"  [fireworks] chunk {chunk_num} failed after {max_retries} retries")
        return None

    # -----------------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------------

    def _parse_response(self, raw, verbose: bool) -> ChunkResult:
        """Convert raw Fireworks/OpenAI API response to ChunkResult."""
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
                    id=getattr(seg, "id", i),
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    text=getattr(seg, "text", ""),
                    avg_logprob=float(getattr(seg, "avg_logprob", 0.0)),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                    compression_ratio=float(getattr(seg, "compression_ratio", 1.6)),
                    tokens=list(getattr(seg, "tokens", [])),
                )
            )

        return ChunkResult(
            text=text.strip(),
            segments=segments,
            detected_language=detected_language,
            duration=float(duration) if duration else None,
        )
