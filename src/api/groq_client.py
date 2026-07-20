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

    # Free-tier: 7200 audio-sec/hr rolling → 500s × 14 chunks ≈ full window
    CHUNK_SECONDS = 500

    def __init__(self) -> None:
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
            print(
                f"[groq] '{resolved}' does not support translation — "
                f"switching to {self.TRANSLATION_MODEL}"
            )
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

                # Build kwargs — only include optional params when they have values
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
                    print(
                        f"  [groq] rate limit  chunk {chunk_num}/{total_chunks}  "
                        f"attempt {attempt}/{max_retries}  "
                        f"→ sleeping {wait + 1:.0f}s"
                    )
                    time.sleep(wait + 1)
                    continue
                print(f"  [groq] error chunk {chunk_num}: {e}")
                return None

        print(f"  [groq] chunk {chunk_num} failed after {max_retries} retries")
        return None

    # -----------------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------------

    def _parse_response(self, raw, verbose: bool) -> ChunkResult:
        """Convert raw Groq API response to ChunkResult."""
        if not verbose:
            # response_format="text" → raw is a plain string
            return ChunkResult(text=str(raw).strip())

        # response_format="verbose_json" → raw is an object with attributes
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
