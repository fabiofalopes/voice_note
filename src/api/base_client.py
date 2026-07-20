"""
Base STT client interface.

All provider clients (Groq, modelos) inherit from this.
Defines the shared pipeline: chunked extraction, partial-file crash-safety,
output serialisation, and the abstract _send_chunk() hook each provider fills.
"""

from __future__ import annotations

import os
import re
import time
import tempfile
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Generator

from audio_processing.utils import get_audio_duration
from emitter import Emitter, HumanEmitter
from i18n import normalize_language


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One Whisper segment — mirrors verbose_json response shape."""

    id: int
    start: float
    end: float
    text: str
    offset_seconds: float = 0.0
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    tokens: list = field(default_factory=list)


@dataclass
class ChunkResult:
    """What _send_chunk() returns for a single audio chunk."""

    text: str  # plain concatenated text
    segments: list[Segment] = field(default_factory=list)
    detected_language: Optional[str] = None
    duration: Optional[float] = None  # audio duration as reported by API
    provider_meta: dict = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    """Full result returned from transcribe() / translate()."""

    text: str  # full plain text
    segments: list[Segment] = field(default_factory=list)
    detected_language: Optional[str] = None
    output_file: Optional[str] = None  # path to saved .txt
    srt_file: Optional[str] = None  # path to saved .srt  (if timestamps requested)
    json_file: Optional[str] = None  # path to saved .json (if verbose requested)
    provider_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared chunk-pipeline base
# ---------------------------------------------------------------------------


class BaseSTTClient(ABC):
    """
    Shared pipeline for all STT providers.

    Subclasses must implement:
      - PROVIDER_NAME: str
      - AVAILABLE_MODELS: list[str]
      - _send_chunk(chunk_file, model, mode, chunk_num, total, ...) -> ChunkResult | None

    The base class handles:
      - ffmpeg chunk extraction (opus 16k mono 16kHz)
      - partial-file crash-safe resume
      - prompt chaining between chunks (last ~200 chars of previous chunk)
      - no_speech_prob filtering (skip silent chunks)
      - output serialisation: .txt, .srt, .json
    """

    PROVIDER_NAME: str = "unknown"
    AVAILABLE_MODELS: list[str] = []
    CAPABILITIES: dict[str, bool] = {}

    def __init__(self, emitter: Optional[Emitter] = None) -> None:
        self.emitter = emitter or HumanEmitter()

    # Chunk / audio settings — may be overridden per provider
    CHUNK_SECONDS: int = 500
    OVERLAP_SECONDS: int = 2
    MAX_FILE_SIZE_MB: float = 25.0

    CHUNK_CODEC: str = "libopus"
    CHUNK_BITRATE: str = "16k"
    CHUNK_SAMPLERATE: str = "16000"
    CHUNK_CHANNELS: str = "1"

    # Quality / filtering thresholds
    NO_SPEECH_SKIP_THRESHOLD: float = 0.85  # skip segment if no_speech_prob > this
    LOW_CONFIDENCE_WARN_THRESHOLD: float = -0.8  # warn if avg_logprob < this

    # Prompt chaining — pass tail of previous chunk to next call for continuity
    PROMPT_TAIL_CHARS: int = 200  # how many trailing chars to use as prompt

    # -----------------------------------------------------------------------
    # Abstract hook — each provider implements this
    # -----------------------------------------------------------------------

    @abstractmethod
    def _send_chunk(
        self,
        chunk_file: str,
        model: str,
        mode: str,  # "transcribe" | "translate"
        chunk_num: int,
        total_chunks: int,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        verbose: bool = False,  # request verbose_json with timestamps
        max_retries: int = 10,
    ) -> Optional[ChunkResult]:
        """
        Send one audio chunk to the provider.

        Must handle rate-limit retries internally and return None on permanent
        failure (so the pipeline can skip the chunk and continue).
        """
        ...

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def transcribe(
        self,
        audio_file: str,
        model: Optional[str] = None,
        output_file: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio file.

        Args:
            audio_file:   Path to source audio.
            model:        Provider model name. Defaults to first in AVAILABLE_MODELS.
            output_file:  Base output path (without extension). Derived from audio_file
                          if not given. Saves <base>_transcription.txt always.
                          If verbose=True also saves .srt and .json.
            language:     ISO-639-1 language hint (e.g. "pt", "en"). Auto-detected
                          if not given.
            verbose:      Request timestamps/segments from API. Required for SRT output.
        """
        model = self._resolve_model(model)
        out_base = output_file or self._default_output_base(audio_file, "transcription")
        return self._run_pipeline(
            audio_file, model, "transcribe", out_base, language, verbose
        )

    def translate(
        self,
        audio_file: str,
        model: Optional[str] = None,
        output_file: Optional[str] = None,
        verbose: bool = False,
    ) -> Optional[TranscriptionResult]:
        """
        Translate audio to English.

        Note: translation does not support a source-language hint (API limitation).
        """
        model = self._resolve_model(model, for_translation=True)
        out_base = output_file or self._default_output_base(audio_file, "translation")
        return self._run_pipeline(
            audio_file, model, "translate", out_base, language=None, verbose=verbose
        )

    # -----------------------------------------------------------------------
    # Core pipeline
    # -----------------------------------------------------------------------

    def _run_pipeline(
        self,
        audio_file: str,
        model: str,
        mode: str,
        out_base: str,
        language: Optional[str],
        verbose: bool,
    ) -> Optional[TranscriptionResult]:
        txt_file = out_base + ".txt"
        partial_file = txt_file + ".partial"

        try:
            duration = get_audio_duration(audio_file)
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)

            verb = "Transcribing" if mode == "transcribe" else "Translating"
            lang_hint = f"  |  lang={language}" if language else ""
            self.emitter.log(
                f"[{self.PROVIDER_NAME}] {verb}  |  "
                f"{file_size_mb:.1f} MB  |  {duration:.1f}s  |  "
                f"model={model}  |  chunk={self.CHUNK_SECONDS}s{lang_hint}"
            )
            self.emitter.log(f"  output → {txt_file}")
            if verbose:
                self.emitter.log(f"  srt    → {out_base}.srt")
                self.emitter.log(f"  json   → {out_base}.json")

            resume_start, existing_text = self._load_partial(partial_file, duration)
            accumulated_texts: list[str] = [existing_text] if existing_text else []
            all_segments: list[Segment] = []
            detected_language: Optional[str] = None
            provider_meta: dict = {}
            prompt: Optional[str] = None  # rolling prompt for chunk continuity

            for chunk_result in self._iter_chunks(
                audio_file,
                duration,
                model,
                mode,
                start_offset=resume_start,
                language=language,
                verbose=verbose,
                partial_file=partial_file,
                accumulated=accumulated_texts,
                get_prompt=lambda: prompt,
            ):
                accumulated_texts.append(chunk_result.text)
                all_segments.extend(chunk_result.segments)
                if chunk_result.detected_language:
                    detected_language = chunk_result.detected_language
                provider_meta.update(chunk_result.provider_meta)
                # Update rolling prompt for next chunk
                prompt = self._build_prompt(chunk_result.text)

            if not accumulated_texts:
                warning_codes = {warning.code for warning in self.emitter.warnings}
                chunk_failure_codes = {
                    "CHUNK_FAILED",
                    "CHUNK_EXTRACTION_FAILED",
                    "PROVIDER_CHUNK_ERROR",
                    "PROVIDER_RETRIES_EXHAUSTED",
                }
                if (
                    "SILENT_CHUNK_SKIPPED" in warning_codes
                    and not warning_codes.intersection(chunk_failure_codes)
                ):
                    self.emitter.warning(
                        "ALL_CHUNKS_SILENT",
                        "All audio chunks were silent",
                    )
                else:
                    self.emitter.warning(
                        "ALL_CHUNKS_FAILED",
                        f"[{self.PROVIDER_NAME}] No chunks succeeded.",
                    )
                return None

            full_text = " ".join(accumulated_texts)

            # Save outputs
            self._save_txt(full_text, txt_file)
            srt_path = None
            if verbose and all_segments:
                srt_path = self._save_srt(all_segments, out_base + ".srt")

            self._remove_partial(partial_file)

            return TranscriptionResult(
                text=full_text,
                segments=all_segments,
                detected_language=normalize_language(detected_language),
                output_file=txt_file,
                srt_file=srt_path,
                json_file=None,
                provider_meta=provider_meta,
            )

        except FileNotFoundError:
            self.emitter.error(
                {
                    "code": "FILE_NOT_FOUND",
                    "category": "input",
                    "retryable": False,
                    "message": f"File not found: {audio_file}",
                }
            )
            return None
        except Exception:
            self.emitter.error(
                {
                    "code": "INTERNAL_ERROR",
                    "category": "internal",
                    "retryable": False,
                    "message": f"[{self.PROVIDER_NAME}] Pipeline failed",
                }
            )
            return None

    def _iter_chunks(
        self,
        audio_file: str,
        duration: float,
        model: str,
        mode: str,
        start_offset: float,
        language: Optional[str],
        verbose: bool,
        partial_file: Optional[str],
        accumulated: list[str],
        get_prompt,  # callable → str | None
    ) -> Generator[ChunkResult, None, None]:
        """
        Walk the audio one chunk at a time.
        Extracts → sends → saves partial → yields ChunkResult.
        """
        chunk_dur = self.CHUNK_SECONDS
        total_chunks = max(1, int((duration + chunk_dur - 1) // chunk_dur))
        done_before = int(start_offset // chunk_dur) if start_offset > 0 else 0

        verb = "Transcribing" if mode == "transcribe" else "Translating"
        if start_offset > 0:
            self.emitter.log(
                f"  Resuming from {start_offset:.0f}s  "
                f"(chunks {done_before + 1}–{total_chunks})"
            )
        self.emitter.log(
            f"  {verb} {duration:.0f}s  |  "
            f"{total_chunks} chunks × {chunk_dur}s  |  "
            f"opus {self.CHUNK_BITRATE} mono {self.CHUNK_SAMPLERATE}Hz"
        )

        chunk_start = start_offset
        chunk_num = done_before

        while chunk_start < duration:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_dur, duration)
            seg_duration = chunk_end - chunk_start

            # 1. Extract chunk to temp opus file
            tmp_fd, chunk_file = tempfile.mkstemp(suffix=".opus")
            os.close(tmp_fd)
            try:
                self._extract_chunk(audio_file, chunk_start, seg_duration, chunk_file)
                chunk_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
                self.emitter.log(
                    f"  chunk {chunk_num}/{total_chunks}  "
                    f"({chunk_start:.0f}s–{chunk_end:.0f}s)  "
                    f"{chunk_size_mb:.2f} MB"
                )
            except Exception as e:
                self.emitter.warning(
                    "CHUNK_EXTRACTION_FAILED",
                    f"Could not extract chunk {chunk_num}: {e}",
                    chunk_num - 1,
                )
                try:
                    os.unlink(chunk_file)
                except OSError:
                    pass
                chunk_start = chunk_end - self.OVERLAP_SECONDS
                if chunk_end >= duration:
                    break
                continue

            # 2. Send to provider
            result = self._send_chunk(
                chunk_file,
                model,
                mode,
                chunk_num,
                total_chunks,
                language=language,
                prompt=get_prompt(),
                verbose=verbose,
            )

            # 3. Delete temp file immediately
            try:
                os.unlink(chunk_file)
            except OSError:
                pass

            if result:
                null_fields: set[str] = set()
                for segment in result.segments:
                    segment.offset_seconds = chunk_start
                    if segment.end > seg_duration:
                        original_end = segment.end
                        segment.end = seg_duration
                        self.emitter.warning(
                            "TIMESTAMP_CLAMPED",
                            f"end {original_end:.2f}s clamped to chunk duration {seg_duration:.2f}s",
                            chunk_num - 1,
                        )
                    for field_name in (
                        "avg_logprob",
                        "compression_ratio",
                        "no_speech_prob",
                    ):
                        if getattr(segment, field_name) is None:
                            null_fields.add(field_name)
                if null_fields:
                    self.emitter.warning(
                        "PROVIDER_FIELD_NULL",
                        "Provider returned null for "
                        + ", ".join(sorted(null_fields))
                        + " (preserved as null)",
                        chunk_num - 1,
                    )

                # Filter: if all segments are silence, skip
                if self._is_silent_chunk(result):
                    self.emitter.warning(
                        "SILENT_CHUNK_SKIPPED",
                        f"Chunk {chunk_num}/{total_chunks} skipped because it is silent",
                        chunk_num - 1,
                    )
                else:
                    # Warn on low confidence
                    if result.segments:
                        low_conf = [
                            s
                            for s in result.segments
                            if s.avg_logprob is not None
                            and s.avg_logprob < self.LOW_CONFIDENCE_WARN_THRESHOLD
                        ]
                        if low_conf:
                            self.emitter.warning(
                                "LOW_CONFIDENCE_SEGMENTS",
                                f"{len(low_conf)} segments below threshold "
                                f"{self.LOW_CONFIDENCE_WARN_THRESHOLD}",
                                chunk_num - 1,
                            )

                    self.emitter.log(
                        f"  [ok] chunk {chunk_num}/{total_chunks}: "
                        f"{len(result.text)} chars"
                        + (
                            f"  lang={result.detected_language}"
                            if result.detected_language
                            else ""
                        )
                    )

                    for segment in result.segments:
                        self.emitter.segment(
                            {
                                "chunk_index": chunk_num - 1,
                                "offset_seconds": segment.offset_seconds,
                                "id": segment.id,
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text,
                                "avg_logprob": segment.avg_logprob,
                                "no_speech_prob": segment.no_speech_prob,
                            }
                        )

                    # 4. Persist partial immediately
                    if partial_file:
                        self._append_partial(
                            partial_file, result.text, chunk_end, duration, accumulated
                        )

                    yield result
            else:
                self.emitter.warning(
                    "CHUNK_FAILED",
                    f"Chunk {chunk_num}/{total_chunks} failed and was skipped",
                    chunk_num - 1,
                )

            chunk_start = chunk_end - self.OVERLAP_SECONDS
            if chunk_end >= duration:
                break

    # -----------------------------------------------------------------------
    # ffmpeg extraction
    # -----------------------------------------------------------------------

    def _extract_chunk(
        self, audio_file: str, start: float, duration: float, output_file: str
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            audio_file,
            "-t",
            str(duration),
            "-ar",
            self.CHUNK_SAMPLERATE,
            "-ac",
            self.CHUNK_CHANNELS,
            "-c:a",
            self.CHUNK_CODEC,
            "-b:a",
            self.CHUNK_BITRATE,
            "-vbr",
            "on",
            "-compression_level",
            "10",
            output_file,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))

    # -----------------------------------------------------------------------
    # Partial file (crash-safe resume)
    # -----------------------------------------------------------------------

    def _load_partial(
        self, partial_file: str, total_duration: float
    ) -> tuple[float, Optional[str]]:
        if not os.path.exists(partial_file):
            return 0.0, None
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                text = f.read().strip()
            if header.startswith("PARTIAL:"):
                parts = header[8:].split("/")
                resume_at = float(parts[0])
                pct = resume_at / total_duration * 100 if total_duration else 0
                self.emitter.log(
                    f"  Resuming from partial  {resume_at:.0f}s ({pct:.0f}% done)"
                )
                return resume_at, text if text else None
        except Exception as e:
            self.emitter.warning(
                "PARTIAL_READ_FAILED",
                f"Could not read partial file ({e}); starting fresh",
            )
        return 0.0, None

    def _append_partial(
        self,
        partial_file: str,
        new_text: str,
        chunk_end: float,
        total_duration: float,
        accumulated: list[str],
    ) -> None:
        all_parts = list(accumulated) + [new_text]
        full_so_far = " ".join(all_parts)
        try:
            with open(partial_file, "w", encoding="utf-8") as f:
                f.write(f"PARTIAL:{chunk_end:.1f}/{total_duration:.1f}\n")
                f.write(full_so_far)
        except Exception as e:
            self.emitter.warning(
                "PARTIAL_WRITE_FAILED", f"Could not write partial file: {e}"
            )

    def _remove_partial(self, partial_file: str) -> None:
        try:
            os.unlink(partial_file)
        except OSError:
            pass

    # -----------------------------------------------------------------------
    # Output serialisation
    # -----------------------------------------------------------------------

    def _save_txt(self, text: str, path: str) -> None:
        self._atomic_write(path, text)
        self.emitter.log(f"  saved txt  → {path}  ({len(text)} chars)")

    def _save_srt(self, segments: list[Segment], path: str) -> str:
        """Write standard SRT subtitle file from segment list."""
        blocks = []
        for i, seg in enumerate(segments, 1):
            blocks.append(
                f"{i}\n"
                f"{_fmt_srt_time(seg.start + seg.offset_seconds)} --> "
                f"{_fmt_srt_time(seg.end + seg.offset_seconds)}\n"
                f"{seg.text.strip()}\n"
            )
        self._atomic_write(path, "\n".join(blocks))
        self.emitter.log(f"  saved srt  → {path}  ({len(segments)} segments)")
        return path

    def _atomic_write(self, path: str, content: str) -> None:
        """Write UTF-8 text via fsync and atomic replacement.

        Args:
            path: Final output path.
            content: Complete text to commit.
        """
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{path}.tmp.{self.emitter.request_id}"
        with open(temp_path, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_prompt(self, text: str) -> Optional[str]:
        """Return the trailing PROMPT_TAIL_CHARS of text, or None if empty."""
        if not text:
            return None
        tail = text.strip()[-self.PROMPT_TAIL_CHARS :]
        return tail if tail else None

    def _is_silent_chunk(self, result: ChunkResult) -> bool:
        """True if all segments have high no_speech_prob (chunk is silence/noise)."""
        if not result.segments:
            return False
        return all(
            s.no_speech_prob is not None
            and s.no_speech_prob > self.NO_SPEECH_SKIP_THRESHOLD
            for s in result.segments
        )

    def _default_output_base(self, audio_file: str, suffix: str) -> str:
        base = os.path.splitext(audio_file)[0]
        return f"{base}_{suffix}"

    def _resolve_model(
        self, model: Optional[str], for_translation: bool = False
    ) -> str:
        if model is None:
            return self.AVAILABLE_MODELS[0]
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"[{self.PROVIDER_NAME}] Unknown model '{model}'. "
                f"Available: {self.AVAILABLE_MODELS}"
            )
        return model


# ---------------------------------------------------------------------------
# SRT time formatting helper
# ---------------------------------------------------------------------------


def _fmt_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
