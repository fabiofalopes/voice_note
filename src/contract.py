"""Pydantic models for the voice_note v1.0 output contract."""

from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0"
TOOL_VERSION = "1.0.0"
ErrorCategory = Literal[
    "usage",
    "input",
    "recording",
    "dependency",
    "auth",
    "provider",
    "storage",
    "internal",
    "cancelled",
]
EventLevel = Literal["info", "warning", "error"]


class StrictModel(BaseModel):
    """Base for producer-owned contract models."""

    model_config = ConfigDict(extra="forbid")


class EventType(str, Enum):
    """NDJSON event vocabulary for schema version 1.0."""

    START = "start"
    SEGMENT = "segment"
    WARNING = "warning"
    ERROR = "error"
    END = "end"


class Status(str, Enum):
    """Request completion status."""

    OK = "ok"
    ERROR = "error"
    PARTIAL = "partial"


class StatusCode(str, Enum):
    """Stable symbolic process outcomes exposed to consumers."""

    OK = "OK"
    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    USAGE_ERROR = "USAGE_ERROR"
    PROVIDER_NOT_REGISTERED = "PROVIDER_NOT_REGISTERED"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    AUDIO_INVALID = "AUDIO_INVALID"
    AUDIO_DURATION_UNDETECTABLE = "AUDIO_DURATION_UNDETECTABLE"
    NO_MIC = "NO_MIC"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    PROVIDER_429_EXHAUSTED = "PROVIDER_429_EXHAUSTED"
    PROVIDER_TEMPFAIL = "PROVIDER_TEMPFAIL"
    PROVIDER_UNREACHABLE = "PROVIDER_UNREACHABLE"
    FFMPEG_MISSING = "FFMPEG_MISSING"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    OUTPUT_WRITE_FAILED = "OUTPUT_WRITE_FAILED"
    API_KEY_MISSING = "API_KEY_MISSING"
    API_KEY_INVALID = "API_KEY_INVALID"
    USER_INTERRUPT = "USER_INTERRUPT"
    UNCATEGORIZED = "UNCATEGORIZED"


class ExitCode(IntEnum):
    """Internal POSIX and sysexits-aligned process exit values."""

    OK = 0
    GENERIC = 1
    PUBLIC_USAGE = 2
    USAGE = 64
    DATA_ERROR = 65
    NO_INPUT = 66
    UNAVAILABLE = 69
    SOFTWARE = 70
    CANNOT_CREATE = 73
    TEMPORARY_FAILURE = 75
    NO_PERMISSION = 76
    USER_INTERRUPT = 130


class Word(StrictModel):
    """One word-level timestamp returned by a capable provider."""

    word: str
    start: float
    end: float


class Segment(StrictModel):
    """Canonical provider-neutral transcription segment."""

    id: int = 0
    seek: int = 0
    start: float
    end: float
    offset_seconds: float = 0.0
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    words: list[Word] | None = None
    speaker: str | None = None


class Capabilities(StrictModel):
    """Features supported by the selected provider."""

    word_timestamps: bool
    segment_timestamps: bool
    language_detection: bool
    quality_metrics: bool
    speaker_diarization: bool


class AudioInput(StrictModel):
    """Metadata for the input audio source."""

    audio_file: str
    duration_seconds: float
    size_bytes: int


class Outputs(StrictModel):
    """Successfully promoted output artifact paths."""

    txt: str | None = None
    srt: str | None = None
    json_path: str | None = Field(default=None, alias="json")


class Warning(StrictModel):
    """Structured non-fatal condition."""

    level: Literal["warning"] = "warning"
    code: str
    chunk_index: int | None = None
    detail: str


class Error(StrictModel):
    """Structured fatal or partial-failure detail."""

    code: str
    category: ErrorCategory
    retryable: bool = False
    retry_after_seconds: float | None = None
    message: str
    cause: dict[str, Any] | None = None


class Timing(StrictModel):
    """Wall-clock durations for request stages."""

    record_secs: float = 0.0
    transcribe_secs: float = 0.0
    total_secs: float = 0.0


class Result(StrictModel):
    """Successful or partial transcription result."""

    kind: Literal["transcription", "translation"]
    text: str
    language_detected: str | None = None
    source_language: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    postprocessed: None = None


class Envelope(StrictModel):
    """Complete JSON batch document and on-disk JSON shape."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    tool_version: str = TOOL_VERSION
    request_id: UUID
    mode: Literal["transcribe", "translate"]
    status: Status
    code: str
    message: str
    provider: str
    model: str
    capabilities: Capabilities
    input: AudioInput
    result: Result | None = None
    outputs: Outputs
    warnings: list[Warning] = Field(default_factory=list)
    error: Error | None = None
    timing: Timing
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class Event(StrictModel):
    """One independently parseable NDJSON stream event."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    request_id: UUID
    event_id: UUID
    sequence: int = Field(ge=0)
    time: datetime
    level: EventLevel
    type: EventType
    data: dict[str, Any]


def clamp_end(end: float, duration: float) -> float:
    """Clamp a provider segment end timestamp to input duration.

    Args:
        end: Provider-reported segment end in seconds.
        duration: Input audio duration in seconds.

    Returns:
        ``end`` when valid, otherwise ``duration``.
    """
    return min(end, duration)
