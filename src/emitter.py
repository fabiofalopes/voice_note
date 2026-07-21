"""Contract-aware output renderers for human and machine consumers."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, TextIO
from uuid import UUID, uuid4

from contract import Envelope, Event, EventLevel, EventType, Warning


class Emitter(ABC):
    """Shared event source rendered by each supported output mode."""

    def __init__(
        self,
        request_id: UUID | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        quiet: bool = False,
    ) -> None:
        self.request_id = request_id or uuid4()
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.quiet = quiet
        self.sequence = 0
        self.warnings: list[Warning] = []
        self.segments_emitted = 0
        self.ended = False

    def log(self, message: str) -> None:
        """Render a human progress message when the output mode supports it."""
        if not self.quiet:
            self._render_log(message)

    def start(self, data: dict[str, Any]) -> None:
        self._emit(EventType.START, "info", data)

    def segment(self, data: dict[str, Any]) -> None:
        self.segments_emitted += 1
        self._emit(EventType.SEGMENT, "info", data)

    def warning(self, code: str, detail: str, chunk_index: int | None = None) -> None:
        warning = Warning(code=code, detail=detail, chunk_index=chunk_index)
        self.warnings.append(warning)
        self._emit(
            EventType.WARNING,
            "warning",
            warning.model_dump(exclude_none=True, exclude={"level"}),
        )

    def error(self, data: dict[str, Any]) -> None:
        self._emit(EventType.ERROR, "error", data)

    def end(self, data: dict[str, Any]) -> None:
        if self.ended:
            return
        self._emit(EventType.END, "info", data)
        self.ended = True

    def finalize(self, envelope: Envelope) -> None:
        """Render the final JSON document or human summary."""
        self._render_envelope(envelope)

    def _emit(
        self,
        event_type: EventType,
        level: EventLevel,
        data: dict[str, Any],
    ) -> None:
        if self.ended:
            raise RuntimeError("Events after end are forbidden")
        event = Event(
            request_id=self.request_id,
            event_id=uuid4(),
            sequence=self.sequence,
            time=datetime.now(timezone.utc),
            level=level,
            type=event_type,
            data=data,
        )
        self.sequence += 1
        self._render_event(event)

    @abstractmethod
    def _render_event(self, event: Event) -> None:
        """Render one typed event."""

    @abstractmethod
    def _render_envelope(self, envelope: Envelope) -> None:
        """Render the completed JSON envelope."""

    def _render_log(self, message: str) -> None:
        pass


class HumanEmitter(Emitter):
    """Readable TTY renderer derived from contract events."""

    def _render_event(self, event: Event) -> None:
        if self.quiet:
            return
        if event.type == EventType.WARNING:
            self.stderr.write(f"Warning: {event.data['detail']}\n")
        elif event.type == EventType.ERROR:
            self.stderr.write(f"Error: {event.data['message']}\n")

    def _render_envelope(self, envelope: Envelope) -> None:
        if envelope.result is not None:
            self.stdout.write(envelope.result.text + "\n")
        self.stdout.write(envelope.message + "\n")

    def _render_log(self, message: str) -> None:
        self.stdout.write(message + "\n")


class PlainEmitter(HumanEmitter):
    """Human renderer without ANSI styling."""


class JSONEmitter(Emitter):
    """Atomic single-document JSON renderer."""

    def _render_event(self, event: Event) -> None:
        return

    def _render_envelope(self, envelope: Envelope) -> None:
        self.stdout.write(
            json.dumps(
                envelope_payload(envelope),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


class NDJSONEmitter(Emitter):
    """Streaming renderer that writes one JSON event per line."""

    def _render_event(self, event: Event) -> None:
        self.stdout.write(
            json.dumps(
                event.model_dump(mode="json", by_alias=True, exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self.stdout.flush()

    def _render_envelope(self, envelope: Envelope) -> None:
        return


def envelope_payload(envelope: Envelope) -> dict[str, Any]:
    """Serialize an envelope with contract-specific null handling."""
    payload = envelope.model_dump(mode="json", by_alias=True)
    payload["outputs"] = {
        key: value for key, value in payload["outputs"].items() if value is not None
    }
    return payload
