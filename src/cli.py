#!/usr/bin/env python3
"""
Voice Transcriber CLI

Main command-line interface for the voice transcriber.
"""

import argparse
import sys
import os
import json
import time

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import (
    AudioInput,
    Capabilities,
    Envelope,
    Error,
    ErrorCategory,
    ExitCode,
    Outputs,
    Result,
    Segment,
    Status,
    Timing,
)
from emitter import (
    Emitter,
    HumanEmitter,
    JSONEmitter,
    NDJSONEmitter,
    PlainEmitter,
    envelope_payload,
)


def _build_client(provider: str, emitter: Emitter):
    """Instantiate the right STT client for the given provider name."""
    if provider == "groq":
        from api.groq_client import GroqWhisperClient

        return GroqWhisperClient(emitter)
    elif provider == "modelos":
        from api.modelos_client import ModelosSTTClient

        return ModelosSTTClient(emitter)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'groq' or 'modelos'.")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Transcriber — record and transcribe audio"
    )

    # Positional
    parser.add_argument(
        "file", nargs="?", help="Audio file to transcribe (omit to record)"
    )

    # Provider & model
    parser.add_argument(
        "--provider",
        "-p",
        choices=["groq", "modelos"],
        default="groq",
        help="STT provider to use (default: groq)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Model name. Defaults to provider default. "
            "Groq: whisper-large-v3[-turbo], distil-whisper-large-v3-en. "
            "Modelos: stt-large-v3-turbo."
        ),
    )

    # Transcription options
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        metavar="LANG",
        help="ISO-639-1 language hint, e.g. 'pt', 'en' (improves accuracy + speed)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Save .srt and .json in addition to .txt (uses verbose_json mode)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate to English instead of transcribe",
    )
    parser.add_argument(
        "--output",
        choices=["human", "plain", "json", "ndjson"],
        help="Output mode (default: human in TTY, ndjson when piped)",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON document")
    parser.add_argument("--ndjson", action="store_true", help="Emit streaming NDJSON")
    parser.add_argument("--plain", action="store_true", help="Emit plain human output")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress human stderr output"
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default=None,
        help="Base output path without extension",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Request word timestamps when supported",
    )

    # Recording options (only used when no file is given)
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--test-mic", action="store_true", help="Test microphone input levels"
    )
    parser.add_argument("--device", "-d", type=int, help="Audio input device index")
    parser.add_argument(
        "--record-output",
        default="recording.wav",
        help="Output WAV filename when recording (default: recording.wav)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the legacy recorder instead of the robust recorder (not recommended)",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=5,
        help="Minutes per chunk for robust recording (default: 5)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        help="Maximum recording duration in minutes",
    )

    # Misc
    parser.add_argument(
        "--no-clipboard", action="store_true", help="Don't copy text to clipboard"
    )

    parser.epilog = (
        "Exit codes: 0 success, 1 generic error, 2 usage error, 130 interrupt. "
        "Use the JSON code field for granular semantics."
    )
    args = parser.parse_args()
    emitter = _build_emitter(args)
    started_at = time.monotonic()

    # -----------------------------------------------------------------------
    # Device listing / mic test (no API needed)
    # -----------------------------------------------------------------------
    if args.list_devices:
        try:
            from audio_processing.recorder import list_audio_devices

            list_audio_devices()
        except ImportError as e:
            emitter.error(_error_data("MISSING_DEPENDENCY", "dependency", str(e)))
            return 1
        return 0

    if args.test_mic:
        try:
            from audio_processing.recorder import AudioRecorder

            recorder = AudioRecorder(device_index=args.device)
            return 0 if recorder.test_microphone(device_id=args.device) else 1
        except ImportError as e:
            emitter.error(_error_data("MISSING_DEPENDENCY", "dependency", str(e)))
            return 1

    # -----------------------------------------------------------------------
    # Determine audio source
    # -----------------------------------------------------------------------
    if args.file:
        if not os.path.exists(args.file):
            message = f"Audio file not found: {args.file}"
            _emit_failure(
                emitter,
                args,
                "FILE_NOT_FOUND",
                "input",
                message,
                started_at,
                audio_file=args.file,
            )
            return ExitCode.NO_INPUT
        audio_file = args.file
        emitter.log(f"Transcribing existing file: {audio_file}")
    else:
        # Record new audio
        if args.legacy:
            print(
                "⚠️  WARNING: --legacy uses the old recorder without device-failure recovery. "
                "Audio may be lost on device disconnection. Use the default robust recorder instead.",
                file=sys.stderr,
            )

            # Use standard recorder
            try:
                from audio_processing.recorder import AudioRecorder
            except ImportError as e:
                emitter.error(_error_data("MISSING_DEPENDENCY", "dependency", str(e)))
                return 1

            recorder = AudioRecorder(device_index=args.device)
            if not recorder.recording_method:
                emitter.error(
                    _error_data("NO_MIC", "recording", "No working microphone found")
                )
                return 1

            emitter.log(recorder.get_recording_info())
            audio_file = recorder.record_until_q(args.record_output)
            if not audio_file:
                emitter.error(_error_data("NO_MIC", "recording", "Recording failed"))
                return 1

        else:
            # Use robust recorder with auto-chunking and error recovery
            try:
                from audio_processing.robust_recorder import record_robust
            except ImportError as e:
                emitter.error(_error_data("MISSING_DEPENDENCY", "dependency", str(e)))
                return 1

            chunk_files = record_robust(
                output_dir=os.path.dirname(args.record_output) or "recordings",
                chunk_minutes=args.chunk_minutes,
                max_duration_minutes=args.max_duration,
                device_index=args.device,
                merge_after=True,  # Merge chunks into single file
            )

            if not chunk_files:
                emitter.error(
                    _error_data("DEVICE_DISCONNECTED", "recording", "Recording failed")
                )
                return 1

            audio_file = str(chunk_files[0])  # Use merged file
            emitter.log(f"Final recording: {audio_file}")

    # -----------------------------------------------------------------------
    # Build STT client
    # -----------------------------------------------------------------------
    provider_capabilities = _provider_capabilities(args.provider)
    if args.word_timestamps and not provider_capabilities["word_timestamps"]:
        message = f"Provider '{args.provider}' does not support word timestamps"
        _emit_failure(
            emitter,
            args,
            "CAPABILITY_UNSUPPORTED",
            "usage",
            message,
            started_at,
            audio_file,
            provider_capabilities,
        )
        return ExitCode.USAGE

    try:
        client = _build_client(args.provider, emitter)
    except (ValueError, ImportError) as e:
        code = "API_KEY_MISSING" if "KEY" in str(e).upper() else "MISSING_DEPENDENCY"
        category = "auth" if code == "API_KEY_MISSING" else "dependency"
        _emit_failure(emitter, args, code, category, str(e), started_at, audio_file)
        return ExitCode.NO_PERMISSION if category == "auth" else ExitCode.SOFTWARE

    from audio_processing.utils import get_audio_duration

    duration = get_audio_duration(audio_file)
    try:
        resolved_model = client._resolve_model(args.model, args.translate)
    except ValueError as error:
        _emit_failure(
            emitter,
            args,
            "MODEL_NOT_FOUND",
            "usage",
            str(error),
            started_at,
            audio_file,
            client.CAPABILITIES,
        )
        return ExitCode.USAGE
    emitter.start(
        {
            "command": "translate" if args.translate else "transcribe",
            "mode": "translate" if args.translate else "transcribe",
            "provider": args.provider,
            "model": resolved_model,
            "audio": os.path.abspath(audio_file),
            "duration_seconds": duration,
        }
    )
    if args.translate and args.model and args.model != resolved_model:
        emitter.warning(
            "MODEL_SWITCHED",
            f"Model '{args.model}' does not support translation; using {resolved_model}",
        )

    # -----------------------------------------------------------------------
    # Transcribe / translate
    # -----------------------------------------------------------------------
    action = "Translating" if args.translate else "Transcribing"
    emitter.log(f"{action} with provider={args.provider} ...")

    try:
        if args.translate:
            result = client.translate(
                audio_file,
                model=args.model,
                output_file=args.output_file,
                verbose=args.timestamps,
            )
        else:
            result = client.transcribe(
                audio_file,
                model=args.model,
                output_file=args.output_file,
                language=args.language,
                verbose=args.timestamps,
            )
    except KeyboardInterrupt:
        status = "partial" if emitter.segments_emitted else "error"
        emitter.error(_error_data("USER_INTERRUPT", "cancelled", "Interrupted by user"))
        emitter.end(_end_data(status, "USER_INTERRUPT", args, emitter, started_at))
        return ExitCode.USER_INTERRUPT
    except Exception:
        _emit_failure(
            emitter,
            args,
            "PROVIDER_TEMPFAIL",
            "provider",
            "Provider request failed",
            started_at,
            audio_file,
            client.CAPABILITIES,
            retryable=True,
        )
        return ExitCode.TEMPORARY_FAILURE

    if not result:
        warning_codes = {warning.code for warning in emitter.warnings}
        if "ALL_CHUNKS_SILENT" in warning_codes:
            _emit_no_speech(
                emitter,
                args,
                client,
                resolved_model,
                audio_file,
                duration,
                started_at,
            )
            return ExitCode.OK
        _emit_failure(
            emitter,
            args,
            "PROVIDER_TEMPFAIL",
            "provider",
            "No transcription chunks succeeded",
            started_at,
            audio_file,
            client.CAPABILITIES,
            retryable=True,
        )
        return ExitCode.TEMPORARY_FAILURE

    if not args.no_clipboard:
        try:
            import pyperclip

            pyperclip.copy(result.text)
            emitter.log("Copied to clipboard.")
        except Exception:
            emitter.warning("CLIPBOARD_FAILED", "Could not copy to clipboard")

    elapsed = time.monotonic() - started_at
    contract_result = Result(
        kind="translation" if args.translate else "transcription",
        text=result.text,
        language_detected=result.detected_language,
        source_language=None,
        segments=[
            Segment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                offset_seconds=segment.offset_seconds,
                text=segment.text,
                tokens=segment.tokens,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
            )
            for segment in result.segments
        ],
    )
    json_path = None
    if args.timestamps and result.segments and result.output_file:
        json_path = os.path.splitext(result.output_file)[0] + ".json"
    outputs = Outputs(
        txt=result.output_file,
        srt=result.srt_file,
        json=json_path,
    )
    envelope = Envelope(
        request_id=emitter.request_id,
        mode="translate" if args.translate else "transcribe",
        status=Status.OK,
        code="OK",
        message=f"{action} complete: {len(result.segments)} segments",
        provider=args.provider,
        model=resolved_model,
        capabilities=Capabilities(**client.CAPABILITIES),
        input=AudioInput(
            audio_file=os.path.abspath(audio_file),
            duration_seconds=duration,
            size_bytes=os.path.getsize(audio_file),
        ),
        result=contract_result,
        outputs=outputs,
        warnings=emitter.warnings,
        timing=Timing(transcribe_secs=elapsed, total_secs=elapsed),
        provider_meta=result.provider_meta,
    )

    if json_path:
        _write_envelope_json(json_path, envelope)
        result.json_file = json_path

    emitter.end(
        {
            "status": "ok",
            "code": "OK",
            "mode": envelope.mode,
            "segments_total": len(result.segments),
            "chars_total": len(result.text),
            "outputs": outputs.model_dump(by_alias=True, exclude_none=True),
            "timing": envelope.timing.model_dump(),
            "warnings": [w.model_dump(exclude_none=True) for w in emitter.warnings],
        }
    )
    emitter.finalize(envelope)

    return 0


def _build_emitter(args) -> Emitter:
    """Select the requested output renderer from parsed CLI arguments."""
    mode = "json" if args.json else "ndjson" if args.ndjson else None
    mode = mode or ("plain" if args.plain or args.no_color else args.output)
    mode = mode or ("human" if sys.stdout.isatty() else "ndjson")
    emitter_class = {
        "human": HumanEmitter,
        "plain": PlainEmitter,
        "json": JSONEmitter,
        "ndjson": NDJSONEmitter,
    }[mode]
    return emitter_class(quiet=args.quiet)


def _provider_capabilities(provider: str) -> dict[str, bool]:
    """Read provider capabilities without constructing an API client."""
    if provider == "groq":
        from api.groq_client import GroqWhisperClient

        return GroqWhisperClient.CAPABILITIES
    from api.modelos_client import ModelosSTTClient

    return ModelosSTTClient.CAPABILITIES


def _error_data(
    code: str,
    category: ErrorCategory,
    message: str,
    retryable: bool = False,
) -> dict:
    """Build the stable event payload for one structured error."""
    return {
        "code": code,
        "category": category,
        "retryable": retryable,
        "message": message,
    }


def _end_data(status, code, args, emitter, started_at) -> dict:
    """Build the mandatory NDJSON end-event payload."""
    return {
        "status": status,
        "code": code,
        "mode": "translate" if args.translate else "transcribe",
        "segments_total": emitter.segments_emitted,
        "chars_total": 0,
        "outputs": {},
        "timing": {
            "record_secs": 0.0,
            "transcribe_secs": 0.0,
            "total_secs": time.monotonic() - started_at,
        },
        "warnings": [w.model_dump(exclude_none=True) for w in emitter.warnings],
    }


def _write_envelope_json(path: str, envelope: Envelope) -> None:
    """Atomically write the exact JSON envelope used by JSON stdout."""
    content = (
        json.dumps(
            envelope_payload(envelope),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    temp_path = f"{path}.tmp.{envelope.request_id}"
    with open(temp_path, "w", encoding="utf-8") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _emit_failure(
    emitter: Emitter,
    args,
    code: str,
    category: ErrorCategory,
    message: str,
    started_at: float,
    audio_file: str | None = None,
    capabilities: dict[str, bool] | None = None,
    retryable: bool = False,
) -> None:
    """Emit a complete failure stream and JSON envelope."""
    error_data = _error_data(code, category, message, retryable)
    emitter.error(error_data)
    emitter.end(_end_data("error", code, args, emitter, started_at))
    path = audio_file or getattr(args, "file", None) or ""
    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    envelope = Envelope(
        request_id=emitter.request_id,
        mode="translate" if args.translate else "transcribe",
        status=Status.ERROR,
        code=code,
        message=message,
        provider=args.provider,
        model=args.model or "",
        capabilities=Capabilities(
            **(
                capabilities
                or {
                    "word_timestamps": False,
                    "segment_timestamps": False,
                    "language_detection": False,
                    "quality_metrics": False,
                    "speaker_diarization": False,
                }
            )
        ),
        input=AudioInput(
            audio_file=os.path.abspath(path) if path else "",
            duration_seconds=0.0,
            size_bytes=size,
        ),
        outputs=Outputs(),
        warnings=emitter.warnings,
        error=Error(
            code=code,
            category=category,
            retryable=retryable,
            message=message,
        ),
        timing=Timing(total_secs=time.monotonic() - started_at),
    )
    emitter.finalize(envelope)


def _emit_no_speech(
    emitter: Emitter,
    args,
    client,
    model: str,
    audio_file: str,
    duration: float,
    started_at: float,
) -> None:
    """Emit the valid empty-result outcome for all-silent audio."""
    elapsed = time.monotonic() - started_at
    envelope = Envelope(
        request_id=emitter.request_id,
        mode="translate" if args.translate else "transcribe",
        status=Status.OK,
        code="NO_SPEECH_DETECTED",
        message="No speech detected",
        provider=args.provider,
        model=model,
        capabilities=Capabilities(**client.CAPABILITIES),
        input=AudioInput(
            audio_file=os.path.abspath(audio_file),
            duration_seconds=duration,
            size_bytes=os.path.getsize(audio_file),
        ),
        result=Result(
            kind="translation" if args.translate else "transcription",
            text="",
        ),
        outputs=Outputs(),
        warnings=emitter.warnings,
        timing=Timing(transcribe_secs=elapsed, total_secs=elapsed),
    )
    emitter.end(
        {
            "status": "ok",
            "code": "NO_SPEECH_DETECTED",
            "mode": envelope.mode,
            "segments_total": 0,
            "chars_total": 0,
            "outputs": {},
            "timing": envelope.timing.model_dump(),
            "warnings": [
                warning.model_dump(exclude_none=True) for warning in emitter.warnings
            ],
        }
    )
    emitter.finalize(envelope)


if __name__ == "__main__":
    sys.exit(main())
