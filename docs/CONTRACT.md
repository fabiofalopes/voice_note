# docs/CONTRACT.md — voice_note Output Contract v1.0

> **The output contract is the heart of voice_note v1.0.**
> This document specifies the JSON envelope, NDJSON event stream, exit codes, versioning rules, provider normalisation, and Stream A Definition of Done.
> For agent operating rules: see [../AGENTS.md](../AGENTS.md).
> For project history/state/decisions: see [../MEMORY.md](../MEMORY.md).

**`schema_version: "1.0"`** — Draft. Frozen when Stream A DoD passes.

---

## 1. Output modes (selected by flag, auto-detected by TTY)

| Mode | Flag | stdout | stderr | Use case |
|---|---|---|---|---|
| **Human** (default in TTY) | `--output human` or auto | ANSI progress, banner, preview, file paths | warnings, errors | interactive use via `vn` |
| **Plain human** | `--plain` (or `--output human --no-color`) | same content as Human, no ANSI escapes | warnings, errors | CI logs, redirected output that wants readability |
| **JSON batch** | `--json` (or `--output json`) | **single** JSON document (the §2 envelope), written once at process end | human logs (still visible if TTY) | short atomic commands, `jq` consumers |
| **NDJSON stream** (default in pipe) | `--ndjson` (or `--output ndjson`) | one JSON object per line — pure, no interleaved text | human logs (still visible if TTY) | long-running commands, UI consumers, streaming pipelines |

### 1.1 Auto-detection rules (in order, first match wins)

1. `--json` → JSON batch mode (single document, no streaming)
2. `--ndjson` → NDJSON stream mode
3. `--plain` / `--no-color` → Plain human mode
4. `--output <mode>` → that mode
5. No flag → `sys.stdout.isatty()` true → Human mode; false → NDJSON stream mode

**`--quiet`** suppresses stderr output entirely (stdout unaffected). Useful in CI.

### 1.2 Pipe-closure contract

Consumers that read NDJSON must read until the `end` event. Truncated streams (e.g., `vn … | head`) are inherently incomplete — the contract makes no guarantees about partial reads. The `end` event is the commit point.

---

## 2. JSON status document (batch mode)

This shape is emitted **both** to stdout in `--json` mode AND to the `<audio_basename>_transcription.json` file written to disk (see §7). **One schema, two destinations.** Byte-identical modulo `request_id`.

```json
{
  "schema_version": "1.0",
  "tool_version": "1.0.0",
  "request_id": "5a3e2c8f-1b2d-4e5f-8a9c-0d1e2f3a4b5c",
  "mode": "transcribe",
  "status": "ok",
  "code": "OK",
  "message": "Transcribed 47 segments in 12.34s",
  "provider": "groq",
  "model": "whisper-large-v3",
  "capabilities": {
    "word_timestamps": true,
    "segment_timestamps": true,
    "language_detection": true,
    "quality_metrics": true,
    "speaker_diarization": false
  },
  "input": {
    "audio_file": "/abs/path/recording.wav",
    "duration_seconds": 123.45,
    "size_bytes": 9876543
  },
  "result": {
    "kind": "transcription",
    "text": "Full transcription text concatenated from all segments.",
    "language_detected": "pt",
    "source_language": null,
    "segments": [
      {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": 2.31,
        "offset_seconds": 0.0,
        "text": "...segment text...",
        "tokens": [50364, 543],
        "temperature": 0.0,
        "avg_logprob": -0.42,
        "compression_ratio": 1.6,
        "no_speech_prob": 0.03,
        "words": null,
        "speaker": null
      }
    ],
    "postprocessed": null
  },
  "outputs": {
    "txt": "/abs/path/recording_transcription.txt",
    "srt": "/abs/path/recording_transcription.srt",
    "json": "/abs/path/recording_transcription.json"
  },
  "warnings": [
    {
      "level": "warning",
      "code": "LOW_CONFIDENCE_SEGMENTS",
      "chunk_index": 4,
      "detail": "2 segments below threshold -0.8"
    }
  ],
  "error": null,
  "timing": {
    "record_secs": 0.0,
    "transcribe_secs": 12.34,
    "total_secs": 12.40
  },
  "provider_meta": {
    "x_groq": {"id": "req_abc123"}
  }
}
```

### 2.1 Top-level field rules (single, unambiguous definition each)

- **`schema_version`** (mandatory, string, semver): version of **this contract**. v1.0 = `"1.0"`. Bumps per §6. Renamed from `version` in earlier drafts to disambiguate from `tool_version`.
- **`tool_version`** (mandatory, string): version of the `vn` package itself (separate concern).
- **`request_id`** (mandatory, UUIDv4 string): generated at process start. Same value in every NDJSON event for correlation.
- **`mode`** (mandatory, enum `"transcribe" | "translate"`): without this, a consumer cannot distinguish a translation of Portuguese (text is English) from a transcription of English.
- **`status`** (mandatory, enum `"ok" | "error" | "partial"`):
  - `"ok"` = completed successfully
  - `"error"` = failed; `result` omitted
  - `"partial"` = some chunks succeeded, some failed (still produces output); also used when Ctrl+C interrupts with partial data saved
- **`code`** (mandatory, symbolic string): one code per process. See §5.
- **`message`** (mandatory, string, may be empty when `status=ok`): one-line human summary.
- **`provider`** (mandatory, string): free-form string owned by the provider registry. Not a hardcoded enum; the contract does not validate against a fixed list.
- **`model`** (mandatory, string): the actual model name as the provider received it.
- **`capabilities`** (mandatory, object): provider capability declaration — see §3.
- **`input`** (mandatory, object): `audio_file`, `duration_seconds`, `size_bytes`.
- **`result`** (conditional, object): omitted when `status=error`. Present otherwise.
- **`outputs`** (mandatory, object): always present; `srt` and `json` keys omitted unless `--timestamps`. Only lists files successfully promoted by the transactional write (§8).
- **`warnings`** (mandatory, array, possibly empty): structured. See §2.4.
- **`error`** (conditional, object or null): structured error object when `status=error` or `status=partial`. `null` otherwise. See §4.
- **`timing`** (mandatory, object): always present. All three fields; `record_secs=0.0` for transcribe-existing-file.
- **`provider_meta`** (optional, object, default `{}`): opaque blob for provider-specific extension fields (e.g., `{"x_groq": {"id": "req_…"}}` for Groq). **Consumers MUST NOT depend on specific keys.** Promotion to a typed first-class field requires (a) 3+ provider implementations AND (b) demonstrated consumer need.

### 2.2 `result.kind` discriminator (mandatory, enum)

- `"transcription"` — `mode=transcribe`. `language_detected` = detected/source language.
- `"translation"` — `mode=translate`. `language_detected` = `"en"` (target). `source_language` = detected source (if known, else `null`).
- `"diarization"` — reserved for future WhisperX/diarize pipeline (not emitted in v1.0).

The discriminator exists so consumers can switch on shape without parsing `mode`. Adding a new `kind` is a **minor** version bump (additive). **Note:** `"analysis"` was considered and dropped from v1.0 — no concrete shape exists. Add only when a real LLM-postprocessing pipeline defines one.

`result.postprocessed` is `null` in v1.0. Reserved for future LLM postprocessing. Consumers must tolerate its presence without breaking.

### 2.3 Segment schema (provider-verified, empirically grounded)

| Field | Type | Status | Notes |
|---|---|---|---|
| `id` | `int` | nullable; **defaulted at emission** | Sequential index assigned at emission time if null. Documented structural default (no warning). |
| `seek` | `int` | nullable; **defaulted at emission** | Set to `0` if null. Internal Whisper frame offset; not contract-meaningful. |
| `start`, `end` | `float` | required | Always present. **For modelos: `end` can hallucinate** (empirically observed `end=30.19` on 2.28s audio) — contract MUST clamp `end` to `input.duration_seconds` at emission time and emit `TIMESTAMP_CLAMPED` warning. |
| `offset_seconds` | `float` | required | Chunk-start offset in source audio. Always `0.0` for single-chunk; `chunk_start` for multi-chunk. `start + offset_seconds` = global timestamp. Resolves ROADMAP "per-chunk offsets" tech debt without breaking raw segment data. |
| `text` | `string` | required | Always present. |
| `tokens` | `int[]` | nullable; **defaulted at emission** | Set to `[]` if null. Documented structural default (no warning). |
| `temperature` | `float` | nullable; **defaulted at emission** | Set to `0.0` if null. Documented structural default (no warning). |
| `avg_logprob` | `float` or `null` | **null preserved** | Quality metric, not structural. **null = "unknown," not "0.0".** Do NOT fabricate a default. |
| `compression_ratio` | `float` or `null` | **null preserved** | Quality metric. **null = "unknown," not "1.6".** |
| `no_speech_prob` | `float` or `null` | **null preserved** | Quality metric. **null = "unknown," not "0.0".** |
| `words` | `array` or `null` | optional | Array of `{word, start, end}` when supported and requested. Null otherwise (e.g., modelos). Per-word confidence NOT in the contract (provider divergence: mlx-whisper uses `probability`, WhisperX uses `score`, APIs return nothing). |
| `speaker` | `string` or `null` | optional | Reserved slot. `null` in v1.0. Populated only by future WhisperX/diarize provider. |

**The single null-handling rule, stated once:**

> *Null fields indicate the provider did not return that value. Consumers should treat null as "unknown," not as any particular default.*

This applies to `avg_logprob`, `compression_ratio`, `no_speech_prob`, `words`, `speaker`. Structural fields (`id`, `seek`, `tokens`, `temperature`) receive documented defaults because their semantics are valid as defaults. The `end` field is the one exception: it can be *factually wrong* (not merely absent), so we clamp it and warn.

### 2.4 Warnings array (always present, possibly `[]`)

Each entry:
```json
{
  "level": "warning",
  "code": "LOW_CONFIDENCE_SEGMENTS | SILENT_CHUNK_SKIPPED | PARTIAL_RESUME | PROVIDER_FIELD_NULL | TIMESTAMP_CLAMPED",
  "chunk_index": 4,
  "detail": "human-readable explanation"
}
```

- `PROVIDER_FIELD_NULL` — provider returned null for `avg_logprob` / `compression_ratio` / `no_speech_prob` (quality metrics); preserved as null per the rule above. Warning makes the absence observable, not silent.
- `TIMESTAMP_CLAMPED` — `end` was clamped to `input.duration_seconds` because the provider hallucinated a larger value.
- `LOW_CONFIDENCE_SEGMENTS` — segment `avg_logprob` below threshold (only emitted when provider returned a real value).
- `SILENT_CHUNK_SKIPPED` — chunk dropped due to silence detection (only when provider returned a real `no_speech_prob`).

---

## 3. Provider capabilities object (declarative, fail-fast)

Every provider declares its capabilities at registration time:

```python
capabilities = {
    "word_timestamps": bool,      # Groq: True, modelos: False
    "segment_timestamps": bool,   # both: True
    "language_detection": bool,   # both: True
    "quality_metrics": bool,      # Groq: True, modelos: False (all null)
    "speaker_diarization": bool,  # neither in v1.0: False
}
```

**Capability object in envelope:**
```json
{
  "provider": "modelos",
  "capabilities": {
    "word_timestamps": false,
    "segment_timestamps": true,
    "language_detection": true,
    "quality_metrics": false,
    "speaker_diarization": false
  }
}
```

**Rules:**
- CLI checks capabilities **before** sending request.
- If user requests a feature the active provider doesn't support (e.g., `--word-timestamps` on modelos) → **fail fast** with exit code 64 (`EX_USAGE`), `code: USAGE_ERROR`, and message: `"Provider 'modelos' does not support word timestamps"`.
- **No post-hoc warnings for unsupported features.** The failure happens before any audio is sent.
- Capabilities declared in provider registration (Pattern 1 class or Pattern 2 config entry — Stream C).

---

## 4. Structured error object

When `status` is `"error"` or `"partial"`, the `error` field carries a structured object:

```json
{
  "error": {
    "code": "PROVIDER_429_EXHAUSTED",
    "category": "provider",
    "retryable": true,
    "retry_after_seconds": 12.5,
    "message": "Provider rate limit retry budget exhausted",
    "cause": {
      "provider_http_status": 429,
      "provider_request_id": "req_..."
    }
  }
}
```

**Field rules:**
- `code` — symbolic string (see §5). Stable across versions.
- `category` — coarse enum, stable: `"usage" | "input" | "recording" | "dependency" | "auth" | "provider" | "storage" | "internal" | "cancelled"`.
- `retryable` — operational assertion, not a guess. `true` only when retry is meaningfully possible.
- `retry_after_seconds` — optional, only when known (e.g., Groq 429 wait-time parser).
- `message` — safe, stable, human-readable, **no secret material**.
- `cause` — optional object with provider-specific debugging details (HTTP status, provider request id). Raw provider response bodies are opt-in and never appear in default output (may contain unexpected information).

---

## 5. Exit codes (POSIX + sysexits.h aligned)

Every error path in `cli.py`, `base_client.py`, `recorder.py`, `robust_recorder.py` MUST map to exactly one of these codes. Currently `cli.py` returns `1` on every error path — Stream A replaces each with a specific code.

### 5.1 Public exit codes (documented in `--help` and README)

| Exit | Meaning |
|------|---------|
| `0` | Success (including `NO_SPEECH_DETECTED`; **not** including Ctrl+C) |
| `1` | Generic error — last resort only; every normal path must map to a specific code below |
| `2` | Usage error (bad flags, missing args, unknown provider, capability mismatch) |
| `130` | User interrupt (Ctrl+C / SIGINT) — POSIX convention 128 + 2 |

**The public set is deliberately minimal.** Internal codes below carry granular semantics; the NDJSON `code` field exposes them to scripts that need to branch.

### 5.2 Internal exit codes (used in code, NOT in `--help`; exposed via JSON/NDJSON `code` field)

| Exit | Symbolic name | When | Maps to JSON `code` |
|---|---|---|---|
| 0 | `EX_OK` | Success | `OK`, `NO_SPEECH_DETECTED` |
| 64 | `EX_USAGE` | Bad CLI usage / capability mismatch / unknown provider | `USAGE_ERROR`, `PROVIDER_NOT_REGISTERED`, `CAPABILITY_UNSUPPORTED` |
| 65 | `EX_DATAERR` | Audio malformed / duration undetectable | `AUDIO_INVALID`, `AUDIO_DURATION_UNDETECTABLE` |
| 66 | `EX_NOINPUT` | Input file not found / mic unavailable / device disconnected with no chunks saved | `FILE_NOT_FOUND`, `NO_MIC`, `DEVICE_DISCONNECTED` |
| 69 | `EX_UNAVAILABLE` | Provider exhausted retries / all providers down / ffmpeg missing / local model load failed | `PROVIDER_429_EXHAUSTED`, `ALL_PROVIDERS_FAILED`, `FFMPEG_MISSING`, `MODEL_LOAD_FAILED` |
| 70 | `EX_SOFTWARE` | Internal bug / unhandled exception / missing Python dependency | `INTERNAL_ERROR`, `MISSING_DEPENDENCY` |
| 73 | `EX_CANTCREAT` | Cannot write output file (disk full, permissions) | `OUTPUT_WRITE_FAILED` |
| 75 | `EX_TEMPFAIL` | Transient provider error (retry-able but exhausted) | `PROVIDER_TEMPFAIL` |
| 76 | `EX_NOPERM` | API key missing / invalid / 403 forbidden | `API_KEY_MISSING`, `API_KEY_INVALID` |
| 130 | (signal-derived) | Ctrl+C / SIGINT — **regardless of whether partial data was saved** | `USER_INTERRUPT` |
| 1 | `generic` | Last resort only. Must not appear in normal paths. | `UNCATEGORIZED` |

### 5.3 Special rules (non-negotiable)

- **Ctrl+C during recording or transcription** → exit `130`, `code: USER_INTERRUPT`. Always. Whether or not partial data was saved. POSIX convention: 128 + SIGINT(2). Saving a useful recovery artifact is a success of the *reliability subsystem*, not successful completion of the *request*. The JSON envelope's `status` field is `"partial"` when partial data exists; `"error"` otherwise.
- **All chunks silent** (every chunk hit `chunk_skipped` with `reason: silence`) → exit `0`, `code: NO_SPEECH_DETECTED`, `status: "ok"`. This is a valid outcome, not a failure. Emit `warnings[]` entry with `code: ALL_CHUNKS_SILENT`.
- **Never surface raw Python tracebacks to stdout/stderr.** Catch, log full traceback to stderr only when `--verbose` is set, exit with the mapped code.
- **Symbolic name MUST appear in the `code` field of the JSON/NDJSON output.** The number is for the shell; the string is for parsers.
- **Document the public set (0, 1, 2, 130) in `--help` and in the README under "Exit codes".** Internal codes are implementation detail.

### 5.4 Provider key error mapping

| HTTP / SDK error | `code` | exit |
|---|---|---|
| 401 / 403 (key invalid/forbidden) | `API_KEY_INVALID` | 76 |
| Missing env var when constructing client | `API_KEY_MISSING` | 76 |
| 404 model not found | `MODEL_NOT_FOUND` | 64 |
| 429 (rate limit) exhausted after `max_retries` | `PROVIDER_429_EXHAUSTED` | 69 |
| 5xx after all retries | `PROVIDER_TEMPFAIL` | 75 |
| Network timeout / DNS failure | `PROVIDER_UNREACHABLE` | 69 |

---

## 6. Contract versioning & stability

- **v1.0** is the initial contract defined here. Once shipped and proven, it is frozen.
- **Additive changes** (new event `type` values, new optional `result.kind` values, new optional fields) bump minor (`1.1`, `1.2`).
- **Breaking changes** (renamed fields, removed values, changed shapes, new mandatory fields) bump major (`2.0`) and require a migration note.
- **Consumers MUST tolerate** unknown event `type` values, unknown `result.kind` values, and unknown warning codes by ignoring them.
- **Producers MUST NOT change** the meaning or type of an existing field within a major version.
- The contract lives in `src/contract.py` (to be created in Stream A) as **Pydantic v2 models** + JSON Schema export, and is the single source of truth for both JSON and NDJSON output and for the on-disk `.json` file.

| `schema_version` | Status | Notes |
|---|---|---|
| `1.0` | Draft (this doc) | Initial contract. Will be frozen when Stream A DoD passes. |

---

## 7. On-disk JSON file shape (unification rule)

**Critical**: the `.json` file written to disk at `<audio_basename>_transcription.json` (currently produced by `BaseSTTClient._save_json` in `base_client.py` L500–530) MUST use the **exact same schema** as the §2 JSON document emitted to stdout in `--json` mode. One schema, two destinations.

This is a deliberate change from the current implementation, which writes a stripped-down shape (`{text, language, segments[]}`) that lacks `schema_version`, `request_id`, `warnings`, `timing`, `mode`, etc.

Stream A's DoD includes: rewrite `BaseSTTClient._save_json` to emit the full §2 envelope. The on-disk `.srt` and `.txt` files remain raw Whisper-shaped (they're format-specific outputs, not contract surfaces).

---

## 8. Transactional write rules (atomic output files)

All output files (`.txt`, `.srt`, `.json`) follow this exact sequence:

1. Write to temp file in same directory: `<final_name>.tmp.<request_id>`
2. `fsync()` the temp file to flush to disk
3. **Atomic** rename via `os.replace(temp, final)` — atomic on POSIX
4. Only **after** successful rename, record the path in `outputs.*` of the JSON envelope

**Invariants:**
- If the process crashes before rename, no partial file exists at the final path. The temp file is left behind (cleaned on next run or by OS).
- `outputs.*` in the JSON envelope ONLY lists files that were successfully promoted (renamed). A file in `outputs.*` is a commit guarantee.
- Partial completions write a separate `<audio_basename>.partial.json` manifest, never a normal final artifact that happens to contain partial data.

---

## 9. NDJSON event stream (streaming mode)

One JSON object per line, no extra whitespace, no interleaved text on stdout. **v1.0 ships with 5 event types.** Other event types from prior drafts are deferred (see §10).

### 9.1 Event vocabulary (v1.0 — 5 types)

| `type` | level | When emitted | Mandatory `data` keys |
|---|---|---|---|
| `start` | info | Process start, after argument validation | `command`, `mode`, `provider`, `audio`, `duration_seconds` |
| `segment` | info | One segment parsed and accumulated | `chunk_index`, `offset_seconds`, `id`, `start`, `end`, `text`, `avg_logprob`, `no_speech_prob` |
| `warning` | warning | Non-fatal issue (null field, clamped timestamp, low confidence, silent chunk, provider field null) | `code`, `chunk_index` (optional), `detail` |
| `error` | error | Fatal issue — includes structured error object | `code`, `category`, `retryable`, `message`, `cause` (optional) |
| `end` | info | **Always last line.** Final summary. | `status`, `code`, `mode`, `segments_total`, `chars_total`, `outputs`, `timing`, `warnings` |

### 9.2 Every event carries these envelope fields

- `schema_version` (string) — same as JSON document
- `request_id` (UUIDv4 string) — same across all events in one process
- `event_id` (UUIDv4 string) — unique per event
- `sequence` (int, monotonic starting at 0) — provides ordering even when wall-clock time moves
- `time` (ISO 8601 UTC string) — wall-clock timestamp
- `level` (enum `info | warning | error`)
- `type` (the event type from the table above)
- `data` (object — type-specific per the table above)

### 9.3 Complete NDJSON stream example (Groq, transcribe, success)

```json
{"schema_version":"1.0","request_id":"5a3e2c8f-1b2d-4e5f-8a9c-0d1e2f3a4b5c","event_id":"e0001","sequence":0,"time":"2026-07-19T14:30:00.123Z","level":"info","type":"start","data":{"command":"transcribe","mode":"transcribe","provider":"groq","model":"whisper-large-v3","audio":"/abs/path.wav","duration_seconds":123.45}}
{"schema_version":"1.0","request_id":"5a3e2c8f-1b2d-4e5f-8a9c-0d1e2f3a4b5c","event_id":"e0002","sequence":1,"time":"2026-07-19T14:30:01.456Z","level":"info","type":"segment","data":{"chunk_index":0,"offset_seconds":0.0,"id":0,"start":0.0,"end":2.31,"text":"Okay.","avg_logprob":-0.42,"no_speech_prob":0.03}}
{"schema_version":"1.0","request_id":"5a3e2c8f-1b2d-4e5f-8a9c-0d1e2f3a4b5c","event_id":"e0003","sequence":2,"time":"2026-07-19T14:30:01.789Z","level":"info","type":"segment","data":{"chunk_index":0,"offset_seconds":0.0,"id":1,"start":2.31,"end":5.42,"text":"Testing recording.","avg_logprob":-0.38,"no_speech_prob":0.02}}
{"schema_version":"1.0","request_id":"5a3e2c8f-1b2d-4e5f-8a9c-0d1e2f3a4b5c","event_id":"e0004","sequence":3,"time":"2026-07-19T14:30:12.012Z","level":"warning","type":"warning","data":{"code":"LOW_CONFIDENCE_SEGMENTS","chunk_index":1,"detail":"2 segments below threshold -0.8"}}
{"schema_version":"1.0","request_id":"5a3e2c8f-1b2d-4e5f-8a9c-0d1e2f3a4b5c","event_id":"e0005","sequence":4,"time":"2026-07-19T14:30:12.345Z","level":"info","type":"end","data":{"status":"ok","code":"OK","mode":"transcribe","segments_total":47,"chars_total":12345,"outputs":{"txt":"/abs/path_transcription.txt","srt":"/abs/path_transcription.srt","json":"/abs/path_transcription.json"},"timing":{"record_secs":0.0,"transcribe_secs":12.34,"total_secs":12.40},"warnings":[{"level":"warning","code":"LOW_CONFIDENCE_SEGMENTS","chunk_index":1,"detail":"2 segments below threshold -0.8"}]}}
```

### 9.4 Complete NDJSON stream example (modelos, transcribe, with null fields + clamped timestamp)

```json
{"schema_version":"1.0","request_id":"b7c1d2e3-...","event_id":"e0001","sequence":0,"time":"2026-07-19T14:31:00.000Z","level":"info","type":"start","data":{"command":"transcribe","mode":"transcribe","provider":"modelos","model":"stt-large-v3-turbo","audio":"/abs/short.wav","duration_seconds":2.28}}
{"schema_version":"1.0","request_id":"b7c1d2e3-...","event_id":"e0002","sequence":1,"time":"2026-07-19T14:31:00.234Z","level":"warning","type":"warning","data":{"code":"TIMESTAMP_CLAMPED","chunk_index":0,"detail":"end 30.19s clamped to input duration 2.28s (modelos backend hallucination)"}}
{"schema_version":"1.0","request_id":"b7c1d2e3-...","event_id":"e0003","sequence":2,"time":"2026-07-19T14:31:00.267Z","level":"warning","type":"warning","data":{"code":"PROVIDER_FIELD_NULL","chunk_index":0,"detail":"Provider returned null for avg_logprob, compression_ratio, no_speech_prob (preserved as null)"}}
{"schema_version":"1.0","request_id":"b7c1d2e3-...","event_id":"e0004","sequence":3,"time":"2026-07-19T14:31:00.278Z","level":"info","type":"segment","data":{"chunk_index":0,"offset_seconds":0.0,"id":0,"start":0.0,"end":2.28,"text":"Okay.","avg_logprob":null,"no_speech_prob":null}}
{"schema_version":"1.0","request_id":"b7c1d2e3-...","event_id":"e0005","sequence":4,"time":"2026-07-19T14:31:00.301Z","level":"info","type":"end","data":{"status":"ok","code":"OK","mode":"transcribe","segments_total":1,"chars_total":5,"outputs":{"txt":"/abs/short_transcription.txt"},"timing":{"record_secs":0.0,"transcribe_secs":0.3,"total_secs":0.31},"warnings":[{"level":"warning","code":"TIMESTAMP_CLAMPED","chunk_index":0,"detail":"end 30.19s clamped to input duration 2.28s"},{"level":"warning","code":"PROVIDER_FIELD_NULL","chunk_index":0,"detail":"Provider returned null for avg_logprob, compression_ratio, no_speech_prob"}]}}
```

### 9.5 Rules

- **Every line is independently JSON-parseable.** No multiline JSON.
- `schema_version`, `request_id`, `event_id`, `sequence` MUST appear on every event.
- The `end` event MUST be the last line and MUST contain the same `status`/`code`/`outputs`/`warnings`/`timing` fields as the §2 JSON document — so a UI can `tail -1` and get the final state.
- **Events after `end` are forbidden in v1.0.** Future postprocess pipelines that want to emit after `end` require a contract minor bump to `1.1` AND a documented rule change here.
- **stdout is pure NDJSON.** All human-facing logs (warnings rendered as text, debug) go to stderr. The structured `warnings[]` array is only in the `end` event and the JSON batch document; individual warning conditions on the NDJSON stream use the typed `warning` event.
- **Consumers MUST tolerate unknown `type` values** by ignoring the line (forward-compatibility — see §6).

---

## 10. Deferred event types (NOT in v1.0)

These were in earlier drafts and are explicitly **deferred**. Add only when a real consumer demonstrates need:

- `recording_progress`, `recording_stopped` — useful for UI progress; add in Stream F
- `chunk_extracted`, `chunk_sent`, `chunk_received` — useful for debugging; add if needed
- `chunk_skipped`, `provider_429`, `provider_error`, `low_confidence`, `provider_field_null`, `partial_saved` — covered by `warning`/`error` in v1.0
- `model_loading`, `model_loaded`, `warmup` — Stream E (MLX). When added, use generic names: `engine_initializing`, `engine_ready`, `engine_warmup_started`, `engine_warmup_completed` (so remote providers with cold starts can reuse them).

---

## 11. Provider normalisation (empirically grounded)

The normalisation layer runs in `BaseSTTClient._parse_response` (or a dedicated `_normalise_segment` helper extracted from it) and applies these rules **before** the segment reaches the contract emitter.

### 11.1 Architectural principle — normalisation as a pure boundary

Provider work splits into three distinct layers:

1. **Transport** — auth, multipart upload, HTTP/client SDK, retries, raw response. Provider-specific.
2. **Decode** — provider response into a provider-specific typed structure. Provider-specific.
3. **Normalise** — provider-specific structure to canonical `Segment`. **Pure, deterministic, no side effects, no global mutation.** Testable without invoking live endpoints.

No provider adapter should directly mutate the global transcript. The merger is a separate deterministic component.

### 11.2 Normalisation rules (single, unambiguous definition per field)

| Field | If null/missing from provider | Action | Warning? |
|---|---|---|---|
| `id` | Assign sequential index based on accumulation order | Structural default | No (documented) |
| `seek` | Set to `0` | Structural default | No (documented) |
| `tokens` | Set to `[]` | Structural default | No (documented) |
| `temperature` | Set to `0.0` | Structural default | No (documented) |
| `avg_logprob` | **Preserve `null`** | Quality metric — "unknown," not "0.0" | Yes (`PROVIDER_FIELD_NULL`) |
| `compression_ratio` | **Preserve `null`** | Quality metric | Yes (`PROVIDER_FIELD_NULL`) |
| `no_speech_prob` | **Preserve `null`** | Quality metric | Yes (`PROVIDER_FIELD_NULL`) |
| `end` (if > `input.duration_seconds`) | **Clamp** to `input.duration_seconds` | Factually incorrect, not absent (modelos hallucination) | Yes (`TIMESTAMP_CLAMPED`) |
| `start`, `text` | Always present | If missing: hard error | — |
| `words` | Preserve `null` | Capability-dependent | No (capability object handles this) |
| `speaker` | Preserve `null` | Reserved future | No |
| `language_detected` + `source_language` | Normalise via `langcodes` to ISO 639-1 lowercase | Format normalisation, not default substitution | No |

**The single design rule, elevated:**

> *Never make a consumer infer whether a field is raw, corrected, defaulted, unavailable, or partial.*

This rule resolves every ambiguity: provider nulls preserved (not fabricated), timestamp clamping observable, word-timestamp support declared (not warned post-hoc), partial saves explicit (transactional write + `status: "partial"`), Ctrl+C ≠ success.

### 11.3 Language normalisation

- Groq returns `"English"` (full word). modelos returns `"en"` (ISO code). OpenAI docs are inconsistent.
- **Contract mandates ISO 639-1 lowercase** (`"en"`, `"pt"`, `"es"`, etc.) for both `language_detected` and `source_language`.
- Implementation: `langcodes` library wrapped in a single function in `src/i18n.py`:

```python
# src/i18n.py
from langcodes import Language

def normalize_language(raw: str | None) -> str | None:
    """Normalise a language string to ISO 639-1 lowercase.

    Args:
        raw: Language string from provider (e.g., "English", "en", "en-US").

    Returns:
        ISO 639-1 lowercase code (e.g., "en", "pt") or None if input is None.

    Raises:
        Never — unknown languages pass through as-is.
    """
    if raw is None:
        return None
    try:
        return Language.get(raw).maximize().language
    except Exception:
        return raw.lower()
```

- Wrapping in a single function isolates the dependency. If `langcodes` proves burdensome in Stream B packaging, fall back to inline dict — the function signature stays identical, no contract change.

### 11.4 Provider extension fields (`provider_meta`)

- Some providers return non-standard top-level fields (Groq: `x_groq: {id: "…"}` for request tracing; OpenAI: `system_fingerprint`; LiteLLM: cost).
- These are preserved verbatim under top-level `provider_meta` in the contract document (`{"provider_meta": {"x_groq": {"id": "…"}}}` for Groq, `{}` for modelos).
- **Consumers MUST NOT depend on specific keys inside `provider_meta`** — debugging/forensics only.
- **Promotion rule:** a provider-specific field is promoted to a typed first-class optional contract field only when (a) it appears in 3+ provider implementations AND (b) a real consumer has demonstrated need.
- Adding/removing keys inside `provider_meta` is a minor (additive) contract change.

---

## Stream A — Definition of Done

**Goal**: Implement this contract in code. Replace every `print()` in `src/cli.py` and `src/api/base_client.py` with contract-aware emitters.

**Files to create:**
- `src/contract.py` — **Pydantic v2 models**: `Envelope`, `Result`, `Segment`, `Word`, `Event`, `Capabilities`, `Error`. Plus enums: `EventType`, `StatusCode`, `ExitCode`. Configuration: `extra="forbid"` on producer-owned stable models; `extra="allow"` only on `provider_meta`. Generate JSON Schema as a committed artifact (`schema/envelope-1.0.json`, `schema/event-1.0.json`).
- `src/i18n.py` — `normalize_language(raw) -> str | None` using `langcodes` (see §11.3).
- `src/emitter.py` — `Emitter` ABC + `HumanEmitter`, `PlainEmitter`, `JSONEmitter`, `NDJSONEmitter`. Human output is a **derived view** of the machine stream — not a separate code path.
- `tests/contract/` — golden-file tests asserting exact NDJSON/JSON shape. Must include a modelos fixture (exercises null-field preservation + `TIMESTAMP_CLAMPED`) and a Groq fixture (exercises `provider_meta` preservation + `x_groq.id`).

**Files to modify:**
- `src/cli.py` — add `--output {human,plain,json,ndjson}`, `--json`, `--ndjson`, `--plain`, `--no-color`, `--quiet` flags; auto-detect TTY; route all output through the chosen `Emitter`; replace binary exit codes with the §5 map.
- `src/api/base_client.py` — replace `print()` calls in `_run_pipeline`, `_iter_chunks`, `_save_*` with emitter events; add per-segment `offset_seconds` from `chunk_start`; add the §11.2 normaliser (null pass-through for quality fields; clamping for `end`); add `langcodes` language normalisation; add capability check (§3) before request; add transactional write (§8).
- `requirements.txt` — add `langcodes>=3.0`, `pydantic>=2.0`, `pytest>=8.0`.

**DoD checklist (17 items):**

1. [ ] `vn recording.wav --ndjson | jq 'last | .code'` returns `"OK"` on success
2. [ ] `vn recording.wav --json > /tmp/out.json && jq '.status' /tmp/out.json` returns `"ok"`
3. [ ] `vn recording.wav --json | jq '.mode'` returns `"transcribe"` (or `"translate"` with `--translate`)
4. [ ] `vn nonexistent.wav --ndjson; echo $?` exits `66` and last NDJSON line has `"code":"FILE_NOT_FOUND"`
5. [ ] `vn recording.wav --provider modelos --ndjson` emits a `warning` event with `"code":"PROVIDER_FIELD_NULL"` for `avg_logprob` + `compression_ratio` + `no_speech_prob`, and the `segment` events carry `"avg_logprob":null` (NOT `0.0`)
6. [ ] `vn recording.wav --ndjson 2>/dev/null` produces valid JSON on every line (stdout is pure)
7. [ ] The on-disk `.json` file at `<audio>_transcription.json` is byte-identical (modulo `request_id`) to the stdout of `vn recording.wav --json`
8. [ ] `vn --word-timestamps --provider modelos recording.wav` fails fast with exit `64` and message `"Provider 'modelos' does not support word timestamps"` (capability check, NOT a warning)
9. [ ] `vn recording.wav --provider modelos --ndjson` (with a short audio) emits `TIMESTAMP_CLAMPED` warning AND `end` field in `segment` equals `input.duration_seconds` (NOT the hallucinated value)
10. [ ] Ctrl+C during transcription exits `130`, last NDJSON line `"code":"USER_INTERRUPT"`, `status:"partial"` if any segments were emitted
11. [ ] All chunks silent → exit `0`, `code: "NO_SPEECH_DETECTED"`, `status:"ok"`
12. [ ] All `float(getattr(...))` calls in `_parse_response` removed — normaliser layer in place
13. [ ] `schema_version` field is `"1.0"` and asserted in tests
14. [ ] Output files written via temp → `fsync` → `os.replace` (transactional). Killing the process mid-write leaves no partial file at the final path.
15. [ ] README has a "Programmatic use" section showing `jq` recipes for both `--json` and `--ndjson`
16. [ ] `--help` documents public exit codes (0, 1, 2, 130) and points to JSON `code` field for granular semantics
17. [ ] All tests in `tests/` pass

**Anti-scope (do NOT in Stream A):**
- Do NOT change the recording logic, provider internal logic, or `.txt`/`.srt` file formats. Only the *stdout/stderr* surface and the `.json` file shape change.
- Do NOT add new providers.
- Do NOT refactor `recorder.py` or `utils.py` (pre-existing LSP errors are documented tech debt).
- Do NOT implement the provider registry (Stream C) — keep the existing hardcoded dispatcher.

---

*End of docs/CONTRACT.md. For agent operating rules: [../AGENTS.md](../AGENTS.md). For project history/state/decisions: [../MEMORY.md](../MEMORY.md).*
