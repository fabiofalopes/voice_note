# voice_note — STT Pipeline Roadmap

> **⚠️ DRIFTED (2026-07-19).** This document uses a Phase 0-8 model that has been **replaced by the Stream A/B/C/D/E/F model** in [`AGENTS.md`](./AGENTS.md) §3-4. It also lists Fireworks as live (dropped 2026-07-18 — see [`MEMORY.md`](./MEMORY.md) §3.2 #8) and has zero awareness of the output contract or the modelos provider. **Do NOT use this document for current direction.** Stream A's DoD includes updating/replacing this file (see [`docs/CONTRACT.md`](./docs/CONTRACT.md) Stream A DoD item 16). Retained for phase-tracking archaeology only.

This document is the authoritative state of the project for any agent picking up work.

> **Note:** The claim above is itself stale as of 2026-07-19. The authoritative sources are now AGENTS.md (rules), MEMORY.md (state/history), docs/CONTRACT.md (spec).
Read this first. Each phase is scoped to be completable in a single session.

---

## Architecture

```
transcribe.py           ← entry-point shim (just calls src/cli.py::main)
src/
  cli.py                ← argparse CLI; dispatches to provider client
  api/
    base_client.py      ← abstract BaseSTTClient — all shared pipeline logic
    groq_client.py      ← GroqWhisperClient(BaseSTTClient)
    fireworks_client.py ← FireworksSTTClient(BaseSTTClient)
    __init__.py         ← exports all public classes
  audio_processing/
    recorder.py         ← mic recording (pre-existing LSP errors, do not touch)
    utils.py            ← get_audio_duration (pre-existing LSP errors, do not touch)
  config.py             ← dead code, leave alone
recordings/             ← all audio + output files live here
```

### Class hierarchy

```
BaseSTTClient (ABC)
  ├── GroqWhisperClient      → groq SDK
  └── FireworksSTTClient     → openai SDK @ fireworks base_url
```

`BaseSTTClient` owns: chunking, ffmpeg extraction, partial-file crash-safety,
prompt chaining, silence filtering, output serialisation (txt / srt / json).

Each provider subclass only implements `_send_chunk()` and `_parse_response()`.

### Key data types (base_client.py)

| Type | Fields |
|---|---|
| `Segment` | id, start, end, text, avg_logprob, no_speech_prob, compression_ratio, tokens |
| `ChunkResult` | text, segments, detected_language, duration |
| `TranscriptionResult` | text, segments, detected_language, output_file, srt_file, json_file |

---

## CLI Usage

```bash
# Transcribe (Groq, default)
python transcribe.py recording.wav

# With language hint + SRT output
python transcribe.py recording.wav --language pt --timestamps

# Use Fireworks instead of Groq
python transcribe.py recording.wav --provider fireworks --timestamps

# Translate to English
python transcribe.py recording.wav --translate

# Other flags
--model MODEL          # override provider default model
--output PATH          # base output path (no extension)
--no-clipboard         # skip clipboard copy
--list-devices         # list audio input devices
--test-mic             # test microphone
--device N             # select audio device by index
--record-output FILE   # WAV filename when recording (default: recording.wav)
```

### Output files (saved next to source audio)

| Flag | Files saved |
|---|---|
| (default) | `<name>_transcription.txt` |
| `--timestamps` | + `<name>_transcription.srt`, `<name>_transcription.json` |
| `--translate` | `<name>_translation.txt` (+ .srt/.json if `--timestamps`) |

---

## Completed

### Phase 0 — Foundation (done)
- [x] `BaseSTTClient` with full shared pipeline
- [x] `GroqWhisperClient` — verbose_json, prompt chaining, 429 retry with exact wait
- [x] `FireworksSTTClient` — openai SDK @ fireworks base_url, exponential backoff
- [x] CLI: `--provider`, `--language`, `--timestamps`, `--translate`
- [x] Output: `.txt` always; `.srt` + `.json` when `--timestamps`
- [x] Partial-file crash-safe resume (`.partial` files)
- [x] Silence filtering via `no_speech_prob`
- [x] Low-confidence segment warnings via `avg_logprob`
- [x] `openai>=1.0.0` added to requirements.txt and installed
- [x] Smoke-tested: 30s clip → txt ✓, srt (5 segments) ✓, json ✓, lang detection ✓

---

## Next phases (in recommended order)

### Phase 1 — Fireworks end-to-end test
**Goal:** verify Fireworks provider actually works, not just that the code compiles.

Steps:
1. Set `FIREWORKS_API_KEY` in `.env`
2. Run: `python transcribe.py /tmp/test_30s.wav --provider fireworks --timestamps --no-clipboard`
3. Confirm txt/srt/json written, language detected
4. Test rate-limit path: run 10× in quick succession, confirm backoff works
5. Update this section with findings (latency, quality vs Groq)

---

### Phase 2 — Word-level timestamps (Groq only)
**Goal:** optionally get per-word timestamps from Groq's `timestamp_granularities=["word"]`.

Notes:
- Only works with `whisper-large-v3` (not turbo, not distil)
- Adds ~20% latency
- Returns a `words` list in the verbose_json response
- Add `Segment.words: list[Word]` dataclass where `Word` has start/end/word fields
- Add CLI flag `--word-timestamps` (implies `--timestamps`, selects large-v3 on groq)
- Save word-level data in the `.json` output

Scope: `base_client.py` (add `Word` dataclass), `groq_client.py` (`_send_chunk`: add
`timestamp_granularities`; `_parse_response`: parse words), `cli.py` (new flag).
Fireworks does not support `timestamp_granularities` — silently ignore it there.

---

### Phase 3 — Auto-fallback (Groq → Fireworks)
**Goal:** when Groq returns a 429 and the wait is > N minutes, automatically retry the
same chunk via Fireworks instead of sleeping.

Design:
- Add `FallbackSTTClient(BaseSTTClient)` that wraps a primary + secondary client
- Override `_send_chunk()`: call primary; on 429 with long wait, call secondary
- Or: simpler approach — pass `fallback_client` kwarg to `GroqWhisperClient`

Scope: new file `src/api/fallback_client.py`, update `cli.py` to allow
`--provider groq+fireworks` (or `--fallback fireworks`).

---

### Phase 4 — Local MLX Whisper (Apple Silicon)
**Goal:** add offline provider using Apple's MLX Whisper for Air-gapped / no-API use.

Notes:
- `mlx-whisper` package: `pip install mlx-whisper`
- API: `mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")`
- Returns dict with `text` and `segments` (same shape as OpenAI verbose_json)
- No chunking needed for short files; for long files, chunking still applies
- Much faster than API for short clips; competitive for long clips on M-series

Scope: `src/api/mlx_client.py` (new file), update `__init__.py`, `cli.py`
(`--provider mlx`), `requirements.txt` (optional dependency).

---

### Phase 5 — AssemblyAI provider
**Goal:** add AssemblyAI as a third cloud provider (async, speaker diarisation,
chapters, sentiment).

Notes:
- AssemblyAI has a very different API (upload → poll), so `_send_chunk()` abstraction
  may need a small tweak (async or blocking poll)
- Speaker diarisation is AssemblyAI's key differentiator
- Free tier: 100 hours/month
- `assemblyai` pip package

Scope: `src/api/assemblyai_client.py`, update `__init__.py`, `cli.py`
(`--provider assemblyai`), add `--diarize` flag.

---

### Phase 6 — Streaming transcription
**Goal:** stream mic input directly to Groq's WebSocket endpoint for real-time
transcription while recording.

Notes:
- Groq streaming STT: `wss://api.groq.com/openai/v1/audio/transcriptions` (beta)
- Requires `websockets` package
- Architecture: `StreamingSTTClient` — does NOT extend `BaseSTTClient` (different
  interface: `stream_mic()` → async generator of partial transcripts)
- Keep separate from batch pipeline — do not muddle the abstractions
- `recorder.py` needs a streaming-compatible audio callback mode

Scope: new file `src/api/streaming_client.py`, new CLI subcommand `record-live`,
update `recorder.py` minimally to expose raw audio callback.

---

### Phase 7 — Post-processing & summarisation
**Goal:** after transcription, run an LLM pass to clean up, punctuate, and summarise.

Ideas:
- `--clean`: LLM cleanup pass (fix run-ons, add paragraph breaks)
- `--summarise`: bullet-point summary via Groq's chat API
- `--topics`: extract main topics
- `--chapters`: split into named chapters (feed segments to LLM)

Scope: `src/postprocess/` directory, update CLI.

---

### Phase 8 — Batch processing
**Goal:** transcribe all recordings in a folder in one command.

```bash
python transcribe.py --batch recordings/ --language pt --timestamps
```

Notes:
- Skip files that already have a `.txt` output (unless `--force`)
- Log summary: files done, files skipped, total duration, errors
- Rate-limit awareness: distribute chunks across providers to avoid hitting limits

Scope: new `--batch` mode in `cli.py`.

---

## Known issues / tech debt

- `recorder.py` and `utils.py` have pre-existing LSP type errors — **do not fix**,
  they are pre-existing and out of scope
- `config.py` is dead code — leave alone
- Segment timestamps in multi-chunk files are relative to each chunk, not the full
  file. Phase 1 task: accumulate a global offset and add it to each segment's
  start/end when appending from `_iter_chunks`.
- `_append_partial` joins chunks with a space — works for prose but loses paragraph
  structure. Low priority.

---

## Environment

```
Python:   venv/bin/python  (always use this, not system python)
venv:     /Users/fabiofalopes/projetos/hub/voice_note/venv/
Keys:     .env  (GROQ_API_KEY set; FIREWORKS_API_KEY optional)
ffmpeg:   available on PATH
Platform: macOS (Apple Silicon)
```
