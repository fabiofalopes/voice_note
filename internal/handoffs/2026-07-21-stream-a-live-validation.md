---
created: 2026-07-20
session_goal: Complete Stream A live-provider validation and ship decision
type: handoff
audience: next agent session
author: opencode
status: completed
---

# 2026-07-21 - Stream A Live Validation

## Objective

Complete the remaining live-provider checks for Stream A, then commit the
reviewed implementation if every applicable DoD item passes.

## Completion State

- Stream A implementation is present but uncommitted.
- All 17 DoD items passed on 2026-07-20; contract v1.0 is frozen.
- `./venv/bin/pytest tests/ -q` passes: **29 passed**.
- Groq JSON/NDJSON live paths returned `OK`; Groq request metadata was preserved;
  on-disk JSON matched stdout modulo `request_id`.
- modelos live paths preserved null metrics, emitted `PROVIDER_FIELD_NULL`, and
  emitted `TIMESTAMP_CLAMPED` with the segment end clamped to the 0.35s input.
- Missing-file NDJSON exits 66 with `FILE_NOT_FOUND`; modelos word timestamps
  fail before client construction with exit 64 and `CAPABILITY_UNSUPPORTED`.
- Pyright reports no errors in modified contract modules; compileall, help, and
  `git diff --check` pass.

## Completed Actions

1. Fixed exact-name handling in the external secrets registry and migrated
   provider keys to `GROQ_API_KEY` and `MODELOS_AI_KEY` without exposing values.
2. Executed live provider checks through `secrets run -- <command>`.
3. Recorded final evidence in `MEMORY.md` and checked all contract DoD boxes.

## Live Fixture

Recreate a tiny spoken WAV in the approved temp directory if needed:

```bash
say -o "/var/folders/fb/kjdbjyy54872630pg1xsf37m0000gn/T/opencode/voice-note-dod.aiff" "Hello voice note contract test"
ffmpeg -y -loglevel error \
  -i "/var/folders/fb/kjdbjyy54872630pg1xsf37m0000gn/T/opencode/voice-note-dod.aiff" \
  -ar 16000 -ac 1 \
  "/var/folders/fb/kjdbjyy54872630pg1xsf37m0000gn/T/opencode/voice-note-dod.wav"
```

## Required Live Checks

1. Groq `--json --timestamps`: status `ok`, code `OK`, mode `transcribe`,
   `provider_meta.x_groq.id` present, on-disk JSON byte-identical to stdout.
2. Groq `--ndjson`: every line parses, first successful event is `start`, last
   event is `end`, and `end.data.code == "OK"`.
3. modelos `--ndjson`: null quality metrics remain null, emits
   `PROVIDER_FIELD_NULL`, and hallucinated end values emit `TIMESTAMP_CLAMPED`.
4. Confirm no provider status text contaminates machine stdout.
5. Run all 17 DoD checks in `docs/CONTRACT.md` and record evidence in MEMORY.md.

## Final Verification

```bash
./venv/bin/python -m compileall -q src tests
./venv/bin/pytest tests/ -q
./venv/bin/python transcribe.py --help
git diff --check
git status --short
```

Stream A and contract v1.0 are shipped. Do not mix the still-uncommitted Stream A
cluster with later stream work; commit it atomically when explicitly requested.
