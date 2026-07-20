---
created: 2026-07-20
session_goal: Stream A — implement the output contract
type: handoff
audience: next agent session
author: sisyphus (2026-07-20 handoff)
status: ready
---

# 2026-07-20 — Stream A: Output Contract Implementation

> **Use this prompt to start the next session.**
> This session's goal is to implement the voice_note v1.0 output contract.

---

## Session objective

Implement **docs/CONTRACT.md** in code. Replace every `print()` in `src/cli.py` and `src/api/base_client.py` with contract-aware emitters.

Pre-Stream-A Steps 1 and 2 are both complete:
- **Step 1**: baseline committed in atomic `pre-a(...)` history
- **Step 2**: pytest executable-spec suite exists (17 tests, 6 passing, 9 failing meaningfully)

The failing tests ARE the executable spec for this session. Make them pass.

---

## Read first (in this order)

1. **AGENTS.md** (auto-loaded). Confirm:
   - §3 Current scope (Stream A — the v1.0 ship)
   - §5 Non-goals (what NOT to do)
   - §6 Agent operating protocol (commit conventions, handoff rule)
2. **docs/CONTRACT.md** — the full spec. Pay special attention to:
   - §2 JSON status document (envelope shape)
   - §2.3 Segment schema (null-handling rules)
   - §5 Exit codes
   - §11 Provider normalisation (the rules the failing tests encode)
   - §Stream A Definition of Done (17-item checklist)
3. **MEMORY.md**
   - §2.1–§2.4 for current code reality and provider quirks
   - §11 Validation log for empirically verified provider behavior
4. Run `./venv/bin/pytest tests/ -v` to see the current test state

---

## The failing tests (your executable spec)

```
tests/test_contract_schema.py — 3 failures (src/contract.py doesn't exist)
  test_segment_model_preserves_null_quality_fields
  test_envelope_model_has_required_fields
  test_capabilities_model_has_all_declaration_fields

tests/test_normalization.py — 5 failures (src/i18n.py + src/contract.py don't exist)
  test_normalize_language_english_to_en
  test_normalize_language_already_iso
  test_normalize_language_none_returns_none
  test_end_clamped_to_duration
  test_end_within_range_not_clamped

tests/test_modelos_parsing.py — 1 failure (§2.2 parser gap)
  test_modelos_parse_preserves_null_quality_fields
```

After Stream A, all 17 tests should pass (the 2 skipped modelos tests will un-skip once the parser handles nulls).

---

## Files to create

1. **`src/contract.py`** — Pydantic v2 models:
   - `Envelope` (top-level JSON document, `extra="forbid"`)
   - `Result` (with `kind` discriminator: `"transcription" | "translation"`)
   - `Segment` (null-preserving for `avg_logprob`, `compression_ratio`, `no_speech_prob`; structural defaults for `id`, `seek`, `tokens`, `temperature`)
   - `Capabilities` (5 booleans)
   - `Error` (structured error object)
   - `clamp_end(end, duration) -> float` helper
   - Generate JSON Schema as committed artifact

2. **`src/i18n.py`** — `normalize_language(raw) -> str | None` using `langcodes`:
   - `"English"` → `"en"`
   - `"en"` → `"en"` (pass-through)
   - `None` → `None`

3. **`src/emitter.py`** — `Emitter` ABC + `HumanEmitter`, `PlainEmitter`, `JSONEmitter`, `NDJSONEmitter`:
   - Human output is a **derived view** of the machine stream
   - 5 NDJSON event types: `start`, `segment`, `warning`, `error`, `end`

---

## Files to modify

1. **`src/api/base_client.py`**:
   - Replace `float(getattr(seg, "field", 0.0))` with null-aware normaliser
   - Add per-segment `offset_seconds` from `chunk_start`
   - Add `langcodes` language normalisation
   - Add capability check before request
   - Add transactional write (temp → fsync → os.replace)
   - Replace `print()` calls with emitter events

2. **`src/cli.py`**:
   - Add `--output`, `--json`, `--ndjson`, `--plain`, `--no-color`, `--quiet` flags
   - Auto-detect TTY
   - Route all output through chosen `Emitter`
   - Replace binary exit codes with the CONTRACT.md §5 exit code map

---

## Definition of Done (17 items)

See **docs/CONTRACT.md §Stream A Definition of Done**. Highlights:

1. `vn recording.wav --ndjson | jq 'last | .code'` returns `"OK"` on success
2. `vn nonexistent.wav --ndjson; echo $?` exits `66` with `"code":"FILE_NOT_FOUND"`
3. modelos preserves nulls (`avg_logprob: null`, NOT `0.0`) + emits `PROVIDER_FIELD_NULL` warning
4. `--word-timestamps --provider modelos` fails fast with exit `64` (capability check)
5. modelos `end` hallucination → `TIMESTAMP_CLAMPED` warning + clamped value
6. Ctrl+C → exit `130`, `"code":"USER_INTERRUPT"`, `status:"partial"` if segments emitted
7. On-disk `.json` byte-identical to stdout `--json` (modulo `request_id`)
8. Output files written via temp → `fsync` → `os.replace` (transactional)

---

## Anti-scope (do NOT do in this session)

- ❌ Do NOT change recording logic, provider internal logic, or `.txt`/`.srt` file formats
- ❌ Do NOT add new providers
- ❌ Do NOT refactor `recorder.py` or `utils.py`
- ❌ Do NOT implement provider registry (Stream C)
- ❌ Do NOT flip `--robust` default (Stream D)
- ❌ Do NOT add packaging / `pyproject.toml` (Stream B)
- ❌ Do NOT start UI or local MLX work (Streams F / E)

---

## If interrupted mid-session

1. Re-read AGENTS.md + CONTRACT.md
2. Run `git status` to see what's been created/modified
3. Run `./venv/bin/pytest tests/ -v` to see which tests now pass
4. Resume from the first still-failing test

---

*End of handoff. For agent operating rules: [AGENTS.md](../../AGENTS.md). For project state/history: [MEMORY.md](../../MEMORY.md). For contract spec: [docs/CONTRACT.md](../CONTRACT.md).*
