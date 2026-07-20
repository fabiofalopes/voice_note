# AGENTS.md — voice_note

> **Read this file end-to-end before touching voice_note code. It is authoritative.**
> For history, decisions, and validation evidence: see [MEMORY.md](./internal/MEMORY.md).
> For the output contract spec (Stream A reference): see [docs/CONTRACT.md](./docs/CONTRACT.md).
> For stream-sequencing history and supersession: see [MEMORY.md §Document relationships](./internal/MEMORY.md#document-relationships).

---

## 1. Mission

> Make `voice_note` a **contract-first CLI**: a tool whose entire behaviour is described by a versioned JSON/NDJSON output stream and a structured exit code, so any future UI, script, or downstream consumer is a thin client of a stable surface — never a parser of human text.

Everything else (packaging, provider abstraction, local MLX execution, UI) is a *consequence* of this mission. The output contract ([docs/CONTRACT.md](./docs/CONTRACT.md)) is the heart. Streams B/E/F wait until the contract ships and proves itself.

---

## 2. Execution environment (read before invoking anything)

**Agents do not inherit the user's shell aliases.** The `vn` alias is for humans; agents must invoke the tool directly.

| Concern | Value | Notes |
|---|---|---|
| **Python interpreter** | `./venv/bin/python` | Python 3.14 + all deps installed in venv. Do NOT use `python3` (system Python lacks deps). |
| **Invocation pattern** | `./venv/bin/python transcribe.py <args>` | Always from repo root. `transcribe.py` is the 16-line entry shim that calls `src.cli.main()`. |
| **Stream A DoD commands** translate as follows | `vn recording.wav --ndjson` → `./venv/bin/python transcribe.py recording.wav --ndjson` | Substitute accordingly when running DoD verifications. |
| **Future tests** | `./venv/bin/pytest` | pytest will be installed in venv as part of Pre-Stream-A Step 2. |
| **API keys** | `.env` file at repo root or secure registry | Provides `GROQ_API_KEY` + `MODELOS_AI_KEY`. `.env` is gitignored — never commit. |
| **Working directory** | Always the repo root (`/Users/fabiofalopes/projetos/hub/voice_note/`) | The `vn` alias uses a subshell to preserve `pwd`; agents must `cd` explicitly or use `workdir`. |
| **Audio byproducts** | `recordings/` (24 GB, gitignored), `audio/` (probe outputs, gitignored) | NEVER commit. User manages compression/cleanup of recordings/ manually. |
| **Platform** | macOS (darwin), arm64 | Linux recording path (`parecord`) supported but not the dev environment. |

**Sanity check before Stream A work:**
```bash
./venv/bin/python transcribe.py --help          # CLI works
./venv/bin/python -c "import groq, openai"       # deps available
test -f .env && echo ".env present"              # API keys available
```

---

## 3. Current scope (what is in front of us)

**v1.0 ships on Stream A's Definition of Done.** Everything else is deferred or parallel.

### Pre-Stream A — Foundation (MUST complete first, hard prerequisite)

**Step 1 — Commit baseline.** ✅ Completed 2026-07-20. The provider-abstraction layer, the reliability layer, and all strategy docs were committed in atomic `pre-a(...)` chunks — see [MEMORY.md §2.3 Baseline commit status](./internal/MEMORY.md#23-baseline-commit-status) for the commit list. The Fireworks dead-code baseline committed there has since been removed (user decision 2026-07-18, removed 2026-07-20).

**Step 2 — Write tests.** ✅ Completed 2026-07-20. The pytest executable-spec baseline now covers provider parsing, normalisation, recorder recovery, contract models, emitters, CLI errors, atomic JSON identity, all-silent handling, and interrupts. Current local status: 29 passing tests.

The original minimum target was:
- `tests/test_groq_parsing.py` — parse Groq response fixture, assert all fields populated
- `tests/test_modelos_parsing.py` — parse modelos response fixture, assert nulls preserved (not defaulted)
- `tests/test_normalization.py` — null pass-through, language normalisation via `langcodes`, `end` clamping
- `tests/test_robust_recorder.py` — fault recovery paths
- `tests/test_contract_schema.py` — Pydantic v2 model validation

These tests are the executable spec for the contract. They must fail meaningfully before Stream A implementation, pass after.

### Stream A — Output Contract (✅ shipped 2026-07-20)

**Current status (2026-07-20):** All 17 DoD items passed, including live Groq and modelos success paths. Contract v1.0 is frozen. Validation evidence is recorded in [MEMORY.md §11](./internal/MEMORY.md#11-validation-log-empirically-verified) and [the completed handoff](./internal/handoffs/2026-07-21-stream-a-live-validation.md).

**Goal**: Implement [docs/CONTRACT.md](./docs/CONTRACT.md) in code. Replace every `print()` in `src/cli.py` and `src/api/base_client.py` with contract-aware emitters.

**Files to create:**
- `src/contract.py` — Pydantic v2 models: `Envelope`, `Result`, `Segment`, `Word`, `Event`, `Capabilities`, `Error`. Plus enums. `extra="forbid"` on producer-owned stable models; `extra="allow"` only on `provider_meta`. Generate JSON Schema as committed artifact.
- `src/i18n.py` — `normalize_language(raw) -> str | None` using `langcodes`.
- `src/emitter.py` — `Emitter` ABC + `HumanEmitter`, `PlainEmitter`, `JSONEmitter`, `NDJSONEmitter`. Human output is a **derived view** of the machine stream — not a separate code path.
- `tests/contract/` — golden-file tests with modelos + Groq fixtures.

**Files to modify:**
- `src/cli.py` — add `--output`, `--json`, `--ndjson`, `--plain`, `--no-color`, `--quiet` flags; auto-detect TTY; route all output through chosen `Emitter`; replace binary exit codes with the [docs/CONTRACT.md §Exit codes](./docs/CONTRACT.md#35-exit-codes-posix--sysexitsh-aligned) map.
- `src/api/base_client.py` — replace `print()` with emitter events; add per-segment `offset_seconds`; add normaliser (null pass-through for quality fields; clamping for `end`); add `langcodes` language normalisation; add capability check before request; add transactional write.
- `requirements.txt` — add `langcodes>=3.0`, `pydantic>=2.0`, `pytest>=8.0`.

**Definition of Done** (17 items — full list in [docs/CONTRACT.md §Stream A DoD](./docs/CONTRACT.md#stream-a-definition-of-done)). Highlights:
1. `vn recording.wav --ndjson | jq 'last | .data.code'` returns `"OK"` on success
2. `vn nonexistent.wav --ndjson; echo $?` exits `66` with `"code":"FILE_NOT_FOUND"`
3. modelos preserves nulls (`avg_logprob: null`, NOT `0.0`) + emits `PROVIDER_FIELD_NULL` warning
4. `--word-timestamps --provider modelos` fails fast with exit `64` (capability check, NOT warning)
5. modelos `end` hallucination → `TIMESTAMP_CLAMPED` warning + clamped value
6. Ctrl+C → exit `130`, `"code":"USER_INTERRUPT"`, `status:"partial"` if any segments emitted
7. On-disk `.json` byte-identical to stdout `--json` (modulo `request_id`)
8. Output files written via temp → `fsync` → `os.replace` (transactional)

**Anti-scope (do NOT in Stream A):**
- Do NOT change recording logic, provider internal logic, or `.txt`/`.srt` file formats.
- Do NOT add new providers.
- Do NOT refactor `recorder.py` or `utils.py` (pre-existing LSP errors are documented tech debt).
- Do NOT implement provider registry (Stream C). Keep existing hardcoded dispatcher.

### Stream D — Reliability flip (L0, parallel to A, optional for v1.0 ship)

Closes the 15-month-old CRITICAL handoff. No `vn` invocation should be able to lose more than 5 minutes of audio to a device failure.

**Files to touch:**
- `src/cli.py` — flip default: `robust=True`. Add `--legacy` flag (opt-IN to old recorder, with stderr warning on every use).
- `src/audio_processing/robust_recorder.py` — apply unfixed recommendations from `internal/reports/signal_handling_corruption_analysis.md`: explicit `signal.signal(SIGINT, …)` handler setting `_shutdown_requested` flag, atomic WAV writes (write-to-tmp + `os.replace`), terminal state save/restore around PyAudio sessions.
- `.HANDOFF_NEXT.md` — mark CLOSED once default is flipped.
- `internal/RELIABILITY_FIRST.md`, `ROBUST_RECORDING.md` — update to reflect "robust is now default; `--legacy` available".

**DoD:** See [MEMORY.md](./internal/MEMORY.md) or original Stream D spec.

### Deferred streams

| Stream | Status | Unblocks when |
|---|---|---|
| B (packaging / `pyproject.toml`) | Deferred | v1.0 ship — clone-installable is enough |
| C (provider registry, two-pattern dispatch) | Deferred | After A ships; current hardcoded dispatcher works for 2 providers |
| E (MLX Whisper) | Deferred | Depends on C. Model: `mlx-community/whisper-large-v3-turbo` |
| F (UI) | Deferred | After A v1.0 ships + 2 weeks stability with no contract churn |

---

## 4. Stream dependency graph

```
Pre-Stream A (commit + tests)
    ↓
Stream A (contract) ←── v1.0 SHIP
    ↓
Stream B (packaging)     Stream C (provider registry)     Stream D (reliability flip, parallel)
                              ↓
                         Stream E (MLX)
                              ↓
                         Stream F (UI — blocked by A v1.0 ship + 2 weeks stability)
```

---

## 5. Non-goals (explicit exclusions for v1.0)

**Out of scope for v1.0 ship. Refuse and cite this section if asked to do any of these:**

- ❌ **Stream B / C / E / F** — see §2 above
- ❌ **Fireworks** — dropped per user decision (2026-07-18). Never used. Dead code removed 2026-07-20. Do not re-add.
- ❌ **UI in v1.0** — hard gate. UI is NEVER started before contract v1.0 ships and is proven by at least one external consumer (a 50-line Python script driving `vn --ndjson` end-to-end counts).
- ❌ **Backward compatibility escape hatch** — no `VN_LEGACY_OUTPUT=1` env var. Clean break. Document prominently in README.
- ❌ **`field_status` provenance object** — simplified to null pass-through + documentation. The provenance object is over-engineering for a personal tool; null is already unambiguous.
- ❌ **14 NDJSON event types** — pruned to 5 for v1.0 (`start`, `segment`, `warning`, `error`, `end`).
- ❌ **15 public exit codes** — pruned to 4 public (`0`, `1`, `2`, `130`); full sysexits set is internal, exposed via JSON `code` field.
- ❌ **`result.kind: "analysis"`** — removed. No concrete shape exists. Add in minor bump when a real LLM-postprocessing pipeline defines one.
- ❌ **Word-level confidence field** — omitted. Provider divergence (mlx-whisper `probability`, WhisperX `score`, APIs return nothing).
- ❌ **`pipeline_stages[]` array** — reserved slots (`speaker`, `words`, `postprocessed`) are sufficient for v1.0.
- ❌ **AssemblyAI / streaming mic WebSocket / LLM post-processing / batch folder / auto-fallback** — ROADMAP phases 3/5/6/7/8; defer.
- ❌ **Audio compression / VAD / noise reduction** — abandoned for good reasons. Do not resurrect.
- ❌ **Replacing `argparse` with Click/Typer** — cosmetic; only revisit if subcommands become necessary (Stream F era).
- ❌ **`vn provider probe` conformance suite** — Stream C deliverable.
- ❌ **Raw provider response in default JSON/NDJSON** — opt-in only via future `--include-raw` flag.
- ❌ **Re-introducing local Whisper monolithically** — never. MLX-via-provider-registry is the future local path.

---

## 6. Agent operating protocol (hard rules, non-negotiable)

### 6.1 Before any code change

1. **Read this file (AGENTS.md) end-to-end.** Confirm in your first message which stream(s) your work touches.
2. **Read `.opencode/agent-environment.json`** for runtime context — Python version, OS, audio backend.
3. **Check `git status`.** The working tree currently has uncommitted abstraction-layer work. If your task touches any of those files, propose a commit plan (Pre-Stream-A Step 1) before editing.
4. **Verify your scope against §4 Non-goals.** If asked to do something there, refuse and cite this file.
5. **Consult [docs/CONTRACT.md](./docs/CONTRACT.md) for any contract-shape question** — field definitions, NDJSON event types, exit codes, normalisation rules.

### 6.2 During implementation

6. **One stream per commit cluster.** Do not mix Stream A work with Stream D work.
7. **Never suppress types** (`as any`, `# type: ignore`, bare `except:`). All exceptions must map to a `code` from [docs/CONTRACT.md §Exit codes](./docs/CONTRACT.md#35-exit-codes-posix--sysexitsh-aligned).
8. **Never re-introduce `print()` for status output** in code paths the contract owns. New `print()` calls allowed only in `HumanEmitter` / `PlainEmitter` (the default TTY renderers).
9. **Never edit `recorder.py` or `utils.py`** unless your stream's DoD explicitly requires it. Their pre-existing LSP errors are documented tech debt, not yours to fix.
10. **Never commit secrets.** `.env` is in `.gitignore`; verify before any commit.
11. **Every new public function gets a docstring** consistent with existing `base_client.py` style (Args / Returns / Raises).
12. **Write tests first.** New contract behaviour = new test in `tests/contract/` before the implementation. Tests are the executable spec.

### 6.3 Before declaring done

13. **Run the stream's Definition of Done checklist verbatim.** Each box must be demonstrably true.
14. **Run `lsp_diagnostics` on every changed file** and resolve all errors your change introduced.
15. **Run the tests.** All tests in `tests/` must pass.
16. **Update `internal/ROADMAP.md`** if your stream completed a Phase — mark `[x]` and date it.
17. **If you created new strategy/architecture docs at repo root, you did the wrong thing** — edit AGENTS.md / internal/MEMORY.md / docs/CONTRACT.md instead. New `.md` files at repo root are a smell.
18. **If you touched the contract, bump `schema_version`** and document the change in [MEMORY.md §Changelog](./internal/MEMORY.md#12-changelog).
19. **Write a handoff prompt for the next session** at `internal/handoffs/YYYY-MM-DD-<task-slug>.md`. Date-stamp with the *next* session's expected start date. Use the most recent handoff in `internal/handoffs/` as the template. This is non-negotiable for multi-session continuity — state lives in files, not session context.

### 6.4 Commit message convention

```
<stream-tag>(<scope>): <imperative summary>

<body explaining why, referencing AGENTS.md §X or docs/CONTRACT.md §X>

<DoD checklist status>
```

**Stream tags:** `pre-a`, `contract` (Stream A), `packaging` (Stream B), `registry` (Stream C), `reliability` (Stream D), `mlx` (Stream E), `ui` (Stream F).

**Examples:**
- `pre-a(recorder): commit robust_recorder.py baseline`
- `pre-a(providers): commit BaseSTTClient ABC + Groq + modelos`
- `pre-a(tests): add pytest fixtures for Groq + modelos response parsing`
- `contract(A): emit NDJSON events from base_client pipeline`
- `contract(A): implement Pydantic v2 envelope + segment models`
- `reliability(D): flip --robust default to true, add --legacy opt-in`

---

## 7. Sequencing rules (anti-scope-creep)

1. **No UI work** until contract v1.0 ships and is proven by at least one external consumer.
2. **No new provider** until Stream C (registry) lands. Adding MLX (Stream E) requires the registry.
3. **No refactor of `recorder.py` / `utils.py`** — they carry documented pre-existing LSP errors. Leave them unless Stream D explicitly requires touching them.
4. **No reintroduction of local Whisper** until Stream E (MLX) and only via the provider registry, never as a special case.
5. **No new top-level strategy `.md`** — edit AGENTS.md / internal/MEMORY.md / docs/CONTRACT.md. All prior strategy docs in Obsidian are superseded.
6. **No editing `config.py` to "fix" it** — it gets *deleted* in Stream C, replaced by provider-aware config.

---

## 8. What this file IS and is NOT

### 8.1 What it IS
- The single authoritative source for agent operating rules in voice_note.
- Short enough to be read every session.
- Pointers to deeper reference material (internal/MEMORY.md for history/state, docs/CONTRACT.md for spec).

### 8.2 What it is NOT
- Not the contract spec. That's [docs/CONTRACT.md](./docs/CONTRACT.md).
- Not the history/state file. That's [MEMORY.md](./internal/MEMORY.md).
- Not a code review or tutorial.
- Not an excuse to delay shipping. v1.0 ships when Stream A DoD passes.

---

*This file replaces the 880-line master prompt for agent operating rules. Full history of how this file came to be: see [MEMORY.md §Changelog](./internal/MEMORY.md#12-changelog).*
