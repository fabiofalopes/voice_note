# MEMORY.md — voice_note

> **Project state, history, decisions, and validation evidence.**
> Read this when you need context on *why* things are the way they are, or *what was decided*.
> For agent operating rules: see [AGENTS.md](../AGENTS.md).
> For the output contract spec: see [docs/CONTRACT.md](../docs/CONTRACT.md).

---

## 1. Historical context (why this project is the way it is)

The project has accumulated 18 commits of provider iteration. The current state emerged from pragmatic decisions that were never unified by a structural throughline:

### 1.1 The origin (v1 narrative, 2024-05-22)

The first strategy doc was a raw brain-dump that identified one true conviction:

> **"UI is a rendering of an output contract."**

This conviction — contract-first — survived everything that followed. It is the project's philosophical anchor. Everything else in v1 was either vague intention or naive assumption: Fireworks existed but nobody knew why; "local" meant two different things; the `vn` alias was acknowledged as "kinda pathetic but it works"; the entire local Whisper daemon stack had already been deleted but v1 didn't know this.

### 1.2 Lessons that shaped the project

1. **Local Whisper was tried and removed.** Commit `5bce6f5` deleted an entire local-inference stack (faster-whisper, daemon, benchmarking). Rationale captured in `abandoned/future_features/local_inference_architecture_decision.md`: complexity was disproportionate to value, APIs were faster, installation friction was too high. **This decision is final** — do not reintroduce local inference until the CLI contract is solid (Stream E for the future-MLX path).

2. **Groq became the default provider** because it was free, fast, and worked. Empirically verified (2026-07-18, HTTP 200 in 0.84s, all fields populated including word timestamps).

3. **The "local" the user refers to is `https://modelos.ai.ulusofona.pt`** — a self-hosted LiteLLM gateway on user-controlled infrastructure, NOT on-device execution. Exposed via the `modelos` provider using the OpenAI SDK against a custom `base_url`. Treat as a remote OpenAI-compatible endpoint, not as "local".

4. **Fireworks is dropped.** Added speculatively as an RPM-based complement to Groq's audio-seconds rate limit, but never used in practice. Dead code removed 2026-07-20. The architecture still supports adding any future OpenAI-compatible endpoint via config-only registration, so dropping Fireworks costs nothing in future flexibility.

5. **The two real providers are Groq and modelos.** Both OpenAI-compatible Whisper endpoints. Groq integrated via its native Python SDK (Pattern 1, preserves the precise 429 wait-time parser); modelos integrated via OpenAI SDK at custom `base_url` (Pattern 2, canonical OpenAI-compat pattern).

6. **The `vn` shell alias is a workaround** for the absence of a packaged entry point:
   ```bash
   alias vn='(cd /Users/fabiofalopes/projetos/hub/voice_note && ./venv/bin/python transcribe.py)'
   ```
   The subshell preserves `pwd`. Acknowledged tech debt — Stream B removes the need for it.

7. **A CRITICAL reliability handoff has been open for 15+ months.** `.HANDOFF_NEXT.md` (dated 2025-04-02) mandates flipping `--robust` to the default recording mode after a 4.5-hour data-loss incident (earbuds disconnected mid-recording; the fragile in-RAM recorder kept everything in memory and crashed on device failure). The robust recorder was built; the default was never flipped. This is the single most-debted open item — see Stream D.

### 1.3 The strategy journey (2026-07-18 to 2026-07-19)

v2.0–v2.3 transformed conviction into specification (789 lines of operational detail). The design review requested external feedback (8 questions to senior engineers). Perplexity GPT-5.6 answered all 8 and added an unprompted 16-point critique. Oracle then stress-tested the synthesis and found 7 blind spots (uncommitted code, zero tests, NDJSON scope creep, shim migration, auth strategy, platform gating, solo-dev overcommitment). The settling document resolved the 4 highest-priority contradictions (null pass-through over defaults, capabilities over warnings, transactional writes, SIGINT ≠ success), pruned NDJSON to 5 event types, and pruned scope for solo-dev focus. The result: a tighter, more honest contract that ships in v1.0. The strategy phase ended 2026-07-19. See §12 Changelog.

---

## 2. Current state — verified snapshot (as of 2026-07-20)

Descriptive reality, not aspiration. Every claim grounded in code on disk.

### 2.1 What exists (do not re-build)

| Concern | Where | LOC | Status |
|---|---|---|---|
| Entry-point shim | `transcribe.py` | 16 | Adds `src/` to `sys.path`, calls `src.cli.main()` |
| CLI parser | `src/cli.py` | 629 | `argparse`, hardcoded 2-provider dispatch, output-mode selection, structured exits, and envelope finalisation |
| Provider abstraction | `src/api/base_client.py` | 667 | **ABC with shared chunked pipeline** — chunking, normalisation, contract events, transactional writes, partial-file safety, prompt chaining, and silence handling |
| Provider hook | `BaseSTTClient._send_chunk()` | abstract | Subclasses implement only this + `_parse_response()` |
| Concrete providers | `groq_client.py` (245), `modelos_client.py` (195) | 440 total | **2 active providers.** Groq = native SDK custom adapter (Pattern 1). modelos = OpenAI SDK at custom `base_url` (Pattern 2). |
| Output contract | `src/contract.py`, `src/emitter.py`, `src/i18n.py` | 422 total | Pydantic v2 envelope/events, JSON/NDJSON/human renderers, ISO language normalisation, committed schemas in `schema/` |
| Robust recorder | `src/audio_processing/robust_recorder.py` | 471 | Auto-chunking, device-failure recovery, WAV merge. **Opt-in** via `--robust`. |
| Legacy recorder | `src/audio_processing/recorder.py` | 1029 | PyAudio + parecord (PipeWire/PulseAudio) auto-detection. Carries pre-existing LSP errors — documented tech debt, do not touch outside Stream D. |
| Cross-platform recording | `recorder.py` | (above) | PyAudio on macOS/Windows, parecord on Linux |
| Crash-safe resume | `.partial` files via `_load_partial` / `_append_partial` | — | `PARTIAL:chunk_end/total_duration` header + accumulated text |
| `.opencode/` agent ecosystem | `.opencode/agent/`, `.opencode/agent-environment.json` | — | 5 specialised agents + ground-truth environment file (some files stale — see §13) |

**Stream A implementation adds 422 contract-layer LOC plus contract tests and schema artifacts.**

### 2.2 What is missing (the actual gaps)

| Gap | Evidence | Impact |
|---|---|---|
| **No packaging** | No `pyproject.toml`, no `setup.py`, no `[project.scripts]` | Forces `vn` shell alias; blocks `pipx install` / `uv tool install` |
| **Dead `src/config.py`** | Exports `get_groq_api_key()` etc.; **0 callers**. Clients call `os.getenv()` directly with redundant `load_dotenv()` | Confusing; violates single-source-of-truth |
| **Hardcoded provider dispatch** | `cli.py`: `if/elif/else` chain over the 2 providers | Adding a provider requires editing the dispatcher |
| **`--robust` not default** | `cli.py` L162: `if args.robust:` (default `False`) | Every recording without the flag risks total data loss on device failure |
| **Ctrl+C corruption (unfixed)** | `reports/signal_handling_corruption_analysis.md` recommendations not applied | Terminal state corruption + possible file truncation on interrupt |

### 2.3 Baseline commit status

Pre-Stream-A Step 1 was completed on **2026-07-20**.

The previously uncommitted baseline (provider abstraction layer, robust recorder,
CLI wiring, and reliability docs) is now committed in atomic history:

- `3db1430` — `pre-a(recorder): commit fault-tolerant robust recorder baseline`
- `4c55223` — `pre-a(providers): commit BaseSTTClient ABC with shared chunked pipeline`
- `3b4e21a` — `pre-a(groq): rewrite Groq client as BaseSTTClient subclass`
- `0b1d7b2` — `pre-a(modelos): add modelos client`
- `e4d9887` — `pre-a(fireworks): add Fireworks client dead-code baseline`
- `ad74c68` — `pre-a(cli): wire multi-provider dispatch + config + dependencies`
- `4d51231` — `pre-a(docs): commit reliability + signal-handling docs`

**Current reality:** both Pre-Stream-A prerequisites and all 17 Stream A DoD
items are complete. Live Groq and modelos success paths passed on 2026-07-20,
and output contract v1.0 is frozen. The implementation remains in the working
tree pending an explicit commit request.

### 2.4 Provider quirks (empirically verified — see §11 for raw probe data)

| Provider | Quirk | Impact |
|---|---|---|
| **modelos (vLLM)** | Returns `null` for `id`, `seek`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio`, `no_speech_prob` — only `start`, `end`, `text` populated | Quality metrics unavailable. Silence filter + low-confidence warnings non-functional on modelos. |
| **modelos (vLLM)** | `end` timestamp hallucination: returned `end=30.19` on a 2.28s audio file | Contract must clamp `end` to `input.duration_seconds` at emission time + emit `TIMESTAMP_CLAMPED` warning. |
| **modelos (vLLM)** | Ignores `timestamp_granularities=["word"]` silently; returns `words: null` | Word-level timestamps unavailable on modelos. |
| **modelos (vLLM)** | `language: "en"` (ISO 639-1 code) | Aligned with contract; no normalisation needed. |
| **Groq** | All 10 segment fields populated; word-level timestamps work | Fully compliant; reference provider. |
| **Groq** | `language: "English"` (full word, not ISO code) | Contract normalises to `"en"` via `langcodes`. |
| **Groq** | Includes `x_groq.id` extension field for request tracing | Preserved verbatim under `provider_meta` in the contract. |

---

## 3. Open decisions (carried forward)

### 3.1 Non-blocking (defaults may be overridden by user)

| # | Question | Default | Blocking? |
|---|----------|---------|-----------|
| 1 | Contract v1.0 ship vehicle — ship when Stream A DoD passes, or wait for Stream B packaging so first consumer can `pipx install`? | Ship when Stream A passes (clone-installable; `pip install .` works) | No |
| 2 | Provider default — stay on `groq` or switch to `modelos`? | Stay on `groq` (empirically verified, better quality metrics, faster) | No |
| 3 | MLX model default when Stream E ships? | `mlx-community/whisper-large-v3-turbo` (leading candidate; consumes significant RAM) | No (Stream E deferred) |
| 4 | Binary name when packaged? | `vn` for continuity (alternatives: `voice-note`, `voice_note`) | No (Stream B) |

### 3.2 Resolved decisions (recorded for audit; do not relitigate)

| # | Decision | Date | Resolution |
|---|----------|------|-----------|
| 5 | Groq registration pattern | 2026-07-18 | Pattern 1 custom adapter (native SDK), preserves 429 wait-time parser. MLX will follow same pattern. |
| 6 | `provider_meta` policy | 2026-07-18 | Opaque in v1.0. Promotion to typed field requires (a) 3+ provider implementations AND (b) demonstrated consumer need. |
| 7 | Language normalisation library | 2026-07-18 | `langcodes` wrapped in `src/i18n.py::normalize_language()`. |
| 8 | Fireworks | 2026-07-18 | Dropped. Never used. Dead code removed 2026-07-20. |
| 9 | Groq API key (earlier 403) | 2026-07-18 | Stale key. Re-tested after user confirmed key in place: 0.84s, HTTP 200, all fields populated. Closed. |
| 10 | Null handling | 2026-07-19 | Preserve nulls for quality fields (`avg_logprob`, `compression_ratio`, `no_speech_prob`). Documented defaults for structural fields (`id`, `seek`, `tokens`, `temperature`). Clamp `end` if hallucinated. |
| 11 | Capabilities | 2026-07-19 | Declarative object per provider. Fail-fast (exit 64) on incompatible requests. No post-hoc warnings. |
| 12 | Transactional writes | 2026-07-19 | temp → `fsync` → `os.replace`. `outputs.*` only lists promoted files. |
| 13 | SIGINT handling | 2026-07-19 | Exit `130` (POSIX 128+SIGINT). `status: "partial"` if any data saved; `"error"` otherwise. NOT exit 0. |
| 14 | Public exit codes | 2026-07-19 | 4 public (`0`, `1`, `2`, `130`); full sysexits set internal, exposed via JSON `code` field. |
| 15 | `result.kind: "analysis"` | 2026-07-19 | Removed from v1.0. No concrete shape exists. Add in minor bump when a real LLM-postprocessing pipeline defines one. |
| 16 | NDJSON event types | 2026-07-19 | 5 for v1.0: `start`, `segment`, `warning`, `error`, `end`. Other types deferred. |
| 17 | `field_status` provenance object | 2026-07-19 | Simplified to null pass-through + documentation. Over-engineering for personal tool. |
| 18 | Backward compatibility escape hatch | 2026-07-19 | No `VN_LEGACY_OUTPUT=1`. Clean break. |

### 3.3 Hard prerequisites (NOT optional)

| # | Question | Answer | Blocking? |
|---|----------|--------|-----------|
| 19 | Commit pre-Stream-A baseline first? | **YES** — enforced as AGENTS.md §3 Pre-Stream-A Step 1 | **Yes** |
| 20 | Write tests before Stream A? | **YES** — enforced as AGENTS.md §3 Pre-Stream-A Step 2 | **Yes** |

---

## 4. Stream D — Reliability flip (full spec, parallel to Stream A)

Closes the 15-month-old CRITICAL handoff (`.HANDOFF_NEXT.md`, dated 2025-04-02). No `vn` invocation should be able to lose more than 5 minutes of audio to a device failure.

**Files to touch:**
- `src/cli.py` — flip default: `robust=True`. Add `--legacy` flag (opt-IN to old recorder, with stderr warning on every use).
- `src/audio_processing/robust_recorder.py` — apply unfixed recommendations from `reports/signal_handling_corruption_analysis.md`: explicit `signal.signal(SIGINT, …)` handler setting `_shutdown_requested` flag, atomic WAV writes (write-to-tmp + `os.replace`), terminal state save/restore around PyAudio sessions.
- `.HANDOFF_NEXT.md` — mark CLOSED once default is flipped.
- `RELIABILITY_FIRST.md`, `ROBUST_RECORDING.md` — update to reflect "robust is now default; `--legacy` available".

**DoD:**
- [ ] `vn` (no flags) uses `RobustAudioRecorder` — verified by `--output ndjson` (Stream A required first)
- [ ] Pulling the audio jack mid-recording leaves at most one chunk window (≤ `--chunk-minutes`) of lost data; previous chunks are valid WAVs on disk
- [ ] Ctrl+C during recording cleanly shuts down within 1s, leaves terminal usable, preserves all completed chunks
- [ ] `vn --legacy` prints a stderr warning: `"Using legacy recorder — data loss risk on device failure"`
- [ ] `.HANDOFF_NEXT.md` updated with `"CLOSED on <date>"` header

**Anti-scope:** do NOT touch `recorder.py` beyond what the signal-handling fix strictly requires.

---

## 5. Stream B/C/E/F (deferred — high-level only)

### Stream B — Packaging & entry point (L2, deferred)

**Goal**: Replace `vn` shell alias with proper installable command.

**Files:**
- `pyproject.toml` (NEW) — `[project.scripts]` exposing `vn = "voice_note.cli:main"`.
- `README.md` — replace "use alias" with `pipx install .` / `uv tool install .`.

**DoD**: `pipx install .` puts `vn` on `$PATH`; `uv tool install . -e` works editable; shell alias deprecated.

### Stream C — Provider registry (L3, deferred, depends on A)

**Goal**: Replace `if/elif` dispatcher with self-registering provider map supporting two patterns:
- **Pattern 1 — Custom adapter class** (Groq today, MLX future): native SDK or non-OpenAI APIs.
- **Pattern 2 — OpenAI-compatible config-only** (modelos today, any future OpenAI-compat endpoint): 5 lines of TOML, no code.

**Files**: `src/providers/registry.py`, `src/providers/openai_compat_client.py` (NEW); move `groq_client.py` to `src/providers/`; convert `modelos_client.py` to config entry. (Fireworks client + `FIREWORKS_API_KEY` already removed 2026-07-20 — no longer part of this stream.)

### Stream E — Local execution via MLX (L4, deferred, depends on C)

**Goal**: Add on-device Whisper via `mlx-whisper` as registered provider (Pattern 1). Optional extra in `pyproject.toml`: `pipx install 'vn[mlx]'`. Apple Silicon only.

### Stream F — UI (L5, deferred, blocked by A v1.0 ship + 2 weeks stability)

**Goal**: Build any UI as thin consumer of the contract. MUST invoke `vn --ndjson` as subprocess. Never couple to internal Python modules.

---

## 6. Layered priorities (decision framework)

| Layer | Priority | When | Depends on |
|---|---|---|---|
| **L0: Reliability floor** (Stream D) | CRITICAL | Immediate, parallel to A | — |
| **L1: Output contract** (Stream A) | P0 | Immediate | L0 stable recording |
| **L2: Packaging & entry point** (Stream B) | P1 | Deferred | — |
| **L3: Provider abstraction refinement** (Stream C) | P2 | After L1 | L1 contract types |
| **L4: Local execution** (Stream E) | P3 | After L1 + L3 stable | L3 registry pattern |
| **L5: UI** (Stream F) | Deferred | After L1 v1.0 ship + 2 weeks stability | L1 contract shipped |

**Golden rule**: L5 (UI) is NEVER started before L1 v1.0 is shipped. If you feel the urge to build UI, instead write a failing test that consumes the CLI's NDJSON stream — that test is your UI stand-in.

---

## 7. The journey in one paragraph

v1 (2024-05-22) said "we need a contract." v2.0–v2.3 (2026-07-18) wrote that contract in 789 lines, validated it against live providers, and sequenced its delivery into 6 streams. The design review (2026-07-18) asked 8 questions; Perplexity GPT-5.6 answered all 8 and raised 16 more points. Oracle then stress-tested the synthesis and found 7 blind spots (uncommitted code, zero tests, NDJSON scope creep, shim migration, auth strategy, platform gating, solo-dev overcommitment). The settling document (2026-07-19) resolved the 4 highest-priority contradictions (null pass-through over defaults, capabilities over warnings, transactional writes, SIGINT ≠ success), pruned NDJSON to 5 event types, deferred streams B/E/F for solo-dev focus, and adopted 15 additional recommendations. The strategy phase ended. What remains is execution.

---

## 8. The one-paragraph pitch

> `voice_note` is a Python CLI that records and transcribes audio via cloud Whisper providers (Groq, self-hosted LiteLLM gateway at modelos.ai.ulusofona.pt). It has a provider abstraction (`BaseSTTClient` ABC, ~579 LOC), a robust chunked pipeline with crash-safe resume (`robust_recorder.py`, 471 LOC), and 2 working providers — all currently uncommitted. What it lacks is a **stable output contract**: today every status line is unparseable human text, exit codes are binary, and there are zero tests. The contract spec (see [docs/CONTRACT.md](../docs/CONTRACT.md)) specifies JSON/NDJSON output, Pydantic v2 models, 5 NDJSON event types, 4 public exit codes + internal sysexits set, declarative capabilities, transactional writes, null pass-through for provider quality fields. The provider normalisation layer makes it provider-agnostic. A pruned plan ships v1.0 after committing the baseline and writing ~10 tests. Local MLX execution, packaging, provider registry, and UI are all deferred. The strategy phase ends here. What remains is execution.

---

## 9. Authoritative references & stale markers

Every agent working on voice_note MUST consult this table before citing or acting on any other doc.

### 9.1 ACTIVE — cite as current direction

| Doc | Why it's authoritative |
|---|---|
| `AGENTS.md` (this project) | Master rules for agents |
| `MEMORY.md` (this file) | State, history, decisions, validation |
| `docs/CONTRACT.md` | Output contract spec |
| `ROADMAP.md` | Phase 0 (done) and Phases 1–8 are real pending work — **but see AGENTS.md §5 Non-goals for deferrals** |
| `README.md` | User-facing truth about current CLI |
| `docs/audio-system.md` | Matches current `recorder.py` architecture |
| `docs/pipewire-integration.md` | Matches current Linux recording path |
| `docs/troubleshooting.md` | Current known user-facing issues |
| `reports/signal_handling_corruption_analysis.md` | Unfixed; recommendations feed Stream D |
| `abandoned/future_features/local_inference_architecture_decision.md` | Decision fully executed; informs Stream E boundary |
| `.opencode/agent-environment.json` | Ground-truth environment state file |
| `.opencode/agent/*.md` (5 active agents) | Current agent registry — some stale, see agent file banners |

### 9.2 PARTIALLY ACTIVE — cite with caveat

| Doc | Caveat |
|---|---|
| `RELIABILITY_FIRST.md` | Core philosophy valid; primary mandate (robust-by-default) is Stream D work, not yet done |
| `ROBUST_RECORDING.md` | Accurate for current opt-in `--robust` behaviour; will need rewrite when Stream D flips the default |

### 9.3 STALE / ABANDONED — do NOT act on without explicit user confirmation

| Doc | Verdict |
|---|---|
| `RELIABILITY_FIX.md` | Historical — describes what was built; superseded by `robust_recorder.py` |
| `.HANDOFF_NEXT.md` | 15 months stale; closes when Stream D lands |
| `abandoned/audio_optimization_plan.md` | Abandoned — zero implementation |
| `abandoned/audio_preprocessing_research.md` | Pure research — zero implementation |
| `abandoned/voice_note_deep_research_prompts.md` | Speculative — never acted on |
| `abandoned/FUTURE-automated-compression-feature.md` | Abandoned after Phase 1 script |
| `abandoned/future_features/system_robustness_roadmap.md` | Aspirational — zero checkboxes ticked; superseded by AGENTS.md Stream D |
| `abandoned/future_features/analyzer_insights.md` + `abandoned/future_features/prompts/` | Historical reference only — old LLM post-processing, removed |
| `development-notes.md` | Pre-dates robust recorder and multi-provider era; useful for archaeology only |
| `agents/legacy/OPTIMIZED_PROMPT.md` | Proposed agent renames + `PROJECT_STATE.md` creation — never executed; superseded by current `.opencode/agent/` setup + AGENTS.md/MEMORY.md |

---

## 10. Document relationships

| Document | Role | Status |
|----------|------|--------|
| `AGENTS.md` (in repo) | **Agent operating rules + scope** | **Active — sole authority for agent rules** |
| `MEMORY.md` (this file, in repo) | **State + history + decisions + validation** | **Active — sole authority for context/history** |
| `docs/CONTRACT.md` (in repo) | **Output contract spec** | **Active — sole authority for contract shape** |
| Obsidian: `dev-voice-note-project-strategy-contract-first-development.md` | v1 narrative — historical intent | **Superseded** — content merged into AGENTS.md/MEMORY.md/docs/CONTRACT.md |
| Obsidian: `dev-voice-note-project-strategy-contract-first-development-v2.md` | v2.3 master strategy — operational specification | **Superseded** — content merged here |
| Obsidian: `dev-voice-note-settling-the-journey.md` | Synthesis of v2.3 + external review | **Superseded** — content merged here |
| Obsidian: `dev-voice-note-design-review-request.md` | 8 questions to external engineers | **Superseded** — feedback integrated |
| Obsidian: `perplexity-voice-note-design-review-request.md` | GPT-5.6 external feedback | **Superseded** — feedback integrated |
| Obsidian: `dev-voice-note-master-prompt-v1.md` | Earlier single-file synthesis | **Superseded** — split into AGENTS.md/MEMORY.md/docs/CONTRACT.md |
| `ROADMAP.md` | Phase 0–8 plan | **Active** for phase tracking; AGENTS.md §5 defers most phases |
| `README.md` | User-facing truth about current CLI | **Active** — update in Stream A |
| `docs/audio-system.md`, `docs/pipewire-integration.md`, `docs/troubleshooting.md` | Current architecture docs | **Active** |
| `reports/signal_handling_corruption_analysis.md` | Unfixed recommendations feeding Stream D | **Active** |
| `abandoned/future_features/local_inference_architecture_decision.md` | Decision fully executed; informs Stream E boundary | **Active** |
| `.opencode/agent-environment.json` | Ground-truth environment state | **Active** (needs refresh — see §11) |
| `.opencode/agent/*.md` (5 agents) | Current agent registry | **Partially stale** — each file has a "defer to AGENTS.md" banner |
| `RELIABILITY_FIRST.md`, `ROBUST_RECORDING.md` | Valid for current opt-in `--robust` behaviour | **Partially active** — Stream D will rewrite |
| `.HANDOFF_NEXT.md` | 15-month-old CRITICAL mandate | **Stale** — closes when Stream D lands |
| `RELIABILITY_FIX.md`, `abandoned/audio_optimization_plan.md`, `abandoned/audio_preprocessing_research.md`, `abandoned/voice_note_deep_research_prompts.md`, `abandoned/FUTURE-automated-compression-feature.md`, `abandoned/future_features/system_robustness_roadmap.md`, `abandoned/future_features/analyzer_insights.md`, `development-notes.md`, `agents/legacy/OPTIMIZED_PROMPT.md` | Historical / abandoned | **Stale / abandoned** — do NOT act on without explicit user confirmation |

---

## 11. Validation log (empirically verified)

This section records what was checked before v1.0 was finalised, so future agents can audit the basis of the contract. **Every provider claim in AGENTS.md / docs/CONTRACT.md references this log.**

| Dimension | Method | Result | Date |
|---|---|---|---|
| **modelos endpoint reachability** | `curl https://modelos.ai.ulusofona.pt/v1/models` | ✅ HTTP 200, 1 STT model exposed (`stt-large-v3-turbo`) | 2026-07-18 |
| **modelos verbose_json shape** | Direct SDK call with 2.27s WAV, dumped `TranscriptionVerbose` | Top-level: `duration, language, text, segments, usage, words`. Per-segment: only `start, end, text` populated; `id, seek, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob` ALL `null`. `usage` and `words` are `null` at top level. **`end` hallucinated as 30.19s on 2.28s audio.** | 2026-07-18 |
| **modelos `timestamp_granularities=["word"]`** | Same probe with word request | Silently ignored — `words: null` returned. Declared `word_timestamps: false` in capabilities. | 2026-07-18 |
| **modelos language format** | Direct SDK call | `language: "en"` (ISO 639-1 code) | 2026-07-18 |
| **Groq endpoint** | Direct SDK call with 2.27s WAV, `timestamp_granularities=["segment","word"]` | ✅ **VERIFIED WORKING** (after user confirmed key in place). HTTP 200 in 0.84s. Top-level: `text, task, language, duration, words, segments, x_groq`. Per-segment: all 10 standard fields populated. Word-level works: `{'word': 'Okay.', 'start': 0.54, 'end': 2.06}` (no confidence field). `language: "English"` (normalised to `"en"` via `langcodes`). `x_groq.id` extension field present (preserved under `provider_meta`). | 2026-07-18 |
| **Groq language format** | Direct SDK call | `language: "English"` (full word — normalised to `"en"`) | 2026-07-18 |
| **Fireworks endpoint** | N/A — user decision 2026-07-18: Fireworks dropped from project | Skipped permanently. Dead code removed 2026-07-20. | 2026-07-18 |
| **Whisper verbose_json canonical schema** | Librarian research across OpenAI/Groq/Fireworks/vLLM/mlx-whisper/WhisperX docs | Confirmed convergence matrix. Critical findings: vLLM nulls `no_speech_prob` always; modelos nulls even more fields; WhisperX has very different shape (loses `id`, `seek`, `tokens`, `temperature`); mlx-whisper near-identical to OpenAI; word confidence key differs (`probability` vs `score`); Groq word-mode-quiet-segments bug not reproduced in our test. | 2026-07-18 |
| **Contract design review** | Oracle 12-dimension stress test | 3 critical fixes applied (exit code coverage, mode/kind discriminator, on-disk vs stdout shape unification). 9 minor fixes applied. | 2026-07-18 |
| **External review** | Perplexity GPT-5.6 design review of 8 questions | Direction validated. 4 highest-priority contradictions resolved (see §3.2). 16 additional recommendations considered; 8 adopted. | 2026-07-18 |
| **Existing parser robustness** | Code read of `modelos_client._parse_response` L153–176 + `fireworks_client._parse_response` L146–172 | Uses `float(getattr(seg, "field", 0.0))` pattern. When attribute exists with value `null`, `getattr` returns `null` (not the default), and `float(None)` raises `TypeError`. Either the OpenAI SDK is silently coercing before our code sees it, or current modelos/Fireworks paths are broken at runtime. Flagged as §2.2 gap + Stream A must fix. | 2026-07-18 |
| **Stream A live Groq contract** | `--json --timestamps` plus `--ndjson` against a generated spoken WAV | ✅ JSON status/code/mode were `ok`/`OK`/`transcribe`; `provider_meta.x_groq.id` present; disk JSON matched stdout modulo `request_id`; NDJSON parsed cleanly and ended with `OK`. | 2026-07-20 |
| **Stream A live modelos contract** | `--ndjson --timestamps` against 2s and 0.35s generated spoken WAVs | ✅ Authentication succeeded; null quality metrics remained null with `PROVIDER_FIELD_NULL`; 0.35s fixture emitted `TIMESTAMP_CLAMPED` and segment end equalled `0.3499375`; final code `OK`. | 2026-07-20 |
| **Stream A final local QA** | compileall, pytest, CLI error probes, Pyright on modified contract modules, `git diff --check` | ✅ 29 tests passed; missing file exit 66/`FILE_NOT_FOUND`; modelos words exit 64/`CAPABILITY_UNSUPPORTED`; zero diagnostics in `contract.py`, `emitter.py`, and `cli.py`; clean diff check. | 2026-07-20 |

**Raw probe data locations (ephemeral, deleted on reboot):**
- `/tmp/probe_out/modelos_raw.json`
- `/tmp/probe_out/modelos_text.txt`
- Recreate with `/tmp/probe_schema.py` (script not committed).

---

## 12. Changelog

| Date | Version / Change | Notes |
|---|---|---|
| 2024-05-22 | v1.0 narrative | Original brain-dump. Identified "UI is a rendering of an output contract." Historical only — superseded. |
| 2026-07-18 | v2.0 | Superseded v1 narrative; grounded claims in code reality; defined concrete output contract; sequenced work into streams A–F; inventoried authoritative vs stale docs; added agent operating protocol. |
| 2026-07-18 | v2.1 | Contract validated empirically against modelos + Groq; schema hardened against vLLM null fields; exit codes / event vocabulary expanded per Oracle review; on-disk JSON unified with stdout JSON. |
| 2026-07-18 | v2.2 | Fireworks dropped per user decision; provider architecture reframed around two-pattern registry (custom adapter vs OpenAI-compat config-only); Groq empirically validated working (0.84s, all fields populated). |
| 2026-07-18 | v2.3 | 3 open decisions resolved: Groq stays Pattern 1 custom adapter; `provider_meta` opaque with demand-driven promotion rule; `langcodes` for language normalisation. Stream A scope updated. |
| 2026-07-18 | (review) | Design review request sent to external engineers. 8 questions asked. |
| 2026-07-18 | (review) | Perplexity GPT-5.6 response received. Direction validated. 4 contradictions raised. 16 additional recommendations. |
| 2026-07-19 | settling | Final synthesis of v2.3 + external review. 4 contradictions resolved (null pass-through over defaults; capabilities over warnings; transactional writes; SIGINT ≠ success). NDJSON pruned to 5 event types. Streams B/E/F deferred for solo-dev focus. |
| 2026-07-19 | master prompt v1 | Single 880-line master prompt created. Briefly authoritative. |
| **2026-07-19** | **3-file split (this structure)** | **Master prompt split into `AGENTS.md` (rules), `MEMORY.md` (this file — state/history), `docs/CONTRACT.md` (spec). Cleaner separation of always-loaded rules from on-demand reference material. Adopted all prior resolutions. Test-first + commit-first enforced as AGENTS.md §2 hard prerequisites.** |
| 2026-07-20 | pre-a baseline committed | 7 atomic `pre-a(...)` commits captured the previously uncommitted baseline: robust recorder, BaseSTTClient foundation, Groq, modelos, Fireworks dead-code baseline, CLI wiring, and reliability docs. Pre-Stream-A Step 1 is complete; Step 2 tests remain. |
| 2026-07-20 | Fireworks removed | `fireworks_client.py`, the `FIREWORKS_API_KEY` entry in `.env.example`, and the `cli.py` dispatcher choice deleted per user decision (2026-07-18). Provider count is now 2 (Groq, modelos). Stream C scope reduced to the provider registry only. |
| 2026-07-20 | pre-a step 2 tests | pytest baseline created: 17 tests across 5 files + 2 JSON fixtures. 6 pass (Groq parsing, robust recorder), 9 fail (all meaningful — `src/contract.py` + `src/i18n.py` not yet created, §2.2 null-field parser gap), 2 skipped (modelos blocked by parser crash). Pre-Stream-A Step 2 is complete; Stream A is unblocked. |
| 2026-07-20 | Stream A implementation candidate | Added Pydantic v1.0 contract models, JSON Schema artifacts, human/JSON/NDJSON emitters, null-safe provider normalisation, global offsets, capability fail-fast, structured exits, transactional writes, and golden/black-box tests. 29 tests pass after five-role review fixes. Live provider success-path DoD remains before freezing v1.0. |
| **2026-07-20** | **Stream A / contract v1.0 shipped** | **All 17 DoD items passed. Live Groq JSON/NDJSON and modelos null/clamp paths verified; 29 tests pass; contract `schema_version: "1.0"` frozen.** |

---

*End of MEMORY.md. For agent operating rules: [AGENTS.md](../AGENTS.md). For contract spec: [docs/CONTRACT.md](../docs/CONTRACT.md).*
