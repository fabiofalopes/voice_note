---
created: 2026-07-20
session_goal: Pre-Stream-A Step 2 — write tests
type: handoff
audience: next agent session
author: sisyphus (2026-07-20 handoff)
status: ready
---

# 2026-07-20 — Pre-Stream-A Step 2: Write Tests

> **Use this prompt to start the next session.**
> This session's goal is to create the missing pytest baseline so Stream A can begin safely.

---

## Session objective

Create the initial `tests/` suite described by **AGENTS.md §3 Pre-Stream-A Step 2**.

Pre-Stream-A Step 1 is now complete: the baseline code is committed in atomic `pre-a(...)`
history. The next hard prerequisite is to write the executable-spec test suite that will
fail meaningfully before Stream A and pass after Stream A lands.

Until these tests exist, Stream A contract work remains blocked by **MEMORY.md §3.3 #20**.

---

## Read first (in this order)

1. **AGENTS.md** (auto-loaded by OpenCode). Confirm you've read:
   - §2 Execution environment (venv path, Python 3.14, invocation pattern)
   - §3 Current scope (Pre-Stream-A Step 2 + Stream A)
   - §6 Agent operating protocol (test-first rule, handoff rule, commit conventions)
2. **skills/voice-note-resume.md** — 60-second briefing + first action recipe
3. **MEMORY.md**
   - §2.1–§2.4 for current code reality and provider quirks
   - §3.3 #20 confirming tests are a hard prerequisite
   - §11 Validation log for empirically verified provider behavior to encode in fixtures/tests
4. **docs/CONTRACT.md**
   - especially the normalisation expectations and Stream A DoD items the tests must enforce

---

## Concrete task

The repo still has **no `tests/` directory**. Create it.

Minimum target from **AGENTS.md §3 Pre-Stream-A Step 2**:

- `tests/test_groq_parsing.py`
- `tests/test_modelos_parsing.py`
- `tests/test_normalization.py`
- `tests/test_robust_recorder.py`
- `tests/test_contract_schema.py`

Expected scope: **~10 tests minimum** across those files.

### What the tests must cover

1. **Groq parsing fixture**
   - Parse a Groq verbose response fixture
   - Assert all expected fields populate cleanly
   - Preserve current empirical behavior from `MEMORY.md §11`

2. **modelos parsing fixture**
   - Parse a modelos verbose response fixture
   - Assert null quality fields are preserved as null-like inputs to the future normaliser,
     not defaulted to synthetic values in the contract layer
   - Encode the known `end` hallucination case from `MEMORY.md §2.4`

3. **Normalisation tests**
   - Null pass-through
   - Language normalisation via `langcodes`
   - `end` clamping behavior

4. **Robust recorder tests**
   - Fault-recovery paths
   - Chunk save / merge behavior where testable without real hardware
   - Keep this surgical; do not refactor recorder implementation during test writing

5. **Contract schema tests**
   - Pydantic v2 validation expectations for the future `src/contract.py`
   - These tests should describe the desired contract shape even if they fail before Stream A

### Recommended supporting artifacts

If needed, add fixtures under a sensible test-only path such as:

- `tests/fixtures/groq_verbose_response.json`
- `tests/fixtures/modelos_verbose_response.json`

Keep fixtures small and representative.

---

## Hard rules (non-negotiable)

- **Do not start Stream A implementation.** This session writes tests only.
- **Do not refactor production code** just to make tests easier.
- **Do not touch `recorder.py` or `utils.py`** unless a test absolutely requires import-safe minimal handling; no behavior refactors.
- **Do not suppress failing expectations.** If the code does not satisfy the contract yet, let the tests fail meaningfully.
- **Use `./venv/bin/python` and `./venv/bin/pytest`** — never `python3`, never the `vn` alias.
- **Preserve current scope.** No new providers, no Stream D default flip, no packaging work, no UI work.
- **Commit only test-related artifacts** plus any minimal supporting updates strictly required to run them.

---

## Success criteria (definition of done for that session)

- [ ] `tests/` directory exists with the required target files
- [ ] At least ~10 meaningful tests exist across parsing / normalisation / recorder / contract schema
- [ ] Fixtures exist for Groq + modelos parsing cases
- [ ] Tests encode the known provider quirks from `MEMORY.md §2.4` / §11
- [ ] `./venv/bin/pytest` runs and produces meaningful results
- [ ] Any current failures clearly correspond to not-yet-implemented Stream A behavior, not broken test setup
- [ ] `MEMORY.md §12` updated if a meaningful prerequisite milestone is completed
- [ ] A new handoff is written for the first Stream A session once Step 2 is complete

---

## When done: hand off to Stream A

If Step 2 completes successfully, the next handoff should start the **first Stream A session**.

That next handoff should direct the agent to:

1. Re-read `docs/CONTRACT.md`
2. Use the new tests as the executable spec
3. Implement `src/contract.py`, `src/i18n.py`, `src/emitter.py`, and the Stream A CLI/base-client changes
4. Run the full Stream A DoD checklist

---

## If interrupted mid-session

1. Re-read AGENTS.md + MEMORY.md §2 / §3.3 / §11
2. Run `git status` to see what test artifacts exist
3. Run `./venv/bin/pytest` to distinguish setup failures from expected contract failures
4. Resume from the first missing coverage area

Do not start Stream A early just because some tests already exist.

---

## Anti-scope (do NOT do in that session)

- ❌ Start Stream A contract implementation
- ❌ Refactor provider code beyond what is strictly necessary for importable/testable seams
- ❌ Touch `recorder.py` or `utils.py` as cleanup work
- ❌ Flip `--robust` default (Stream D)
- ❌ Add packaging / `pyproject.toml` / installable CLI work (Stream B)
- ❌ Start any UI or local MLX work (Streams F / E)

---

*End of handoff. For agent operating rules: [AGENTS.md](../../AGENTS.md). For project state/history: [MEMORY.md](../../MEMORY.md). For contract spec: [docs/CONTRACT.md](../CONTRACT.md). For 60-second briefing: [skills/voice-note-resume.md](../../skills/voice-note-resume.md).* 
