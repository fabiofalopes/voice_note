---
created: 2026-07-20
session_goal: Pre-Stream-A Step 1 — commit baseline
type: handoff
audience: next agent session
author: sisyphus (2026-07-20 handoff)
status: ready
---

# 2026-07-20 — Pre-Stream-A Step 1: Commit Baseline

> **Use this prompt to start the next session.**
> This session's goal is to commit the entire uncommitted voice_note baseline in logical chunks, unblocking Stream A.

---

## Session objective

Commit the entire uncommitted voice_note baseline (provider-abstraction layer + robust recorder + reliability docs + supporting code modifications) in logical chunks, per **AGENTS.md §3 Pre-Stream-A Step 1**.

This is the **hard prerequisite** for everything else. Until these 12 uncommitted entries are committed, no Stream A work can safely begin (per MEMORY.md §3.2 #19).

---

## Read first (in this order)

1. **AGENTS.md** (auto-loaded by OpenCode). Confirm you've read:
   - §2 Execution environment (venv path, Python 3.14, invocation pattern)
   - §3 Current scope (Pre-Stream-A + Stream A)
   - §6 Agent operating protocol (hard rules, commit conventions)
2. **skills/voice-note-resume.md** — 60-second briefing + first action recipe
3. **MEMORY.md §2.3** — exact uncommitted file list (verify against current `git status`)
4. **MEMORY.md §3.2 #19** — confirms this is a hard prerequisite, not optional

---

## Concrete task

Working tree currently has 12 uncommitted entries (verify with `git status`).

**Modified (6 files — pre-existing code changes from prior session):**
- `.env.example`
- `requirements.txt`
- `src/api/__init__.py`
- `src/api/groq_client.py`
- `src/cli.py`
- `src/config.py`

**Untracked (6 entries — new code/docs):**
- `ROBUST_RECORDING.md`
- `docs/reports/` (contains `signal_handling_corruption_analysis.md`)
- `src/api/base_client.py` (the ABC, 579 LOC)
- `src/api/fireworks_client.py` (179 LOC, **dead code — since removed 2026-07-20**)
- `src/api/modelos_client.py` (165 LOC)
- `src/audio_processing/robust_recorder.py` (471 LOC)

### Recommended chunking (6 commits)

Adapt if a cleaner grouping emerges from reading the code:

| # | Commit message | Files |
|---|---|---|
| 1 | `pre-a(recorder): commit fault-tolerant robust recorder baseline` | `src/audio_processing/robust_recorder.py` |
| 2 | `pre-a(providers): commit BaseSTTClient ABC with shared chunked pipeline` | `src/api/base_client.py`, `src/api/__init__.py` |
| 3 | `pre-a(groq): rewrite Groq client as BaseSTTClient subclass (Pattern 1)` | `src/api/groq_client.py` |
| 4 | `pre-a(modelos): add modelos client (Pattern 2 OpenAI-compat)` | `src/api/modelos_client.py` |
| 5 | `pre-a(fireworks): add Fireworks client (dead on arrival, since removed 2026-07-20)` | `src/api/fireworks_client.py` |
| 6 | `pre-a(cli+docs): wire provider dispatch + commit reliability + signal-handling docs` | `src/cli.py`, `src/config.py`, `.env.example`, `requirements.txt`, `ROBUST_RECORDING.md`, `docs/reports/` |

---

## Hard rules (non-negotiable)

- **One logical chunk per commit.** Don't lump recorder + providers + docs together. Smaller, atomic commits are easier to review and revert.
- **Verify each commit doesn't include secrets.** `.env` is gitignored — verify with `git check-ignore .env` (should return `.gitignore:9:.env .env`). NEVER use `git add .` or `git add -A` — stage files explicitly by name.
- **Use the commit message convention from AGENTS.md §6.4**: `pre-a(<scope>): <imperative summary>` + body explaining why + DoD status.
- **Don't refactor anything during these commits.** This is "commit existing work", not "improve existing work". Refactoring belongs to Stream A (or later). Pre-existing LSP errors in `recorder.py` / `utils.py` are documented tech debt — leave them alone.
- **Don't write tests in this session.** That's Pre-Stream-A Step 2 — a separate session.
- **Don't start Stream A.** Pre-Stream-A must complete first (both Step 1 AND Step 2).
- **Don't touch `recordings/` or `audio/`** — both are gitignored byproducts. The user manages compression/cleanup manually.

---

## Success criteria (definition of done for this session)

- [ ] `git status` shows clean working tree (no uncommitted entries)
- [ ] `git log --oneline -10` shows ~6 new `pre-a(...)` commits (number may vary if chunking adapts)
- [ ] Each commit is atomic and logically grouped (no mixed concerns)
- [ ] Each commit message body references AGENTS.md §3 Pre-Stream-A Step 1
- [ ] `.env` was NEVER staged (verify: `git log --all --diff-filter=A -- .env` returns nothing)
- [ ] After all commits: `./venv/bin/python transcribe.py --help` still works (sanity check)
- [ ] MEMORY.md §2.3 updated to mark baseline as COMMITTED (was: 6 modified + 12 untracked; should reflect new reality)
- [ ] MEMORY.md §12 Changelog gets a new entry: `2026-07-20 | pre-a baseline committed | ...`

---

## When done: write the next handoff

Before ending the session, create the next handoff prompt for **Pre-Stream-A Step 2 (write tests)** at:

```
docs/handoffs/YYYY-MM-DD-pre-stream-a-step2-tests.md  <!-- historical path; current location: internal/handoffs/ -->
```

Use the **same structure** as this file:
- Frontmatter: `created`, `session_goal`, `type: handoff`, `status: ready`
- Session objective
- Read-first list
- Concrete task (Pre-Stream-A Step 2 details from AGENTS.md §3)
- Hard rules
- Success criteria
- Handoff-to-next instructions (the next session after Step 2 starts Stream A)

**Date-stamp with the actual date the next session starts**, not today's date.

---

## If interrupted mid-session

1. Re-read AGENTS.md + MEMORY.md §2.3
2. Run `git status` to see what's been committed vs not
3. Run `git log --oneline -10` to see which chunks are already done
4. Resume from the next unstarted chunk

Don't restart from scratch. Don't re-commit work that's already committed. Don't `git reset` unless something is genuinely broken.

---

## Anti-scope (do NOT do in this session)

- ❌ Write tests (Step 2's job)
- ❌ Start Stream A contract implementation (Step 1 + Step 2 prerequisites not yet met)
- ❌ Refactor any code (Stream A's job or later)
- ❌ Touch `recorder.py` or `utils.py` (documented tech debt per AGENTS.md §7 #3)
- ❌ Edit strategy docs that are banner-marked STALE/DRIFTED/ABANDONED
- ❌ Add new providers
- ❌ Flip `--robust` default (Stream D's job, parallel/optional for v1.0)
- ❌ Commit anything not listed in §"Concrete task" above

---

*End of handoff. For agent operating rules: [AGENTS.md](../../AGENTS.md). For project state/history: [MEMORY.md](../MEMORY.md). For contract spec: [docs/CONTRACT.md](../../docs/CONTRACT.md). For 60-second briefing: [skills/voice-note-resume.md](../../skills/voice-note-resume.md).*
