# voice-note-resume

> **Resume skill for voice_note multi-session development.**
> Load this first when picking up voice_note work. Gives the 60-second briefing + first action.

## When to load

Load this skill when:
- Starting a new session that will touch voice_note code
- User says "continue voice_note", "pick up vn", "resume", or similar
- You see `/Users/fabiofalopes/projetos/hub/voice_note/` as the working directory and aren't sure what to do first

## 60-second briefing

voice_note is a **contract-first CLI** for voice recording + transcription. Stream A's 17-item DoD passed and contract v1.0 was frozen on 2026-07-20.

**Three authoritative docs (read in this order if unfamiliar):**
1. [`AGENTS.md`](../AGENTS.md) — agent rules + scope + execution environment + non-goals (loaded automatically)
2. [`docs/CONTRACT.md`](../docs/CONTRACT.md) — the output contract spec (load when working Stream A)
3. [`MEMORY.md`](../MEMORY.md) — state, history, validation, decisions (load when context needed)

## Current phase: Stream A shipped, commit pending

Both Pre-Stream-A prerequisites and all Stream A DoD checks are complete. Live
Groq/modelos success paths passed, 29 tests pass after five-role review, and
contract v1.0 is frozen. The Stream A cluster remains uncommitted; do not mix it
with a later stream and do not commit without an explicit user request.

Resume from:

- [`docs/handoffs/2026-07-21-stream-a-live-validation.md`](../docs/handoffs/2026-07-21-stream-a-live-validation.md)
- [`MEMORY.md §2.3`](../MEMORY.md#23-baseline-commit-status)

Treat the completed handoff and `MEMORY.md §11` as the validation evidence.

## First action recipe

```bash
# 1. Always run this first — orient yourself
cd /Users/fabiofalopes/projetos/hub/voice_note
git status                                      # see uncommitted state
git log --oneline -10                           # see recent commits

# 2. Verify execution environment
./venv/bin/python transcribe.py --help          # CLI works
./venv/bin/python -c "import groq, openai"       # deps available
test -f .env && echo ".env present"              # API keys available

# 3. Read what you need
# - AGENTS.md is auto-loaded by OpenCode (you've already read it)
# - Skim MEMORY.md §2 Current state if you need grounding
# - Open docs/CONTRACT.md if working Stream A
# - Check .opencode/agent-environment.json for runtime state

# 4. Preserve the completed Stream A cluster
#    - Read docs/handoffs/2026-07-21-stream-a-live-validation.md
#    - Do not mix later-stream changes into the uncommitted Stream A work
#    - Commit only when the user explicitly requests it
```

## Hard rules (non-negotiable)

- **Never commit secrets** — `.env` is gitignored; verify before any commit
- **Never use `python3`** — use `./venv/bin/python` (Python 3.14 + deps in venv)
- **Never use the `vn` alias** — agents don't inherit user shell aliases; invoke `./venv/bin/python transcribe.py` directly
- **Never edit `recorder.py` or `utils.py`** — pre-existing LSP errors are documented tech debt
- **Never suppress types** (`# type: ignore`, bare `except:`)
- **One stream per commit cluster** — don't mix Stream A work with Stream D
- **Verify scope against AGENTS.md §5 Non-goals** before starting any task

## Stream A DoD quick reference

The full 17-item Definition of Done lives in [`docs/CONTRACT.md §Stream A DoD`](../docs/CONTRACT.md#stream-a-definition-of-done). Highlights:

```bash
# 1. NDJSON success
./venv/bin/python transcribe.py recording.wav --ndjson | jq 'last | .data.code'
# → must return "OK"

# 2. FILE_NOT_FOUND exit code
./venv/bin/python transcribe.py nonexistent.wav --ndjson; echo $?
# → must exit 66 with "code":"FILE_NOT_FOUND"

# 3. modelos nulls preserved (NOT defaulted to 0.0)
./venv/bin/python transcribe.py recording.wav --provider modelos --ndjson | jq 'first | select(.type=="segment") | .data.avg_logprob'
# → must be null, not 0.0

# 4. Capability check fails fast
./venv/bin/python transcribe.py recording.wav --word-timestamps --provider modelos; echo $?
# → must exit 64 with "Provider 'modelos' does not support word timestamps"

# 5. Ctrl+C → exit 130 (NOT 0)
# interrupt mid-transcription; echo $? → 130
```

## Common pitfalls

- **Forgetting to translate `vn` → `./venv/bin/python transcribe.py`** in DoD verifications
- **Working from wrong directory** — always `cd /Users/fabiofalopes/projetos/hub/voice_note` first
- **Treating any strategy doc other than AGENTS/MEMORY/CONTRACT as authoritative** — they all have STALE/DRIFTED banners for a reason
- **Reopening frozen v1.0 fields casually** — additive changes require a minor bump; breaking changes require a major bump
- **Suppressing null quality fields with defaults** — the contract preserves null per docs/CONTRACT.md §11

## What to do if interrupted mid-task

1. Re-run `git status` to see what's changed
2. Re-read AGENTS.md (auto-loaded) + MEMORY.md §2 to re-orient
3. Check the relevant DoD checklist (Stream A in docs/CONTRACT.md, Stream D in MEMORY.md §4)
4. Find the first unchecked item
5. Verify prerequisites, then continue

## Handoff protocol (end of session)

Before ending a session:
1. Run `git status` — know what's uncommitted
2. If work is complete and verified → commit only when explicitly requested, following AGENTS.md §6.4
3. If work is incomplete → ensure files are in a recoverable state on disk
4. Update MEMORY.md §12 Changelog if a meaningful unit was completed
5. (Optional) Create a session handoff note in `docs/handoffs/` for complex in-progress state

---

*This skill is the entry point. The deep content lives in AGENTS.md / MEMORY.md / docs/CONTRACT.md.*
