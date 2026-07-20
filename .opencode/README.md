# voice_note Agentic Environment

> **⚠️ AUTHORITATIVE DOCS MOVED (2026-07-19).** Strategy, rules, and contract spec are now at the repo root:
> - **`/AGENTS.md`** — agent operating rules, scope, non-goals, protocol (read every session)
> - **/MEMORY.md** — state, history, decisions, validation log (loaded when context needed)
> - **/docs/CONTRACT.md** — output contract spec (loaded for Stream A work)
>
> The `.opencode/agent/*.md` files below are **partially stale** — each has a "defer to AGENTS.md" banner. Full resync deferred until Stream A starts.

Production-grade, state-aware agent registry for systems-level audio engineering.

## Architecture

```
.opencode/
├── agent-environment.json        # GROUND TRUTH - read first, always
├── agent/
│   ├── environment-steward.md    # Refreshes state at session start
│   ├── audio-systems-architect.md # Signal handling, audio subsystems
│   ├── groq-pipeline-validator.md # API calls, transcription quality
│   ├── robustness-engineer.md    # Edge case hunting (read-only)
│   ├── integration-mapper.md     # Clipboard, shell, downstream tools
│   └── legacy/                   # Archived original agents
├── hub-excerpts/
│   └── patterns-synthesis.md     # Extracted patterns (reference)
└── README.md
```

## Core Principle

**State-first operation.** Every agent reads `agent-environment.json` before acting. This file contains:
- Runtime environment (Python version, OS, audio backend)
- Platform-specific behaviors
- Known issues with severity levels
- Whisper model registry with trade-offs

## Agents

### environment-steward
**Trigger**: Session start, environment changes
**Action**: Refreshes `agent-environment.json` with current runtime state

### audio-systems-architect (PRIMARY)
**Use for**: Recording failures, signal handling, Ctrl+C corruption, platform issues
**Expertise**: PyAudio/PortAudio, UNIX signals, terminal control, WAV format

### groq-pipeline-validator
**Use for**: Transcription failures, API errors, model selection, chunking
**Expertise**: Whisper API, rate limiting, file size constraints, retry patterns

### robustness-engineer (READ-ONLY)
**Use for**: Finding edge cases, failure analysis, test scenario design
**Expertise**: Race conditions, platform comparison, stress testing
**Note**: Does NOT make changes - analysis only

### integration-mapper
**Use for**: Clipboard operations, file system events, shell integration
**Expertise**: pbcopy/xclip, fswatch, shell aliases, output formatting

## Key Reference

**Consulting report**: `docs/reports/signal_handling_corruption_analysis.md`

Contains deep technical analysis of current issues with recommended fixes. Agents reference this report for task-specific context.

## Design Decisions

1. **Specialists over generalists** - Each agent has narrow, deep expertise
2. **Concise communication** - CLI-optimized, 1-3 sentences default
3. **Platform-aware** - macOS vs Linux differences explicitly handled
4. **State is ground truth** - `agent-environment.json` is authoritative
5. **Personality in agents, context in docs** - Agents are reusable across sessions

## Patterns Source

Agent architecture extracted from:
- `agentic-forge` - Agent creation methodology, consulting patterns
- `~/.config/opencode/hub/` - Production prompts (Claude Code, Jules, Bolt.new)

## Legacy Agents

Original agents preserved in `agent/legacy/` for reference:
- `systems-audio.md` → replaced by `audio-systems-architect.md`
- `groq-integration.md` → replaced by `groq-pipeline-validator.md`
- `robustness-tester.md` → replaced by `robustness-engineer.md`
