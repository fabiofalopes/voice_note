# Optimized Agentic Environment Prompt

**Generated**: 2025-12-07  
**Source**: Synthesis of agentic-forge patterns, voice_note agents, ~/.config/opencode production configs

---

## Core Philosophy

> **Agents are tools, not personas.** Each agent serves a specific purpose with clear boundaries. They are experts that enhance interactions within this development environment.

The goal: Build a bulletproof, consulting-grade agentic environment where specialized agents embody production-level patterns for conciseness, tool usage, edge-case handling, and deep technical expertise.

---

## Naming Convention (Fixed)

| OLD Name | NEW Name | Rationale |
|----------|----------|-----------|
| `groq-integration` | `transcription-api` | Describes function, not vendor lock-in |
| `systems-audio` | `systems-audio` | Already descriptive (keep) |
| `robustness-tester` | `robustness-analyzer` | "Analyzer" implies read-only better |

---

## Environment State File

**NEW**: Create `PROJECT_STATE.md` - a single source of truth that ALL agents read at session start.

```markdown
# voice_note Project State

**Last Updated**: 2025-12-07 14:30:00 UTC
**Updated By**: systems-audio agent

## Current Version
- **Version**: 0.3.0
- **Python**: 3.11+
- **Dependencies**: pyaudio, groq, sounddevice

## Feature Status

| Feature | Status | Last Modified |
|---------|--------|---------------|
| Basic recording | ✅ Working | 2025-11-15 |
| Groq transcription | ✅ Working | 2025-12-01 |
| Signal handling | ⚠️ Issues | 2025-12-07 |
| Cross-platform | 🔧 Partial | 2025-12-05 |

## Known Issues (Active)
1. **Ctrl+C corruption** - Terminal state not restored on interrupt
2. **WAV truncation** - File corruption on abrupt stop

## Toolbox
- **Audio backend**: PyAudio with PortAudio
- **Transcription**: Groq Whisper API (large-v3)
- **Config**: Environment variables (.env)

## Platform Support
| Platform | Status | Notes |
|----------|--------|-------|
| macOS | ✅ Primary | CoreAudio via PortAudio |
| Linux | ✅ Tested | PipeWire/PulseAudio |
| Windows | ❌ Untested | Not prioritized |

## Roadmap (Next)
1. Fix signal handling (Phase 1 from consulting report)
2. Add retry/backoff to Groq client
3. Cross-device mobile recording (future)
```

**Session Protocol**: Every agent reads `PROJECT_STATE.md` at session start and the `state-updater` agent updates it at session end.

---

## Agent Architecture

### Tier 1: Primary Development

#### `systems-audio` (unchanged, validated)
- **Role**: Core systems engineering
- **Permissions**: Full write/edit/bash
- **Model**: claude-sonnet-4.5 (balanced)
- **Tools**: All enabled
- **Session duty**: Update PROJECT_STATE.md when making changes

#### `transcription-api` (renamed from groq-integration)
- **Role**: API integration specialist
- **Permissions**: Full write/edit/bash
- **Model**: claude-sonnet-4.5
- **Tools**: +sequential_thinking, +memory, +gh_grep (ADDED)
- **Focus**: Reliability, retry logic, error handling

### Tier 2: Analysis (Read-Only)

#### `robustness-analyzer` (renamed from robustness-tester)
- **Role**: Edge case hunter
- **Permissions**: READ-ONLY (edit: deny)
- **Mode**: `all` (not `primary` - FIXED)
- **Model**: claude-haiku-4.5 (cheap, analysis)
- **Output**: Structured findings with severity

### Tier 3: Meta Agents

#### `state-updater` (NEW)
- **Role**: Maintain PROJECT_STATE.md
- **Permissions**: Write only to PROJECT_STATE.md
- **Model**: claude-haiku-4.5 (cheap)
- **Trigger**: End of session, after significant changes
- **Tools**: read, glob, write (restricted path)

---

## Agent Definition Standard

All agents MUST follow this template:

```yaml
---
description: One clear sentence describing purpose
mode: primary | subagent | all
model: provider/model-name
temperature: 0.0-1.0
maxSteps: 20
tools:
  # Core
  write: true/false
  edit: true/false
  bash: true/false
  read: true/false
  grep: true/false
  glob: true/false
  list: true/false
  # Task tracking
  todowrite: true/false
  todoread: true/false
  # MCP
  memory: true/false
  sequential_thinking: true/false
  context7: true/false
  gh_grep: true/false
  code_interpreter: true/false
  fetch: true/false
  duckduckgo: true/false
permission:
  edit: allow | ask | deny
  bash:
    "safe-command*": allow
    "risky-command*": ask
    "dangerous*": deny
    "*": ask
  webfetch: allow | ask | deny
---

# Agent Name

## Identity
One paragraph: who you are, what you do, what you don't do.

## Session Protocol
1. Read PROJECT_STATE.md
2. Check TODO.md if exists
3. [Do work]
4. Update PROJECT_STATE.md if you made changes

## Core Principles
1. [Principle 1]
2. [Principle 2]
3. [Principle 3]

## Communication Style
- Concise: 1-3 sentences default
- No preamble ("Based on...", "Let me help...")
- Direct: Question → Answer
- CLI-friendly output

## Tool Preferences
| Task | Tool | Why |
|------|------|-----|
| Search code | Grep | Not bash rg |
| Find files | Glob | Not bash find |
| Before edit | Read | Check existing patterns |

## Anti-Patterns
❌ [What NOT to do]
❌ [What NOT to do]

## Project Context
Reference: [relevant docs]
```

---

## Critical Fixes to Apply

### 1. Rename `groq-integration.md` → `transcription-api.md`

### 2. Add Missing Tools to `transcription-api`
```yaml
tools:
  # ADD these:
  sequential_thinking: true  # Complex API debugging
  memory: true               # Store API quirks across sessions
  gh_grep: true              # Find Groq usage patterns
```

### 3. Fix `robustness-analyzer` Mode
```yaml
mode: all  # Was: primary (incorrect for read-only)
```

### 4. Create `state-updater.md` Agent
New meta-agent to maintain project state.

### 5. Create `PROJECT_STATE.md`
Single source of truth for all agents.

---

## Lifecycle Integration with agentic-forge

```
agentic-forge/              voice_note/.opencode/
     |                             |
     |  [Pattern Library]          |  [Production Instance]
     |                             |
     v                             v
examples/agents/  -------->  agent/
     |                             |
     | [When mature]               |
     |                             |
     +<--------------------------- + [Graduate back]
```

**Flow**:
1. Develop agents locally in `voice_note/.opencode/agent/`
2. Test and iterate with real usage
3. When patterns are validated, extract to `agentic-forge/examples/agents/`
4. agentic-forge becomes the reference library for new projects

---

## Prompt Patterns to Enforce

### Pattern 1: Session Start
```
Every agent begins with:
1. Read PROJECT_STATE.md
2. Acknowledge current state briefly (1 line)
3. Proceed with task
```

### Pattern 2: Session End
```
state-updater agent runs:
1. Diff what changed during session
2. Update PROJECT_STATE.md
3. Summarize in 2-3 bullets
```

### Pattern 3: Sub-Agent Invocation
```
When systems-audio needs API help:
@transcription-api "Check if Groq client handles 429 errors properly"

Sub-agent does focused work, returns finding.
Primary agent integrates.
```

### Pattern 4: Robustness Check
```
Before merging any systems change:
@robustness-analyzer "Review signal_handler changes for edge cases"

Returns structured finding report.
```

---

## Validation Checklist

Before deploying any agent:

- [ ] `description` is one clear sentence
- [ ] `mode` matches use pattern (primary/subagent/all)
- [ ] `permission` follows least-privilege
- [ ] Communication style section exists
- [ ] Anti-patterns documented
- [ ] No task-specific content (belongs in project docs)
- [ ] References PROJECT_STATE.md in session protocol

---

## Implementation Order

1. **Now**: Rename + fix existing agents
2. **Now**: Create PROJECT_STATE.md
3. **Now**: Create state-updater agent
4. **Later**: Add XML workflow examples to complex agents
5. **Later**: Graduate mature patterns to agentic-forge

---

## Files to Create/Modify

| Action | File | Priority |
|--------|------|----------|
| Rename | groq-integration.md → transcription-api.md | HIGH |
| Edit | transcription-api.md (add tools) | HIGH |
| Edit | robustness-tester.md → robustness-analyzer.md | HIGH |
| Create | PROJECT_STATE.md | HIGH |
| Create | state-updater.md | MEDIUM |
| Edit | README.md (update references) | MEDIUM |

---

## Success Criteria

- [ ] All agents read PROJECT_STATE.md at session start
- [ ] Naming is function-descriptive (not vendor-specific)
- [ ] Read-only agents have `mode: all` + `edit: deny`
- [ ] Development agents have full tooling
- [ ] Meta-agent maintains project state automatically
- [ ] Patterns are documented for graduation to agentic-forge
