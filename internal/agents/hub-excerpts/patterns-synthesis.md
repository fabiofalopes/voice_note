# Agentic Patterns Synthesis

> **⚠️ PRE-CONTRACT ERA (2026-07-19).** This is meta-documentation about agent design patterns, retained for reference. The agent ecosystem itself has banners deferring to [`../../../AGENTS.md`](../../../AGENTS.md). Do not act on roadmap references below without checking current scope in [`../../MEMORY.md`](../../MEMORY.md).

**Created**: 2025-12-07  
**Purpose**: Consolidated patterns from agentic-forge and hub research  
**Mission**: Cross-pollinate excellence into voice_note specialized agents

---

## Sources Analyzed

### Phase 1: Agentic-Forge (Pokeindex Research)
- `docs/AGENT_CREATION_REFERENCE.md` (1052 lines)
- `docs/AGENTIC_SETUP_GUIDE.md` (307 lines)
- `patterns/agentic-prompt-patterns.md` (70 lines)
- `examples/agents/`: security, container, scraper (750+ lines combined)
- `CONTEXT.md` (178 lines)

### Phase 2: Hub Prompts
- `awesome-ai-system-prompts/Claude-Code/System.js` (133 lines)
- `agentic-system-prompts/CLAUDE.md` (290 lines)
- `Prompt_Engineering/cot-prompting.ipynb`
- `Prompt_Engineering/task-decomposition-prompts.ipynb`
- `cursor-commands/.cursor/commands/security-review.md`

**Total Material**: ~3000+ lines analyzed

---

## Pattern Categories

### 1. Agent Configuration Patterns

#### 1.1 Mode Validation (Agentic-Forge)
**Source**: `AGENT_CREATION_REFERENCE.md`

```yaml
# CORRECT
mode: primary    # Full dev capabilities
mode: subagent   # Task-specific invocation
mode: all        # Universal access

# INCORRECT (common mistakes)
mode: secondary  # Does not exist!
mode: readonly   # Use `all` with read-only permissions instead
```

**Application to voice_note**:
- ✅ `systems-audio.md`: Use `primary` (main systems work)
- ✅ `groq-integration.md`: Use `primary` (API integration work)
- ✅ `robustness-tester.md`: Use `all` with read-only permissions

#### 1.2 Permission Granularity (Agentic-Forge)
**Source**: `examples/agents/security.md`, `container.md`

```yaml
permission:
  edit: allow
  bash:
    # Dangerous operations - always ask
    "rm -rf*": ask
    "git push*": ask
    "docker system prune*": ask
    
    # Safe, reversible - allow
    "git status*": allow
    "docker ps*": allow
    "npm install*": allow
    
    # Default - ask
    "*": ask
```

**Insight**: Use pattern matching with wildcards, not blanket allows.

**Application to voice_note**:
```yaml
# systems-audio.md - audio/signal tools
bash:
  "pactl*": allow      # PulseAudio inspection
  "pw-*": allow        # PipeWire tools
  "dmesg*": allow      # Kernel logs
  "kill -9*": ask      # Process killing
  "rm*": ask           # File deletion
  "*": ask
```

#### 1.3 Tool Batching (Experimental, Agentic-Forge)
**Source**: `AGENT_CREATION_REFERENCE.md` line 789

```yaml
features:
  tool_batching: true  # Parallel tool execution
```

**Benefit**: Read multiple files simultaneously, run parallel analyses.

**Application to voice_note**: Enable for all 3 agents (research-heavy work).

---

### 2. Prompt Structure Patterns

#### 2.1 XML Structure for Complex Prompts (Replit/Claude Code)
**Source**: `awesome-ai-system-prompts/Claude-Code/System.js`

```xml
<example>
user: What files are in src/?
assistant: [runs ls and sees foo.c, bar.c]
src/foo.c
src/bar.c
src/baz.c
</example>

<example>
user: which file contains foo?
assistant: src/foo.c
</example>
```

**Insight**: Use XML tags for multi-turn examples, not markdown.

**Application to voice_note**: Add workflow examples in agents:

```xml
<example_workflow>
<scenario>User reports Ctrl+C doesn't stop recording</scenario>
<analysis>
1. Read src/audio_processing/recorder.py (signal handlers)
2. Check for PyAudio blocking in record loop
3. Examine exception nesting in cli.py
4. Propose threading.Event + signal-safe flag
</analysis>
</example_workflow>
```

#### 2.2 Conciseness Rules (Claude Code)
**Source**: `System.js` lines 24-65

```txt
IMPORTANT: Minimize output tokens. Answer in 1-3 sentences unless detail requested.
IMPORTANT: No preamble/postamble ("Based on...", "Here is...")
IMPORTANT: One-word answers are best.

Examples:
user: 2 + 2
assistant: 4

user: is 11 prime?
assistant: Yes
```

**Application to voice_note**: Add to all 3 agents:

```markdown
## Response Style
- **Concise**: 1-3 sentences unless detail requested
- **No preamble**: "Based on..." or "Here is..." → direct answer
- **One-word when possible**: "Yes" not "Yes, 11 is prime"
- **CLI-optimized**: Monospace-friendly, no emoji unless asked
```

#### 2.3 Task Decomposition Template (Prompt Engineering)
**Source**: `task-decomposition-prompts.ipynb`

```markdown
## Workflow: Debugging Signal Handling

When investigating signal-related bugs:
1. **List Facts**: Enumerate signals, handlers, blocking calls
2. **Identify Conditions**: C extension blocking? Nested handlers?
3. **Note Constraints**: POSIX rules, platform differences
4. **Generate Scenarios**: Test each hypothesis
5. **Test Scenarios**: Minimal reproducers
6. **Eliminate Inconsistent**: Discard contradictions
7. **Conclude Solution**: State fix with reasoning
8. **Provide Clear Answer**: Implementation steps
```

**Insight**: Systematic breakdown > generic "think step by step".

**Application to voice_note**: Add domain-specific workflows to `systems-audio.md`.

---

### 3. Tool Usage Patterns

#### 3.1 Search Hierarchy (Claude Code)
**Source**: `System.js` lines 101-102

```txt
When doing file search, prefer ${Hv} tool to reduce context usage.
Use ${jw} to run calls in parallel (batch tool calls).
```

**Translation for voice_note**:
- Prefer `Grep` over bash `rg` (built-in, faster)
- Use `Glob` for filename patterns
- Batch multiple `Read` calls when exploring

**Application**: Add to all agents:

```markdown
## Tool Preferences
- **Search**: Use Grep tool, not bash rg
- **Find files**: Use Glob, not bash find
- **Read multiple**: Batch Read calls in one request
- **Bash**: Only for execution (pyaudio, pytest), not search
```

#### 3.2 Convention Following (Claude Code)
**Source**: `System.js` lines 82-86

```txt
When making changes:
1. First understand file's code conventions
2. Mimic code style
3. Use existing libraries and utilities
4. Follow existing patterns
5. NEVER assume library is available - check first
```

**Application to voice_note**:

```markdown
## Before Editing Code

1. **Read first**: Always use Read tool before Edit
2. **Check imports**: What libraries are already used?
3. **Mimic style**: 
   - PyAudio patterns in recorder.py
   - Exception handling style in cli.py
   - Logging format from existing code
4. **Verify dependencies**: Check requirements.txt before suggesting new libs
```

---

### 4. Workflow Patterns

#### 4.1 Security Checklist (Cursor Commands)
**Source**: `cursor-commands/security-review.md`

```markdown
## Security Review Checklist
- [ ] Verified authentication
- [ ] Checked authorization
- [ ] Reviewed session management
- [ ] SQL injection scan
- [ ] XSS/CSRF checks
- [ ] Sensitive data encryption
- [ ] Dependency vulnerabilities
```

**Insight**: Checklists ensure completeness.

**Application to robustness-tester.md**:

```markdown
## Robustness Testing Checklist
- [ ] Signal handling edge cases
- [ ] Terminal state cleanup on abort
- [ ] WAV file corruption scenarios
- [ ] PyAudio resource leaks
- [ ] Cross-platform behavior (macOS/Linux)
- [ ] Race conditions in threading
- [ ] Error propagation paths
```

#### 4.2 Chain-of-Thought Prompting (Prompt Engineering)
**Source**: `cot-prompting.ipynb`

```txt
Solve step by step. For each step:
1. State what you're going to calculate
2. Write the formula (if applicable)
3. Perform the calculation
4. Explain the result
```

**Application to systems-audio.md** (signal debugging):

```markdown
## Debugging Workflow: Signal Delivery Failure

When Ctrl+C doesn't interrupt:
1. **Identify blocking call**: Where is Python blocked?
   - Check PyAudio pa.read() in recorder loop
   - Note: C extensions block signal delivery
2. **Trace signal path**: How does SIGINT reach handler?
   - Python signal.signal() registration
   - Nested try/except shadow zones
3. **Verify handler logic**: What does handler do?
   - Sets self.stop_flag = True
   - Expects loop to check flag
4. **Find the break**: Why doesn't loop see flag?
   - pa.read() blocks before flag check
   - No async interrupt of blocking read
5. **Design solution**: How to make interruptible?
   - threading.Event instead of boolean
   - signal-safe flag + timeout reads
```

---

### 5. Documentation Patterns

#### 5.1 Attribution Requirements (Agentic System Prompts)
**Source**: `agentic-system-prompts/CLAUDE.md`

```markdown
Every file must include:
- **Source URL**: Direct link with line numbers
- **Extraction date**: YYYY-MM-DD
- **Agent version**: If available
- **Conversion notes**: Format changes
```

**Application to voice_note**: When copying patterns:

```markdown
---
# Pattern: XML Workflow Examples
# Source: ~/.config/opencode/hub/awesome-ai-system-prompts/Claude-Code/System.js#L55-79
# Extracted: 2025-12-07
# Adapted for: voice_note systems-audio agent
---
```

#### 5.2 Jinja2 Template Format (Agentic System Prompts)
**Source**: `agentic-system-prompts/CLAUDE.md`, `patterns/agentic-prompt-patterns.md`

```jinja2
{#
  Agent: systems-audio
  Version: 1.0
  Created: 2025-12-07
  Purpose: OS signal + audio stack specialist
  Tools: read, edit, bash, grep
#}

You are a {{ agent_role }} focused on {{ domain }}.

{% if platform == "darwin" %}
Primary audio stack: CoreAudio
{% elif platform == "linux" %}
Primary audio stack: {{ audio_backend }} (ALSA/PipeWire/PulseAudio)
{% endif %}
```

**Insight**: Dynamic prompts > static prompts for cross-platform work.

**Application**: Future enhancement (Phase 3+), not critical for MVP.

#### 5.3 Folder Structure Standards (Agentic Setup Guide)
**Source**: `docs/AGENTIC_SETUP_GUIDE.md`

```
.opencode/
├── agent/              # Agent definitions
├── hub-excerpts/       # Curated patterns
├── tool/               # Custom tools (if needed)
└── docs/               # Reference docs (if needed)
```

**Current voice_note status**: ✅ Following convention

---

### 6. Quality Patterns

#### 6.1 Read-Only Agent Pattern (Agentic-Forge)
**Source**: `examples/agents/security.md`

```yaml
description: Security audit agent (read-only)
mode: all
tools:
  write: false
  edit: false
  bash:
    "cat*": allow
    "grep*": allow
    "ls*": allow
    "*": deny
```

**Insight**: Analysis-only agents prevent accidental changes.

**Application to robustness-tester.md**: ✅ Already implemented

#### 6.2 Severity Classification (Security Agent)
**Source**: `examples/agents/security.md` workflow

```markdown
## Finding Report Format

### Critical (P0)
- Immediate data loss/corruption risk
- Example: WAV file corruption on SIGINT

### High (P1)
- Degrades user experience significantly
- Example: Terminal corruption requiring reset

### Medium (P2)
- Edge case, workaround exists
- Example: Race condition in stop_flag check

### Low (P3)
- Cosmetic, documentation gaps
```

**Application to robustness-tester.md**:

```markdown
## Output Format

Report findings as:
**[Severity]** Description
- **Impact**: What breaks?
- **Trigger**: How to reproduce?
- **Mitigation**: Immediate workaround?
- **Fix**: Proper solution?
```

---

## Top 10 Patterns to Apply

### Immediate (Phase 3 - This Session)

1. **Conciseness rules** → Add to all 3 agents
2. **XML workflow examples** → Add to systems-audio.md
3. **Tool preferences hierarchy** → Add to all 3 agents
4. **Convention-following checklist** → Add to all 3 agents
5. **Robustness checklist** → Add to robustness-tester.md

### Next Session (Phase 4)

6. **Chain-of-thought debugging workflows** → systems-audio.md
7. **Severity classification** → robustness-tester.md
8. **Granular bash permissions** → Refine all 3 agents
9. **Tool batching feature** → Enable in frontmatter
10. **Attribution headers** → Add to hub-excerpts/

---

## What NOT to Apply

### ❌ Avoid Generic Drift

**From agentic-forge security.md**: "Focus: security ONLY, no feature work"

**Application**: Keep voice_note agents specialized:
- systems-audio: OS/audio ONLY
- groq-integration: Groq API ONLY
- robustness-tester: Testing ONLY

### ❌ Don't Over-Engineer

**From Claude Code System.js**: "Minimize output tokens"

**Application**: Don't add features users didn't ask for:
- No proactive documentation generation
- No unsolicited refactoring
- No emoji unless requested

### ❌ Don't Copy Everything

**From AGENTIC_SETUP_GUIDE.md**: "Extract relevant patterns, not full prompts"

**Application**: Use patterns as inspiration, not templates. voice_note has unique needs (signal handling, audio corruption) not found in web dev agents.

---

## Validation Against Roadmap

**Reference**: `../../abandoned/future_features/system_robustness_roadmap.md`

| Roadmap Phase | Relevant Patterns | Status |
|---------------|-------------------|--------|
| Phase 1: Signal Handling | CoT debugging workflow, XML examples | ✅ Applicable |
| Phase 2: Terminal Safety | bash permission granularity, read-first | ✅ Applicable |
| Phase 3: WAV Integrity | Robustness checklist, severity classification | ✅ Applicable |
| Phase 4: Cross-Platform | Convention following, tool preferences | ✅ Applicable |
| Phase 5: Testing | Security checklist → robustness checklist | ✅ Applicable |

**Alignment**: 100% - All discovered patterns support robustness goals.

---

## Next Actions

### Phase 3: Apply Top 5 Patterns

1. **Enhance systems-audio.md**:
   - Add conciseness rules
   - Add XML workflow examples (signal debugging)
   - Add tool preferences
   - Add convention-following checklist

2. **Enhance groq-integration.md**:
   - Add conciseness rules
   - Add tool preferences
   - Add convention-following checklist

3. **Enhance robustness-tester.md**:
   - Add robustness testing checklist
   - Add severity classification format
   - Confirm read-only permissions

4. **Update .opencode/README.md**:
   - Reference this synthesis document
   - Document pattern sources
   - Add methodology notes

5. **Create agentic-inspiration/ (outside .opencode/)**:
   - Copy top 5 source files with attribution
   - Maintain for future reference
   - No runtime impact

---

## Success Metrics

- ✅ All 3 agents enhanced with patterns
- ✅ Specialization preserved (no generic drift)
- ✅ CLI-optimization maintained
- ✅ Robustness roadmap alignment confirmed
- ✅ Attribution complete for all sources
- ✅ Documentation updated

**Estimated Enhancement Time**: 2-3 hours (focused editing)

---

**Last Updated**: 2025-12-07  
**Status**: Ready for Phase 3 implementation
