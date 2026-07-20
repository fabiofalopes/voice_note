# Agentic Inspiration

Curated patterns extracted from production AI systems. Reference material for agent design.

## Sources

- **Claude Code** - Ultra-concise CLI communication
- **Google Jules** - Verify-before-modify workflows  
- **Bolt.new** - File operation batching
- **Trae Agent** - Sequential thinking for debugging
- **agentic-forge** - Agent registry patterns

## Key Patterns

### 1. Concise CLI Communication (Claude Code)

```
WRONG:
"I'll help you fix that signal handling issue. Let me analyze the code..."

RIGHT:
"Signal handler missing SIGINT. Add handler to recorder.py line 47."
```

**Rules:**
- Sub-4-line responses default
- No preamble ("I'll help you...")
- No postamble ("Let me know if...")
- Direct answers first

### 2. State-First Operation

Every agent session:
1. Read `agent-environment.json`
2. Validate runtime matches expected
3. Check known issues list
4. Then proceed with task

### 3. Verify Before Modify (Google Jules)

Before editing any file:
1. Read current content
2. Understand existing patterns
3. Plan minimal change
4. Execute edit
5. Verify result

### 4. Tool Batching (Bolt.new)

```
WRONG:
- Read file A
- Wait for response
- Read file B
- Wait for response

RIGHT:
- Read file A, B, C simultaneously
- Process all responses together
```

### 5. Structured Problem Decomposition (Trae Agent)

For complex issues:
```
1. Identify symptoms
2. List possible causes
3. Design minimal reproduction
4. Test hypotheses systematically
5. Implement fix
6. Verify fix addresses root cause
```

### 6. Agent Specialization (agentic-forge)

**Good agent:**
- Narrow expertise domain
- Clear trigger conditions
- Explicit limitations
- Defined output format

**Bad agent:**
- "General purpose helper"
- No clear boundaries
- Tries to do everything

## Decision Frameworks

### When to Create New Agent

✅ Create agent if:
- Domain requires deep specialized knowledge
- Task pattern repeats across sessions
- Current agents lack expertise
- Clear trigger condition exists

❌ Don't create agent if:
- One-off task
- Existing agent covers domain
- Task is simple lookup
- No clear specialization

### Agent Communication Style Matrix

| Context | Style | Example |
|---------|-------|---------|
| Error found | Direct statement | "Buffer overflow at line 23" |
| Multiple options | Numbered list | "Options: 1. Fix A, 2. Fix B" |
| Complex analysis | Structured sections | "## Problem\n## Cause\n## Fix" |
| Success | Single line | "Fixed. Test with `python test.py`" |

## Anti-Patterns to Avoid

1. **Generic personality** - "I'm a helpful AI assistant"
2. **Explaining what you'll do** - "Let me read the file..."
3. **Asking permission repeatedly** - "Should I proceed?"
4. **Over-documenting** - Creating README for every change
5. **Hedging language** - "This might possibly be..."

## File Organization

```
agentic-inspiration/
├── README.md              # This file
├── claude-code-style.md   # Communication patterns
├── verification-flows.md  # Jules-style workflows
└── agent-registry.md      # agentic-forge patterns
```

---

*These patterns inform agent design but don't belong in agent prompts directly. Agents embody these patterns through their structure and behavior.*
