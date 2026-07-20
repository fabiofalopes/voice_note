# Future Features & Preserved Assets

> **⚠️ MOSTLY STALE (2026-07-19).** LLM post-processing is ROADMAP Phase 7, deferred per [../AGENTS.md](../../../AGENTS.md) §5 Non-goals. Historical assets preserved for reference only — do NOT treat as active direction. For current scope, see [../MEMORY.md](../../MEMORY.md).

This folder contains valuable assets from the original complex implementation that we want to preserve for future development.

## Planned Features

### LLM Post-Processing Pipeline
- Implement using LlamaIndex for better orchestration
- Sentiment analysis
- Text summarization  
- Task extraction and analysis
- Thinking tags generation

### Current Post-Processing Capabilities (to be reimplemented)
The original `post_processing/analyzer.py` had these features:
- Dynamic model selection from available Groq LLMs
- Fallback mechanism when models are unavailable
- Structured JSON responses for analysis
- Multiple analysis types in one pipeline

### Integration Points
- Keep the `--raw-transcription` flag for simple use
- Add `--analyze` flag for post-processing
- Maintain clipboard integration
- Add structured output options

## Assets Preserved

- `prompts/` - System prompts and examples
- `analyzer_insights.md` - ⚠️ **Historical only** — old LLM post-processing, removed. See [../MEMORY.md](../../MEMORY.md) §9.3.
- Model selection and fallback strategies