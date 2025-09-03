# Future Features & Preserved Assets

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
- `analyzer_insights.md` - Key learnings from original implementation
- Model selection and fallback strategies