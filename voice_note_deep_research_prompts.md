# VoiceNote Deep Research Prompts: Advanced Audio Preprocessing for ASR

> **⚠️ SPECULATIVE / NEVER ACTED ON (2026-07-19).** Audio preprocessing / compression / VAD explicitly excluded from v1.0 — see [`AGENTS.md`](./AGENTS.md) §5 Non-goals. Do not act on these prompts without explicit user confirmation.

This file contains deep research prompts for emerging areas in audio preprocessing to enhance speech recognition for VoiceNote. Each section provides a comprehensive prompt designed for deep research agents to explore specific topics in depth. These prompts are crafted to leverage available tools like web search, academic paper analysis, code repository exploration, and pattern-based reasoning (e.g., Fabric's deep research optimizing pattern).

The prompts assume access to the same context as this project: VoiceNote's focus on optimizing voice input for transcription (prior research now archived in `audio_preprocessing_research.md` — abandoned, see [`AGENTS.md`](./AGENTS.md) §5), and the goal of making voices "pop" and cut through better for better ASR performance.

## Prompt 1: Automatic Audio Mixing for ASR

**Deep Research Query:**

Conduct an exhaustive investigation into applying radio engineering principles of automatic audio mixing to speech preprocessing for automatic speech recognition (ASR) systems, specifically for VoiceNote's use case of optimizing voice input before sending to GROQ API.

**Research Objectives:**
1. Analyze how radio technicians use equalization, compression, limiting, and gating to make voices prominent in broadcasts
2. Identify which of these techniques can be automated for real-time or near-real-time speech processing
3. Explore how these techniques improve ASR accuracy, especially for distant speakers, varying volumes, and multi-speaker scenarios
4. Investigate existing implementations in audio processing libraries and research papers
5. Assess computational requirements and feasibility for integration into VoiceNote

**Key Areas to Explore:**
- Dynamic range compression algorithms and their impact on speech intelligibility
- Multiband compression for frequency-specific enhancement
- Automatic gain control (AGC) systems adapted for speech
- De-essing and sibilance reduction techniques
- Spectral shaping for voice prominence
- Integration with existing noise reduction and normalization methods

**Tools and Methods:**
- Search arXiv for papers on "speech enhancement" + "compression" + "ASR"
- Analyze GitHub repositories for audio processing libraries (e.g., pydub, librosa, webrtcvad)
- Use web search for radio engineering tutorials and audio mixing best practices
- Examine patents and industry standards for broadcast audio processing
- Test simple implementations on sample audio to measure ASR improvement

**Expected Outputs:**
- Comprehensive analysis of applicable mixing techniques
- Prototype code snippets for key algorithms
- Performance benchmarks showing WER improvements
- Recommendations for VoiceNote integration with risk assessments

## Prompt 2: Prosody-Aware Enhancement

**Deep Research Query:**

Investigate prosody-aware audio enhancement techniques that preserve and enhance speech prosodic features (pitch, timing, stress, intonation) to improve automatic speech recognition accuracy, particularly for VoiceNote's transcription optimization.

**Research Objectives:**
1. Understand how prosodic features contribute to ASR performance and human speech perception
2. Identify enhancement methods that boost prosodic clarity without distorting natural speech
3. Explore machine learning approaches for prosody detection and enhancement
4. Analyze the impact on different speaker types (age, accent, emotional state)
5. Assess real-time processing requirements for VoiceNote

**Key Areas to Explore:**
- Pitch contour enhancement and smoothing
- Rhythm and timing regularization
- Stress pattern amplification
- Intonation preservation in noisy environments
- Prosody transfer from clean to degraded speech
- Integration with existing voice activity detection

**Tools and Methods:**
- Search academic databases for "prosody enhancement" + "speech recognition"
- Analyze prosody analysis libraries (e.g., Praat, parselmouth, pyworld)
- Examine research on prosody in ASR systems (Whisper, wav2vec2)
- Use web search for linguistics papers on prosody perception
- Implement prosody feature extraction and enhancement experiments

**Expected Outputs:**
- Detailed taxonomy of prosodic enhancement techniques
- Quantitative analysis of prosody's impact on WER
- Machine learning models for prosody-aware processing
- Integration strategies for VoiceNote pipeline
- Ethical considerations for prosody modification

## Prompt 3: Real-time Voice Isolation

**Deep Research Query:**

Explore real-time voice isolation and separation techniques for multi-speaker environments to enhance ASR performance in VoiceNote, focusing on extracting target speech from overlapping or background audio streams.

**Research Objectives:**
1. Survey state-of-the-art voice separation algorithms suitable for real-time processing
2. Analyze performance trade-offs between quality and latency
3. Investigate speaker identification and tracking in continuous audio streams
4. Explore integration with existing VoiceNote recording pipeline
5. Assess computational requirements for mobile/desktop deployment

**Key Areas to Explore:**
- Blind source separation algorithms (ICA, NMF, deep clustering)
- Speaker diarization integration with separation
- Real-time beamforming and spatial audio processing
- Neural network architectures for fast inference (e.g., SepFormer, TF-GridNet)
- Handling of overlapping speech and reverberation
- Voice activity detection for selective processing

**Tools and Methods:**
- Search arXiv for "real-time speech separation" + "ASR"
- Analyze open-source implementations (e.g., Asteroid, Spleeter, Lalal.ai)
- Examine GitHub repositories for real-time audio processing
- Use web search for commercial voice isolation tools and benchmarks
- Conduct experiments with multi-speaker datasets (LibriMix, WSJ0-2mix)

**Expected Outputs:**
- Comparative analysis of real-time separation methods
- Latency and quality benchmarks for ASR improvement
- Optimized pipeline designs for VoiceNote
- Hardware acceleration strategies (GPU, DSP)
- User experience implications for real-time processing

## Prompt 4: Adaptive Enhancement

**Deep Research Query:**

Research adaptive audio enhancement systems that dynamically adjust preprocessing based on detected audio characteristics, speaker conditions, and environmental factors to optimize speech for ASR in VoiceNote.

**Research Objectives:**
1. Design adaptive systems that detect and respond to audio quality issues in real-time
2. Explore machine learning approaches for automatic parameter selection
3. Analyze the balance between adaptation complexity and performance gains
4. Investigate personalization for individual speakers and environments
5. Assess robustness to unseen audio conditions

**Key Areas to Explore:**
- Audio quality assessment metrics for adaptive processing
- Reinforcement learning for parameter optimization
- Speaker adaptation and voice profile learning
- Environmental noise classification and countermeasure selection
- Dynamic switching between enhancement techniques
- User feedback integration for continuous improvement

**Tools and Methods:**
- Search for "adaptive speech enhancement" + "ASR" in academic literature
- Analyze adaptive audio processing libraries and frameworks
- Examine research on personalized speech processing
- Use web search for adaptive systems in hearing aids and communications
- Implement adaptive pipelines with machine learning components

**Expected Outputs:**
- Architecture designs for adaptive enhancement systems
- Machine learning models for audio condition detection
- Performance evaluation across diverse audio scenarios
- Integration roadmap for VoiceNote with minimal latency impact
- Privacy and data usage considerations for adaptive features

---

*These prompts are designed to be executed by deep research agents with access to web search, academic databases, code repositories, and analytical tools. Each prompt builds on the VoiceNote context of optimizing voice input for transcription while maintaining simplicity and efficiency.*