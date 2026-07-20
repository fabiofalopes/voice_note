# Audio Preprocessing & Compression Research Findings

> **⚠️ ABANDONED (2026-07-19).** Pure research, zero implementation. Audio compression / preprocessing explicitly excluded from v1.0 — see [`AGENTS.md`](./AGENTS.md) §5 Non-goals. Retained for historical reference only.

# Audio Preprocessing & Compression Research Findings

## Investigation Overview
This document contains research findings on audio preprocessing techniques to improve speech recognition quality and compressed audio formats for efficient GROQ API usage. The goal is to identify safe, effective methods to optimize audio before sending to GROQ's Whisper models.

## Research Questions
1. What audio preprocessing techniques can improve speech recognition accuracy?
2. How do compressed audio formats (MP3, AAC, etc.) perform vs lossless formats for transcription?
3. What are the trade-offs between file size, quality, and API efficiency?

## Methodology
- Web research using webfetch for documentation and articles
- GitHub code examples using gh_grep
- Analysis of real-world implementations
- Testing with GROQ API where possible

## Findings

### Section 1: Audio Preprocessing Techniques for Better Speech Recognition

#### From OpenAI Whisper Documentation & Research
- **Whisper's Built-in Preprocessing**: Loads audio, resamples to 16kHz, pads/trims to 30s, creates log-Mel spectrograms
- **Language Detection**: Uses mel spectrograms to detect spoken language
- **Sliding Window**: Processes audio in 30-second chunks with autoregressive prediction

#### Real-World Preprocessing Techniques Found
**Noise Reduction:**
- `noisereduce` library: Spectral gating algorithms for background noise removal
- Used in Pipecat, stable-ts, audio-webui projects
- Stationary and non-stationary noise reduction options
- Example: `noisereduce.reduce_noise(y=audio, sr=sample_rate)`

**Voice Activity Detection (VAD):**
- `webrtcvad`: WebRTC's VAD for detecting speech segments
- Used in ESPnet, fairseq, Real-Time-Voice-Cloning
- Aggressiveness levels (0-3) for different noise environments
- Helps trim silence and reduce API costs

**Audio Normalization:**
- Volume normalization to consistent levels (-20 LUFS)
- Dynamic range compression
- Found in MockingBird, audio preprocessing pipelines

**High-Pass Filtering:**
- Remove low-frequency noise (<80Hz)
- Used in various speech processing pipelines
- Can be done with `ffmpeg -af highpass=f=80`

#### From Research Papers (arXiv):
- **Quantization Studies**: Multiple papers on 2-bit, 1-bit quantization of Whisper/Conformer models
- **Compression Techniques**: Low-rank approximation, pruning, knowledge distillation
- **Edge Deployment**: Model compression for resource-constrained devices
- **Audio Enhancement**: Bandwidth expansion, denoising, neural network-based speech enhancement
- **Voice Enhancement**: Techniques to improve speech clarity and intelligibility

#### Voice Enhancement Techniques
**From Research & Code:**
- **Bandwidth Expansion**: Using GANs to expand narrowband speech to wideband
- **Neural Speech Enhancement**: Deep learning models for noise reduction
- **Dynamic Range Compression**: PCEN (Per-Channel Energy Normalization)
- **Spectral Enhancement**: Improving speech frequency components
- **Automatic Mixing Techniques**: Equalization, compression, and limiting to make voices "pop" and cut through
- **Proximity Effect Compensation**: Adjusting levels for speakers at different distances
- **Multi-speaker Voice Isolation**: Beamforming and source separation for overlapping speech
- **Prosody Enhancement**: Boosting pitch and timing variations for better intelligibility

**Advanced Ideas for Making Voices Prevalent:**
- **Dynamic Range Compression (DRC)**: Compress loud parts, boost quiet parts to even out volume variations
- **Multiband Compression**: Apply compression to different frequency bands to enhance speech clarity
- **De-essing**: Reduce harsh "s" sounds that can mask other speech
- **Voice Activity Detection (VAD) with Level Adjustment**: Automatically boost segments with detected speech
- **Spectral Shaping**: Boost mid-range frequencies where human speech is most prominent
- **Reverberation Control**: Reduce room echo that can bury speech in background noise
- **Adaptive Gain Control**: Continuously adjust levels based on input characteristics

**Risk Level**: High - Can improve or degrade quality significantly
**Recommendation**: Start with simple techniques (normalization, basic compression), test complex ones carefully. Focus on preserving natural speech while making it more prominent.

### Section 2: Compressed Audio Formats Analysis

#### GROQ API Format Support
GROQ accepts: `flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm`
- **Recommended**: FLAC for lossless compression, WAV for lower latency
- **All formats**: Automatically downsampled to 16kHz mono

#### Format Performance Research
**From GitHub Examples:**
- Many projects send MP3 directly to GROQ API
- Examples: `file=("audio-file.mp3", bytes)` in multiple repos
- No apparent quality issues reported

**From Research Papers:**
- **Machine Perceptual Quality (MPQ)**: Study showing compressed data can maintain machine perception
- **Audio Compression Impact**: Research on how lossy compression affects ASR performance
- **Neural Codecs**: Advanced compression techniques that preserve speech features

**Compression Ratios Observed:**
- **FLAC**: 40-60% reduction (lossless)
- **MP3**: 80-90% reduction (lossy, depends on bitrate)
- **AAC/M4A**: Similar to MP3 performance

#### Key Findings on Format Impact
- **Lossless (FLAC/WAV)**: No quality loss, but larger files
- **Lossy (MP3/AAC)**: Potential quality trade-offs, but often negligible for speech recognition
- **GROQ Optimization**: All formats converted to 16kHz mono anyway

### Section 3: Recommendations & Implementation Plan

#### Safe Preprocessing Techniques (Low Risk)
1. **16kHz Mono Recording**: Match GROQ's native format
2. **Simple Normalization**: Volume leveling without aggressive compression
3. **Basic Noise Reduction**: Stationary noise only (non-stationary can distort speech)
4. **Silence Trimming**: Remove long silence periods

#### Compressed Format Recommendations
1. **Primary**: FLAC (lossless, GROQ-recommended, 40-60% size reduction)
2. **Secondary**: MP3 128-192kbps (if FLAC too large, test quality impact)
3. **Fallback**: WAV (current, reliable but largest)

#### Implementation Strategy
1. **Phase 1**: Change recording to 16kHz mono (immediate 64% size reduction)
2. **Phase 2**: Add FLAC conversion option (additional 40-60% reduction)
3. **Phase 3**: Test MP3 with quality validation (potential 80-90% reduction)
4. **Phase 4**: Add selective preprocessing (noise reduction, normalization)

#### Risk Assessment
- **Low Risk**: Format conversion, sample rate changes
- **Medium Risk**: Lossy compression (test thoroughly)
- **High Risk**: Aggressive preprocessing (can degrade quality)

## Key Insights

1. **GROQ Already Optimizes**: All audio → 16kHz mono, so pre-processing saves upload time
2. **Compression Likely Safe**: Research suggests lossy formats work well for ASR
3. **Voice Enhancement Potential**: Automatic mixing techniques can significantly improve transcription of distant/multiple speakers
4. **Simple Wins First**: Recording format change gives biggest bang for buck
5. **Test Everything**: Measure WER before/after any changes, especially for enhancement techniques
6. **User Experience Priority**: Never degrade quality for size alone; enhancement should improve clarity

## Next Steps
1. Implement 16kHz mono recording
2. Test FLAC vs WAV transcription quality
3. Add basic preprocessing options (normalization, simple compression)
4. Research and prototype voice enhancement techniques for multi-speaker/distant speech
5. Measure API cost savings and quality improvements
6. Document all findings for the main optimization plan

## Emerging Research Areas
- **Automatic Audio Mixing for ASR**: Applying radio engineering principles to speech preprocessing
- **Prosody-Aware Enhancement**: Boosting speech features that transcription models rely on
- **Real-time Voice Isolation**: Processing streaming audio to enhance target speakers
- **Adaptive Enhancement**: Adjusting processing based on detected speech characteristics

---

*Research conducted on: November 21, 2025*
*Sources: OpenAI Whisper docs, arXiv papers, GitHub repositories, GROQ API documentation, Wikipedia speech enhancement, additional arXiv searches on speech enhancement for ASR*