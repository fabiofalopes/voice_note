# 🎯 Audio Optimization Vision & Mission

> **⚠️ ABANDONED (2026-07-19).** Zero implementation. Audio compression / VAD / noise reduction explicitly excluded from v1.0 — see [`AGENTS.md`](./AGENTS.md) §5 Non-goals. Do not resurrect.

## Our Core Mission
**Make VoiceNote the most efficient voice transcription system** by sending optimized audio to GROQ API at the lowest cost while maintaining quality. Keep it simple, keep it working.

**VoiceNote is a transcription mic, not an audio conversion tool.** We optimize the "pen" (voice input) for seamless API transcription. Our focus is minimal file sizes by default, with options for lossless when needed.

## The Big Opportunity
GROQ's Whisper models downsample all audio to **16kHz mono** before transcription. We're currently recording at **44.1kHz mono** (or stereo on some systems), then sending WAV files. This wastes bandwidth, storage, and API costs.

**Simple fix:** Record at 16kHz mono from the start. Match what GROQ expects.

## Current State Assessment
✅ **Already Working Well:**
- Smart chunking for large files (>25MB)
- Multiple Whisper models (large-v3, large-v3-turbo, distil-whisper-large-v3-en)
- Cross-platform recording (PyAudio + PipeWire fallback for Linux)
- Robust error handling and device auto-detection

✅ **Current Format:**
- Recording: 44.1kHz mono, WAV, pyaudio.paInt16 (16-bit PCM)
- File format: Uncompressed WAV
- Chunking: Uses 16kHz mono WAV for chunks (already optimized!)

❌ **Low-Hanging Fruit to Optimize:**
1. **Recording sample rate:** Change default from 44.1kHz → 16kHz
2. **Default to lossy compression:** Use MP3 by default for minimal file sizes, add flag for lossless (FLAC)
3. **Pre-processing:** Optimize existing recordings before upload when using lossless

## The Vision: Simplicity First, Optimization Second

### 🎯 **What We're Actually Solving**
GROQ accepts these file formats: `flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm`

They **downsample everything to 16kHz mono**. We prioritize minimal file sizes by default:
- **Default:** MP3 (lossy) for smallest possible files
- **Option:** FLAC (lossless) via flag when quality is critical
- **Client-side optimization:** Convert to 16kHz mono yourself to save upload time

### 💡 **Key Insight from Real-World Usage**
Looking at how others use GROQ's API:
- Most send files directly without pre-processing
- Many use MP3 for smaller sizes (file=("audio-file.mp3", bytes))
- WAV is common but not universal
- **Nobody seems to worry about format** - they all work
- **Our approach:** Default to lossy (MP3) for efficiency, lossless (FLAC) as option

### 🔬 **Questions We Need to Answer**
1. **Does format matter for quality?** Will MP3/FLAC transcribe as accurately as WAV?
2. **What's the size difference?** 16kHz mono WAV vs FLAC vs MP3
3. **Does recording at 16kHz work on all platforms?** Some hardware might not support it
4. **What compression level is safe?** Where's the sweet spot for size vs quality?

### 📊 **Expected Impact (Conservative Estimates)**
- **16kHz recording:** ~64% smaller files (44.1kHz → 16kHz)
- **MP3 compression (default):** 80-90% reduction (lossy)
- **FLAC compression (option):** Additional 40-60% reduction (lossless)
- **Fewer chunks:** Less API overhead, faster processing

## Implementation Roadmap (One Step at a Time)

### Phase 1: Change Recording Sample Rate (Safest First Step)
**What:** Change default from 44.1kHz → 16kHz in recorder.py line 219
**Why:** GROQ downsamples anyway, so we're wasting upload time
**Risk:** Low - just one number change, easy to revert
**Test:** Record on macOS/Linux/Windows, verify quality is same
**Impact:** ~64% smaller files instantly

**Tasks:**
- [ ] Change `self.rate = 44100` → `self.rate = 16000`
- [ ] Test on current platform (macOS)
- [ ] Test microphone still works
- [ ] Compare transcription quality with same audio at both rates
- [ ] Document findings

### Phase 2: Default to Lossy Compression (MP3)
**What:** Change default format from WAV to MP3 for minimal file sizes
**Why:** Send as little data as possible by default, optimizing for efficiency
**Risk:** Low - MP3 is widely supported and tested with GROQ
**Test:** Verify MP3 transcription quality matches WAV

**Tasks:**
- [ ] Change default recording format to MP3 (16kHz mono)
- [ ] Add CLI flag `--lossless` to use FLAC instead
- [ ] Optimize FLAC settings for best compression when lossless is used
- [ ] Test on current platform (macOS)
- [ ] Compare transcription quality and file sizes

### Phase 3: Optimize Lossless Option (Build on Lossy Default)
**What:** Fine-tune FLAC compression for when lossless is needed
**Why:** Even lossless should be as optimized as possible
**Risk:** Very low - FLAC is lossless, just optimizing compression level
**Test:** Measure FLAC file sizes at different compression levels

**Tasks:**
- [ ] Test FLAC compression levels (0-8) for size vs speed
- [ ] Choose optimal FLAC settings for balance
- [ ] Document size differences from MP3 default
- [ ] Ensure lossless flag works with existing recordings

### Phase 3: Optimize Chunking (Build on Lossy Default)
**What:** Use MP3 for chunks by default, FLAC when lossless flag is used
**Why:** Chunks are already created client-side, optimize with our new defaults
**Risk:** Low - chunking already works, just changing format
**Test:** Large file chunking with MP3/FLAC

**Tasks:**
- [ ] Update chunk creation to use MP3 by default
- [ ] Support FLAC chunks when lossless flag is used
- [ ] Test with files >100MB
- [ ] Verify chunk reassembly quality
- [ ] Measure total API call reduction

### Phase 4: Audio Enhancement (Future Exploration)
**What:** Voice isolation, noise reduction, etc.
**Why:** Better input = better transcription
**Risk:** High - can degrade quality if done wrong
**Test:** Extensive A/B testing needed

**NOT started yet** - focus on simple wins first

## How We'll Validate Everything

### Testing Approach
1. **Never break the working system** - Test changes in isolation first
2. **Measure before and after** - File sizes, transcription quality, upload time
3. **Compare apples to apples** - Same audio content in different formats
4. **Document everything** - What worked, what didn't, why

### Key Questions to Answer
| Question | How to Test | Success Criteria |
|----------|-------------|------------------|
| Does 16kHz sound OK? | Record test audio, listen to it | Acceptable voice quality |
| Does MP3 work as default? | Same audio → MP3 vs WAV → transcribe | <2% WER difference |
| What's the size win? | Measure file sizes | Document compression ratios |
| Does it work cross-platform? | Test on macOS/Linux/Windows | No recording failures |
| Are chunks still OK? | Large file test | Transcription is coherent |

### Success Metrics (Realistic)
- **File Size:** 50-80% reduction (conservative target)
- **Quality:** No worse than current system (maintain, don't degrade)
- **Simplicity:** No added complexity to user workflow
- **Compatibility:** Works on all current platforms

## Principles We Follow

### 1. Simplicity Over Cleverness
- One change at a time
- Test before committing
- Easy to understand, easy to revert

### 2. Don't Fix What Ain't Broke
- Current system works well
- Optimizations are extras, not requirements
- If a change breaks something, we roll it back

### 3. Measure, Don't Guess
- Test with real audio
- Compare actual file sizes
- Verify transcription quality

### 4. User Experience First
- Recording should "just work"
- Transcription should be accurate
- Speed improvements are nice-to-have

## What We're NOT Doing (Yet)

- ❌ Real-time audio effects during recording (too complex)
- ❌ AI-powered voice enhancement (not simple enough)
- ❌ Automatic silence trimming (can break natural speech)
- ❌ Custom audio codecs (keep using standard ffmpeg)
- ❌ Multi-pass transcription (over-engineering)
- ❌ Extensive format testing (MP3 works, FLAC as option)

These might come later, but only after simpler wins are proven.

## Current Status: Ready for Phase 1

The recorder already has good platform detection and error handling. Changing the sample rate is the safest first optimization - one number in one place. Then we'll default to MP3 for minimal sizes, with lossless as an option.

**VoiceNote optimizes the voice input for transcription, not audio processing.** Start small. Test thoroughly. Document results. Repeat.