# 🎤 VoiceNote - Automatic Compression Feature

> **⚠️ ABANDONED (2026-07-19).** Abandoned after Phase 1 script. Audio compression / VAD / noise reduction explicitly excluded from v1.0 — see [`AGENTS.md`](./AGENTS.md) §5 Non-goals.

**Status:** 📝 Idea/Backlog
**Created:** 2026-04-03
**Context:** Currently 32 GB of WAV recordings

---

## 🎯 Feature: Automatic Audio Compression

### Problem
Voice recordings accumulate quickly:
- Current: 941 recordings = 32 GB
- Each recording: ~10-100 MB (uncompressed WAV)
- Storage grows unbounded

### Solution
Implement automatic compression workflow:
1. **After transcription:** Compress WAV → Opus 24kbps
2. **Delete original WAV** (keep Opus)
3. **Create archive** for backup
4. **Space savings:** ~96% reduction (32 GB → ~1.3 GB)

---

## 🔧 Implementation Plan

### Phase 1: Current Manual Script ✅
- [x] Compression script created
- [x] Only compresses files WITH transcriptions
- [x] Verifies output with ffprobe
- [x] Creates tar.gz archive

### Phase 2: Integrate into VoiceNote App
- [ ] Add compression to post-transcription workflow
- [ ] Make compression configurable (bitrate, codec)
- [ ] Add progress indicator
- [ ] Handle errors gracefully
- [ ] Support batch processing

### Phase 3: Storage Management
- [ ] Auto-archive old recordings (configurable age)
- [ ] Cloud storage integration (optional)
- [ ] Compression stats dashboard
- [ ] Storage cleanup recommendations

---

## 📊 Compression Options

| Codec | Bitrate | Quality | Size Reduction |
|-------|---------|---------|----------------|
| **Opus** | 24 kbps | Good for speech | 96% |
| **Opus** | 32 kbps | Better speech | 94% |
| **Opus** | 64 kbps | Very good | 89% |
| **FLAC** | Lossless | Perfect | 30-40% |

**Current choice:** Opus 24kbps (optimized for transcription, saves 96%)

---

## 🛠️ Technical Details

### Command Used
```bash
ffmpeg -i input.wav -c:a libopus -b:a 24k -ar 16000 -ac 1 output.opus
```

**Parameters:**
- `-c:a libopus`: Opus codec (best speech compression)
- `-b:a 24k`: 24 kilobits per second (speech optimized)
- `-ar 16000`: 16 kHz sample rate (speech quality)
- `-ac 1`: Mono (single channel)

### Quality Notes
- 24 kbps Opus: Good for speech, transcription, voice notes
- Still intelligible for playback
- Small file size: ~1.5 MB per hour of audio
- 96% smaller than WAV

---

## 📁 File Structure

**Before:**
```
recordings/
├── recording_20260203_010104.wav (1.3 GB)
├── recording_20260203_010104_transcription.txt
├── recording_20260226_045536.wav (93 MB)
└── recording_20260226_045536_transcription.txt
```

**After:**
```
recordings/
├── recording_20260203_010104.opus (45 MB) ✅
├── recording_20260203_010104_transcription.txt ✅
├── recording_20260226_045536.opus (2.8 MB) ✅
└── recording_20260226_045536_transcription.txt ✅
```

**Archived:**
```
~/backups/voice_note_recordings_2026-04-03.tar.gz
```

---

## 🚀 Future Enhancements

### Ideas
1. **Automatic compression after transcription**
   - Hook into transcription workflow
   - Run in background
   - Notify user when complete

2. **Selective compression**
   - User chooses quality level
   - Keep important recordings at higher quality
   - Auto-delete after X days

3. **Cloud backup**
   - Upload compressed files to S3/GDrive
   - Keep local cache of recent recordings
   - Stream older recordings from cloud

4. **Compression dashboard**
   - Show storage savings
   - Compression statistics
   - Recordings by date/size

5. **Batch recompression**
   - Change quality settings
   - Re-compress all existing files
   - Verify integrity

---

## 📝 Development Notes

### Safety First
- ✅ Only compress files WITH transcriptions
- ✅ Verify output before deleting original
- ✅ Create backup before batch operations
- ✅ Preserve transcription files

### Performance
- Current: ~1 hour per GB (serial processing)
- Could parallelize with GNU parallel
- Consider GPU acceleration (ffmpeg supports)

### User Experience
- Show progress indicator
- Allow cancellation
- Preview quality before committing
- Rollback capability

---

## 🎯 Success Criteria

- [x] Manual script works correctly
- [ ] Integrated into VoiceNote app
- [ ] Automatic compression post-transcription
- [ ] Configurable quality settings
- [ ] Storage savings visible in UI
- [ ] Archive/restore functionality

---

**Related:** `~/backups/compression-log.txt`
**Script:** `/tmp/compress-voice.sh`
