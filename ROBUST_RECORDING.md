# Robust Recording Guide

## The Problem

When recording long audio sessions, the legacy recorder can lose everything if:
- Your Bluetooth earbuds run out of battery
- The audio device disconnects
- The process crashes
- You press Ctrl+C before the file is saved

**Why?** The legacy recorder keeps ALL audio in memory and only saves when you stop. 4.5 hours = 1.3GB in RAM - if anything goes wrong, it's all gone.

## The Solution

The **robust recorder** is now the **DEFAULT**. It:

✅ **Saves every 5 minutes** (configurable) to separate files
✅ **Handles device disconnection** gracefully - saves what it has
✅ **Auto-merges chunks** after recording into one file
✅ **Keeps individual chunks** as backup
✅ **No data loss** - worst case you lose only the current chunk
✅ **Atomic WAV writes** - temp file + fsync + os.replace (no corrupt files on crash)
✅ **Terminal state protection** - saves/restores terminal settings around PyAudio
✅ **Async-signal-safe shutdown** - Ctrl+C sets a flag checked between audio reads

## Usage

### Basic Usage (Default — No Flag Needed)

```bash
# Record with robust mode (saves every 5 minutes) — this is the default
vn

# Specify chunk size (e.g., every 10 minutes)
vn --chunk-minutes 10

# Record with max duration (e.g., 2 hours)
vn --max-duration 120

# Combine with transcription
vn --provider groq --model whisper-large-v3
```

### Legacy Recorder (Not Recommended)

```bash
# Opt into the old recorder (prints a warning to stderr)
vn --legacy
```

> ⚠️ `--legacy` uses the old recorder without device-failure recovery. Audio may be lost on device disconnection. Only use if you have a specific reason to avoid the robust recorder.

### Recording Only

```bash
# Record for later transcription
vn --record-output my_session.wav
```

### What Happens

1. Recording starts with progress indicators
2. Every 5 minutes (or your `--chunk-minutes`), it saves a chunk:
   ```
   📼 Chunk 1: Recording to recording_20260402_194800_chunk001.wav
      Progress: 100.0% | 300s elapsed
   ✅ Saved chunk: recording_20260402_194800_chunk001.wav (5.2MB)
   ```

3. When you stop (Ctrl+C), it merges all chunks:
   ```
   🔗 Merging 12 chunks into recording_20260402_195120_merged.wav...
   ✅ Merged file created: recording_20260402_195120_merged.wav (62.4MB)
   ```

4. Individual chunks are kept as backup in the recordings folder

### What If Device Fails?

If your earbuds die or device disconnects:

```
⚠️  Device error detected (1/50)
⚠️  Device error detected (2/50)
...
⚠️  Device error detected (50/50)
⚠️  Device disconnected after 50 errors
💡 Tip: Reconnect your audio device and restart recording
✅ Chunk 3 complete: 287.3s recorded
✅ Saved chunk: recording_20260402_194800_chunk003.wav (3.1MB)
```

**Result:** You keep chunks 1-3 (only lose the current 4th chunk)

## Comparison

| Feature | Legacy Recorder (`--legacy`) | Robust Recorder (default) |
|---------|------------------|-----------------|
| Memory usage | ALL in RAM | Saves to disk |
| Device failure | LOSES EVERYTHING | Saves current chunk |
| Process crash | LOSES EVERYTHING | Keeps all saved chunks |
| Long recordings | Risky | Safe |
| File output | Single file | Merged + individual chunks |
| WAV write safety | Direct (corrupt on crash) | Atomic (temp + fsync + rename) |
| Terminal state | May corrupt on crash | Saved/restored |
| Ctrl+C handling | KeyboardInterrupt (may miss during C call) | Flag-based (checked between reads) |

## Tips

1. **Robust is the default** — no flag needed for safe recording
2. **Chunk size**: 5-10 minutes is good balance (`--chunk-minutes`)
3. **Individual chunks**: Keep them as backup until you verify the merged file
4. **Earbuds**: Keep them charged! 😄
5. **`--legacy`**: Only use if you have a specific reason; it prints a warning

## Examples

```bash
# Quick voice note (5 min) — robust by default
vn

# Long meeting (1 hour) — larger chunks
vn --chunk-minutes 10

# Lecture recording (2 hours) — robust + transcription
vn --max-duration 120 --chunk-minutes 10

# Recording from specific device
vn --device 0 --chunk-minutes 5

# Just record, transcribe later
vn --record-output important_session.wav

# Legacy recorder (not recommended, prints warning)
vn --legacy
```

## Recovery from Failed Standard Recording

If you already have a failed recording (like what happened to you):

```bash
# Check what was saved
ls -lh recordings/

# If you see a .partial file or .wav file, try to transcribe it
vn recordings/failed_recording.wav
```

The partial file might have some recoverable audio!
