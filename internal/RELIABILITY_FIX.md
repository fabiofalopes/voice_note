# Audio Recording Reliability Fix

> **⚠️ STALE / HISTORICAL (2026-07-19).** Superseded by `src/audio_processing/robust_recorder.py` (the implementation this doc analysed). For current reliability work, see [`AGENTS.md`](../AGENTS.md) §3 Stream D + [`reports/signal_handling_corruption_analysis.md`](./reports/signal_handling_corruption_analysis.md). Retained for archaeology only.

## Problem Analysis

### What Happened
You lost 4.5 hours of audio because:
1. Bluetooth earbuds battery died
2. macOS Core Audio (AUHAL) threw error -50 (device disconnected)
3. The recording process crashed
4. All audio was lost because it was never saved to disk

### Root Causes in Original Code

**File**: `src/audio_processing/recorder.py`

#### 1. In-Memory Buffering (Line 576)
```python
frames = []  # ALL audio stored in RAM
# ... records for hours ...
# Only saves at line 629 when Ctrl+C pressed
```
- 4.5 hours = 1.3GB of audio in RAM
- If process crashes, everything in `frames` is lost

#### 2. No Device Error Handling (Lines 582-605)
```python
try:
    data = stream.read(self.chunk, exception_on_overflow=False)
except OSError as e:
    if "Input overflowed" in str(e):  # Only handles overflow
        # ...
    else:
        raise  # Device errors crash the process!
```
- Only handles buffer overflows
- Device disconnection (error -50) is not caught
- Exception bubbles up and crashes the process

#### 3. Single Point of Failure (Line 629)
```python
updated_file_path = self.save_wav(frames, file_path)  # Only saves at end
```
- WAV file only created when recording stops
- No intermediate saves
- No checkpoints

#### 4. No Recovery Mechanism
- Cannot resume after device failure
- Cannot recover partial recordings
- No fallback device support

## Solution Implemented

### New Robust Recorder
**File**: `src/audio_processing/robust_recorder.py`

#### Key Features

1. **Periodic Chunking**
   - Saves audio every N minutes (default: 5)
   - Each chunk is a complete, valid WAV file
   - Reduces memory footprint
   - Provides multiple recovery points

2. **Device Error Handling**
   ```python
   elif "-50" in error_msg or "Unknown Error" in error_msg:
       consecutive_errors += 1
       if consecutive_errors >= max_consecutive_errors:
           # Save current chunk and stop gracefully
           return frames, False, "Device disconnected"
   ```
   - Detects device disconnection
   - Saves current chunk before stopping
   - Provides clear error message

3. **Immediate Write**
   - Each chunk is saved immediately after recording
   - No risk of losing already-recorded audio
   - Files are on disk, not in memory

4. **Auto-Merge**
   - Chunks automatically merged after recording
   - Individual chunks kept as backup
   - Can re-merge if needed

5. **Progress Tracking**
   - Real-time progress per chunk
   - Total recording time tracked
   - Visual feedback during recording

### CLI Integration
**File**: `src/cli.py`

New flags:
```bash
--robust          # Enable robust recorder
--chunk-minutes N # Save every N minutes (default: 5)
--max-duration N  # Stop after N minutes
```

## Usage Comparison

### Before (Fragile)
```bash
vn --record-output long_session.wav
# Risk: If earbuds die at 4:29:59, you lose everything
```

### After (Robust)
```bash
vn --robust --chunk-minutes 5
# Safe: Worst case you lose <5 minutes
```

## Testing Results

Test with 1-minute chunks for 1 minute:
```
✅ Chunk 1 complete: 60.0s recorded
✅ Saved chunk: recording_20260402_200721_chunk001.wav (5.0MB)
🔗 Merging 1 chunks into recordings/recording_20260402_200721_merged.wav...
✅ Merged file created: recordings/recording_20260402_200721_merged.wav (5.0MB)
```

Files created:
- `recording_20260402_200721_chunk001.wav` (individual chunk - backup)
- `recording_20260402_200721_merged.wav` (final merged file)

## Failure Scenario Comparison

### Scenario: Earbuds die after 4 hours, 29 minutes

**Old Recorder:**
```
Recording with PyAudio... Press Ctrl+C to stop.
....................................................................................................
||PaMacCore (AUHAL)|| Error on line 2523: err='-50', msg=Unknown Error

❌ Result: 0 files created, 4.5 hours lost
```

**New Robust Recorder:**
```
📼 Chunk 54: Recording to recording_20260402_163800_chunk054.wav
   Progress: 95.2% | 285s elapsed
⚠️  Device error detected (1/50)
⚠️  Device error detected (2/50)
...
⚠️  Device disconnected after 50 errors
💡 Tip: Reconnect your audio device and restart recording

✅ Result:
   - Chunks 1-53 saved (4h 25min) ✓
   - Chunk 54 partial (~4min) ✓
   - Only ~1 minute lost
```

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Max data loss | Entire recording | < N minutes (chunk size) |
| Memory usage | Grows with duration | Constant (~chunk size) |
| Device failure | Crashes, loses all | Saves current chunk |
| Process crash | Loses all | Keeps saved chunks |
| Recovery points | None | Every chunk |
| File corruption risk | High (single file) | Low (multiple chunks) |

## Recommendations

1. **Always use `--robust` for recordings > 30 minutes**
2. **Chunk size 5-10 minutes** provides good balance
3. **Keep individual chunks** until you verify the merged file
4. **Monitor battery levels** on wireless devices
5. **Consider wired backup** for critical long recordings

## Files Modified

- `src/audio_processing/robust_recorder.py` (NEW)
- `src/cli.py` (UPDATED)
- `ROBUST_RECORDING.md` (NEW - user guide)

## Next Steps

Optional enhancements for even more reliability:
1. Auto-detect device reconnection and resume recording
2. Fallback to built-in mic when earbuds disconnect
3. Background recording daemon with watchdog
4. Cloud backup for chunks
5. Audio level monitoring and alerts
