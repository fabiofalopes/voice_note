# 🎯 RELIABILITY FIRST - Core Project Principle

> **⚠️ PARTIALLY DRIFTED (2026-07-19).** Core philosophy valid. Factual claim (robust is still opt-in, not default) is accurate. **However**: the urgency framing ("UNACCEPTABLE", top priority) is at odds with current sequencing — Stream A (output contract) is the v1.0 ship target, Stream D (reliability flip) is parallel/optional. See [`AGENTS.md`](../AGENTS.md) §3. References to `RELIABILITY_FIX.md` below are stale (that doc is itself historical).

## The Non-Negotiable Standard

**voice_note MUST be reliable by default. Users should NEVER lose recordings due to preventable failures.**

## The Incident That Changed Everything

```
┌───(🐍)-[~/projetos/hub]
└─$ vn
🎤 PyAudio → Device 0
🎤 PyAudio → Device 0
🎤 Using PyAudio device: 0
   Format: 1ch, 44100Hz, chunk=1024
Recording with PyAudio... Press Ctrl+C to stop.
.......................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................||PaMacCore (AUHAL)|| Error on line 2523: err='-50', msg=Unknown Error
^C
```

**Result**: 4.5 hours of audio lost. User was discussing important topics about vehicle network security.

**Cause**: Bluetooth earbuds battery died → device disconnected → recording crashed → ALL audio in RAM was lost.

**This is UNACCEPTABLE.**

---

## The Mandate

### Current State (UNACCEPTABLE)
- **Default behavior**: Fragile recording that can lose everything
- **User action required**: Must remember to add `--robust` flag
- **Failure mode**: Total data loss
- **Memory pattern**: All audio in RAM, single point of failure

### Required State (NON-NEGOTIABLE)
- **Default behavior**: Robust recording with auto-chunking
- **User action required**: Nothing - it just works safely
- **Failure mode**: Maximum loss = one chunk (default: 5 minutes)
- **Memory pattern**: Immediate writes to disk, multiple recovery points

---

## Implementation Goal

**Make `--robust` the DEFAULT behavior for ALL recordings.**

### Required Changes

1. **CLI Default Behavior** (`src/cli.py`)
   - [ ] Change default: `robust=True` instead of `robust=False`
   - [ ] Add `--legacy` or `--unsafe` flag for old behavior (advanced users only)
   - [ ] Update help text to reflect robust is default
   - [ ] Add warning when using `--legacy`: "⚠️  Risk of total data loss"

2. **Sensible Defaults**
   - [ ] Default chunk size: 5 minutes (good balance)
   - [ ] Auto-merge chunks after recording
   - [ ] Keep individual chunks as backup until user deletes
   - [ ] Clear progress indicators

3. **User Communication**
   - [ ] Update README to emphasize reliability
   - [ ] Add warning when NOT using robust mode
   - [ ] Document why robust is now default
   - [ ] Examples should show robust behavior

4. **Testing**
   - [ ] Test device disconnection handling
   - [ ] Test process crash recovery
   - [ ] Verify all recordings produce files
   - [ ] Test with various durations (1min to 4+ hours)
   - [ ] Test merge functionality

---

## Technical Implementation Notes

### Architecture Decision

**Option RECOMMENDED: Make robust the only recorder**

```python
# In src/cli.py - recording section
def record_audio(args):
    # Always use robust recorder
    from audio_processing.robust_recorder import record_robust

    chunk_files = record_robust(
        output_dir=args.output_dir,
        chunk_minutes=args.chunk_minutes,  # Default: 5
        device_index=args.device
    )

    return chunk_files[0]  # Return merged file
```

**Rationale**:
- Simplest code path
- No parallel implementations to maintain
- Clear message: reliability is mandatory
- Reduces testing surface area

### Backward Compatibility

For existing scripts that might depend on old behavior:
```bash
# Old way (now requires explicit flag)
vn --legacy  # ⚠️  WARNING: Risk of data loss

# New way (automatic)
vn  # Robust by default
```

---

## Success Criteria

### Must Have (Blocking)
- [x] Robust recorder implemented ✅ (COMPLETE)
- [ ] Robust is the DEFAULT behavior
- [ ] No recording can lose more than 5 minutes of audio
- [ ] Device failures are handled gracefully
- [ ] User MUST see files created, even on failure
- [ ] Documentation updated

### Should Have (Important)
- [ ] Individual chunks kept as backup
- [ ] Clear error messages on device failure
- [ ] Progress indicators during recording
- [ ] Auto-merge after recording
- [ ] Performance is acceptable (no noticeable lag)

### Could Have (Nice to Have)
- [ ] Config file for user preferences
- [ ] Auto-resume on device reconnection
- [ ] Fallback to built-in mic
- [ ] Audio level monitoring

---

## The Philosophy

### voice_note Project Principles

1. **Reliability First**
   - Data loss is a critical bug, not a feature request
   - Default behavior must be safe
   - Users shouldn't need to be experts to have safe recordings

2. **Fail Gracefully**
   - Worst case = lose minimal data
   - Never crash without saving what we have
   - Clear error messages with actionable guidance

3. **Progressive Enhancement**
   - Works safely out of the box
   - Advanced options available for power users
   - No unsafe defaults

4. **Learn From Failures**
   - When data loss occurs, FIX IT
   - Make it impossible to repeat
   - Share the lessons

---

## Handoff Checklist

For the next session working on voice_note:

- [ ] Read [`AGENTS.md`](../AGENTS.md) §3 Stream D scope + DoD (current authority)
- [ ] Read `ROBUST_RECORDING.md` for user guide (current opt-in behaviour)
- [ ] Review `src/audio_processing/robust_recorder.py` implementation
- [ ] Review `src/audio_processing/recorder.py` (old implementation)
- [ ] Implement: Make robust the default
- [ ] Test: Device disconnection scenarios
- [ ] Test: Various recording durations
- [ ] Update: README with new default behavior
- [ ] Update: Help text and documentation
- [ ] Verify: No recording can lose everything

---

## The Promise

**We will not rest until users can trust voice_note with their most important recordings.**

A 4-hour lecture about vehicle network security should NOT be lost because earbuds died.

A 2-hour podcast recording should NOT vanish because of a device error.

An important meeting recording should NOT disappear because of a process crash.

**This is the standard. This is non-negotiable. This is voice_note.**

---

## References

- **Current authority**: [`AGENTS.md`](../AGENTS.md) §3 Stream D + [`MEMORY.md`](./MEMORY.md)
- **Technical analysis (historical)**: `RELIABILITY_FIX.md` ⚠️ stale — see [`reports/signal_handling_corruption_analysis.md`](./reports/signal_handling_corruption_analysis.md) instead
- **User guide**: `ROBUST_RECORDING.md`
- **Implementation**: `src/audio_processing/robust_recorder.py`
- **Current CLI**: `src/cli.py`
- **Original issue**: Line 576-634 in `src/audio_processing/recorder.py`

---

*Last updated: 2025-04-02*
*Status: IN PROGRESS - Robust recorder implemented, needs to be made default*
*Priority: CRITICAL*
