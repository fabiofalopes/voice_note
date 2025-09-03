#!/bin/bash

# Run transcriber with suppressed ALSA warnings for cleaner output

# Suppress ALSA warnings
export ALSA_PCM_CARD=0
export ALSA_PCM_DEVICE=0

# Run the transcriber with stderr filtered
python3 transcribe.py "$@" 2> >(grep -v "ALSA lib\|Cannot connect to server\|jack server\|JackShmReadWritePtr" >&2)