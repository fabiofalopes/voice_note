"""
faster_whisper_daemon - Module for running FasterWhisper as a background service

This package contains modules for starting, testing, and interacting with the FasterWhisperDaemon,
which loads whisper models into memory for faster transcription. The daemon architecture
significantly improves performance by:

1. Loading the model only once and keeping it in memory
2. Processing multiple transcription requests without reloading the model
3. Reducing memory usage and startup time

Components:
- daemon.py: Core daemon and client implementation
- server.py: Server implementation for the daemon
- client.py: Client for connecting to the daemon
- service.py: Service for managing whisper models
- api.py: Direct API for using whisper models without the daemon
- cli.py: Unified CLI for daemon operations
"""

# Re-export key classes for easier imports
from .daemon import FasterWhisperDaemon, FasterWhisperClient
from .service import FasterWhisperService
from .api import FasterWhisperAPI
from .client import find_latest_socket 