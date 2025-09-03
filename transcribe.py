#!/usr/bin/env python3
"""
Voice Transcriber - Main entry point

Simple entry point that calls the CLI module.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())