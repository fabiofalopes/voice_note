#!/usr/bin/env python3
"""
faster_whisper_daemon/server.py - Start the FasterWhisperDaemon service

This script starts the FasterWhisperDaemon service, which loads the faster-whisper model
once and keeps it in memory, listening for transcription requests.

Usage:
    python -m faster_whisper_daemon.server [--model MODEL] [--device DEVICE] [--compute_type COMPUTE_TYPE] [--use_tcp]
"""

import os
import sys
import argparse
import signal
import time

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .daemon import FasterWhisperDaemon

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Stopping daemon...")
    if daemon and daemon.running:
        daemon.stop()
    sys.exit(0)

def main():
    """Main function to start the daemon"""
    parser = argparse.ArgumentParser(description="Start the FasterWhisperDaemon")
    
    # Socket options
    parser.add_argument("--socket-path", help="Path to the daemon socket")
    parser.add_argument("--use-tcp", action="store_true", help="Use TCP instead of Unix socket")
    parser.add_argument("--host", default="localhost", help="Host to bind to (if using TCP)")
    parser.add_argument("--port", type=int, default=9876, help="Port to bind to (if using TCP)")
    
    # Model options
    parser.add_argument("--model", default="large-v3", help="Model to load")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--compute-type", default="float16", help="Compute type (float32, float16, int8_float16, int8)")
    parser.add_argument("--cpu-threads", type=int, help="Number of CPU threads to use")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")
    parser.add_argument("--download-root", help="Directory to download models to")
    parser.add_argument("--local-files-only", action="store_true", help="Only use local files, don't download")
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start daemon
    global daemon
    daemon = FasterWhisperDaemon(
        socket_path=args.socket_path,
        host=args.host,
        port=args.port,
        use_tcp=args.use_tcp,
        model_id=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
        download_root=args.download_root,
        local_files_only=args.local_files_only,
        verbose=True
    )
    
    # Start daemon
    daemon.start()
    
    # Keep running until interrupted
    try:
        while daemon.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping daemon...")
        daemon.stop()
    
if __name__ == "__main__":
    main() 