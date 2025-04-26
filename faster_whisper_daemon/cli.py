#!/usr/bin/env python3
"""
faster_whisper_daemon/cli.py - Unified CLI for the FasterWhisperDaemon

This module provides a unified command-line interface for all daemon operations,
including starting, stopping, checking status, and testing the daemon.

Usage:
    python -m faster_whisper_daemon.cli start [--model MODEL] [--device DEVICE] [--compute_type COMPUTE_TYPE]
    python -m faster_whisper_daemon.cli stop
    python -m faster_whisper_daemon.cli status
    python -m faster_whisper_daemon.cli test [--audio_file FILE]
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import glob
import tempfile

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .daemon import FasterWhisperDaemon, FasterWhisperClient
from .client import find_latest_socket
from audio_processing.recorder import AudioRecorder

def start_daemon(args):
    """Start the daemon"""
    # Build command to start the daemon in a new process
    cmd = [
        sys.executable, "-m", "faster_whisper_daemon.server",
        "--model", args.model,
        "--device", args.device,
        "--compute-type", args.compute_type,
    ]
    
    # Add optional arguments
    if args.socket_path:
        cmd.extend(["--socket-path", args.socket_path])
    if args.use_tcp:
        cmd.append("--use-tcp")
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.num_workers:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.cpu_threads:
        cmd.extend(["--cpu-threads", str(args.cpu_threads)])
    if args.download_root:
        cmd.extend(["--download-root", args.download_root])
    if args.local_files_only:
        cmd.append("--local-files-only")
    
    # Start the daemon
    print(f"🚀 Starting daemon with model: {args.model}")
    print(f"   Device: {args.device}, Compute type: {args.compute_type}")
    
    if args.detach:
        # Start in background
        if os.name == 'posix':  # Unix/Linux/Mac
            cmd_str = " ".join(cmd)
            os.system(f"nohup {cmd_str} > daemon.log 2>&1 &")
            print("✅ Daemon started in background. Output redirected to daemon.log")
        else:  # Windows
            subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS)
            print("✅ Daemon started in background")
    else:
        # Start in foreground
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n🛑 Daemon stopped by user")

def stop_daemon(args):
    """Stop the daemon"""
    # Connect to the daemon
    try:
        client = FasterWhisperClient(
            socket_path=args.socket_path,
            host=args.host,
            port=args.port,
            use_tcp=args.use_tcp
        )
        
        # Get daemon status
        status = client.get_status()
        if status.get("status") == "running":
            print("🛑 Stopping daemon...")
            # We can't actually stop the daemon through the client API
            # So we'll need to find the process and kill it
            if os.name == 'posix':  # Unix/Linux/Mac
                # Find daemon process
                try:
                    # Try to find the process by looking for python processes running the server
                    ps_output = subprocess.check_output(
                        ["ps", "aux"], 
                        universal_newlines=True
                    )
                    for line in ps_output.splitlines():
                        if "faster_whisper_daemon.server" in line and "python" in line:
                            pid = int(line.split()[1])
                            os.kill(pid, signal.SIGTERM)
                            print(f"✅ Sent SIGTERM to daemon process (PID: {pid})")
                            return
                    print("❌ Could not find daemon process")
                except Exception as e:
                    print(f"❌ Error stopping daemon: {e}")
            else:
                print("❌ Stopping the daemon is only supported on Unix/Linux/Mac")
        else:
            print("❌ Daemon is not running")
            
    except Exception as e:
        print(f"❌ Error connecting to daemon: {e}")
        print("   Is the daemon running?")

def status_daemon(args):
    """Check daemon status"""
    # Connect to the daemon
    try:
        # Find socket path if not provided
        socket_path = args.socket_path
        if not socket_path and not args.use_tcp:
            socket_path = find_latest_socket()
            if not socket_path:
                print("❌ No daemon socket found. Is the daemon running?")
                return
                
        client = FasterWhisperClient(
            socket_path=socket_path,
            host=args.host,
            port=args.port,
            use_tcp=args.use_tcp
        )
        
        # Get daemon status
        status = client.get_status()
        
        if status.get("status") == "running":
            print("✅ Daemon is running")
            
            # Print connection info
            if args.use_tcp:
                print(f"   Connection: TCP {args.host}:{args.port}")
            else:
                print(f"   Connection: Unix socket {socket_path}")
                
            # Print model info
            info = status.get("info", {})
            model_info = info.get("model", {})
            if model_info.get("loaded"):
                print(f"   Model: {model_info.get('id')}")
                print(f"   Load time: {model_info.get('load_time', 0):.2f} seconds")
            else:
                print("   No model loaded")
                
            # Print system info
            system_info = info.get("system", {})
            if system_info:
                print(f"   Platform: {system_info.get('platform', 'unknown')}")
                if system_info.get("cuda_available"):
                    print(f"   GPU: CUDA {system_info.get('cuda_version', 'unknown')}")
                    print(f"   Device: {system_info.get('cuda_device_name', 'unknown')}")
                elif system_info.get("mps_available"):
                    print(f"   GPU: Apple Silicon (MPS)")
                else:
                    print(f"   GPU: Not available")
        else:
            print("❌ Daemon is not running")
            
    except Exception as e:
        print(f"❌ Error connecting to daemon: {e}")
        print("   Is the daemon running?")

def test_daemon(args):
    """Test the daemon"""
    # Connect to the daemon
    try:
        # Find socket path if not provided
        socket_path = args.socket_path
        if not socket_path and not args.use_tcp:
            socket_path = find_latest_socket()
            if not socket_path:
                print("❌ No daemon socket found. Is the daemon running?")
                print("   Start it with: python -m faster_whisper_daemon.cli start")
                return
                
        client = FasterWhisperClient(
            socket_path=socket_path,
            host=args.host,
            port=args.port,
            use_tcp=args.use_tcp
        )
        
        # Check daemon status
        status = client.get_status()
        print(f"✅ Connected to daemon. Status: {status.get('status', 'unknown')}")
        
        # Print model info
        info = status.get("info", {})
        model_info = info.get("model", {})
        if model_info.get("loaded"):
            print(f"   Model: {model_info.get('id')}")
        else:
            print("   No model loaded")
            
        # Record audio if requested
        audio_file = args.audio_file
        if args.record:
            print(f"🎤 Recording for {args.duration} seconds...")
            recorder = AudioRecorder()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            audio_file = f"recording_{timestamp}.wav"
            recorder.record(args.duration, audio_file)
            print(f"✅ Recording saved to {audio_file}")
            
        # Transcribe audio if available
        if audio_file:
            print(f"🔊 Transcribing {audio_file}...")
            
            try:
                # Submit transcription job
                response = client.transcribe(audio_file)
                
                if 'error' in response:
                    print(f"❌ Error: {response['error']}")
                    return
                    
                job_id = response.get('job_id')
                print(f"✅ Job submitted. ID: {job_id}")
                
                # Wait for job to complete
                print("⏳ Waiting for transcription to complete...")
                start_time = time.time()
                
                while True:
                    job_status = client.get_job_status(job_id)
                    
                    if 'error' in job_status:
                        print(f"❌ Error: {job_status['error']}")
                        break
                        
                    status = job_status.get('status')
                    progress = job_status.get('progress', 0)
                    
                    print(f"   Status: {status}, Progress: {progress}%", end="\r")
                    
                    if status == 'completed':
                        result = job_status.get('result', {})
                        elapsed = time.time() - start_time
                        print(f"\n✅ Transcription completed in {elapsed:.2f} seconds")
                        
                        # Display the result
                        print(f"\n🔤 Transcription result:")
                        print(f"\n{result.get('text', 'No text returned')}")
                        
                        # Clean up job
                        client.cleanup_job(job_id)
                        break
                        
                    elif status == 'failed':
                        print(f"\n❌ Transcription failed: {job_status.get('error')}")
                        break
                        
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error connecting to daemon: {e}")
        print("   Is the daemon running? Start it with: python -m faster_whisper_daemon.cli start")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="FasterWhisperDaemon CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the daemon")
    start_parser.add_argument("--model", default="large-v3", help="Model to load")
    start_parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    start_parser.add_argument("--compute-type", default="float16", help="Compute type (float32, float16, int8_float16, int8)")
    start_parser.add_argument("--socket-path", help="Path to the daemon socket")
    start_parser.add_argument("--use-tcp", action="store_true", help="Use TCP instead of Unix socket")
    start_parser.add_argument("--host", default="localhost", help="Host to bind to (if using TCP)")
    start_parser.add_argument("--port", type=int, default=9876, help="Port to bind to (if using TCP)")
    start_parser.add_argument("--num-workers", type=int, help="Number of workers for parallel processing")
    start_parser.add_argument("--cpu-threads", type=int, help="Number of CPU threads to use")
    start_parser.add_argument("--download-root", help="Directory to download models to")
    start_parser.add_argument("--local-files-only", action="store_true", help="Only use local files, don't download")
    start_parser.add_argument("--detach", action="store_true", help="Run daemon in background")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the daemon")
    stop_parser.add_argument("--socket-path", help="Path to the daemon socket")
    stop_parser.add_argument("--use-tcp", action="store_true", help="Use TCP instead of Unix socket")
    stop_parser.add_argument("--host", default="localhost", help="Daemon host (if using TCP)")
    stop_parser.add_argument("--port", type=int, default=9876, help="Daemon port (if using TCP)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check daemon status")
    status_parser.add_argument("--socket-path", help="Path to the daemon socket")
    status_parser.add_argument("--use-tcp", action="store_true", help="Use TCP instead of Unix socket")
    status_parser.add_argument("--host", default="localhost", help="Daemon host (if using TCP)")
    status_parser.add_argument("--port", type=int, default=9876, help="Daemon port (if using TCP)")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test the daemon")
    test_parser.add_argument("--socket-path", help="Path to the daemon socket")
    test_parser.add_argument("--use-tcp", action="store_true", help="Use TCP instead of Unix socket")
    test_parser.add_argument("--host", default="localhost", help="Daemon host (if using TCP)")
    test_parser.add_argument("--port", type=int, default=9876, help="Daemon port (if using TCP)")
    test_parser.add_argument("--audio-file", help="Audio file to transcribe")
    test_parser.add_argument("--record", action="store_true", help="Record audio before transcribing")
    test_parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "start":
        start_daemon(args)
    elif args.command == "stop":
        stop_daemon(args)
    elif args.command == "status":
        status_daemon(args)
    elif args.command == "test":
        test_daemon(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 