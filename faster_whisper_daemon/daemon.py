#!/usr/bin/env python3
"""
faster_whisper_daemon/daemon.py - Daemon implementation for faster-whisper

This module provides the FasterWhisperDaemon class, which runs the FasterWhisperService
as a daemon process that listens for requests over a Unix socket or TCP connection.
It also provides the FasterWhisperClient class for connecting to the daemon.
"""

import os
import json
import socket
import threading
import time
import tempfile
from pathlib import Path
import uuid
from typing import Dict, Any, Optional, Union

from .service import FasterWhisperService

class FasterWhisperDaemon:
    """Run the FasterWhisperService as a daemon process that listens for requests"""
    
    def __init__(self, 
                 socket_path=None, 
                 host='localhost',
                 port=9876,
                 use_tcp=False,
                 **service_params):
        """
        Initialize the daemon.
        
        Args:
            socket_path: Path to the Unix socket (default: auto-generated in temp dir)
            host: Host for TCP socket (if use_tcp=True)
            port: Port for TCP socket (if use_tcp=True)
            use_tcp: Whether to use TCP instead of Unix socket
            **service_params: Parameters to pass to FasterWhisperService
        """
        self.use_tcp = use_tcp
        self.host = host
        self.port = port
        
        # Generate socket path if not provided
        if not use_tcp and socket_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            socket_path = os.path.join(tempfile.gettempdir(), f"whisper_daemon_{timestamp}.sock")
        
        self.socket_path = socket_path
        self.service = FasterWhisperService(**service_params)
        
        # Initialize daemon state
        self.running = False
        self.server_socket = None
        self.thread = None
        self.clients = {}
        self.jobs = {}
        self._lock = threading.Lock()
    
    def start(self):
        """Start the daemon"""
        if self.running:
            return
            
        # Create socket
        if self.use_tcp:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
        else:
            # Remove socket file if it exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
                
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            
            # Set permissions on socket file
            os.chmod(self.socket_path, 0o777)
            
        # Listen for connections
        self.server_socket.listen(5)
        self.running = True
        
        # Start listener thread
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        
        # Print connection info
        if self.use_tcp:
            print(f"🚀 Daemon started on {self.host}:{self.port}")
        else:
            print(f"🚀 Daemon started on {self.socket_path}")
            
        return True
    
    def stop(self):
        """Stop the daemon"""
        if not self.running:
            return
            
        self.running = False
        
        # Close all client connections
        with self._lock:
            for client_id, client in self.clients.items():
                try:
                    client.close()
                except:
                    pass
            self.clients = {}
            
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
            
        # Remove socket file
        if not self.use_tcp and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            
        # Wait for thread to exit
        if self.thread:
            self.thread.join(timeout=1.0)
            
        return True
    
    def _listen(self):
        """Listen for connections"""
        while self.running:
            try:
                # Accept connection
                client_socket, _ = self.server_socket.accept()
                
                # Generate client ID
                client_id = str(uuid.uuid4())
                
                # Store client
                with self._lock:
                    self.clients[client_id] = client_socket
                    
                # Start client handler thread
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id, client_socket),
                    daemon=True
                )
                thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
                    time.sleep(0.1)
    
    def _handle_client(self, client_id, client_socket):
        """Handle client connection"""
        try:
            while self.running:
                # Receive data
                data = b""
                while True:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in chunk:
                        break
                        
                if not data:
                    break
                    
                # Parse request
                try:
                    request = json.loads(data.decode("utf-8"))
                    command = request.get("command")
                    
                    # Handle command
                    if command == "status":
                        response = self._handle_status()
                    elif command == "load_model":
                        response = self._handle_load_model(request)
                    elif command == "transcribe":
                        response = self._handle_transcribe(request)
                    elif command == "job_status":
                        response = self._handle_job_status(request)
                    elif command == "cleanup_job":
                        response = self._handle_cleanup_job(request)
                    else:
                        response = {"error": f"Unknown command: {command}"}
                        
                except json.JSONDecodeError:
                    response = {"error": "Invalid JSON"}
                except Exception as e:
                    response = {"error": str(e)}
                    
                # Send response
                try:
                    client_socket.sendall(json.dumps(response).encode("utf-8") + b"\n")
                except:
                    break
                    
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
            
        finally:
            # Close connection
            try:
                client_socket.close()
            except:
                pass
                
            # Remove client
            with self._lock:
                if client_id in self.clients:
                    del self.clients[client_id]
    
    def _handle_status(self):
        """Handle status command"""
        return {
            "status": "running",
            "info": self.service.get_info()
        }
    
    def _handle_load_model(self, request):
        """Handle load_model command"""
        model_id = request.get("model_id")
        if not model_id:
            return {"error": "Missing model_id"}
            
        success = self.service.load_model(model_id)
        if success:
            return {"status": "ok", "model_id": model_id}
        else:
            return {"error": f"Failed to load model: {self.service.model_load_error}"}
    
    def _handle_transcribe(self, request):
        """Handle transcribe command"""
        audio_path = request.get("audio_path")
        if not audio_path:
            return {"error": "Missing audio_path"}
            
        # Check if file exists
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
            
        # Get transcription options
        options = request.get("options", {})
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job
        job = {
            "id": job_id,
            "status": "pending",
            "audio_path": audio_path,
            "options": options,
            "result": None,
            "error": None,
            "progress": 0,
            "thread": None
        }
        
        # Store job
        with self._lock:
            self.jobs[job_id] = job
            
        # Start job thread
        job["thread"] = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True
        )
        job["thread"].start()
        
        return {"status": "ok", "job_id": job_id}
    
    def _handle_job_status(self, request):
        """Handle job_status command"""
        job_id = request.get("job_id")
        if not job_id:
            return {"error": "Missing job_id"}
            
        # Get job
        with self._lock:
            job = self.jobs.get(job_id)
            
        if not job:
            return {"error": f"Job not found: {job_id}"}
            
        # Return job status
        return {
            "status": job["status"],
            "progress": job["progress"],
            "result": job["result"],
            "error": job["error"]
        }
    
    def _handle_cleanup_job(self, request):
        """Handle cleanup_job command"""
        job_id = request.get("job_id")
        if not job_id:
            return {"error": "Missing job_id"}
            
        # Remove job
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                
        return {"status": "ok"}
    
    def _run_job(self, job_id):
        """Run transcription job"""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return
                
            # Update job status
            job["status"] = "running"
            job["progress"] = 0
            
        try:
            # Load model if needed
            model_id = job["options"].get("model_id")
            if model_id:
                self.service.load_model(model_id)
                
            # Update progress
            with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["progress"] = 10
                    
            # Transcribe audio
            result = self.service.transcribe(job["audio_path"], **job["options"])
            
            # Update job status
            with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = "completed"
                    self.jobs[job_id]["result"] = result
                    self.jobs[job_id]["progress"] = 100
                    
        except Exception as e:
            # Update job status
            with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["error"] = str(e)
                    
class FasterWhisperClient:
    """Client for connecting to the FasterWhisperDaemon"""
    
    def __init__(self, socket_path=None, host='localhost', port=9876, use_tcp=False):
        """
        Initialize the client.
        
        Args:
            socket_path: Path to the Unix socket
            host: Host for TCP socket (if use_tcp=True)
            port: Port for TCP socket (if use_tcp=True)
            use_tcp: Whether to use TCP instead of Unix socket
        """
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self.use_tcp = use_tcp
    
    def _connect(self):
        """Connect to the daemon"""
        if self.use_tcp:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
            
        return sock
    
    def _send_request(self, request):
        """Send request to daemon and get response"""
        try:
            # Connect to daemon
            sock = self._connect()
            
            # Send request
            sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
            
            # Receive response
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break
                    
            # Parse response
            response = json.loads(data.decode("utf-8"))
            
            # Close connection
            sock.close()
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self):
        """Get daemon status"""
        return self._send_request({"command": "status"})
    
    def load_model(self, model_id):
        """Load a model"""
        return self._send_request({
            "command": "load_model",
            "model_id": model_id
        })
    
    def transcribe(self, audio_path, model_id=None, **options):
        """
        Transcribe audio.
        
        Args:
            audio_path: Path to audio file
            model_id: Model ID to use
            **options: Transcription options
            
        Returns:
            Response with job_id
        """
        # Prepare options
        transcribe_options = dict(options)
        if model_id:
            transcribe_options["model_id"] = model_id
            
        # Send request
        return self._send_request({
            "command": "transcribe",
            "audio_path": audio_path,
            "options": transcribe_options
        })
    
    def get_job_status(self, job_id):
        """Get job status"""
        return self._send_request({
            "command": "job_status",
            "job_id": job_id
        })
    
    def cleanup_job(self, job_id):
        """Clean up job"""
        return self._send_request({
            "command": "cleanup_job",
            "job_id": job_id
        }) 