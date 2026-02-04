#!/usr/bin/env python3
"""
stop.py

Gracefully stops USB-AI services.
"""

import logging
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

OLLAMA_PORT = 11434
WEBUI_PORT = 3000


def find_process_by_port(port: int) -> list:
    """Find process IDs using a specific port."""
    system = platform.system().lower()
    pids = []
    
    try:
        if system == "windows":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pids.append(int(parts[-1]))
        else:
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.strip().splitlines():
                if line.isdigit():
                    pids.append(int(line))
    except Exception as e:
        log.debug(f"Error finding process: {e}")
        
    return list(set(pids))


def kill_process(pid: int) -> bool:
    """Terminate a process by PID."""
    system = platform.system().lower()
    
    try:
        if system == "windows":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True
            )
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception as e:
        log.debug(f"Error killing process {pid}: {e}")
        return False


def stop_service(name: str, port: int) -> bool:
    """Stop a service running on a specific port."""
    pids = find_process_by_port(port)
    
    if not pids:
        log.info(f"{name} not running")
        return True
        
    log.info(f"Stopping {name} (PIDs: {pids})")
    
    for pid in pids:
        kill_process(pid)
        
    log.info(f"{name} stopped")
    return True


def main() -> int:
    """Entry point."""
    print("")
    print("=" * 50)
    print("           USB-AI Stopping")
    print("=" * 50)
    print("")
    
    stop_service("Open WebUI", WEBUI_PORT)
    stop_service("Ollama", OLLAMA_PORT)
    
    print("")
    print("=" * 50)
    print("          USB-AI stopped")
    print("   Safe to remove USB drive")
    print("=" * 50)
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
