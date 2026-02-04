#!/usr/bin/env python3
"""
status.py

Check status of USB-AI services.
"""

import json
import logging
import socket
import sys
import urllib.request
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


def check_port(port: int) -> bool:
    """Check if port is open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        return result == 0
    finally:
        sock.close()


def check_ollama() -> dict:
    """Check Ollama server status."""
    status = {
        "name": "Ollama",
        "port": OLLAMA_PORT,
        "running": False,
        "models": []
    }
    
    if not check_port(OLLAMA_PORT):
        return status
        
    status["running"] = True
    
    try:
        url = f"http://127.0.0.1:{OLLAMA_PORT}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            status["models"] = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        log.debug(f"Error getting models: {e}")
        
    return status


def check_webui() -> dict:
    """Check Open WebUI status."""
    status = {
        "name": "Open WebUI",
        "port": WEBUI_PORT,
        "running": False
    }
    
    if not check_port(WEBUI_PORT):
        return status
        
    status["running"] = True
    return status


def print_status(service: dict):
    """Print service status."""
    name = service["name"]
    port = service["port"]
    running = service["running"]
    
    status_text = "RUNNING" if running else "STOPPED"
    status_color = "\033[92m" if running else "\033[91m"
    reset_color = "\033[0m"
    
    print(f"  {name}")
    print(f"    Port: {port}")
    print(f"    Status: {status_color}{status_text}{reset_color}")
    
    if "models" in service and service["models"]:
        print(f"    Models: {len(service['models'])}")
        for model in service["models"]:
            print(f"      - {model}")


def main() -> int:
    """Entry point."""
    print("")
    print("=" * 50)
    print("           USB-AI Status")
    print("=" * 50)
    print("")
    
    ollama = check_ollama()
    webui = check_webui()
    
    print_status(ollama)
    print("")
    print_status(webui)
    
    print("")
    print("=" * 50)
    
    if ollama["running"] and webui["running"]:
        print(f"  Chat: http://127.0.0.1:{WEBUI_PORT}")
    else:
        print("  Run start.py to launch services")
        
    print("=" * 50)
    print("")
    
    all_running = ollama["running"] and webui["running"]
    return 0 if all_running else 1


if __name__ == "__main__":
    sys.exit(main())
