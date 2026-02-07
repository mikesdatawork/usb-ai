#!/usr/bin/env python3
"""
start.py

Cross-platform launcher for USB-AI.
Starts Ollama server and Open WebUI.
"""

import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

__version__ = "1.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

OLLAMA_PORT = 11434
WEBUI_PORT = 3000
STARTUP_TIMEOUT = 60
HEALTH_CHECK_INTERVAL = 2


class USBAILauncher:
    """Manages USB-AI service lifecycle."""

    def __init__(self):
        self.root_path = self._find_root()
        self.config = self._load_config()
        self.ollama_process: Optional[subprocess.Popen] = None
        self.webui_process: Optional[subprocess.Popen] = None
        self.warmup_manager = None
        self.system = platform.system().lower()
        
    def _find_root(self) -> Path:
        """Locate USB-AI root directory."""
        script_dir = Path(__file__).parent.resolve()

        # scripts/launchers -> scripts -> root
        if (script_dir.parent.parent / "modules").exists():
            return script_dir.parent.parent
        # scripts -> root (if moved)
        if (script_dir.parent / "modules").exists():
            return script_dir.parent
        # direct (if script at root)
        if (script_dir / "modules").exists():
            return script_dir

        log.error("Cannot locate USB-AI root directory")
        sys.exit(1)
        
    def _load_config(self) -> dict:
        """Load launcher configuration."""
        config_path = self.root_path / "modules" / "config" / "system.json"
        
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        
        return {
            "auto_open_browser": True,
            "startup_timeout_seconds": STARTUP_TIMEOUT,
        }
        
    def _get_ollama_binary(self) -> Path:
        """Get platform-specific Ollama binary path."""
        arch = platform.machine().lower()
        
        arch_map = {
            "x86_64": "amd64",
            "amd64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }
        
        arch_name = arch_map.get(arch, "amd64")
        
        if self.system == "darwin":
            binary_name = "ollama"
            platform_dir = f"darwin-{arch_name}"
        elif self.system == "windows":
            binary_name = "ollama.exe"
            platform_dir = f"windows-{arch_name}"
        else:
            binary_name = "ollama"
            platform_dir = f"linux-{arch_name}"
            
        binary_path = (
            self.root_path / "modules" / "ollama-portable" / 
            "bin" / platform_dir / binary_name
        )
        
        return binary_path
        
    def _get_models_path(self) -> Path:
        """Get models directory path."""
        return self.root_path / "modules" / "models"
        
    def _check_port(self, port: int) -> bool:
        """Check if a port is responding."""
        import socket
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            return result == 0
        finally:
            sock.close()
            
    def _wait_for_service(self, port: int, name: str, timeout: int) -> bool:
        """Wait for a service to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._check_port(port):
                log.info(f"{name} is ready on port {port}")
                return True
            time.sleep(HEALTH_CHECK_INTERVAL)
            
        log.error(f"{name} failed to start within {timeout} seconds")
        return False
        
    def start_ollama(self) -> bool:
        """Start Ollama server."""
        binary = self._get_ollama_binary()
        
        if not binary.exists():
            log.error(f"Ollama binary not found: {binary}")
            return False
            
        if self._check_port(OLLAMA_PORT):
            log.info("Ollama already running")
            return True
            
        log.info("Starting Ollama server...")
        
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
        env["OLLAMA_MODELS"] = str(self._get_models_path())
        
        try:
            self.ollama_process = subprocess.Popen(
                [str(binary), "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.error(f"Failed to start Ollama: {e}")
            return False
            
        return self._wait_for_service(OLLAMA_PORT, "Ollama", 30)

    def start_warmup_manager(self) -> bool:
        """Start the model warmup manager for zero cold start latency."""
        try:
            # Import warmup manager
            sys.path.insert(0, str(self.root_path))
            from scripts.performance.model_warmup import WarmupManager

            log.info("Starting model warmup manager...")

            # Get default model from config
            default_model = self.config.get("default_model")

            # Get keep_alive setting
            keep_alive = self.config.get("keep_alive", "30m")

            self.warmup_manager = WarmupManager(
                ollama_host=f"http://127.0.0.1:{OLLAMA_PORT}",
                keep_alive=keep_alive,
                default_model=default_model
            )

            # Start warmup manager in background
            self.warmup_manager.start()
            log.info("Model warmup manager started")
            return True

        except ImportError as e:
            log.warning(f"Model warmup not available: {e}")
            return False
        except Exception as e:
            log.warning(f"Failed to start warmup manager: {e}")
            return False
        
    def start_webui(self) -> bool:
        """Start USB-AI Chat UI (Flask + HTMX)."""
        webui_path = self.root_path / "modules" / "webui-portable"
        chat_ui = webui_path / "chat_ui.py"

        if self._check_port(WEBUI_PORT):
            log.info("Chat UI already running")
            return True

        if not chat_ui.exists():
            log.error(f"Chat UI not found: {chat_ui}")
            return False

        log.info("Starting USB-AI Chat UI...")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(webui_path / "app")

        try:
            self.webui_process = subprocess.Popen(
                [
                    sys.executable, str(chat_ui),
                    "--port", str(WEBUI_PORT),
                    "--host", "127.0.0.1"
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.error(f"Failed to start Chat UI: {e}")
            return False

        return self._wait_for_service(WEBUI_PORT, "Chat UI", 30)
        
    def open_browser(self):
        """Open browser to WebUI."""
        url = f"http://127.0.0.1:{WEBUI_PORT}"
        log.info(f"Opening browser: {url}")
        webbrowser.open(url)
        
    def stop_all(self):
        """Stop all services."""
        log.info("Stopping services...")

        # Stop warmup manager first
        if self.warmup_manager:
            try:
                self.warmup_manager.stop()
                log.info("Warmup manager stopped")
            except Exception as e:
                log.warning(f"Error stopping warmup manager: {e}")

        if self.webui_process:
            self.webui_process.terminate()
            try:
                self.webui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.webui_process.kill()
            log.info("Open WebUI stopped")

        if self.ollama_process:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
            log.info("Ollama stopped")
            
    def run(self):
        """Main execution flow."""
        print("")
        print("=" * 50)
        print("             USB-AI Starting")
        print("=" * 50)
        print("")
        
        if not self.start_ollama():
            log.error("Failed to start Ollama")
            return 1

        # Start warmup manager to eliminate cold start latency
        if self.config.get("enable_warmup", True):
            self.start_warmup_manager()

        if not self.start_webui():
            log.error("Failed to start Open WebUI")
            self.stop_all()
            return 1

        if self.config.get("auto_open_browser", True):
            self.open_browser()
            
        print("")
        print("=" * 50)
        print("            USB-AI is running")
        print(f"         http://127.0.0.1:{WEBUI_PORT}")
        print("=" * 50)
        print("")
        print("Press Ctrl+C to stop")
        
        def signal_handler(sig, frame):
            print("")
            self.stop_all()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while True:
            time.sleep(1)


def main() -> int:
    """Entry point."""
    launcher = USBAILauncher()
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
