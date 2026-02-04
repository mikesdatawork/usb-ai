#!/usr/bin/env python3
"""
s003_download_models.py

Downloads AI models using Ollama.
Models are stored in modules/models/

Requires Ollama to be available (either system-installed or from s002).
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

MODELS = [
    {
        "name": "dolphin-llama3:8b",
        "description": "General purpose, uncensored",
        "size_gb": 4.7,
        "priority": "critical",
    },
    {
        "name": "llama3.2:8b",
        "description": "General purpose, balanced",
        "size_gb": 4.7,
        "priority": "high",
    },
    {
        "name": "qwen2.5:14b",
        "description": "High quality, complex tasks",
        "size_gb": 8.9,
        "priority": "normal",
    },
]

OLLAMA_PORT = 11434
SERVER_STARTUP_TIMEOUT = 30


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def get_ollama_binary() -> Optional[Path]:
    """Find Ollama binary - prefer local, fall back to system."""
    root = get_root_path()
    system = platform.system().lower()
    machine = platform.machine().lower()

    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    arch = arch_map.get(machine, "amd64")

    if system == "darwin":
        local_binary = root / "modules" / "ollama-portable" / "bin" / f"darwin-{arch}" / "ollama"
    elif system == "windows":
        local_binary = root / "modules" / "ollama-portable" / "bin" / "windows-amd64" / "ollama.exe"
    else:
        local_binary = root / "modules" / "ollama-portable" / "bin" / "linux-amd64" / "ollama"

    if local_binary.exists():
        log.info(f"Using local Ollama: {local_binary}")
        return local_binary

    try:
        result = subprocess.run(
            ["which", "ollama"] if system != "windows" else ["where", "ollama"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            system_binary = Path(result.stdout.strip().split('\n')[0])
            log.info(f"Using system Ollama: {system_binary}")
            return system_binary
    except Exception:
        pass

    return None


def check_ollama_server(port: int = OLLAMA_PORT) -> bool:
    """Check if Ollama server is running."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)

    try:
        result = sock.connect_ex(("127.0.0.1", port))
        return result == 0
    finally:
        sock.close()


def start_ollama_server(binary: Path, models_path: Path) -> Optional[subprocess.Popen]:
    """Start Ollama server."""
    if check_ollama_server():
        log.info("Ollama server already running")
        return None

    log.info("Starting Ollama server...")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    env["OLLAMA_MODELS"] = str(models_path)

    try:
        process = subprocess.Popen(
            [str(binary), "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for i in range(SERVER_STARTUP_TIMEOUT):
            if check_ollama_server():
                log.info("Ollama server started")
                return process
            time.sleep(1)

        log.error("Ollama server failed to start")
        process.terminate()
        return None

    except Exception as e:
        log.error(f"Failed to start Ollama: {e}")
        return None


def stop_ollama_server(process: Optional[subprocess.Popen]):
    """Stop Ollama server if we started it."""
    if process:
        log.info("Stopping Ollama server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def pull_model(binary: Path, model_name: str, models_path: Path) -> bool:
    """Pull a model using Ollama."""
    log.info(f"Pulling model: {model_name}")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    env["OLLAMA_MODELS"] = str(models_path)

    try:
        process = subprocess.Popen(
            [str(binary), "pull", model_name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                if "pulling" in line.lower() or "%" in line:
                    print(f"  {line}", end='\r')
                elif "success" in line.lower():
                    print(f"  {line}")

        process.wait()
        print("")

        if process.returncode == 0:
            log.info(f"Successfully pulled: {model_name}")
            return True
        else:
            log.error(f"Failed to pull: {model_name}")
            return False

    except Exception as e:
        log.error(f"Error pulling model: {e}")
        return False


def list_models(binary: Path, models_path: Path) -> list:
    """List installed models."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    env["OLLAMA_MODELS"] = str(models_path)

    try:
        result = subprocess.run(
            [str(binary), "list"],
            env=env,
            capture_output=True,
            text=True,
        )

        models = []
        for line in result.stdout.strip().split('\n')[1:]:
            if line.strip():
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models

    except Exception as e:
        log.error(f"Error listing models: {e}")
        return []


def update_models_config(root: Path, installed_models: list):
    """Update models configuration file."""
    config_path = root / "modules" / "models" / "config" / "models.json"

    config = {
        "default_model": "dolphin-llama3:8b",
        "available_models": []
    }

    for model in MODELS:
        if model["name"] in installed_models:
            config["available_models"].append({
                "name": model["name"],
                "size_gb": model["size_gb"],
                "description": model["description"],
                "installed": True
            })

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Updated models config: {config_path}")
    except Exception as e:
        log.error(f"Failed to update config: {e}")


def print_summary(results: dict, installed_models: list):
    """Print download summary."""
    print("")
    print("=" * 50)
    print("        Model Download Complete")
    print("=" * 50)
    print("")

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for model, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model}: {status}")

    print("")
    print(f"Downloaded: {success_count}/{total_count} models")
    print("")

    if installed_models:
        print("Installed models:")
        for model in installed_models:
            print(f"  - {model}")
        print("")

    print("Next step:")
    print("  python scripts/build/s004_setup_webui.py")
    print("")
    print("=" * 50)


def main() -> int:
    """Entry point."""
    log.info("USB-AI Model Download")
    log.info(f"Platform: {platform.system()} {platform.machine()}")
    print("")

    root = get_root_path()
    models_path = root / "modules" / "models"

    log.info(f"Root: {root}")
    log.info(f"Models dir: {models_path}")
    print("")

    binary = get_ollama_binary()
    if not binary:
        log.error("Ollama not found. Run s002_download_ollama.py first.")
        return 1

    models_path.mkdir(parents=True, exist_ok=True)

    server_process = start_ollama_server(binary, models_path)

    if not check_ollama_server():
        log.error("Ollama server not available")
        return 1

    print("")
    results = {}

    try:
        for model in MODELS:
            results[model["name"]] = pull_model(binary, model["name"], models_path)
            print("")

    finally:
        stop_ollama_server(server_process)

    installed_models = list_models(binary, models_path)
    update_models_config(root, installed_models)

    print_summary(results, installed_models)

    critical_models = [m["name"] for m in MODELS if m["priority"] == "critical"]
    critical_success = all(results.get(m, False) for m in critical_models)

    return 0 if critical_success else 1


if __name__ == "__main__":
    sys.exit(main())
