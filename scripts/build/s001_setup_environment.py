#!/usr/bin/env python3
"""
s001_setup_environment.py

Sets up the build environment for USB-AI.
Creates modular directory structure.
Run this first before any other build scripts.
"""

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

REQUIRED_PYTHON = (3, 10)


def check_python_version() -> bool:
    """Verify Python version meets requirements."""
    current = sys.version_info[:2]
    
    if current < REQUIRED_PYTHON:
        log.error(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required. "
            f"Found {current[0]}.{current[1]}"
        )
        return False
        
    log.info(f"Python {current[0]}.{current[1]} OK")
    return True


def check_git() -> bool:
    """Verify git is available."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log.info(f"Git: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
        
    log.error("Git not found. Please install git.")
    return False


def create_directory_structure(root: Path) -> bool:
    """Create modular directory structure."""
    log.info("Creating directory structure...")
    
    directories = [
        "modules/ollama-portable/bin/darwin-arm64",
        "modules/ollama-portable/bin/darwin-amd64",
        "modules/ollama-portable/bin/linux-amd64",
        "modules/ollama-portable/bin/windows-amd64",
        "modules/ollama-portable/config",
        "modules/webui-portable/app",
        "modules/webui-portable/data",
        "modules/webui-portable/static/css",
        "modules/webui-portable/config",
        "modules/models/manifests",
        "modules/models/blobs",
        "modules/models/config",
        "modules/launchers",
        "modules/config",
        "scripts/build",
        "scripts/utils",
        "docs",
    ]
    
    for dir_path in directories:
        full_path = root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        log.info(f"  Created: {dir_path}")
        
    return True


def create_config_files(root: Path) -> bool:
    """Create initial configuration files."""
    log.info("Creating configuration files...")
    
    system_config = {
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "modules": {
            "ollama": "../ollama-portable",
            "webui": "../webui-portable",
            "models": "../models",
            "launchers": "../launchers"
        }
    }
    
    config_path = root / "modules" / "config" / "system.json"
    with open(config_path, "w") as f:
        json.dump(system_config, f, indent=2)
    log.info(f"  Created: system.json")
    
    user_config = {
        "default_model": "dolphin-llama3:8b",
        "auto_start_browser": True,
        "theme": "dark"
    }
    
    user_path = root / "modules" / "config" / "user.json"
    with open(user_path, "w") as f:
        json.dump(user_config, f, indent=2)
    log.info(f"  Created: user.json")
    
    models_config = {
        "default_model": "dolphin-llama3:8b",
        "available_models": [
            {
                "name": "dolphin-llama3:8b",
                "size_gb": 4.7,
                "parameters": "8B",
                "use_case": "general"
            },
            {
                "name": "llama3.2:8b",
                "size_gb": 4.7,
                "parameters": "8B",
                "use_case": "general"
            },
            {
                "name": "qwen2.5:14b",
                "size_gb": 8.9,
                "parameters": "14B",
                "use_case": "complex"
            }
        ]
    }
    
    models_path = root / "modules" / "models" / "config" / "models.json"
    with open(models_path, "w") as f:
        json.dump(models_config, f, indent=2)
    log.info(f"  Created: models.json")
    
    return True


def setup_git_worktrees(root: Path) -> bool:
    """Initialize git worktrees for build and release."""
    log.info("Setting up git worktrees...")
    
    try:
        subprocess.run(
            ["git", "branch", "build"],
            cwd=root,
            capture_output=True
        )
        
        subprocess.run(
            ["git", "branch", "release"],
            cwd=root,
            capture_output=True
        )
        
        build_path = root.parent / "usb-ai-build"
        if not build_path.exists():
            subprocess.run(
                ["git", "worktree", "add", str(build_path), "build"],
                cwd=root,
                capture_output=True
            )
            log.info(f"  Created worktree: {build_path}")
        else:
            log.info(f"  Worktree exists: {build_path}")
            
        release_path = root.parent / "usb-ai-release"
        if not release_path.exists():
            subprocess.run(
                ["git", "worktree", "add", str(release_path), "release"],
                cwd=root,
                capture_output=True
            )
            log.info(f"  Created worktree: {release_path}")
        else:
            log.info(f"  Worktree exists: {release_path}")
            
    except Exception as e:
        log.warning(f"Git worktree setup failed: {e}")
        log.info("Continuing without worktrees...")
        
    return True


def print_summary(root: Path):
    """Print setup summary."""
    print("")
    print("=" * 50)
    print("        Environment Setup Complete")
    print("=" * 50)
    print("")
    print("Directory structure created:")
    print(f"  {root}/")
    print("    modules/")
    print("      ollama-portable/  (LLM runtime)")
    print("      webui-portable/   (Chat interface)")
    print("      models/           (AI models)")
    print("      launchers/        (Start scripts)")
    print("      config/           (Settings)")
    print("    scripts/            (Build automation)")
    print("    docs/               (Documentation)")
    print("")
    print("Next steps:")
    print("  1. python scripts/build/s002_download_ollama.py")
    print("  2. python scripts/build/s003_download_models.py")
    print("  3. python scripts/build/s004_setup_webui.py")
    print("  4. python scripts/build/s005_apply_theme.py")
    print("")
    print("=" * 50)


def main() -> int:
    """Entry point."""
    log.info("USB-AI Environment Setup")
    log.info(f"Platform: {platform.system()} {platform.machine()}")
    print("")
    
    if not check_python_version():
        return 1
        
    if not check_git():
        return 1
    
    root = Path(__file__).parent.parent.parent.resolve()
    log.info(f"Root: {root}")
    print("")
    
    if not create_directory_structure(root):
        return 1
        
    if not create_config_files(root):
        return 1
        
    setup_git_worktrees(root)
    
    print_summary(root)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
