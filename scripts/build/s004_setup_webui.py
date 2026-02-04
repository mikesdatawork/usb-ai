#!/usr/bin/env python3
"""
s004_setup_webui.py

Sets up Open WebUI for portable operation.
Installs to modules/webui-portable/
"""

import json
import logging
import platform
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

WEBUI_PACKAGE = "open-webui"
OLLAMA_PORT = 11434
WEBUI_PORT = 3000


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def check_pip() -> bool:
    """Verify pip is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log.info(f"pip: {result.stdout.strip()}")
            return True
    except Exception:
        pass

    log.error("pip not found")
    return False


def install_webui(target_dir: Path) -> bool:
    """Install Open WebUI to target directory."""
    log.info(f"Installing Open WebUI to: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                WEBUI_PACKAGE,
                "--target", str(target_dir),
                "--upgrade",
                "--no-warn-script-location",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            log.info("Open WebUI installed successfully")
            return True
        else:
            log.error(f"Installation failed: {result.stderr}")
            return False

    except Exception as e:
        log.error(f"Installation error: {e}")
        return False


def create_directories(webui_path: Path) -> bool:
    """Create required directories."""
    log.info("Creating directories...")

    directories = [
        webui_path / "app",
        webui_path / "data",
        webui_path / "data" / "uploads",
        webui_path / "data" / "cache",
        webui_path / "static" / "css",
        webui_path / "config",
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        log.info(f"  Created: {dir_path.relative_to(webui_path)}")

    return True


def create_start_script(webui_path: Path) -> bool:
    """Create WebUI start script."""
    log.info("Creating start script...")

    script_content = '''#!/usr/bin/env python3
"""
start_webui.py

Starts Open WebUI server.
"""

import os
import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.resolve()

    # Add app directory to Python path
    app_dir = script_dir / "app"
    if app_dir.exists():
        sys.path.insert(0, str(app_dir))

    # Set environment
    os.environ["DATA_DIR"] = str(script_dir / "data")
    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    os.environ["WEBUI_AUTH"] = "False"
    os.environ["ENABLE_SIGNUP"] = "False"

    # Import and run
    try:
        from open_webui.main import app
        import uvicorn

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=3000,
            log_level="warning",
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure Open WebUI is installed in the app/ directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    script_path = webui_path / "start_webui.py"

    try:
        with open(script_path, "w") as f:
            f.write(script_content)

        if platform.system() != "Windows":
            import stat
            script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        log.info(f"Created: {script_path}")
        return True

    except Exception as e:
        log.error(f"Failed to create start script: {e}")
        return False


def create_config(webui_path: Path) -> bool:
    """Create WebUI configuration."""
    log.info("Creating configuration...")

    config = {
        "host": "127.0.0.1",
        "port": WEBUI_PORT,
        "ollama_url": f"http://127.0.0.1:{OLLAMA_PORT}",
        "data_dir": "./data",
        "enable_signup": False,
        "enable_login": False,
        "default_model": "dolphin-llama3:8b",
        "theme": "dark"
    }

    config_path = webui_path / "config" / "webui.json"

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Created: {config_path}")
        return True

    except Exception as e:
        log.error(f"Failed to create config: {e}")
        return False


def create_env_script(webui_path: Path) -> bool:
    """Create environment setup script."""
    log.info("Creating environment script...")

    bash_content = '''#!/bin/bash
# Open WebUI environment configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$SCRIPT_DIR/app:$PYTHONPATH"
export DATA_DIR="$SCRIPT_DIR/data"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export WEBUI_AUTH="False"
export ENABLE_SIGNUP="False"

echo "WebUI environment configured"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  DATA_DIR: $DATA_DIR"
'''

    bash_path = webui_path / "config" / "webui_env.sh"

    try:
        with open(bash_path, "w") as f:
            f.write(bash_content)

        if platform.system() != "Windows":
            import stat
            bash_path.chmod(bash_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        log.info(f"Created: {bash_path}")
        return True

    except Exception as e:
        log.error(f"Failed to create env script: {e}")
        return False


def verify_installation(webui_path: Path) -> bool:
    """Verify Open WebUI installation."""
    log.info("Verifying installation...")

    app_dir = webui_path / "app"

    check_paths = [
        app_dir / "open_webui",
        webui_path / "data",
        webui_path / "config" / "webui.json",
        webui_path / "start_webui.py",
    ]

    all_ok = True
    for path in check_paths:
        if path.exists():
            log.info(f"  OK: {path.name}")
        else:
            log.warning(f"  Missing: {path}")
            all_ok = False

    return all_ok


def print_summary(success: bool, webui_path: Path):
    """Print setup summary."""
    print("")
    print("=" * 50)
    print("        Open WebUI Setup Complete")
    print("=" * 50)
    print("")

    if success:
        print(f"Installed to: {webui_path}")
        print("")
        print("Components:")
        print("  app/          - Open WebUI application")
        print("  data/         - User data and chat history")
        print("  static/css/   - Custom themes")
        print("  config/       - Configuration files")
        print("")
        print("Next step:")
        print("  python scripts/build/s005_apply_theme.py")
    else:
        print("Installation incomplete. Check errors above.")

    print("")
    print("=" * 50)


def main() -> int:
    """Entry point."""
    log.info("USB-AI Open WebUI Setup")
    log.info(f"Platform: {platform.system()} {platform.machine()}")
    print("")

    root = get_root_path()
    webui_path = root / "modules" / "webui-portable"
    app_path = webui_path / "app"

    log.info(f"Root: {root}")
    log.info(f"WebUI dir: {webui_path}")
    print("")

    if not check_pip():
        return 1

    print("")

    if not create_directories(webui_path):
        return 1

    print("")

    if not install_webui(app_path):
        return 1

    print("")

    create_start_script(webui_path)
    create_config(webui_path)
    create_env_script(webui_path)

    print("")

    success = verify_installation(webui_path)

    print_summary(success, webui_path)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
