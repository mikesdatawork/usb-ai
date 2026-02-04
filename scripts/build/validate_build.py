#!/usr/bin/env python3
"""
validate_build.py

Validates the USB-AI build to ensure all components are present and functional.
Run this after completing all build scripts.
"""

import json
import logging
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


class ValidationResult:
    """Stores validation results."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warnings: List[str] = []

    def add_pass(self, check: str):
        self.passed.append(check)
        log.info(f"  [PASS] {check}")

    def add_fail(self, check: str):
        self.failed.append(check)
        log.error(f"  [FAIL] {check}")

    def add_warn(self, check: str):
        self.warnings.append(check)
        log.warning(f"  [WARN] {check}")

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def check_directory_structure(root: Path, results: ValidationResult):
    """Validate directory structure exists."""
    log.info("Checking directory structure...")

    required_dirs = [
        "modules/ollama-portable/bin",
        "modules/ollama-portable/config",
        "modules/webui-portable/app",
        "modules/webui-portable/data",
        "modules/webui-portable/static/css",
        "modules/webui-portable/config",
        "modules/models",
        "modules/launchers",
        "modules/config",
        "scripts/build",
        "scripts/launchers",
    ]

    for dir_path in required_dirs:
        full_path = root / dir_path
        if full_path.exists() and full_path.is_dir():
            results.add_pass(f"Directory: {dir_path}")
        else:
            results.add_fail(f"Directory missing: {dir_path}")


def check_ollama_binaries(root: Path, results: ValidationResult):
    """Validate Ollama binaries exist."""
    log.info("Checking Ollama binaries...")

    platforms = [
        ("darwin-arm64", "ollama"),
        ("darwin-amd64", "ollama"),
        ("linux-amd64", "ollama"),
        ("windows-amd64", "ollama.exe"),
    ]

    bin_dir = root / "modules" / "ollama-portable" / "bin"

    for plat, binary in platforms:
        binary_path = bin_dir / plat / binary
        if binary_path.exists():
            size_mb = binary_path.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Should be at least 10MB
                results.add_pass(f"Ollama binary: {plat} ({size_mb:.1f} MB)")
            else:
                results.add_warn(f"Ollama binary small: {plat} ({size_mb:.1f} MB)")
        else:
            results.add_fail(f"Ollama binary missing: {plat}")


def check_webui_installation(root: Path, results: ValidationResult):
    """Validate Open WebUI installation."""
    log.info("Checking Open WebUI installation...")

    webui_dir = root / "modules" / "webui-portable"
    app_dir = webui_dir / "app"

    # Check for open_webui package
    open_webui_path = app_dir / "open_webui"
    if open_webui_path.exists():
        results.add_pass("Open WebUI package installed")
    else:
        # Check if it might be installed differently
        if any(app_dir.glob("**/open_webui")):
            results.add_pass("Open WebUI package found")
        else:
            results.add_fail("Open WebUI package not found")

    # Check config
    config_path = webui_dir / "config" / "webui.json"
    if config_path.exists():
        results.add_pass("WebUI config exists")
    else:
        results.add_warn("WebUI config missing")

    # Check theme
    theme_path = webui_dir / "static" / "css" / "custom-theme.css"
    if theme_path.exists():
        results.add_pass("Custom theme CSS exists")
    else:
        results.add_warn("Custom theme CSS missing")


def check_models_directory(root: Path, results: ValidationResult):
    """Validate models directory."""
    log.info("Checking models directory...")

    models_dir = root / "modules" / "models"

    if not models_dir.exists():
        results.add_fail("Models directory missing")
        return

    # Check for model files (blobs)
    blobs_dir = models_dir / "blobs"
    if blobs_dir.exists():
        blob_count = len(list(blobs_dir.glob("sha256-*")))
        if blob_count > 0:
            results.add_pass(f"Model blobs found: {blob_count}")
        else:
            results.add_warn("No model blobs found (run s003_download_models.py)")
    else:
        results.add_warn("Model blobs directory missing")

    # Check config
    config_path = models_dir / "config" / "models.json"
    if config_path.exists():
        results.add_pass("Models config exists")
        try:
            with open(config_path) as f:
                config = json.load(f)
                models = config.get("available_models", [])
                results.add_pass(f"Models configured: {len(models)}")
        except Exception as e:
            results.add_warn(f"Could not read models config: {e}")
    else:
        results.add_warn("Models config missing")


def check_launcher_scripts(root: Path, results: ValidationResult):
    """Validate launcher scripts."""
    log.info("Checking launcher scripts...")

    # Python launchers
    python_launchers = [
        "scripts/launchers/start.py",
        "scripts/launchers/stop.py",
        "scripts/launchers/status.py",
    ]

    for script in python_launchers:
        script_path = root / script
        if script_path.exists():
            results.add_pass(f"Python launcher: {script}")
        else:
            results.add_fail(f"Python launcher missing: {script}")

    # Platform launchers
    platform_launchers = [
        "modules/launchers/start_macos.command",
        "modules/launchers/start_windows.bat",
        "modules/launchers/start_linux.sh",
        "modules/launchers/stop_all.sh",
    ]

    for script in platform_launchers:
        script_path = root / script
        if script_path.exists():
            results.add_pass(f"Platform launcher: {script}")
        else:
            results.add_fail(f"Platform launcher missing: {script}")


def check_build_scripts(root: Path, results: ValidationResult):
    """Validate build scripts."""
    log.info("Checking build scripts...")

    scripts = [
        "scripts/build/s001_setup_environment.py",
        "scripts/build/s002_download_ollama.py",
        "scripts/build/s003_download_models.py",
        "scripts/build/s004_setup_webui.py",
        "scripts/build/s005_apply_theme.py",
    ]

    for script in scripts:
        script_path = root / script
        if script_path.exists():
            results.add_pass(f"Build script: {script}")
        else:
            results.add_fail(f"Build script missing: {script}")


def check_config_files(root: Path, results: ValidationResult):
    """Validate configuration files."""
    log.info("Checking configuration files...")

    configs = [
        ("modules/config/system.json", True),
        ("modules/config/user.json", True),
        ("modules/ollama-portable/config/ollama.json", False),
        ("modules/models/config/models.json", False),
    ]

    for config, required in configs:
        config_path = root / config
        if config_path.exists():
            results.add_pass(f"Config: {config}")
        elif required:
            results.add_fail(f"Config missing: {config}")
        else:
            results.add_warn(f"Config missing (optional): {config}")


def check_documentation(root: Path, results: ValidationResult):
    """Validate documentation files."""
    log.info("Checking documentation...")

    docs = [
        "README.md",
        "CLAUDE.md",
        "docs/PRD.md",
        "docs/BUILD_PROCESS.md",
    ]

    for doc in docs:
        doc_path = root / doc
        if doc_path.exists():
            results.add_pass(f"Documentation: {doc}")
        else:
            results.add_warn(f"Documentation missing: {doc}")


def print_summary(results: ValidationResult):
    """Print validation summary."""
    print("")
    print("=" * 60)
    print("              USB-AI Build Validation Report")
    print("=" * 60)
    print("")

    total = len(results.passed) + len(results.failed) + len(results.warnings)

    print(f"Total checks: {total}")
    print(f"  Passed:   {len(results.passed)}")
    print(f"  Failed:   {len(results.failed)}")
    print(f"  Warnings: {len(results.warnings)}")
    print("")

    if results.failed:
        print("FAILED CHECKS:")
        for check in results.failed:
            print(f"  - {check}")
        print("")

    if results.warnings:
        print("WARNINGS:")
        for check in results.warnings:
            print(f"  - {check}")
        print("")

    if results.success:
        print("=" * 60)
        print("       BUILD VALIDATION: PASSED")
        print("")
        print("  USB-AI is ready to use!")
        print("  Run: python scripts/launchers/start.py")
        print("=" * 60)
    else:
        print("=" * 60)
        print("       BUILD VALIDATION: FAILED")
        print("")
        print("  Please fix the failed checks above.")
        print("=" * 60)

    print("")


def main() -> int:
    """Entry point."""
    log.info("USB-AI Build Validation")
    log.info(f"Platform: {platform.system()} {platform.machine()}")
    print("")

    root = get_root_path()
    log.info(f"Root: {root}")
    print("")

    results = ValidationResult()

    check_directory_structure(root, results)
    print("")

    check_build_scripts(root, results)
    print("")

    check_ollama_binaries(root, results)
    print("")

    check_webui_installation(root, results)
    print("")

    check_models_directory(root, results)
    print("")

    check_launcher_scripts(root, results)
    print("")

    check_config_files(root, results)
    print("")

    check_documentation(root, results)

    print_summary(results)

    return 0 if results.success else 1


if __name__ == "__main__":
    sys.exit(main())
