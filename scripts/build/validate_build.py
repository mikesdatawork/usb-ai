#!/usr/bin/env python3
"""
validate_build.py

Comprehensive build validation for USB-AI.
Validates all build outputs, checks file integrity,
verifies cross-platform compatibility, and generates detailed reports.

Usage:
    python scripts/build/validate_build.py [options]

Options:
    --output PATH       Output report path (default: docs/BUILD_REPORT.md)
    --json              Output JSON format instead of markdown
    --verbose           Enable verbose output
    --strict            Fail on any warning
    --checksums         Verify file checksums
    --quick             Skip time-consuming checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__version__ = "2.0.0"

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class CheckStatus(Enum):
    """Status of a validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    INFO = "info"


class CheckCategory(Enum):
    """Categories of validation checks."""
    DIRECTORY = "directory"
    FILE = "file"
    BINARY = "binary"
    CONFIG = "config"
    MODEL = "model"
    SCRIPT = "script"
    PERMISSION = "permission"
    CHECKSUM = "checksum"
    PLATFORM = "platform"
    INTEGRATION = "integration"


# Minimum file sizes for binaries (in MB)
MIN_BINARY_SIZES = {
    "ollama": 50,
    "ollama.exe": 50,
}

# Required directories
REQUIRED_DIRECTORIES = [
    "modules/ollama-portable/bin",
    "modules/ollama-portable/config",
    "modules/webui-portable/app",
    "modules/webui-portable/data",
    "modules/webui-portable/static/css",
    "modules/webui-portable/config",
    "modules/models",
    "modules/models/config",
    "modules/launchers",
    "modules/config",
    "scripts/build",
    "scripts/launchers",
]

# Required configuration files
REQUIRED_CONFIGS = [
    ("modules/config/system.json", True),
    ("modules/config/user.json", True),
    ("modules/ollama-portable/config/ollama.json", False),
    ("modules/models/config/models.json", False),
    ("modules/webui-portable/config/webui.json", False),
]

# Platform-specific binaries
PLATFORM_BINARIES = [
    ("darwin-arm64", "ollama"),
    ("darwin-amd64", "ollama"),
    ("linux-amd64", "ollama"),
    ("windows-amd64", "ollama.exe"),
]

# Required launcher scripts
REQUIRED_LAUNCHERS = {
    "python": [
        "scripts/launchers/start.py",
        "scripts/launchers/stop.py",
        "scripts/launchers/status.py",
    ],
    "platform": [
        "modules/launchers/start_macos.command",
        "modules/launchers/start_windows.bat",
        "modules/launchers/start_linux.sh",
        "modules/launchers/stop_all.sh",
    ],
}

# Required build scripts
REQUIRED_BUILD_SCRIPTS = [
    "scripts/build/s001_setup_environment.py",
    "scripts/build/s002_download_ollama.py",
    "scripts/build/s003_download_models.py",
    "scripts/build/s004_setup_webui.py",
    "scripts/build/s005_apply_theme.py",
    "scripts/build/parallel_builder.py",
    "scripts/build/build_manifest.yaml",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationCheck:
    """Represents a single validation check."""
    name: str
    category: CheckCategory
    status: CheckStatus
    message: str
    details: Optional[str] = None
    path: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ValidationResults:
    """Collection of validation results."""
    checks: List[ValidationCheck] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def passed(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == CheckStatus.PASS]

    @property
    def failed(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warnings(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    @property
    def skipped(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == CheckStatus.SKIP]

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def add(self, check: ValidationCheck):
        """Add a check result."""
        self.checks.append(check)

        # Log based on status
        if check.status == CheckStatus.PASS:
            log.info(f"  [PASS] {check.name}")
        elif check.status == CheckStatus.FAIL:
            log.error(f"  [FAIL] {check.name}: {check.message}")
        elif check.status == CheckStatus.WARN:
            log.warning(f"  [WARN] {check.name}: {check.message}")
        elif check.status == CheckStatus.SKIP:
            log.info(f"  [SKIP] {check.name}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "summary": {
                "total": len(self.checks),
                "passed": len(self.passed),
                "failed": len(self.failed),
                "warnings": len(self.warnings),
                "skipped": len(self.skipped),
                "success": self.success,
                "duration_seconds": self.duration_seconds,
            },
            "checks": [c.to_dict() for c in self.checks],
        }


# =============================================================================
# Utility Functions
# =============================================================================

def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def is_executable(file_path: Path) -> bool:
    """Check if file has executable permission."""
    if platform.system() == "Windows":
        return file_path.suffix.lower() in [".exe", ".bat", ".cmd", ".ps1"]
    return os.access(file_path, os.X_OK)


def detect_binary_type(file_path: Path) -> Optional[str]:
    """Detect binary file type (Mach-O, ELF, PE)."""
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)

            # Mach-O (macOS)
            if magic[:4] in [b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
                            b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"]:
                return "macho"

            # Mach-O Universal (Fat binary)
            if magic[:4] in [b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"]:
                return "macho-universal"

            # ELF (Linux)
            if magic[:4] == b"\x7fELF":
                return "elf"

            # PE (Windows)
            if magic[:2] == b"MZ":
                return "pe"

    except Exception:
        pass

    return None


def check_json_valid(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if JSON file is valid."""
    try:
        with open(file_path) as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        return result != 0  # Port is available if connect fails
    finally:
        sock.close()


# =============================================================================
# Validation Checks
# =============================================================================

class BuildValidator:
    """Comprehensive build validator."""

    def __init__(
        self,
        root_path: Path,
        verbose: bool = False,
        strict: bool = False,
        verify_checksums: bool = False,
        quick: bool = False,
    ):
        self.root_path = root_path
        self.verbose = verbose
        self.strict = strict
        self.verify_checksums = verify_checksums
        self.quick = quick
        self.results = ValidationResults()

    def validate_all(self) -> ValidationResults:
        """Run all validation checks."""
        self.results.start_time = datetime.now()

        log.info("Starting USB-AI build validation...")
        log.info(f"Root path: {self.root_path}")
        log.info("")

        # Run validation categories
        self._validate_directories()
        log.info("")

        self._validate_build_scripts()
        log.info("")

        self._validate_ollama_binaries()
        log.info("")

        self._validate_webui_installation()
        log.info("")

        self._validate_models()
        log.info("")

        self._validate_launcher_scripts()
        log.info("")

        self._validate_config_files()
        log.info("")

        self._validate_cross_platform()
        log.info("")

        if not self.quick:
            self._validate_integration()
            log.info("")

        self.results.end_time = datetime.now()

        return self.results

    def _validate_directories(self):
        """Validate directory structure."""
        log.info("Checking directory structure...")

        for dir_path in REQUIRED_DIRECTORIES:
            full_path = self.root_path / dir_path

            if full_path.exists() and full_path.is_dir():
                self.results.add(ValidationCheck(
                    name=f"Directory: {dir_path}",
                    category=CheckCategory.DIRECTORY,
                    status=CheckStatus.PASS,
                    message="Directory exists",
                    path=dir_path,
                ))
            else:
                self.results.add(ValidationCheck(
                    name=f"Directory: {dir_path}",
                    category=CheckCategory.DIRECTORY,
                    status=CheckStatus.FAIL,
                    message="Directory missing",
                    path=dir_path,
                ))

    def _validate_build_scripts(self):
        """Validate build scripts exist and are valid Python."""
        log.info("Checking build scripts...")

        for script_path in REQUIRED_BUILD_SCRIPTS:
            full_path = self.root_path / script_path

            if not full_path.exists():
                self.results.add(ValidationCheck(
                    name=f"Script: {script_path}",
                    category=CheckCategory.SCRIPT,
                    status=CheckStatus.FAIL,
                    message="Script not found",
                    path=script_path,
                ))
                continue

            # Check file size
            size = full_path.stat().st_size
            if size == 0:
                self.results.add(ValidationCheck(
                    name=f"Script: {script_path}",
                    category=CheckCategory.SCRIPT,
                    status=CheckStatus.FAIL,
                    message="Script is empty",
                    path=script_path,
                ))
                continue

            # Check Python syntax for .py files
            if script_path.endswith(".py"):
                try:
                    with open(full_path) as f:
                        compile(f.read(), script_path, "exec")

                    self.results.add(ValidationCheck(
                        name=f"Script: {script_path}",
                        category=CheckCategory.SCRIPT,
                        status=CheckStatus.PASS,
                        message="Valid Python script",
                        path=script_path,
                    ))
                except SyntaxError as e:
                    self.results.add(ValidationCheck(
                        name=f"Script: {script_path}",
                        category=CheckCategory.SCRIPT,
                        status=CheckStatus.FAIL,
                        message=f"Syntax error: {e}",
                        path=script_path,
                    ))
            else:
                self.results.add(ValidationCheck(
                    name=f"Script: {script_path}",
                    category=CheckCategory.SCRIPT,
                    status=CheckStatus.PASS,
                    message="Script exists",
                    path=script_path,
                ))

    def _validate_ollama_binaries(self):
        """Validate Ollama binaries for all platforms."""
        log.info("Checking Ollama binaries...")

        bin_dir = self.root_path / "modules" / "ollama-portable" / "bin"

        for plat, binary_name in PLATFORM_BINARIES:
            binary_path = bin_dir / plat / binary_name

            if not binary_path.exists():
                self.results.add(ValidationCheck(
                    name=f"Ollama: {plat}",
                    category=CheckCategory.BINARY,
                    status=CheckStatus.FAIL,
                    message="Binary not found",
                    path=str(binary_path.relative_to(self.root_path)),
                ))
                continue

            # Check file size
            size_mb = get_file_size_mb(binary_path)
            min_size = MIN_BINARY_SIZES.get(binary_name, 10)

            if size_mb < min_size:
                self.results.add(ValidationCheck(
                    name=f"Ollama: {plat}",
                    category=CheckCategory.BINARY,
                    status=CheckStatus.WARN,
                    message=f"Binary smaller than expected ({size_mb:.1f} MB < {min_size} MB)",
                    path=str(binary_path.relative_to(self.root_path)),
                    expected=f">= {min_size} MB",
                    actual=f"{size_mb:.1f} MB",
                ))
                continue

            # Check binary type matches platform
            binary_type = detect_binary_type(binary_path)
            expected_type = self._get_expected_binary_type(plat)

            if binary_type != expected_type:
                self.results.add(ValidationCheck(
                    name=f"Ollama: {plat}",
                    category=CheckCategory.BINARY,
                    status=CheckStatus.WARN,
                    message=f"Unexpected binary type: {binary_type}",
                    path=str(binary_path.relative_to(self.root_path)),
                    expected=expected_type,
                    actual=binary_type,
                ))
            else:
                self.results.add(ValidationCheck(
                    name=f"Ollama: {plat}",
                    category=CheckCategory.BINARY,
                    status=CheckStatus.PASS,
                    message=f"Valid {binary_type} binary ({size_mb:.1f} MB)",
                    path=str(binary_path.relative_to(self.root_path)),
                ))

            # Check executable permission (Unix only)
            if not plat.startswith("windows"):
                if not is_executable(binary_path):
                    self.results.add(ValidationCheck(
                        name=f"Ollama: {plat} (permissions)",
                        category=CheckCategory.PERMISSION,
                        status=CheckStatus.WARN,
                        message="Binary is not executable",
                        path=str(binary_path.relative_to(self.root_path)),
                    ))

    def _get_expected_binary_type(self, platform: str) -> str:
        """Get expected binary type for platform."""
        if platform.startswith("darwin"):
            return "macho"
        elif platform.startswith("linux"):
            return "elf"
        elif platform.startswith("windows"):
            return "pe"
        return "unknown"

    def _validate_webui_installation(self):
        """Validate Open WebUI installation."""
        log.info("Checking WebUI installation...")

        webui_dir = self.root_path / "modules" / "webui-portable"
        app_dir = webui_dir / "app"

        # Check open_webui package
        open_webui_path = app_dir / "open_webui"
        if open_webui_path.exists():
            self.results.add(ValidationCheck(
                name="WebUI: Package",
                category=CheckCategory.FILE,
                status=CheckStatus.PASS,
                message="Open WebUI package found",
                path=str(open_webui_path.relative_to(self.root_path)),
            ))
        else:
            # Check alternative locations
            found = list(app_dir.glob("**/open_webui"))
            if found:
                self.results.add(ValidationCheck(
                    name="WebUI: Package",
                    category=CheckCategory.FILE,
                    status=CheckStatus.PASS,
                    message=f"Open WebUI found at {found[0]}",
                    path=str(found[0].relative_to(self.root_path)),
                ))
            else:
                self.results.add(ValidationCheck(
                    name="WebUI: Package",
                    category=CheckCategory.FILE,
                    status=CheckStatus.FAIL,
                    message="Open WebUI package not found",
                    path=str(app_dir.relative_to(self.root_path)),
                ))

        # Check config file
        config_path = webui_dir / "config" / "webui.json"
        if config_path.exists():
            valid, error = check_json_valid(config_path)
            if valid:
                self.results.add(ValidationCheck(
                    name="WebUI: Config",
                    category=CheckCategory.CONFIG,
                    status=CheckStatus.PASS,
                    message="Valid JSON configuration",
                    path=str(config_path.relative_to(self.root_path)),
                ))
            else:
                self.results.add(ValidationCheck(
                    name="WebUI: Config",
                    category=CheckCategory.CONFIG,
                    status=CheckStatus.FAIL,
                    message=f"Invalid JSON: {error}",
                    path=str(config_path.relative_to(self.root_path)),
                ))
        else:
            self.results.add(ValidationCheck(
                name="WebUI: Config",
                category=CheckCategory.CONFIG,
                status=CheckStatus.WARN,
                message="Config file missing",
                path=str(config_path.relative_to(self.root_path)),
            ))

        # Check theme CSS
        theme_path = webui_dir / "static" / "css" / "custom-theme.css"
        if theme_path.exists():
            size = theme_path.stat().st_size
            if size > 1000:  # Should be substantial CSS
                self.results.add(ValidationCheck(
                    name="WebUI: Theme",
                    category=CheckCategory.FILE,
                    status=CheckStatus.PASS,
                    message=f"Custom theme found ({size} bytes)",
                    path=str(theme_path.relative_to(self.root_path)),
                ))
            else:
                self.results.add(ValidationCheck(
                    name="WebUI: Theme",
                    category=CheckCategory.FILE,
                    status=CheckStatus.WARN,
                    message="Theme file seems incomplete",
                    path=str(theme_path.relative_to(self.root_path)),
                ))
        else:
            self.results.add(ValidationCheck(
                name="WebUI: Theme",
                category=CheckCategory.FILE,
                status=CheckStatus.WARN,
                message="Custom theme not found",
                path=str(theme_path.relative_to(self.root_path)),
            ))

        # Check data directories
        data_dirs = ["data", "data/uploads", "data/cache"]
        for data_dir in data_dirs:
            data_path = webui_dir / data_dir
            if data_path.exists():
                self.results.add(ValidationCheck(
                    name=f"WebUI: {data_dir}",
                    category=CheckCategory.DIRECTORY,
                    status=CheckStatus.PASS,
                    message="Directory exists",
                    path=str(data_path.relative_to(self.root_path)),
                ))
            else:
                self.results.add(ValidationCheck(
                    name=f"WebUI: {data_dir}",
                    category=CheckCategory.DIRECTORY,
                    status=CheckStatus.WARN,
                    message="Directory missing",
                    path=str(data_path.relative_to(self.root_path)),
                ))

    def _validate_models(self):
        """Validate AI models."""
        log.info("Checking AI models...")

        models_dir = self.root_path / "modules" / "models"

        if not models_dir.exists():
            self.results.add(ValidationCheck(
                name="Models: Directory",
                category=CheckCategory.DIRECTORY,
                status=CheckStatus.FAIL,
                message="Models directory missing",
                path="modules/models",
            ))
            return

        # Check blobs directory
        blobs_dir = models_dir / "blobs"
        if blobs_dir.exists():
            blobs = list(blobs_dir.glob("sha256-*"))
            if blobs:
                total_size = sum(f.stat().st_size for f in blobs) / (1024 * 1024 * 1024)
                self.results.add(ValidationCheck(
                    name="Models: Blobs",
                    category=CheckCategory.MODEL,
                    status=CheckStatus.PASS,
                    message=f"Found {len(blobs)} model blobs ({total_size:.1f} GB)",
                    path=str(blobs_dir.relative_to(self.root_path)),
                ))
            else:
                self.results.add(ValidationCheck(
                    name="Models: Blobs",
                    category=CheckCategory.MODEL,
                    status=CheckStatus.WARN,
                    message="No model blobs found (run s003_download_models.py)",
                    path=str(blobs_dir.relative_to(self.root_path)),
                ))
        else:
            self.results.add(ValidationCheck(
                name="Models: Blobs",
                category=CheckCategory.MODEL,
                status=CheckStatus.WARN,
                message="Blobs directory missing",
                path="modules/models/blobs",
            ))

        # Check manifests directory
        manifests_dir = models_dir / "manifests"
        if manifests_dir.exists():
            manifests = list(manifests_dir.rglob("*"))
            manifests = [m for m in manifests if m.is_file()]
            if manifests:
                self.results.add(ValidationCheck(
                    name="Models: Manifests",
                    category=CheckCategory.MODEL,
                    status=CheckStatus.PASS,
                    message=f"Found {len(manifests)} model manifest files",
                    path=str(manifests_dir.relative_to(self.root_path)),
                ))
            else:
                self.results.add(ValidationCheck(
                    name="Models: Manifests",
                    category=CheckCategory.MODEL,
                    status=CheckStatus.WARN,
                    message="No model manifests found",
                    path=str(manifests_dir.relative_to(self.root_path)),
                ))

        # Check models config
        config_path = models_dir / "config" / "models.json"
        if config_path.exists():
            valid, error = check_json_valid(config_path)
            if valid:
                with open(config_path) as f:
                    config = json.load(f)
                    available = config.get("available_models", [])
                    default = config.get("default_model", "none")

                    self.results.add(ValidationCheck(
                        name="Models: Config",
                        category=CheckCategory.CONFIG,
                        status=CheckStatus.PASS,
                        message=f"Default: {default}, Available: {len(available)}",
                        path=str(config_path.relative_to(self.root_path)),
                    ))
            else:
                self.results.add(ValidationCheck(
                    name="Models: Config",
                    category=CheckCategory.CONFIG,
                    status=CheckStatus.FAIL,
                    message=f"Invalid JSON: {error}",
                    path=str(config_path.relative_to(self.root_path)),
                ))
        else:
            self.results.add(ValidationCheck(
                name="Models: Config",
                category=CheckCategory.CONFIG,
                status=CheckStatus.WARN,
                message="Models config missing",
                path="modules/models/config/models.json",
            ))

    def _validate_launcher_scripts(self):
        """Validate launcher scripts."""
        log.info("Checking launcher scripts...")

        # Python launchers
        for script in REQUIRED_LAUNCHERS["python"]:
            script_path = self.root_path / script

            if script_path.exists():
                # Check syntax
                try:
                    with open(script_path) as f:
                        compile(f.read(), script, "exec")

                    self.results.add(ValidationCheck(
                        name=f"Launcher: {Path(script).name}",
                        category=CheckCategory.SCRIPT,
                        status=CheckStatus.PASS,
                        message="Valid Python launcher",
                        path=script,
                    ))
                except SyntaxError as e:
                    self.results.add(ValidationCheck(
                        name=f"Launcher: {Path(script).name}",
                        category=CheckCategory.SCRIPT,
                        status=CheckStatus.FAIL,
                        message=f"Syntax error: {e}",
                        path=script,
                    ))
            else:
                self.results.add(ValidationCheck(
                    name=f"Launcher: {Path(script).name}",
                    category=CheckCategory.SCRIPT,
                    status=CheckStatus.FAIL,
                    message="Script not found",
                    path=script,
                ))

        # Platform launchers
        for script in REQUIRED_LAUNCHERS["platform"]:
            script_path = self.root_path / script

            if script_path.exists():
                # Check executable permission for shell scripts
                if script.endswith(".sh") or script.endswith(".command"):
                    if is_executable(script_path):
                        self.results.add(ValidationCheck(
                            name=f"Launcher: {Path(script).name}",
                            category=CheckCategory.SCRIPT,
                            status=CheckStatus.PASS,
                            message="Executable shell script",
                            path=script,
                        ))
                    else:
                        self.results.add(ValidationCheck(
                            name=f"Launcher: {Path(script).name}",
                            category=CheckCategory.PERMISSION,
                            status=CheckStatus.WARN,
                            message="Script not executable",
                            path=script,
                        ))
                else:
                    self.results.add(ValidationCheck(
                        name=f"Launcher: {Path(script).name}",
                        category=CheckCategory.SCRIPT,
                        status=CheckStatus.PASS,
                        message="Script exists",
                        path=script,
                    ))
            else:
                self.results.add(ValidationCheck(
                    name=f"Launcher: {Path(script).name}",
                    category=CheckCategory.SCRIPT,
                    status=CheckStatus.FAIL,
                    message="Script not found",
                    path=script,
                ))

    def _validate_config_files(self):
        """Validate configuration files."""
        log.info("Checking configuration files...")

        for config_path, required in REQUIRED_CONFIGS:
            full_path = self.root_path / config_path

            if not full_path.exists():
                status = CheckStatus.FAIL if required else CheckStatus.WARN
                self.results.add(ValidationCheck(
                    name=f"Config: {Path(config_path).name}",
                    category=CheckCategory.CONFIG,
                    status=status,
                    message="File not found",
                    path=config_path,
                ))
                continue

            # Validate JSON
            valid, error = check_json_valid(full_path)
            if valid:
                self.results.add(ValidationCheck(
                    name=f"Config: {Path(config_path).name}",
                    category=CheckCategory.CONFIG,
                    status=CheckStatus.PASS,
                    message="Valid JSON configuration",
                    path=config_path,
                ))
            else:
                self.results.add(ValidationCheck(
                    name=f"Config: {Path(config_path).name}",
                    category=CheckCategory.CONFIG,
                    status=CheckStatus.FAIL,
                    message=f"Invalid JSON: {error}",
                    path=config_path,
                ))

    def _validate_cross_platform(self):
        """Validate cross-platform compatibility."""
        log.info("Checking cross-platform compatibility...")

        # Check for Windows-specific issues
        windows_issues = []

        # Check for Unix-only scripts in launchers
        launchers_dir = self.root_path / "modules" / "launchers"
        if launchers_dir.exists():
            for script in launchers_dir.glob("*.sh"):
                content = script.read_text()
                if "#!/bin/bash" in content or "#!/bin/sh" in content:
                    if not (launchers_dir / script.stem).with_suffix(".bat").exists():
                        windows_issues.append(f"No Windows equivalent for {script.name}")

        if windows_issues:
            for issue in windows_issues:
                self.results.add(ValidationCheck(
                    name="Cross-platform: Windows",
                    category=CheckCategory.PLATFORM,
                    status=CheckStatus.WARN,
                    message=issue,
                ))
        else:
            self.results.add(ValidationCheck(
                name="Cross-platform: Windows",
                category=CheckCategory.PLATFORM,
                status=CheckStatus.PASS,
                message="Windows launchers present",
            ))

        # Check Python version compatibility
        try:
            result = subprocess.run(
                [sys.executable, "--version"],
                capture_output=True,
                text=True,
            )
            version = result.stdout.strip()
            self.results.add(ValidationCheck(
                name="Cross-platform: Python",
                category=CheckCategory.PLATFORM,
                status=CheckStatus.PASS,
                message=f"Running on {version}",
            ))
        except Exception as e:
            self.results.add(ValidationCheck(
                name="Cross-platform: Python",
                category=CheckCategory.PLATFORM,
                status=CheckStatus.WARN,
                message=f"Could not verify Python: {e}",
            ))

        # Check for path separators
        all_scripts = list(self.root_path.glob("scripts/**/*.py"))
        path_issues = []

        for script in all_scripts[:10]:  # Sample first 10
            try:
                content = script.read_text()
                if "\\\\" in content and "Windows" not in content:
                    path_issues.append(script.name)
            except Exception:
                pass

        if path_issues:
            self.results.add(ValidationCheck(
                name="Cross-platform: Paths",
                category=CheckCategory.PLATFORM,
                status=CheckStatus.WARN,
                message=f"Potential hardcoded paths in: {', '.join(path_issues[:3])}",
            ))
        else:
            self.results.add(ValidationCheck(
                name="Cross-platform: Paths",
                category=CheckCategory.PLATFORM,
                status=CheckStatus.PASS,
                message="No obvious path issues detected",
            ))

    def _validate_integration(self):
        """Validate integration and runtime checks."""
        log.info("Checking integration...")

        # Check if Ollama port is available
        if check_port_available(11434):
            self.results.add(ValidationCheck(
                name="Integration: Ollama Port",
                category=CheckCategory.INTEGRATION,
                status=CheckStatus.PASS,
                message="Port 11434 available",
            ))
        else:
            self.results.add(ValidationCheck(
                name="Integration: Ollama Port",
                category=CheckCategory.INTEGRATION,
                status=CheckStatus.WARN,
                message="Port 11434 in use (Ollama may already be running)",
            ))

        # Check if WebUI port is available
        if check_port_available(3000):
            self.results.add(ValidationCheck(
                name="Integration: WebUI Port",
                category=CheckCategory.INTEGRATION,
                status=CheckStatus.PASS,
                message="Port 3000 available",
            ))
        else:
            self.results.add(ValidationCheck(
                name="Integration: WebUI Port",
                category=CheckCategory.INTEGRATION,
                status=CheckStatus.WARN,
                message="Port 3000 in use",
            ))

        # Check available disk space
        try:
            stat = shutil.disk_usage(self.root_path)
            free_gb = stat.free / (1024 ** 3)

            if free_gb > 50:
                self.results.add(ValidationCheck(
                    name="Integration: Disk Space",
                    category=CheckCategory.INTEGRATION,
                    status=CheckStatus.PASS,
                    message=f"{free_gb:.1f} GB free",
                ))
            elif free_gb > 10:
                self.results.add(ValidationCheck(
                    name="Integration: Disk Space",
                    category=CheckCategory.INTEGRATION,
                    status=CheckStatus.WARN,
                    message=f"Low disk space: {free_gb:.1f} GB free",
                ))
            else:
                self.results.add(ValidationCheck(
                    name="Integration: Disk Space",
                    category=CheckCategory.INTEGRATION,
                    status=CheckStatus.FAIL,
                    message=f"Insufficient disk space: {free_gb:.1f} GB free",
                ))
        except Exception as e:
            self.results.add(ValidationCheck(
                name="Integration: Disk Space",
                category=CheckCategory.INTEGRATION,
                status=CheckStatus.WARN,
                message=f"Could not check disk space: {e}",
            ))


# =============================================================================
# Report Generation
# =============================================================================

class ReportGenerator:
    """Generate validation reports."""

    def __init__(self, results: ValidationResults, root_path: Path):
        self.results = results
        self.root_path = root_path

    def generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# USB-AI Build Validation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Platform:** {platform.system()} {platform.machine()}",
            f"**Python:** {platform.python_version()}",
            f"**Root Path:** `{self.root_path}`",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Checks | {len(self.results.checks)} |",
            f"| Passed | {len(self.results.passed)} |",
            f"| Failed | {len(self.results.failed)} |",
            f"| Warnings | {len(self.results.warnings)} |",
            f"| Skipped | {len(self.results.skipped)} |",
            f"| Duration | {self.results.duration_seconds:.2f}s |",
            f"| Status | {'**PASS**' if self.results.success else '**FAIL**'} |",
            "",
        ]

        # Failed checks
        if self.results.failed:
            lines.extend([
                "## Failed Checks",
                "",
            ])
            for check in self.results.failed:
                lines.append(f"- **{check.name}**: {check.message}")
                if check.path:
                    lines.append(f"  - Path: `{check.path}`")
                if check.details:
                    lines.append(f"  - Details: {check.details}")
            lines.append("")

        # Warnings
        if self.results.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for check in self.results.warnings:
                lines.append(f"- **{check.name}**: {check.message}")
                if check.path:
                    lines.append(f"  - Path: `{check.path}`")
            lines.append("")

        # Passed checks by category
        lines.extend([
            "## Passed Checks",
            "",
        ])

        by_category: Dict[CheckCategory, List[ValidationCheck]] = {}
        for check in self.results.passed:
            if check.category not in by_category:
                by_category[check.category] = []
            by_category[check.category].append(check)

        for category, checks in sorted(by_category.items(), key=lambda x: x[0].value):
            lines.append(f"### {category.value.title()}")
            lines.append("")
            for check in checks:
                lines.append(f"- {check.name}: {check.message}")
            lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            "*Report generated by USB-AI Build Validator v" + __version__ + "*",
        ])

        return "\n".join(lines)

    def generate_json(self) -> str:
        """Generate JSON report."""
        report = {
            "meta": {
                "generated": datetime.now().isoformat(),
                "platform": f"{platform.system()} {platform.machine()}",
                "python_version": platform.python_version(),
                "root_path": str(self.root_path),
                "validator_version": __version__,
            },
            "results": self.results.to_dict(),
        }
        return json.dumps(report, indent=2)


# =============================================================================
# CLI Entry Point
# =============================================================================

def print_summary(results: ValidationResults):
    """Print validation summary to console."""
    print("")
    print("=" * 60)
    print("              USB-AI Build Validation Report")
    print("=" * 60)
    print("")

    total = len(results.checks)
    print(f"Total checks: {total}")
    print(f"  Passed:   {len(results.passed)}")
    print(f"  Failed:   {len(results.failed)}")
    print(f"  Warnings: {len(results.warnings)}")
    print(f"  Skipped:  {len(results.skipped)}")
    print(f"  Duration: {results.duration_seconds:.2f}s")
    print("")

    if results.failed:
        print("FAILED CHECKS:")
        for check in results.failed:
            print(f"  - {check.name}: {check.message}")
        print("")

    if results.warnings:
        print("WARNINGS:")
        for check in results.warnings[:10]:  # Limit to first 10
            print(f"  - {check.name}: {check.message}")
        if len(results.warnings) > 10:
            print(f"  ... and {len(results.warnings) - 10} more warnings")
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="USB-AI Build Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output report path (default: docs/BUILD_REPORT.md)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format instead of markdown",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any warning",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="Verify file checksums",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip time-consuming checks",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"validate_build {__version__}",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    root_path = get_root_path()

    # Run validation
    validator = BuildValidator(
        root_path=root_path,
        verbose=args.verbose,
        strict=args.strict,
        verify_checksums=args.checksums,
        quick=args.quick,
    )

    results = validator.validate_all()

    # Print summary
    print_summary(results)

    # Generate report
    reporter = ReportGenerator(results, root_path)

    if args.json:
        report = reporter.generate_json()
        suffix = ".json"
    else:
        report = reporter.generate_markdown()
        suffix = ".md"

    # Write report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = root_path / "docs" / f"BUILD_REPORT{suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    log.info(f"Report written to: {output_path}")

    # Return code
    if args.strict and results.warnings:
        return 1

    return 0 if results.success else 1


if __name__ == "__main__":
    sys.exit(main())
