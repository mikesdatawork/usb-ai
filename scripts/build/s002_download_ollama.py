#!/usr/bin/env python3
"""
s002_download_ollama.py

Downloads Ollama binaries for all supported platforms.
Binaries are saved to modules/ollama-portable/bin/
"""

import json
import logging
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

__version__ = "1.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Get latest version from GitHub API or use fallback
OLLAMA_VERSION = "v0.15.4"

OLLAMA_DOWNLOADS = {
    "darwin-arm64": {
        "url": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-darwin.tgz",
        "filename": "ollama",
        "archive": "tgz",
        "executable": True,
    },
    "darwin-amd64": {
        "url": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-darwin.tgz",
        "filename": "ollama",
        "archive": "tgz",
        "executable": True,
    },
    "linux-amd64": {
        "url": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst",
        "filename": "ollama",
        "archive": "tar.zst",
        "executable": True,
    },
    "windows-amd64": {
        "url": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-windows-amd64.zip",
        "filename": "ollama.exe",
        "archive": "zip",
        "executable": False,
    },
}

CHUNK_SIZE = 8192


class DownloadProgress:
    """Track and display download progress."""

    def __init__(self, total_size: int, filename: str):
        self.total_size = total_size
        self.filename = filename
        self.downloaded = 0
        self.last_percent = -1

    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        if self.total_size > 0:
            percent = int((self.downloaded / self.total_size) * 100)
            if percent != self.last_percent and percent % 10 == 0:
                self.last_percent = percent
                mb_downloaded = self.downloaded / (1024 * 1024)
                mb_total = self.total_size / (1024 * 1024)
                log.info(f"  {self.filename}: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def download_file(url: str, dest_path: Path, filename: str) -> bool:
    """Download a file with progress reporting."""
    log.info(f"Downloading: {url}")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "USB-AI-Builder/1.0"}
        )

        with urllib.request.urlopen(request, timeout=300) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            progress = DownloadProgress(total_size, filename)

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))

        log.info(f"  Saved: {dest_path}")
        return True

    except Exception as e:
        log.error(f"  Download failed: {e}")
        return False


def make_executable(path: Path) -> bool:
    """Make a file executable on Unix systems."""
    if platform.system() == "Windows":
        return True

    try:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        log.info(f"  Made executable: {path.name}")
        return True
    except Exception as e:
        log.error(f"  Failed to set executable: {e}")
        return False


def verify_binary(path: Path) -> bool:
    """Verify downloaded binary is valid."""
    if not path.exists():
        return False

    size = path.stat().st_size
    if size < 1024 * 1024:  # Less than 1MB is suspicious
        log.warning(f"  Binary suspiciously small: {size} bytes")
        return False

    log.info(f"  Verified: {path.name} ({size / (1024*1024):.1f} MB)")
    return True


def extract_archive(archive_path: Path, dest_dir: Path, archive_type: str) -> bool:
    """Extract archive to destination directory."""
    log.info(f"  Extracting {archive_type} archive...")

    try:
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
        elif archive_type == "tgz" or archive_type == "tar.gz":
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(dest_dir)
        elif archive_type == "tar":
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(dest_dir)
        elif archive_type == "tar.zst":
            # Use system tar with zstd support
            result = subprocess.run(
                ["tar", "--zstd", "-xf", str(archive_path), "-C", str(dest_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Try alternative: decompress first, then extract
                log.info("  Trying alternative extraction method...")
                zst_decompress = subprocess.run(
                    ["zstd", "-d", str(archive_path), "-o", str(archive_path.with_suffix(''))],
                    capture_output=True,
                    text=True
                )
                if zst_decompress.returncode == 0:
                    tar_path = archive_path.with_suffix('')
                    with tarfile.open(tar_path, 'r') as tf:
                        tf.extractall(dest_dir)
                    tar_path.unlink(missing_ok=True)
                else:
                    log.error(f"  zstd extraction failed: {result.stderr}")
                    return False
        else:
            log.error(f"  Unknown archive type: {archive_type}")
            return False

        log.info(f"  Extracted to: {dest_dir}")
        return True
    except Exception as e:
        log.error(f"  Extraction failed: {e}")
        return False


def download_platform(plat: str, info: dict, bin_dir: Path) -> bool:
    """Download Ollama for a specific platform."""
    log.info(f"Platform: {plat}")

    dest_dir = bin_dir / plat
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / info["filename"]

    if dest_path.exists() and verify_binary(dest_path):
        log.info(f"  Already exists, skipping")
        return True

    archive_type = info.get("archive")

    if archive_type:
        # Download to temp file, then extract
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{archive_type}") as tmp:
            tmp_path = Path(tmp.name)

        if not download_file(info["url"], tmp_path, f"ollama-{plat}.{archive_type}"):
            tmp_path.unlink(missing_ok=True)
            return False

        if not extract_archive(tmp_path, dest_dir, archive_type):
            tmp_path.unlink(missing_ok=True)
            return False

        tmp_path.unlink(missing_ok=True)

        # Find the binary in extracted files
        if not dest_path.exists():
            # Look for ollama binary in extracted contents
            for f in dest_dir.rglob("ollama*"):
                if f.is_file() and f.name == info["filename"]:
                    if f != dest_path:
                        shutil.move(str(f), str(dest_path))
                    break
    else:
        # Direct binary download
        if not download_file(info["url"], dest_path, info["filename"]):
            return False

    if info.get("executable", False):
        make_executable(dest_path)

    return verify_binary(dest_path)


def create_ollama_config(config_dir: Path) -> bool:
    """Create Ollama configuration file."""
    import json

    config = {
        "host": "127.0.0.1",
        "port": 11434,
        "models_path": "../models",
        "gpu_enabled": False,
        "num_parallel": 1,
        "max_loaded_models": 1
    }

    config_path = config_dir / "ollama.json"

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Created config: {config_path}")
        return True
    except Exception as e:
        log.error(f"Failed to create config: {e}")
        return False


def print_summary(results: dict):
    """Print download summary."""
    print("")
    print("=" * 50)
    print("        Ollama Download Complete")
    print("=" * 50)
    print("")

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for plat, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {plat}: {status}")

    print("")
    print(f"Downloaded: {success_count}/{total_count} platforms")
    print("")
    print("Next step:")
    print("  python scripts/build/s003_download_models.py")
    print("")
    print("=" * 50)


def main() -> int:
    """Entry point."""
    log.info("USB-AI Ollama Download")
    log.info(f"Platform: {platform.system()} {platform.machine()}")
    print("")

    root = get_root_path()
    bin_dir = root / "modules" / "ollama-portable" / "bin"
    config_dir = root / "modules" / "ollama-portable" / "config"

    log.info(f"Root: {root}")
    log.info(f"Bin dir: {bin_dir}")
    print("")

    results = {}

    for plat, info in OLLAMA_DOWNLOADS.items():
        results[plat] = download_platform(plat, info, bin_dir)
        print("")

    create_ollama_config(config_dir)

    print_summary(results)

    all_success = all(results.values())
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
