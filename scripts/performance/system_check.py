#!/usr/bin/env python3
"""
system_check.py

System hardware analysis and compatibility check for USB-AI.
Detects hardware capabilities, validates USB speed, and recommends
optimal settings for portable AI operation.
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.END}"
    return text


@dataclass
class CPUInfo:
    """CPU hardware information."""
    model: str
    vendor: str
    physical_cores: int
    logical_cores: int
    frequency_mhz: float
    architecture: str
    features: List[str] = field(default_factory=list)
    cache_l3_mb: float = 0.0


@dataclass
class RAMInfo:
    """Memory information."""
    total_gb: float
    available_gb: float
    used_gb: float
    swap_total_gb: float
    swap_used_gb: float


@dataclass
class GPUInfo:
    """GPU hardware information."""
    available: bool
    name: str
    vendor: str
    vram_gb: float
    driver_version: str
    compute_capability: str


@dataclass
class StorageInfo:
    """Storage/USB information."""
    mount_point: str
    total_gb: float
    available_gb: float
    filesystem: str
    is_removable: bool
    usb_version: str
    read_speed_mbps: float
    write_speed_mbps: float


@dataclass
class SystemReport:
    """Complete system analysis report."""
    platform: str
    os_version: str
    cpu: CPUInfo
    ram: RAMInfo
    gpu: GPUInfo
    storage: Optional[StorageInfo]
    recommendations: List[str]
    warnings: List[str]
    optimal_profile: str
    max_model_size: str
    overall_score: int  # 0-100


class CPUDetector:
    """Detects CPU information across platforms."""

    def __init__(self):
        self.system = platform.system().lower()

    def detect(self) -> CPUInfo:
        """Detect CPU information."""
        info = CPUInfo(
            model="Unknown",
            vendor="Unknown",
            physical_cores=os.cpu_count() or 2,
            logical_cores=os.cpu_count() or 2,
            frequency_mhz=0.0,
            architecture=platform.machine(),
            features=[],
            cache_l3_mb=0.0
        )

        try:
            if self.system == "linux":
                self._detect_linux(info)
            elif self.system == "darwin":
                self._detect_darwin(info)
            elif self.system == "windows":
                self._detect_windows(info)
        except Exception as e:
            log.warning(f"CPU detection error: {e}")

        return info

    def _detect_linux(self, info: CPUInfo):
        """Detect CPU on Linux."""
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()

            # Model name
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if model_match:
                info.model = model_match.group(1).strip()

            # Vendor
            vendor_match = re.search(r"vendor_id\s*:\s*(.+)", cpuinfo)
            if vendor_match:
                info.vendor = vendor_match.group(1).strip()

            # Physical cores
            physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
            cores_per_socket = re.search(r"cpu cores\s*:\s*(\d+)", cpuinfo)
            if cores_per_socket and physical_ids:
                info.physical_cores = int(cores_per_socket.group(1)) * max(1, len(physical_ids))

            # Frequency
            freq_match = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            if freq_match:
                info.frequency_mhz = float(freq_match.group(1))

            # CPU features (important for SIMD support)
            flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo)
            if flags_match:
                flags = flags_match.group(1).split()
                important_flags = ["avx", "avx2", "avx512f", "sse4_2", "fma"]
                info.features = [f for f in important_flags if f in flags]

            # L3 Cache
            cache_match = re.search(r"cache size\s*:\s*(\d+)\s*KB", cpuinfo)
            if cache_match:
                info.cache_l3_mb = int(cache_match.group(1)) / 1024

        except Exception as e:
            log.warning(f"Linux CPU detection error: {e}")

    def _detect_darwin(self, info: CPUInfo):
        """Detect CPU on macOS."""
        try:
            # Model
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.model = result.stdout.strip()

            # Vendor
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.vendor"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.vendor = result.stdout.strip()

            # Physical cores
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.physical_cores = int(result.stdout.strip())

            # Logical cores
            result = subprocess.run(
                ["sysctl", "-n", "hw.logicalcpu"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.logical_cores = int(result.stdout.strip())

            # Frequency
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.frequency_mhz = int(result.stdout.strip()) / 1_000_000

            # Features
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                features = result.stdout.lower().split()
                important = ["avx", "avx2", "avx512", "sse4"]
                info.features = [f for f in important if any(f in feat for feat in features)]

            # Check for Apple Silicon
            if "apple" in info.model.lower() or info.architecture in ("arm64", "aarch64"):
                info.vendor = "Apple"
                info.features.append("apple_silicon")

        except Exception as e:
            log.warning(f"macOS CPU detection error: {e}")

    def _detect_windows(self, info: CPUInfo):
        """Detect CPU on Windows."""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    info.model = lines[1].strip()

            result = subprocess.run(
                ["wmic", "cpu", "get", "manufacturer"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    info.vendor = lines[1].strip()

            result = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    info.physical_cores = int(lines[1].strip())

            result = subprocess.run(
                ["wmic", "cpu", "get", "MaxClockSpeed"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    info.frequency_mhz = float(lines[1].strip())

        except Exception as e:
            log.warning(f"Windows CPU detection error: {e}")


class RAMDetector:
    """Detects memory information."""

    def __init__(self):
        self.system = platform.system().lower()

    def detect(self) -> RAMInfo:
        """Detect RAM information."""
        info = RAMInfo(
            total_gb=8.0,
            available_gb=4.0,
            used_gb=4.0,
            swap_total_gb=0.0,
            swap_used_gb=0.0
        )

        try:
            if self.system == "linux":
                self._detect_linux(info)
            elif self.system == "darwin":
                self._detect_darwin(info)
            elif self.system == "windows":
                self._detect_windows(info)
        except Exception as e:
            log.warning(f"RAM detection error: {e}")

        return info

    def _detect_linux(self, info: RAMInfo):
        """Detect RAM on Linux."""
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()

            total_match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
            if total_match:
                info.total_gb = int(total_match.group(1)) / 1024 / 1024

            available_match = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
            if available_match:
                info.available_gb = int(available_match.group(1)) / 1024 / 1024

            info.used_gb = info.total_gb - info.available_gb

            swap_total = re.search(r"SwapTotal:\s+(\d+)\s+kB", meminfo)
            if swap_total:
                info.swap_total_gb = int(swap_total.group(1)) / 1024 / 1024

            swap_free = re.search(r"SwapFree:\s+(\d+)\s+kB", meminfo)
            if swap_free:
                info.swap_used_gb = info.swap_total_gb - int(swap_free.group(1)) / 1024 / 1024

        except Exception as e:
            log.warning(f"Linux RAM detection error: {e}")

    def _detect_darwin(self, info: RAMInfo):
        """Detect RAM on macOS."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.total_gb = int(result.stdout.strip()) / 1024 / 1024 / 1024

            # Approximate available from vm_stat
            result = subprocess.run(["vm_stat"], capture_output=True, text=True)
            if result.returncode == 0:
                free_match = re.search(r"Pages free:\s+(\d+)", result.stdout)
                inactive_match = re.search(r"Pages inactive:\s+(\d+)", result.stdout)
                speculative_match = re.search(r"Pages speculative:\s+(\d+)", result.stdout)

                pages = 0
                if free_match:
                    pages += int(free_match.group(1))
                if inactive_match:
                    pages += int(inactive_match.group(1))
                if speculative_match:
                    pages += int(speculative_match.group(1))

                info.available_gb = pages * 4096 / 1024 / 1024 / 1024

            info.used_gb = info.total_gb - info.available_gb

        except Exception as e:
            log.warning(f"macOS RAM detection error: {e}")

    def _detect_windows(self, info: RAMInfo):
        """Detect RAM on Windows."""
        try:
            result = subprocess.run(
                ["wmic", "OS", "get", "TotalVisibleMemorySize"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    info.total_gb = int(lines[1].strip()) / 1024 / 1024

            result = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    info.available_gb = int(lines[1].strip()) / 1024 / 1024

            info.used_gb = info.total_gb - info.available_gb

        except Exception as e:
            log.warning(f"Windows RAM detection error: {e}")


class GPUDetector:
    """Detects GPU information."""

    def __init__(self):
        self.system = platform.system().lower()

    def detect(self) -> GPUInfo:
        """Detect GPU information."""
        info = GPUInfo(
            available=False,
            name="None",
            vendor="None",
            vram_gb=0.0,
            driver_version="N/A",
            compute_capability="N/A"
        )

        # Try NVIDIA first
        if self._detect_nvidia(info):
            return info

        # Try AMD ROCm
        if self._detect_amd(info):
            return info

        # Check for Apple Silicon
        if self._detect_apple_silicon(info):
            return info

        return info

    def _detect_nvidia(self, info: GPUInfo) -> bool:
        """Detect NVIDIA GPU."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    parts = output.split(",")
                    info.available = True
                    info.name = parts[0].strip()
                    info.vendor = "NVIDIA"
                    if len(parts) > 1:
                        info.vram_gb = float(parts[1].strip()) / 1024
                    if len(parts) > 2:
                        info.driver_version = parts[2].strip()
                    if len(parts) > 3:
                        info.compute_capability = parts[3].strip()
                    return True
        except FileNotFoundError:
            pass
        return False

    def _detect_amd(self, info: GPUInfo) -> bool:
        """Detect AMD GPU with ROCm."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                info.available = True
                info.vendor = "AMD"
                info.name = "AMD GPU (ROCm)"

                # Try to get memory info
                mem_result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True
                )
                if mem_result.returncode == 0:
                    # Parse VRAM info
                    match = re.search(r"(\d+)\s*MB", mem_result.stdout)
                    if match:
                        info.vram_gb = int(match.group(1)) / 1024

                return True
        except FileNotFoundError:
            pass
        return False

    def _detect_apple_silicon(self, info: GPUInfo) -> bool:
        """Detect Apple Silicon GPU."""
        if self.system == "darwin":
            machine = platform.machine().lower()
            if machine in ("arm64", "aarch64"):
                info.available = True
                info.vendor = "Apple"
                info.name = "Apple Silicon (Metal)"
                info.compute_capability = "Metal 3"

                # Get unified memory (shared with GPU)
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        total_mem = int(result.stdout.strip()) / 1024 / 1024 / 1024
                        # Apple Silicon shares memory, GPU can use ~75%
                        info.vram_gb = total_mem * 0.75
                except:
                    info.vram_gb = 8.0

                return True
        return False


class StorageDetector:
    """Detects storage and USB information."""

    def __init__(self):
        self.system = platform.system().lower()

    def detect(self, path: Optional[Path] = None) -> Optional[StorageInfo]:
        """Detect storage information for the given path or current drive."""
        if path is None:
            path = Path(__file__).resolve()

        info = StorageInfo(
            mount_point=str(path.anchor or "/"),
            total_gb=0.0,
            available_gb=0.0,
            filesystem="unknown",
            is_removable=False,
            usb_version="unknown",
            read_speed_mbps=0.0,
            write_speed_mbps=0.0
        )

        try:
            if self.system == "linux":
                self._detect_linux(info, path)
            elif self.system == "darwin":
                self._detect_darwin(info, path)
            elif self.system == "windows":
                self._detect_windows(info, path)

            # Run speed test
            self._test_speed(info, path)

        except Exception as e:
            log.warning(f"Storage detection error: {e}")

        return info

    def _detect_linux(self, info: StorageInfo, path: Path):
        """Detect storage on Linux."""
        try:
            # Get mount point and filesystem
            result = subprocess.run(
                ["df", "-T", str(path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 7:
                        info.filesystem = parts[1]
                        info.total_gb = int(parts[2]) / 1024 / 1024
                        info.available_gb = int(parts[4]) / 1024 / 1024
                        info.mount_point = parts[6]

            # Check if removable
            device = self._get_device_for_path(path)
            if device:
                removable_path = Path(f"/sys/block/{device}/removable")
                if removable_path.exists():
                    with open(removable_path) as f:
                        info.is_removable = f.read().strip() == "1"

                # Get USB version
                usb_version = self._get_usb_version_linux(device)
                if usb_version:
                    info.usb_version = usb_version

        except Exception as e:
            log.warning(f"Linux storage detection error: {e}")

    def _get_device_for_path(self, path: Path) -> Optional[str]:
        """Get block device name for a path on Linux."""
        try:
            result = subprocess.run(
                ["df", str(path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    device_path = lines[1].split()[0]
                    # Extract device name (e.g., /dev/sda1 -> sda)
                    match = re.search(r"/dev/(\w+)", device_path)
                    if match:
                        device = match.group(1)
                        # Remove partition number
                        device = re.sub(r"\d+$", "", device)
                        return device
        except:
            pass
        return None

    def _get_usb_version_linux(self, device: str) -> Optional[str]:
        """Get USB version for a device on Linux."""
        try:
            # Check if device is a USB device
            usb_path = Path(f"/sys/block/{device}/device")
            if usb_path.exists():
                # Look for USB speed in the chain
                current = usb_path.resolve()
                while current != Path("/"):
                    speed_path = current / "speed"
                    if speed_path.exists():
                        with open(speed_path) as f:
                            speed = f.read().strip()
                            if speed == "5000":
                                return "USB 3.0"
                            elif speed == "10000":
                                return "USB 3.1"
                            elif speed == "20000":
                                return "USB 3.2"
                            elif speed == "480":
                                return "USB 2.0"
                            elif speed == "12":
                                return "USB 1.1"
                    current = current.parent
        except:
            pass
        return None

    def _detect_darwin(self, info: StorageInfo, path: Path):
        """Detect storage on macOS."""
        try:
            result = subprocess.run(
                ["df", "-k", str(path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 6:
                        info.total_gb = int(parts[1]) / 1024 / 1024
                        info.available_gb = int(parts[3]) / 1024 / 1024
                        info.mount_point = " ".join(parts[5:])

            # Get filesystem type
            result = subprocess.run(
                ["diskutil", "info", str(path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                fs_match = re.search(r"Type \(Bundle\):\s+(\w+)", result.stdout)
                if fs_match:
                    info.filesystem = fs_match.group(1)

                removable_match = re.search(r"Removable Media:\s+(\w+)", result.stdout)
                if removable_match:
                    info.is_removable = removable_match.group(1).lower() == "yes"

        except Exception as e:
            log.warning(f"macOS storage detection error: {e}")

    def _detect_windows(self, info: StorageInfo, path: Path):
        """Detect storage on Windows."""
        try:
            drive = str(path.anchor).rstrip("\\")

            result = subprocess.run(
                ["wmic", "logicaldisk", "where", f"DeviceID='{drive}'",
                 "get", "Size,FreeSpace,FileSystem,DriveType"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        drive_type = parts[0]
                        info.filesystem = parts[1]
                        info.available_gb = int(parts[2]) / 1024 / 1024 / 1024
                        info.total_gb = int(parts[3]) / 1024 / 1024 / 1024
                        info.is_removable = drive_type == "2"  # Removable disk
                        info.mount_point = drive

        except Exception as e:
            log.warning(f"Windows storage detection error: {e}")

    def _test_speed(self, info: StorageInfo, path: Path):
        """Test read/write speed of storage."""
        test_file = path.parent / ".usb_ai_speed_test"
        test_size = 10 * 1024 * 1024  # 10 MB
        test_data = os.urandom(test_size)

        try:
            # Write test
            start = time.time()
            with open(test_file, "wb") as f:
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())
            write_time = time.time() - start
            info.write_speed_mbps = (test_size / 1024 / 1024) / write_time

            # Read test
            start = time.time()
            with open(test_file, "rb") as f:
                _ = f.read()
            read_time = time.time() - start
            info.read_speed_mbps = (test_size / 1024 / 1024) / read_time

        except Exception as e:
            log.warning(f"Speed test error: {e}")
        finally:
            try:
                test_file.unlink()
            except:
                pass


class SystemAnalyzer:
    """Analyzes system and generates recommendations."""

    def __init__(self):
        self.cpu_detector = CPUDetector()
        self.ram_detector = RAMDetector()
        self.gpu_detector = GPUDetector()
        self.storage_detector = StorageDetector()

    def analyze(self, storage_path: Optional[Path] = None) -> SystemReport:
        """Perform full system analysis."""
        cpu = self.cpu_detector.detect()
        ram = self.ram_detector.detect()
        gpu = self.gpu_detector.detect()
        storage = self.storage_detector.detect(storage_path)

        recommendations = []
        warnings = []

        # Analyze and generate recommendations
        profile, score = self._determine_profile(cpu, ram, gpu, storage)
        max_model = self._determine_max_model(ram, gpu)

        # CPU recommendations
        if cpu.physical_cores < 4:
            warnings.append("Low core count may limit performance")
        if "avx2" not in cpu.features and "apple_silicon" not in cpu.features:
            warnings.append("No AVX2 support - inference will be slower")

        # RAM recommendations
        if ram.available_gb < 4:
            warnings.append("Low available RAM - close other applications")
            recommendations.append("Use 'minimal' performance profile")
        elif ram.available_gb < 8:
            recommendations.append("Consider 'balanced' profile for best stability")
        elif ram.available_gb >= 16:
            recommendations.append("Sufficient RAM for 'performance' profile")

        # GPU recommendations
        if gpu.available:
            if gpu.vram_gb >= 8:
                recommendations.append(f"GPU acceleration available ({gpu.name})")
            else:
                recommendations.append("Limited VRAM - partial GPU offload recommended")
        else:
            recommendations.append("No GPU detected - CPU-only inference")

        # Storage recommendations
        if storage:
            if storage.is_removable:
                if storage.usb_version == "USB 2.0":
                    warnings.append("USB 2.0 detected - slow model loading expected")
                    recommendations.append("Upgrade to USB 3.0+ for better performance")
                elif "3" in storage.usb_version:
                    recommendations.append(f"{storage.usb_version} detected - good performance")

            if storage.read_speed_mbps < 50:
                warnings.append(f"Slow storage: {storage.read_speed_mbps:.1f} MB/s read")
            elif storage.read_speed_mbps < 100:
                recommendations.append("Moderate storage speed - model preloading recommended")

            if storage.available_gb < 20:
                warnings.append(f"Low storage space: {storage.available_gb:.1f} GB available")

        return SystemReport(
            platform=platform.system(),
            os_version=platform.version(),
            cpu=cpu,
            ram=ram,
            gpu=gpu,
            storage=storage,
            recommendations=recommendations,
            warnings=warnings,
            optimal_profile=profile,
            max_model_size=max_model,
            overall_score=score
        )

    def _determine_profile(
        self,
        cpu: CPUInfo,
        ram: RAMInfo,
        gpu: GPUInfo,
        storage: Optional[StorageInfo]
    ) -> Tuple[str, int]:
        """Determine optimal profile and score."""
        score = 50  # Base score

        # CPU scoring
        if cpu.physical_cores >= 8:
            score += 15
        elif cpu.physical_cores >= 4:
            score += 10
        elif cpu.physical_cores >= 2:
            score += 5

        if "avx2" in cpu.features or "apple_silicon" in cpu.features:
            score += 10

        # RAM scoring
        if ram.available_gb >= 32:
            score += 20
        elif ram.available_gb >= 16:
            score += 15
        elif ram.available_gb >= 8:
            score += 10
        elif ram.available_gb >= 4:
            score += 5
        else:
            score -= 10

        # GPU scoring
        if gpu.available:
            if gpu.vram_gb >= 12:
                score += 15
            elif gpu.vram_gb >= 8:
                score += 10
            elif gpu.vram_gb >= 4:
                score += 5

        # Storage scoring
        if storage:
            if storage.read_speed_mbps >= 200:
                score += 10
            elif storage.read_speed_mbps >= 100:
                score += 5
            elif storage.read_speed_mbps < 50:
                score -= 5

        # Determine profile
        score = max(0, min(100, score))

        if score >= 80:
            profile = "max"
        elif score >= 60:
            profile = "performance"
        elif score >= 40:
            profile = "balanced"
        else:
            profile = "minimal"

        return profile, score

    def _determine_max_model(self, ram: RAMInfo, gpu: GPUInfo) -> str:
        """Determine maximum recommended model size."""
        # Use available RAM as primary constraint
        available = ram.available_gb

        # If GPU available, consider VRAM too
        if gpu.available and gpu.vram_gb > 0:
            # With GPU, we can run slightly larger models
            available = max(available, gpu.vram_gb * 1.2)

        if available >= 64:
            return "70B"
        elif available >= 32:
            return "33B"
        elif available >= 16:
            return "14B"
        elif available >= 8:
            return "8B"
        elif available >= 4:
            return "3B"
        else:
            return "1B"


def print_report(report: SystemReport):
    """Print formatted system report."""
    print("")
    print("=" * 70)
    print(colorize("                    USB-AI System Check", Colors.BOLD))
    print("=" * 70)
    print("")

    # Platform info
    print(f"  Platform:        {report.platform}")
    print(f"  OS Version:      {report.os_version}")
    print("")

    # CPU info
    print(colorize("  CPU", Colors.BLUE))
    print(f"    Model:         {report.cpu.model}")
    print(f"    Cores:         {report.cpu.physical_cores} physical, {report.cpu.logical_cores} logical")
    if report.cpu.frequency_mhz > 0:
        print(f"    Frequency:     {report.cpu.frequency_mhz:.0f} MHz")
    if report.cpu.features:
        print(f"    Features:      {', '.join(report.cpu.features)}")
    print("")

    # RAM info
    print(colorize("  Memory", Colors.BLUE))
    print(f"    Total:         {report.ram.total_gb:.1f} GB")
    print(f"    Available:     {report.ram.available_gb:.1f} GB")
    print(f"    Used:          {report.ram.used_gb:.1f} GB")
    if report.ram.swap_total_gb > 0:
        print(f"    Swap:          {report.ram.swap_used_gb:.1f} / {report.ram.swap_total_gb:.1f} GB")
    print("")

    # GPU info
    print(colorize("  GPU", Colors.BLUE))
    if report.gpu.available:
        print(f"    Name:          {report.gpu.name}")
        print(f"    Vendor:        {report.gpu.vendor}")
        print(f"    VRAM:          {report.gpu.vram_gb:.1f} GB")
        if report.gpu.driver_version != "N/A":
            print(f"    Driver:        {report.gpu.driver_version}")
        if report.gpu.compute_capability != "N/A":
            print(f"    Compute:       {report.gpu.compute_capability}")
    else:
        print("    Status:        Not detected")
    print("")

    # Storage info
    if report.storage:
        print(colorize("  Storage", Colors.BLUE))
        print(f"    Mount:         {report.storage.mount_point}")
        print(f"    Total:         {report.storage.total_gb:.1f} GB")
        print(f"    Available:     {report.storage.available_gb:.1f} GB")
        print(f"    Filesystem:    {report.storage.filesystem}")
        print(f"    Removable:     {'Yes' if report.storage.is_removable else 'No'}")
        if report.storage.usb_version != "unknown":
            print(f"    USB Version:   {report.storage.usb_version}")
        if report.storage.read_speed_mbps > 0:
            print(f"    Read Speed:    {report.storage.read_speed_mbps:.1f} MB/s")
        if report.storage.write_speed_mbps > 0:
            print(f"    Write Speed:   {report.storage.write_speed_mbps:.1f} MB/s")
        print("")

    # Recommendations
    print("=" * 70)
    print(colorize("                      Recommendations", Colors.BOLD))
    print("=" * 70)
    print("")

    # Score and profile
    score_color = Colors.GREEN if report.overall_score >= 60 else Colors.YELLOW if report.overall_score >= 40 else Colors.RED
    print(f"  Overall Score:   {colorize(f'{report.overall_score}/100', score_color)}")
    print(f"  Optimal Profile: {colorize(report.optimal_profile, Colors.GREEN)}")
    print(f"  Max Model Size:  {colorize(report.max_model_size, Colors.GREEN)}")
    print("")

    # Warnings
    if report.warnings:
        print(colorize("  Warnings:", Colors.YELLOW))
        for warning in report.warnings:
            print(f"    - {warning}")
        print("")

    # Recommendations
    if report.recommendations:
        print(colorize("  Suggestions:", Colors.BLUE))
        for rec in report.recommendations:
            print(f"    - {rec}")
        print("")

    print("=" * 70)


def main() -> int:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="USB-AI System Hardware Check"
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to check storage for (default: script location)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output profile recommendation"
    )

    args = parser.parse_args()

    analyzer = SystemAnalyzer()
    report = analyzer.analyze(args.path)

    if args.json:
        # Convert to JSON-serializable format
        output = {
            "platform": report.platform,
            "os_version": report.os_version,
            "cpu": asdict(report.cpu),
            "ram": asdict(report.ram),
            "gpu": asdict(report.gpu),
            "storage": asdict(report.storage) if report.storage else None,
            "recommendations": report.recommendations,
            "warnings": report.warnings,
            "optimal_profile": report.optimal_profile,
            "max_model_size": report.max_model_size,
            "overall_score": report.overall_score,
        }
        print(json.dumps(output, indent=2))
    elif args.quiet:
        print(report.optimal_profile)
    else:
        print_report(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
