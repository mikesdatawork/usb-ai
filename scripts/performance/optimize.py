#!/usr/bin/env python3
"""
optimize.py

Performance optimization for USB-AI portable operation.
Configures Ollama for optimal USB performance with:
- CPU thread optimization based on hardware detection
- Memory limit configuration
- GPU offloading when available
- Low-latency startup optimization
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""
    cpu_cores: int
    cpu_threads: int
    cpu_model: str
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_vram_gb: float
    platform: str
    architecture: str


@dataclass
class OptimizationConfig:
    """Optimized configuration for Ollama."""
    num_threads: int
    num_gpu: int
    gpu_layers: int
    main_gpu: int
    memory_limit_gb: float
    batch_size: int
    context_length: int
    mlock: bool
    mmap: bool
    flash_attention: bool
    keep_alive: str
    numa: bool


class HardwareDetector:
    """Detects system hardware capabilities."""

    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()

    def detect(self) -> HardwareInfo:
        """Detect all hardware capabilities."""
        cpu_info = self._detect_cpu()
        ram_info = self._detect_ram()
        gpu_info = self._detect_gpu()

        return HardwareInfo(
            cpu_cores=cpu_info["cores"],
            cpu_threads=cpu_info["threads"],
            cpu_model=cpu_info["model"],
            total_ram_gb=ram_info["total_gb"],
            available_ram_gb=ram_info["available_gb"],
            gpu_available=gpu_info["available"],
            gpu_name=gpu_info["name"],
            gpu_vram_gb=gpu_info["vram_gb"],
            platform=self.system,
            architecture=self.machine,
        )

    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        cores = os.cpu_count() or 4
        threads = cores  # Default assumption
        model = "Unknown CPU"

        try:
            if self.system == "linux":
                # Get CPU model
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()

                model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
                if model_match:
                    model = model_match.group(1).strip()

                # Count physical cores vs threads
                physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
                cores_per_socket = re.search(r"cpu cores\s*:\s*(\d+)", cpuinfo)

                if cores_per_socket and physical_ids:
                    cores = int(cores_per_socket.group(1)) * len(physical_ids)

                siblings = re.search(r"siblings\s*:\s*(\d+)", cpuinfo)
                if siblings and physical_ids:
                    threads = int(siblings.group(1)) * len(physical_ids)
                else:
                    threads = os.cpu_count() or cores

            elif self.system == "darwin":
                # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    model = result.stdout.strip()

                result = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    cores = int(result.stdout.strip())

                result = subprocess.run(
                    ["sysctl", "-n", "hw.logicalcpu"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    threads = int(result.stdout.strip())

            elif self.system == "windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        model = lines[1].strip()

                result = subprocess.run(
                    ["wmic", "cpu", "get", "NumberOfCores"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        cores = int(lines[1].strip())

                threads = os.cpu_count() or cores

        except Exception as e:
            log.warning(f"CPU detection fallback: {e}")

        return {"cores": cores, "threads": threads, "model": model}

    def _detect_ram(self) -> Dict[str, float]:
        """Detect RAM information."""
        total_gb = 8.0  # Default fallback
        available_gb = 4.0

        try:
            if self.system == "linux":
                with open("/proc/meminfo") as f:
                    meminfo = f.read()

                total_match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
                if total_match:
                    total_gb = int(total_match.group(1)) / 1024 / 1024

                available_match = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
                if available_match:
                    available_gb = int(available_match.group(1)) / 1024 / 1024

            elif self.system == "darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    total_gb = int(result.stdout.strip()) / 1024 / 1024 / 1024

                # Approximate available memory
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    free_match = re.search(r"Pages free:\s+(\d+)", result.stdout)
                    inactive_match = re.search(r"Pages inactive:\s+(\d+)", result.stdout)

                    free_pages = int(free_match.group(1)) if free_match else 0
                    inactive_pages = int(inactive_match.group(1)) if inactive_match else 0

                    # Page size is typically 4096 bytes on macOS
                    available_gb = (free_pages + inactive_pages) * 4096 / 1024 / 1024 / 1024

            elif self.system == "windows":
                result = subprocess.run(
                    ["wmic", "OS", "get", "TotalVisibleMemorySize"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        total_gb = int(lines[1].strip()) / 1024 / 1024

                result = subprocess.run(
                    ["wmic", "OS", "get", "FreePhysicalMemory"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        available_gb = int(lines[1].strip()) / 1024 / 1024

        except Exception as e:
            log.warning(f"RAM detection fallback: {e}")

        return {"total_gb": total_gb, "available_gb": available_gb}

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information."""
        result = {
            "available": False,
            "name": "None",
            "vram_gb": 0.0,
        }

        try:
            # Check for NVIDIA GPU
            nvidia_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            if nvidia_result.returncode == 0:
                output = nvidia_result.stdout.strip()
                if output:
                    parts = output.split(",")
                    result["available"] = True
                    result["name"] = parts[0].strip()
                    if len(parts) > 1:
                        result["vram_gb"] = float(parts[1].strip()) / 1024
                    return result

        except FileNotFoundError:
            pass

        try:
            # Check for AMD GPU (ROCm)
            rocm_result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True
            )
            if rocm_result.returncode == 0:
                output = rocm_result.stdout.strip()
                if "GPU" in output:
                    result["available"] = True
                    result["name"] = "AMD GPU (ROCm)"
                    result["vram_gb"] = 8.0  # Approximate
                    return result

        except FileNotFoundError:
            pass

        # Check for Apple Silicon
        if self.system == "darwin" and self.machine in ("arm64", "aarch64"):
            result["available"] = True
            result["name"] = "Apple Silicon (Metal)"
            # Unified memory - estimate GPU portion
            try:
                mem_result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True
                )
                if mem_result.returncode == 0:
                    total_mem = int(mem_result.stdout.strip()) / 1024 / 1024 / 1024
                    # Apple Silicon shares memory, estimate GPU portion
                    result["vram_gb"] = total_mem * 0.75
            except:
                result["vram_gb"] = 8.0

        return result


class PerformanceOptimizer:
    """Optimizes Ollama configuration for USB operation."""

    # Performance profiles
    PROFILES = {
        "minimal": {
            "thread_ratio": 0.25,
            "memory_ratio": 0.3,
            "batch_size": 256,
            "context_length": 2048,
            "gpu_layers": 0,
        },
        "balanced": {
            "thread_ratio": 0.5,
            "memory_ratio": 0.5,
            "batch_size": 512,
            "context_length": 4096,
            "gpu_layers": 20,
        },
        "performance": {
            "thread_ratio": 0.75,
            "memory_ratio": 0.7,
            "batch_size": 1024,
            "context_length": 8192,
            "gpu_layers": 35,
        },
        "max": {
            "thread_ratio": 1.0,
            "memory_ratio": 0.85,
            "batch_size": 2048,
            "context_length": 16384,
            "gpu_layers": 99,
        },
    }

    def __init__(self, hardware: HardwareInfo, profile: str = "balanced"):
        self.hardware = hardware
        self.profile_name = profile
        self.profile = self.PROFILES.get(profile, self.PROFILES["balanced"])

    def optimize(self) -> OptimizationConfig:
        """Generate optimized configuration."""
        log.info(f"Generating optimization for profile: {self.profile_name}")

        # Calculate optimal thread count
        # Use physical cores for compute, avoid hyperthreading overhead for LLM
        optimal_threads = max(1, int(self.hardware.cpu_cores * self.profile["thread_ratio"]))

        # For USB operation, leave some CPU headroom for I/O
        if self.profile_name in ("minimal", "balanced"):
            optimal_threads = max(1, optimal_threads - 1)

        # Memory configuration
        available_for_model = self.hardware.available_ram_gb * self.profile["memory_ratio"]
        memory_limit = max(2.0, min(available_for_model, self.hardware.total_ram_gb * 0.8))

        # GPU configuration
        num_gpu = 1 if self.hardware.gpu_available else 0
        gpu_layers = self.profile["gpu_layers"] if self.hardware.gpu_available else 0

        # Adjust GPU layers based on VRAM
        if self.hardware.gpu_available and self.hardware.gpu_vram_gb > 0:
            # Approximate: 8B model needs ~5GB VRAM for full offload
            max_layers = int((self.hardware.gpu_vram_gb / 5.0) * 35)
            gpu_layers = min(gpu_layers, max_layers)

        # Context length based on available memory
        context_length = self.profile["context_length"]
        if self.hardware.available_ram_gb < 8:
            context_length = min(context_length, 4096)
        if self.hardware.available_ram_gb < 4:
            context_length = min(context_length, 2048)

        # Flash attention for memory efficiency
        flash_attention = self.hardware.gpu_available or self.hardware.available_ram_gb >= 16

        # Memory mapping for faster model loading from USB
        # mmap is beneficial for USB as it reduces initial load time
        mmap = True

        # mlock: pin model in RAM to prevent swapping
        # Only enable if we have sufficient RAM
        mlock = self.hardware.available_ram_gb >= 16 and self.profile_name in ("performance", "max")

        # NUMA awareness for multi-socket systems
        numa = self.hardware.cpu_cores >= 16

        # Keep alive: how long to keep model loaded
        keep_alive = "5m" if self.profile_name == "minimal" else "15m"

        return OptimizationConfig(
            num_threads=optimal_threads,
            num_gpu=num_gpu,
            gpu_layers=gpu_layers,
            main_gpu=0,
            memory_limit_gb=memory_limit,
            batch_size=self.profile["batch_size"],
            context_length=context_length,
            mlock=mlock,
            mmap=mmap,
            flash_attention=flash_attention,
            keep_alive=keep_alive,
            numa=numa,
        )

    def get_environment_variables(self, config: OptimizationConfig) -> Dict[str, str]:
        """Generate environment variables for Ollama."""
        env_vars = {
            "OLLAMA_NUM_THREADS": str(config.num_threads),
            "OLLAMA_NUM_GPU": str(config.num_gpu),
            "OLLAMA_GPU_LAYERS": str(config.gpu_layers),
            "OLLAMA_MAIN_GPU": str(config.main_gpu),
            "OLLAMA_FLASH_ATTENTION": "1" if config.flash_attention else "0",
            "OLLAMA_KEEP_ALIVE": config.keep_alive,
            "OLLAMA_NOPRUNE": "1",  # Don't auto-prune models
        }

        # Platform-specific optimizations
        if self.hardware.platform == "darwin" and "arm64" in self.hardware.architecture:
            env_vars["OLLAMA_METAL"] = "1"

        return env_vars

    def get_modelfile_options(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Generate Modelfile PARAMETER options."""
        return {
            "num_thread": config.num_threads,
            "num_gpu": config.num_gpu,
            "num_ctx": config.context_length,
            "num_batch": config.batch_size,
            "mlock": config.mlock,
            "mmap": config.mmap,
            "numa": config.numa,
        }


class ConfigWriter:
    """Writes optimization configuration files."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.config_path = root_path / "modules" / "config"

    def write_performance_config(
        self,
        hardware: HardwareInfo,
        config: OptimizationConfig,
        env_vars: Dict[str, str]
    ) -> Path:
        """Write performance configuration JSON."""
        output = {
            "version": __version__,
            "hardware": asdict(hardware),
            "optimization": asdict(config),
            "environment": env_vars,
        }

        output_path = self.config_path / "performance.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        log.info(f"Written: {output_path}")
        return output_path

    def write_env_script(self, env_vars: Dict[str, str], platform_name: str) -> Path:
        """Write platform-specific environment script."""
        scripts_path = self.root_path / "scripts" / "performance"
        scripts_path.mkdir(parents=True, exist_ok=True)

        if platform_name == "windows":
            # Windows batch file
            script_path = scripts_path / "set_performance.bat"
            lines = ["@echo off", "REM USB-AI Performance Settings"]
            for key, value in env_vars.items():
                lines.append(f"set {key}={value}")
            content = "\r\n".join(lines)
        else:
            # Unix shell script
            script_path = scripts_path / "set_performance.sh"
            lines = ["#!/bin/bash", "# USB-AI Performance Settings"]
            for key, value in env_vars.items():
                lines.append(f'export {key}="{value}"')
            content = "\n".join(lines)

        with open(script_path, "w") as f:
            f.write(content)

        # Make executable on Unix
        if platform_name != "windows":
            os.chmod(script_path, 0o755)

        log.info(f"Written: {script_path}")
        return script_path


def find_root() -> Path:
    """Locate USB-AI root directory."""
    script_dir = Path(__file__).parent.resolve()

    # scripts/performance -> scripts -> root
    if (script_dir.parent.parent / "modules").exists():
        return script_dir.parent.parent
    # scripts -> root
    if (script_dir.parent / "modules").exists():
        return script_dir.parent
    # direct
    if (script_dir / "modules").exists():
        return script_dir

    log.error("Cannot locate USB-AI root directory")
    sys.exit(1)


def main() -> int:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize USB-AI performance settings"
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "balanced", "performance", "max"],
        default="balanced",
        help="Performance profile to apply (default: balanced)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show optimization without writing files"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration as JSON"
    )

    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("           USB-AI Performance Optimizer")
    print("=" * 60)
    print("")

    # Detect hardware
    log.info("Detecting hardware capabilities...")
    detector = HardwareDetector()
    hardware = detector.detect()

    print(f"  Platform:    {hardware.platform} ({hardware.architecture})")
    print(f"  CPU:         {hardware.cpu_model}")
    print(f"  Cores:       {hardware.cpu_cores} physical, {hardware.cpu_threads} logical")
    print(f"  RAM:         {hardware.total_ram_gb:.1f} GB total, {hardware.available_ram_gb:.1f} GB available")
    if hardware.gpu_available:
        print(f"  GPU:         {hardware.gpu_name} ({hardware.gpu_vram_gb:.1f} GB VRAM)")
    else:
        print("  GPU:         Not detected")
    print("")

    # Generate optimization
    log.info(f"Generating '{args.profile}' optimization profile...")
    optimizer = PerformanceOptimizer(hardware, args.profile)
    config = optimizer.optimize()
    env_vars = optimizer.get_environment_variables(config)

    print("")
    print("Optimized Configuration:")
    print(f"  Threads:         {config.num_threads}")
    print(f"  GPU Layers:      {config.gpu_layers}")
    print(f"  Memory Limit:    {config.memory_limit_gb:.1f} GB")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Context Length:  {config.context_length}")
    print(f"  Flash Attention: {'Yes' if config.flash_attention else 'No'}")
    print(f"  Memory Map:      {'Yes' if config.mmap else 'No'}")
    print(f"  Memory Lock:     {'Yes' if config.mlock else 'No'}")
    print(f"  Keep Alive:      {config.keep_alive}")
    print("")

    if args.json:
        output = {
            "hardware": asdict(hardware),
            "optimization": asdict(config),
            "environment": env_vars,
        }
        print(json.dumps(output, indent=2))
        return 0

    if args.dry_run:
        print("Dry run - no files written")
        return 0

    # Write configuration files
    root_path = find_root()
    writer = ConfigWriter(root_path)

    writer.write_performance_config(hardware, config, env_vars)
    writer.write_env_script(env_vars, hardware.platform)

    print("")
    print("=" * 60)
    print("         Performance Optimization Complete")
    print("=" * 60)
    print("")
    print("Configuration written to:")
    print(f"  {root_path}/modules/config/performance.json")
    print(f"  {root_path}/scripts/performance/set_performance.sh")
    print("")
    print("To apply settings, source the environment script before")
    print("starting Ollama, or the launcher will apply them automatically.")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
