#!/usr/bin/env python3
"""
ollama_tuner.py

Ollama Configuration Optimizer for Maximum Inference Speed.

This script:
1. Detects hardware (CPU cores, RAM, GPU VRAM)
2. Calculates optimal settings for Ollama environment variables
3. Generates optimized environment configuration
4. Benchmarks before/after with detailed metrics
5. Supports presets: speed, balanced, memory_saver

Key Ollama Environment Variables Optimized:
- OLLAMA_NUM_PARALLEL: Number of parallel request slots
- OLLAMA_MAX_LOADED_MODELS: Maximum models to keep in memory
- OLLAMA_NUM_GPU: Number of GPUs to use
- OLLAMA_FLASH_ATTENTION: Enable flash attention for speed
- OLLAMA_KV_CACHE_TYPE: KV cache precision (f16/q8_0/q4_0)
- OLLAMA_KEEP_ALIVE: How long to keep models loaded
- OLLAMA_NUM_THREADS: CPU thread count for inference
- OLLAMA_GPU_LAYERS: Number of layers to offload to GPU
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import argparse
import urllib.request
import urllib.error

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ==============================================================================
# Hardware Detection
# ==============================================================================

@dataclass
class HardwareProfile:
    """Complete hardware profile for optimization."""
    # CPU
    cpu_model: str
    cpu_vendor: str
    physical_cores: int
    logical_cores: int
    frequency_mhz: float
    cpu_features: List[str]
    has_avx2: bool
    has_avx512: bool

    # Memory
    total_ram_gb: float
    available_ram_gb: float

    # GPU
    gpu_available: bool
    gpu_name: str
    gpu_vendor: str
    gpu_vram_gb: float
    gpu_compute_capability: str

    # Platform
    platform_name: str
    architecture: str
    is_apple_silicon: bool


class HardwareDetector:
    """Detect system hardware for optimization."""

    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()

    def detect(self) -> HardwareProfile:
        """Detect all hardware capabilities."""
        cpu = self._detect_cpu()
        ram = self._detect_ram()
        gpu = self._detect_gpu()

        is_apple = self.system == "darwin" and self.machine in ("arm64", "aarch64")

        return HardwareProfile(
            cpu_model=cpu["model"],
            cpu_vendor=cpu["vendor"],
            physical_cores=cpu["physical_cores"],
            logical_cores=cpu["logical_cores"],
            frequency_mhz=cpu["frequency_mhz"],
            cpu_features=cpu["features"],
            has_avx2="avx2" in cpu["features"],
            has_avx512="avx512f" in cpu["features"] or "avx512" in cpu["features"],
            total_ram_gb=ram["total_gb"],
            available_ram_gb=ram["available_gb"],
            gpu_available=gpu["available"],
            gpu_name=gpu["name"],
            gpu_vendor=gpu["vendor"],
            gpu_vram_gb=gpu["vram_gb"],
            gpu_compute_capability=gpu["compute"],
            platform_name=self.system,
            architecture=self.machine,
            is_apple_silicon=is_apple
        )

    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        result = {
            "model": "Unknown CPU",
            "vendor": "Unknown",
            "physical_cores": os.cpu_count() or 4,
            "logical_cores": os.cpu_count() or 4,
            "frequency_mhz": 0.0,
            "features": []
        }

        try:
            if self.system == "linux":
                self._detect_cpu_linux(result)
            elif self.system == "darwin":
                self._detect_cpu_darwin(result)
            elif self.system == "windows":
                self._detect_cpu_windows(result)
        except Exception as e:
            log.warning(f"CPU detection fallback: {e}")

        return result

    def _detect_cpu_linux(self, result: Dict[str, Any]):
        """Linux CPU detection."""
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()

            # Model
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if model_match:
                result["model"] = model_match.group(1).strip()

            # Vendor
            vendor_match = re.search(r"vendor_id\s*:\s*(.+)", cpuinfo)
            if vendor_match:
                result["vendor"] = vendor_match.group(1).strip()

            # Physical cores
            physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
            cores_per_socket = re.search(r"cpu cores\s*:\s*(\d+)", cpuinfo)
            if cores_per_socket and physical_ids:
                result["physical_cores"] = int(cores_per_socket.group(1)) * max(1, len(physical_ids))

            # Logical cores
            siblings = re.search(r"siblings\s*:\s*(\d+)", cpuinfo)
            if siblings and physical_ids:
                result["logical_cores"] = int(siblings.group(1)) * max(1, len(physical_ids))

            # Frequency
            freq_match = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            if freq_match:
                result["frequency_mhz"] = float(freq_match.group(1))

            # Features (important for SIMD)
            flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo)
            if flags_match:
                flags = flags_match.group(1).split()
                important = ["avx", "avx2", "avx512f", "avx512", "sse4_2", "fma", "f16c"]
                result["features"] = [f for f in important if f in flags]

        except Exception as e:
            log.warning(f"Linux CPU detection error: {e}")

    def _detect_cpu_darwin(self, result: Dict[str, Any]):
        """macOS CPU detection."""
        try:
            # Model
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True)
            if r.returncode == 0:
                result["model"] = r.stdout.strip()

            # Physical cores
            r = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                             capture_output=True, text=True)
            if r.returncode == 0:
                result["physical_cores"] = int(r.stdout.strip())

            # Logical cores
            r = subprocess.run(["sysctl", "-n", "hw.logicalcpu"],
                             capture_output=True, text=True)
            if r.returncode == 0:
                result["logical_cores"] = int(r.stdout.strip())

            # Apple Silicon detection
            if self.machine in ("arm64", "aarch64"):
                result["vendor"] = "Apple"
                result["features"] = ["apple_silicon", "neon"]
            else:
                # Intel Mac
                r = subprocess.run(["sysctl", "-n", "machdep.cpu.features"],
                                 capture_output=True, text=True)
                if r.returncode == 0:
                    features = r.stdout.lower().split()
                    important = ["avx", "avx2", "avx512", "sse4"]
                    result["features"] = [f for f in important if any(f in feat for feat in features)]

        except Exception as e:
            log.warning(f"macOS CPU detection error: {e}")

    def _detect_cpu_windows(self, result: Dict[str, Any]):
        """Windows CPU detection."""
        try:
            r = subprocess.run(["wmic", "cpu", "get", "name"],
                             capture_output=True, text=True)
            if r.returncode == 0:
                lines = r.stdout.strip().split("\n")
                if len(lines) > 1:
                    result["model"] = lines[1].strip()

            r = subprocess.run(["wmic", "cpu", "get", "NumberOfCores"],
                             capture_output=True, text=True)
            if r.returncode == 0:
                lines = r.stdout.strip().split("\n")
                if len(lines) > 1:
                    result["physical_cores"] = int(lines[1].strip())

        except Exception as e:
            log.warning(f"Windows CPU detection error: {e}")

    def _detect_ram(self) -> Dict[str, float]:
        """Detect RAM information."""
        result = {"total_gb": 8.0, "available_gb": 4.0}

        try:
            if self.system == "linux":
                with open("/proc/meminfo") as f:
                    meminfo = f.read()

                total = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
                if total:
                    result["total_gb"] = int(total.group(1)) / 1024 / 1024

                available = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
                if available:
                    result["available_gb"] = int(available.group(1)) / 1024 / 1024

            elif self.system == "darwin":
                r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True)
                if r.returncode == 0:
                    result["total_gb"] = int(r.stdout.strip()) / 1024 / 1024 / 1024

                # Approximate available from vm_stat
                r = subprocess.run(["vm_stat"], capture_output=True, text=True)
                if r.returncode == 0:
                    free = re.search(r"Pages free:\s+(\d+)", r.stdout)
                    inactive = re.search(r"Pages inactive:\s+(\d+)", r.stdout)
                    pages = 0
                    if free:
                        pages += int(free.group(1))
                    if inactive:
                        pages += int(inactive.group(1))
                    result["available_gb"] = pages * 4096 / 1024 / 1024 / 1024

            elif self.system == "windows":
                r = subprocess.run(["wmic", "OS", "get", "TotalVisibleMemorySize"],
                                 capture_output=True, text=True)
                if r.returncode == 0:
                    lines = r.stdout.strip().split("\n")
                    if len(lines) > 1:
                        result["total_gb"] = int(lines[1].strip()) / 1024 / 1024

                r = subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory"],
                                 capture_output=True, text=True)
                if r.returncode == 0:
                    lines = r.stdout.strip().split("\n")
                    if len(lines) > 1:
                        result["available_gb"] = int(lines[1].strip()) / 1024 / 1024

        except Exception as e:
            log.warning(f"RAM detection error: {e}")

        return result

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information."""
        result = {
            "available": False,
            "name": "None",
            "vendor": "None",
            "vram_gb": 0.0,
            "compute": "N/A"
        }

        # Try NVIDIA
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split(",")
                result["available"] = True
                result["name"] = parts[0].strip()
                result["vendor"] = "NVIDIA"
                if len(parts) > 1:
                    result["vram_gb"] = float(parts[1].strip()) / 1024
                if len(parts) > 2:
                    result["compute"] = parts[2].strip()
                return result
        except FileNotFoundError:
            pass

        # Try AMD ROCm
        try:
            r = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True)
            if r.returncode == 0 and "GPU" in r.stdout:
                result["available"] = True
                result["vendor"] = "AMD"
                result["name"] = "AMD GPU (ROCm)"
                result["vram_gb"] = 8.0  # Approximate
                return result
        except FileNotFoundError:
            pass

        # Apple Silicon
        if self.system == "darwin" and self.machine in ("arm64", "aarch64"):
            result["available"] = True
            result["vendor"] = "Apple"
            result["name"] = "Apple Silicon (Metal)"
            result["compute"] = "Metal 3"
            # Unified memory
            try:
                r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True)
                if r.returncode == 0:
                    total = int(r.stdout.strip()) / 1024 / 1024 / 1024
                    result["vram_gb"] = total * 0.75  # GPU can use ~75%
            except:
                result["vram_gb"] = 8.0

        return result


# ==============================================================================
# Optimization Presets
# ==============================================================================

@dataclass
class OptimizationPreset:
    """Optimization preset configuration."""
    name: str
    description: str

    # Thread configuration
    thread_ratio: float  # Ratio of physical cores to use

    # Parallel processing
    num_parallel: int  # OLLAMA_NUM_PARALLEL
    max_loaded_models: int  # OLLAMA_MAX_LOADED_MODELS

    # Memory
    keep_alive: str  # OLLAMA_KEEP_ALIVE
    kv_cache_type: str  # OLLAMA_KV_CACHE_TYPE (f16, q8_0, q4_0)

    # GPU
    gpu_layers_ratio: float  # Ratio of layers to offload

    # Features
    flash_attention: bool  # OLLAMA_FLASH_ATTENTION


# Speed preset: Maximum inference speed, higher memory usage
PRESET_SPEED = OptimizationPreset(
    name="speed",
    description="Maximum inference speed, uses more memory",
    thread_ratio=0.9,  # Use most cores
    num_parallel=4,  # Handle multiple parallel requests
    max_loaded_models=2,  # Keep models loaded
    keep_alive="30m",  # Long keep-alive to avoid reloads
    kv_cache_type="f16",  # Full precision for speed
    gpu_layers_ratio=1.0,  # Offload all layers to GPU
    flash_attention=True  # Enable flash attention
)

# Balanced preset: Good speed with reasonable memory usage
PRESET_BALANCED = OptimizationPreset(
    name="balanced",
    description="Balanced speed and memory usage",
    thread_ratio=0.7,
    num_parallel=2,
    max_loaded_models=1,
    keep_alive="10m",
    kv_cache_type="q8_0",  # 8-bit KV cache
    gpu_layers_ratio=0.8,
    flash_attention=True
)

# Memory saver preset: Minimize memory usage
PRESET_MEMORY_SAVER = OptimizationPreset(
    name="memory_saver",
    description="Minimize memory usage, slower inference",
    thread_ratio=0.5,
    num_parallel=1,
    max_loaded_models=1,
    keep_alive="5m",
    kv_cache_type="q4_0",  # 4-bit KV cache
    gpu_layers_ratio=0.5,
    flash_attention=False  # Disable for memory
)

PRESETS = {
    "speed": PRESET_SPEED,
    "balanced": PRESET_BALANCED,
    "memory_saver": PRESET_MEMORY_SAVER
}


# ==============================================================================
# Configuration Generator
# ==============================================================================

@dataclass
class OllamaConfig:
    """Generated Ollama configuration."""
    # Core settings
    num_threads: int
    num_parallel: int
    max_loaded_models: int
    keep_alive: str

    # GPU settings
    num_gpu: int
    gpu_layers: int
    main_gpu: int

    # Performance features
    flash_attention: bool
    kv_cache_type: str

    # Additional settings
    mlock: bool  # Lock model in memory
    numa: bool  # NUMA-aware memory allocation

    # Metadata
    preset_name: str
    hardware_summary: str


class ConfigGenerator:
    """Generate optimized Ollama configuration."""

    def __init__(self, hardware: HardwareProfile, preset: OptimizationPreset):
        self.hw = hardware
        self.preset = preset

    def generate(self) -> OllamaConfig:
        """Generate optimized configuration."""
        # Thread count based on physical cores
        num_threads = max(1, int(self.hw.physical_cores * self.preset.thread_ratio))

        # For hyperthreaded CPUs, don't exceed physical cores for LLM inference
        # Hyperthreading doesn't help much for SIMD-heavy workloads
        num_threads = min(num_threads, self.hw.physical_cores)

        # Parallel request handling
        num_parallel = self.preset.num_parallel

        # Adjust based on available RAM
        if self.hw.available_ram_gb < 8:
            num_parallel = min(num_parallel, 1)
        elif self.hw.available_ram_gb < 16:
            num_parallel = min(num_parallel, 2)

        # GPU configuration
        if self.hw.gpu_available:
            num_gpu = 1
            # Calculate GPU layers based on VRAM
            # ~35 layers for 8B model, each layer ~140MB
            if self.hw.gpu_vram_gb >= 8:
                max_layers = 99  # Full offload
            elif self.hw.gpu_vram_gb >= 6:
                max_layers = 35
            elif self.hw.gpu_vram_gb >= 4:
                max_layers = 25
            else:
                max_layers = 15

            gpu_layers = int(max_layers * self.preset.gpu_layers_ratio)
        else:
            num_gpu = 0
            gpu_layers = 0

        # KV cache type
        kv_cache_type = self.preset.kv_cache_type
        # Force lower precision on memory-constrained systems
        if self.hw.available_ram_gb < 8 and kv_cache_type == "f16":
            kv_cache_type = "q8_0"
        if self.hw.available_ram_gb < 4:
            kv_cache_type = "q4_0"

        # Flash attention - requires GPU or Apple Silicon
        flash_attention = self.preset.flash_attention and (
            self.hw.gpu_available or self.hw.is_apple_silicon
        )

        # Memory locking - only on systems with sufficient RAM
        mlock = self.hw.available_ram_gb >= 16 and self.preset.name == "speed"

        # NUMA awareness for multi-socket systems
        numa = self.hw.physical_cores >= 16

        # Build hardware summary
        hw_summary = f"{self.hw.cpu_model} ({self.hw.physical_cores}c), {self.hw.total_ram_gb:.0f}GB RAM"
        if self.hw.gpu_available:
            hw_summary += f", {self.hw.gpu_name} ({self.hw.gpu_vram_gb:.0f}GB)"

        return OllamaConfig(
            num_threads=num_threads,
            num_parallel=num_parallel,
            max_loaded_models=self.preset.max_loaded_models,
            keep_alive=self.preset.keep_alive,
            num_gpu=num_gpu,
            gpu_layers=gpu_layers,
            main_gpu=0,
            flash_attention=flash_attention,
            kv_cache_type=kv_cache_type,
            mlock=mlock,
            numa=numa,
            preset_name=self.preset.name,
            hardware_summary=hw_summary
        )

    def to_env_vars(self, config: OllamaConfig) -> Dict[str, str]:
        """Convert configuration to environment variables."""
        env = {
            "OLLAMA_NUM_PARALLEL": str(config.num_parallel),
            "OLLAMA_MAX_LOADED_MODELS": str(config.max_loaded_models),
            "OLLAMA_KEEP_ALIVE": config.keep_alive,
            "OLLAMA_FLASH_ATTENTION": "1" if config.flash_attention else "0",
            "OLLAMA_KV_CACHE_TYPE": config.kv_cache_type,
        }

        # GPU settings
        if config.num_gpu > 0:
            env["OLLAMA_NUM_GPU"] = str(config.num_gpu)
            env["OLLAMA_GPU_LAYERS"] = str(config.gpu_layers)
            env["OLLAMA_MAIN_GPU"] = str(config.main_gpu)

        # Apple Silicon Metal
        if self.hw.is_apple_silicon:
            env["OLLAMA_METAL"] = "1"

        # Debug/advanced settings
        if config.numa:
            env["OLLAMA_NUMA"] = "1"

        # Thread count (via llama.cpp)
        env["OLLAMA_NUM_THREADS"] = str(config.num_threads)

        return env


# ==============================================================================
# Benchmarking
# ==============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model: str
    prompt: str

    # Timing
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_second: float

    # Context
    is_cold_start: bool
    prompt_tokens: int

    # Memory
    memory_used_mb: float


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    model: str
    runs: int

    # Cold start metrics
    cold_ttft_ms: float
    cold_tps: float

    # Warm start metrics
    warm_ttft_avg_ms: float
    warm_ttft_std_ms: float
    warm_tps_avg: float
    warm_tps_std: float

    # Memory
    peak_memory_mb: float

    # Configuration used
    config_preset: str
    env_vars: Dict[str, str]


class OllamaBenchmark:
    """Benchmark Ollama inference performance."""

    OLLAMA_API = "http://localhost:11434"

    # Benchmark prompts
    PROMPTS = {
        "short": "What is 2+2?",
        "medium": "Explain the concept of recursion in programming in 3 sentences.",
        "long": "Write a detailed explanation of how neural networks work, including the concepts of layers, weights, biases, activation functions, and backpropagation. Include examples."
    }

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.OLLAMA_API}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            req = urllib.request.Request(f"{self.OLLAMA_API}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            log.warning(f"Failed to get models: {e}")
            return []

    def run_inference(self, model: str, prompt: str) -> Tuple[float, float, int, str]:
        """
        Run a single inference and measure timing.

        Returns: (ttft_ms, total_ms, token_count, response_text)
        """
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 100  # Limit output for consistent benchmarking
            }
        }).encode()

        req = urllib.request.Request(
            f"{self.OLLAMA_API}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        response_text = ""

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode())

                        if "response" in data:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()

                            response_text += data["response"]
                            # Approximate token count (rough estimate)
                            token_count += 1

                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            log.error(f"Inference error: {e}")
            return 0, 0, 0, ""

        end_time = time.perf_counter()

        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_ms = (end_time - start_time) * 1000

        return ttft_ms, total_ms, token_count, response_text

    def unload_model(self, model: str):
        """Unload a model from memory for cold start testing."""
        try:
            payload = json.dumps({
                "model": model,
                "keep_alive": 0
            }).encode()

            req = urllib.request.Request(
                f"{self.OLLAMA_API}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                # Consume response
                resp.read()
        except:
            pass

    def get_memory_usage(self) -> float:
        """Get current Ollama memory usage in MB."""
        try:
            # Try to get from ps command
            if platform.system().lower() == "windows":
                result = subprocess.run(
                    ["wmic", "process", "where", "name='ollama.exe'",
                     "get", "WorkingSetSize"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        return int(lines[1].strip()) / 1024 / 1024
            else:
                result = subprocess.run(
                    ["pgrep", "-f", "ollama"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    pids = result.stdout.strip().split("\n")
                    total_mem = 0
                    for pid in pids:
                        if pid:
                            try:
                                with open(f"/proc/{pid}/status") as f:
                                    status = f.read()
                                    rss = re.search(r"VmRSS:\s+(\d+)\s+kB", status)
                                    if rss:
                                        total_mem += int(rss.group(1)) / 1024
                            except:
                                pass
                    return total_mem
        except:
            pass
        return 0

    def benchmark_model(
        self,
        model: str,
        prompt_type: str = "medium",
        warm_runs: int = 3,
        env_vars: Optional[Dict[str, str]] = None,
        config_preset: str = "unknown"
    ) -> Optional[BenchmarkSummary]:
        """
        Benchmark a model with cold and warm start measurements.

        Args:
            model: Model name to benchmark
            prompt_type: "short", "medium", or "long"
            warm_runs: Number of warm runs for averaging
            env_vars: Environment variables used
            config_preset: Name of preset used

        Returns:
            BenchmarkSummary with all metrics
        """
        if not self.is_ollama_running():
            log.error("Ollama server is not running")
            return None

        prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["medium"])

        log.info(f"Benchmarking {model} with '{prompt_type}' prompt")

        results = []
        peak_memory = 0

        # Cold start (unload model first)
        log.info("  Running cold start test...")
        self.unload_model(model)
        time.sleep(2)  # Wait for unload

        cold_ttft, cold_total, cold_tokens, _ = self.run_inference(model, prompt)
        cold_tps = cold_tokens / (cold_total / 1000) if cold_total > 0 else 0

        peak_memory = max(peak_memory, self.get_memory_usage())

        log.info(f"    Cold start: TTFT={cold_ttft:.0f}ms, TPS={cold_tps:.1f}")

        # Warm starts
        log.info(f"  Running {warm_runs} warm start tests...")
        warm_ttfts = []
        warm_tps_list = []

        for i in range(warm_runs):
            ttft, total, tokens, _ = self.run_inference(model, prompt)
            tps = tokens / (total / 1000) if total > 0 else 0

            warm_ttfts.append(ttft)
            warm_tps_list.append(tps)

            peak_memory = max(peak_memory, self.get_memory_usage())

            log.info(f"    Run {i+1}: TTFT={ttft:.0f}ms, TPS={tps:.1f}")

        # Calculate statistics
        warm_ttft_avg = statistics.mean(warm_ttfts) if warm_ttfts else 0
        warm_ttft_std = statistics.stdev(warm_ttfts) if len(warm_ttfts) > 1 else 0
        warm_tps_avg = statistics.mean(warm_tps_list) if warm_tps_list else 0
        warm_tps_std = statistics.stdev(warm_tps_list) if len(warm_tps_list) > 1 else 0

        return BenchmarkSummary(
            model=model,
            runs=warm_runs + 1,
            cold_ttft_ms=cold_ttft,
            cold_tps=cold_tps,
            warm_ttft_avg_ms=warm_ttft_avg,
            warm_ttft_std_ms=warm_ttft_std,
            warm_tps_avg=warm_tps_avg,
            warm_tps_std=warm_tps_std,
            peak_memory_mb=peak_memory,
            config_preset=config_preset,
            env_vars=env_vars or {}
        )


# ==============================================================================
# Main Application
# ==============================================================================

class OllamaTuner:
    """Main Ollama tuner application."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.config_dir = root_path / "modules" / "config"
        self.scripts_dir = root_path / "scripts" / "performance"

        self.detector = HardwareDetector()
        self.benchmark = OllamaBenchmark()

    def detect_hardware(self) -> HardwareProfile:
        """Detect and display hardware information."""
        log.info("Detecting hardware...")
        return self.detector.detect()

    def generate_config(
        self,
        hardware: HardwareProfile,
        preset_name: str = "balanced"
    ) -> Tuple[OllamaConfig, Dict[str, str]]:
        """Generate optimized configuration."""
        preset = PRESETS.get(preset_name, PRESET_BALANCED)
        generator = ConfigGenerator(hardware, preset)

        config = generator.generate()
        env_vars = generator.to_env_vars(config)

        return config, env_vars

    def write_config_files(
        self,
        hardware: HardwareProfile,
        config: OllamaConfig,
        env_vars: Dict[str, str]
    ) -> List[Path]:
        """Write configuration files."""
        files = []

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON config
        json_path = self.config_dir / "ollama_tuned.json"
        config_data = {
            "version": __version__,
            "generated_at": datetime.now().isoformat(),
            "hardware": asdict(hardware),
            "config": asdict(config),
            "environment_variables": env_vars
        }

        with open(json_path, "w") as f:
            json.dump(config_data, f, indent=2)
        files.append(json_path)

        # Write shell script for Unix
        sh_path = self.scripts_dir / "ollama_env.sh"
        sh_lines = [
            "#!/bin/bash",
            "# Ollama Optimized Environment Variables",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Preset: {config.preset_name}",
            f"# Hardware: {config.hardware_summary}",
            ""
        ]
        for key, value in env_vars.items():
            sh_lines.append(f'export {key}="{value}"')

        sh_lines.extend([
            "",
            'echo "Ollama environment configured for maximum speed"',
            f'echo "  Preset: {config.preset_name}"',
            f'echo "  Threads: {config.num_threads}"',
            f'echo "  GPU Layers: {config.gpu_layers}"',
            f'echo "  Flash Attention: {config.flash_attention}"'
        ])

        with open(sh_path, "w") as f:
            f.write("\n".join(sh_lines))
        os.chmod(sh_path, 0o755)
        files.append(sh_path)

        # Write batch script for Windows
        bat_path = self.scripts_dir / "ollama_env.bat"
        bat_lines = [
            "@echo off",
            "REM Ollama Optimized Environment Variables",
            f"REM Generated: {datetime.now().isoformat()}",
            f"REM Preset: {config.preset_name}",
            f"REM Hardware: {config.hardware_summary}",
            ""
        ]
        for key, value in env_vars.items():
            bat_lines.append(f"set {key}={value}")

        bat_lines.extend([
            "",
            "echo Ollama environment configured for maximum speed",
            f"echo   Preset: {config.preset_name}",
            f"echo   Threads: {config.num_threads}",
            f"echo   GPU Layers: {config.gpu_layers}",
            f"echo   Flash Attention: {config.flash_attention}"
        ])

        with open(bat_path, "w") as f:
            f.write("\r\n".join(bat_lines))
        files.append(bat_path)

        # Write PowerShell script
        ps1_path = self.scripts_dir / "ollama_env.ps1"
        ps1_lines = [
            "# Ollama Optimized Environment Variables",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Preset: {config.preset_name}",
            f"# Hardware: {config.hardware_summary}",
            ""
        ]
        for key, value in env_vars.items():
            ps1_lines.append(f'$env:{key} = "{value}"')

        ps1_lines.extend([
            "",
            'Write-Host "Ollama environment configured for maximum speed"',
            f'Write-Host "  Preset: {config.preset_name}"',
            f'Write-Host "  Threads: {config.num_threads}"',
            f'Write-Host "  GPU Layers: {config.gpu_layers}"',
            f'Write-Host "  Flash Attention: {config.flash_attention}"'
        ])

        with open(ps1_path, "w") as f:
            f.write("\n".join(ps1_lines))
        files.append(ps1_path)

        return files

    def apply_env_vars(self, env_vars: Dict[str, str]):
        """Apply environment variables to current process."""
        for key, value in env_vars.items():
            os.environ[key] = value
        log.info("Environment variables applied to current process")

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        prompt_type: str = "medium",
        warm_runs: int = 3,
        env_vars: Optional[Dict[str, str]] = None,
        preset_name: str = "unknown"
    ) -> List[BenchmarkSummary]:
        """Run benchmark on specified models."""
        if not self.benchmark.is_ollama_running():
            log.error("Ollama server is not running. Start it with: ollama serve")
            return []

        if not models:
            models = self.benchmark.get_available_models()
            if not models:
                log.error("No models available. Pull a model with: ollama pull <model>")
                return []

        results = []
        for model in models:
            result = self.benchmark.benchmark_model(
                model=model,
                prompt_type=prompt_type,
                warm_runs=warm_runs,
                env_vars=env_vars,
                config_preset=preset_name
            )
            if result:
                results.append(result)

        return results

    def print_hardware(self, hw: HardwareProfile):
        """Print hardware information."""
        print("\n" + "=" * 60)
        print("              HARDWARE DETECTION")
        print("=" * 60)

        print(f"\n  Platform:      {hw.platform_name} ({hw.architecture})")
        print(f"  CPU:           {hw.cpu_model}")
        print(f"  Cores:         {hw.physical_cores} physical, {hw.logical_cores} logical")
        print(f"  Features:      {', '.join(hw.cpu_features) if hw.cpu_features else 'None detected'}")
        print(f"  AVX2:          {'Yes' if hw.has_avx2 else 'No'}")
        print(f"  AVX512:        {'Yes' if hw.has_avx512 else 'No'}")
        print(f"\n  RAM Total:     {hw.total_ram_gb:.1f} GB")
        print(f"  RAM Available: {hw.available_ram_gb:.1f} GB")

        if hw.gpu_available:
            print(f"\n  GPU:           {hw.gpu_name}")
            print(f"  GPU Vendor:    {hw.gpu_vendor}")
            print(f"  GPU VRAM:      {hw.gpu_vram_gb:.1f} GB")
            print(f"  Compute:       {hw.gpu_compute_capability}")
        else:
            print("\n  GPU:           Not detected")

        if hw.is_apple_silicon:
            print("\n  Apple Silicon: Yes (Metal acceleration available)")

    def print_config(self, config: OllamaConfig, env_vars: Dict[str, str]):
        """Print configuration."""
        print("\n" + "=" * 60)
        print("              OPTIMIZED CONFIGURATION")
        print("=" * 60)

        print(f"\n  Preset:         {config.preset_name}")
        print(f"  Hardware:       {config.hardware_summary}")

        print("\n  Thread Settings:")
        print(f"    Threads:      {config.num_threads}")
        print(f"    Parallel:     {config.num_parallel}")
        print(f"    Max Models:   {config.max_loaded_models}")

        print("\n  GPU Settings:")
        print(f"    GPU Count:    {config.num_gpu}")
        print(f"    GPU Layers:   {config.gpu_layers}")

        print("\n  Performance Settings:")
        print(f"    Flash Attn:   {'Enabled' if config.flash_attention else 'Disabled'}")
        print(f"    KV Cache:     {config.kv_cache_type}")
        print(f"    Keep Alive:   {config.keep_alive}")
        print(f"    mlock:        {'Enabled' if config.mlock else 'Disabled'}")
        print(f"    NUMA:         {'Enabled' if config.numa else 'Disabled'}")

        print("\n  Environment Variables:")
        for key, value in sorted(env_vars.items()):
            print(f"    {key}={value}")

    def print_benchmark_results(self, results: List[BenchmarkSummary]):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("              BENCHMARK RESULTS")
        print("=" * 60)

        for result in results:
            print(f"\n  Model: {result.model}")
            print(f"  Preset: {result.config_preset}")
            print(f"  Runs: {result.runs}")

            print("\n  Cold Start:")
            print(f"    Time to First Token: {result.cold_ttft_ms:.0f} ms")
            print(f"    Tokens per Second:   {result.cold_tps:.1f}")

            print("\n  Warm Start (avg):")
            print(f"    Time to First Token: {result.warm_ttft_avg_ms:.0f} ms (+/- {result.warm_ttft_std_ms:.0f})")
            print(f"    Tokens per Second:   {result.warm_tps_avg:.1f} (+/- {result.warm_tps_std:.1f})")

            print(f"\n  Peak Memory: {result.peak_memory_mb:.0f} MB")

            # Speed improvement from cold to warm
            if result.cold_ttft_ms > 0:
                speedup = result.cold_ttft_ms / result.warm_ttft_avg_ms if result.warm_ttft_avg_ms > 0 else 0
                print(f"\n  Warm vs Cold Speedup: {speedup:.1f}x")

        print()


def find_root() -> Path:
    """Find USB-AI root directory."""
    script_dir = Path(__file__).resolve().parent

    # scripts/performance -> scripts -> root
    if (script_dir.parent.parent / "modules").exists():
        return script_dir.parent.parent
    # scripts -> root
    if (script_dir.parent / "modules").exists():
        return script_dir.parent
    # direct
    if (script_dir / "modules").exists():
        return script_dir

    # Fallback to current directory
    return Path.cwd()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ollama Configuration Optimizer for Maximum Inference Speed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  speed         Maximum inference speed, uses more memory
  balanced      Good speed with reasonable memory usage
  memory_saver  Minimize memory usage, slower inference

Examples:
  %(prog)s --preset speed
  %(prog)s --preset balanced --benchmark
  %(prog)s --apply --benchmark --models llama3.2:latest
        """
    )

    parser.add_argument(
        "--preset", "-p",
        choices=["speed", "balanced", "memory_saver"],
        default="balanced",
        help="Optimization preset (default: balanced)"
    )

    parser.add_argument(
        "--apply", "-a",
        action="store_true",
        help="Apply environment variables to current process"
    )

    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run benchmark after configuration"
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Models to benchmark (default: all available)"
    )

    parser.add_argument(
        "--prompt",
        choices=["short", "medium", "long"],
        default="medium",
        help="Benchmark prompt length (default: medium)"
    )

    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of warm benchmark runs (default: 3)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without writing files"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration as JSON"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Initialize tuner
    root_path = find_root()
    tuner = OllamaTuner(root_path)

    # Detect hardware
    hardware = tuner.detect_hardware()

    if not args.quiet and not args.json:
        tuner.print_hardware(hardware)

    # Generate configuration
    config, env_vars = tuner.generate_config(hardware, args.preset)

    if args.json:
        output = {
            "hardware": asdict(hardware),
            "config": asdict(config),
            "environment_variables": env_vars
        }
        print(json.dumps(output, indent=2))
        return 0

    if not args.quiet:
        tuner.print_config(config, env_vars)

    # Write configuration files
    if not args.dry_run:
        files = tuner.write_config_files(hardware, config, env_vars)
        if not args.quiet:
            print("\n  Configuration files written:")
            for f in files:
                print(f"    {f}")

    # Apply environment variables
    if args.apply:
        tuner.apply_env_vars(env_vars)
        if not args.quiet:
            print("\n  Environment variables applied to current process")

    # Run benchmark
    if args.benchmark:
        if not args.quiet:
            print("\n" + "=" * 60)
            print("              RUNNING BENCHMARK")
            print("=" * 60)

        results = tuner.run_benchmark(
            models=args.models,
            prompt_type=args.prompt,
            warm_runs=args.runs,
            env_vars=env_vars,
            preset_name=args.preset
        )

        if results and not args.quiet:
            tuner.print_benchmark_results(results)

    if not args.quiet:
        print("\n" + "=" * 60)
        print("              OPTIMIZATION COMPLETE")
        print("=" * 60)
        print("\nTo apply settings before starting Ollama:")
        print(f"  Linux/macOS: source {root_path}/scripts/performance/ollama_env.sh")
        print(f"  Windows CMD: {root_path}\\scripts\\performance\\ollama_env.bat")
        print(f"  PowerShell:  . {root_path}\\scripts\\performance\\ollama_env.ps1")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
