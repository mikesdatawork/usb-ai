#!/usr/bin/env python3
"""
gpu_optimizer.py

GPU acceleration optimizer for USB-AI portable LLM inference.
Maximizes GPU utilization for optimal inference performance.

Features:
    - GPU detection (NVIDIA CUDA, AMD ROCm, Apple Metal)
    - VRAM-based layer calculation
    - Flash attention configuration
    - GPU memory management
    - Benchmark GPU vs CPU inference
    - Platform-specific optimizations
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    NONE = "none"


class AccelerationType(Enum):
    """GPU acceleration backend types."""
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    VULKAN = "vulkan"
    CPU = "cpu"


@dataclass
class GPUDevice:
    """Detailed GPU device information."""
    index: int
    name: str
    vendor: GPUVendor
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    compute_capability: str
    driver_version: str
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    utilization_percent: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    cuda_cores: int = 0
    tensor_cores: int = 0
    is_integrated: bool = False


@dataclass
class GPUCapabilities:
    """System GPU capabilities summary."""
    has_gpu: bool
    acceleration_type: AccelerationType
    devices: List[GPUDevice]
    total_vram_gb: float
    free_vram_gb: float
    supports_flash_attention: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    supports_int4: bool
    max_batch_size: int
    recommended_layers: int
    cuda_graphs_supported: bool = False
    tensorrt_available: bool = False
    metal_version: Optional[str] = None


@dataclass
class VRAMEstimate:
    """VRAM requirements for a model."""
    model_name: str
    parameter_count: str
    quantization: str
    model_size_gb: float
    context_buffer_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_required_gb: float
    fits_in_vram: bool
    max_gpu_layers: int
    recommended_gpu_layers: int


@dataclass
class BenchmarkResult:
    """Benchmark results for inference."""
    mode: str  # "gpu" or "cpu"
    model_name: str
    tokens_per_second: float
    prompt_eval_tokens_per_second: float
    total_time_seconds: float
    first_token_latency_ms: float
    memory_used_gb: float
    gpu_layers: int


@dataclass
class GPUOptimizationConfig:
    """GPU optimization configuration."""
    enabled: bool
    acceleration_type: AccelerationType
    gpu_layers: int
    main_gpu: int
    flash_attention: bool
    cuda_graphs: bool
    kv_cache_type: str  # "f16", "f32", "q8_0", "q4_0"
    batch_size: int
    num_gpu: int
    tensor_split: List[float]
    low_vram_mode: bool
    mmap: bool
    mlock: bool
    numa: bool
    environment_vars: Dict[str, str]


class NVIDIADetector:
    """Detects NVIDIA GPU capabilities using nvidia-smi and CUDA."""

    @staticmethod
    def is_available() -> bool:
        """Check if NVIDIA GPU tools are available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def detect_devices(self) -> List[GPUDevice]:
        """Detect all NVIDIA GPU devices."""
        devices = []

        if not self.is_available():
            return devices

        try:
            # Query comprehensive GPU information
            query_fields = [
                "index",
                "name",
                "memory.total",
                "memory.free",
                "memory.used",
                "compute_cap",
                "driver_version",
                "temperature.gpu",
                "power.draw",
                "power.limit",
                "utilization.gpu",
            ]

            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(query_fields)}",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return devices

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue

                try:
                    device = GPUDevice(
                        index=int(parts[0]),
                        name=parts[1],
                        vendor=GPUVendor.NVIDIA,
                        vram_total_gb=float(parts[2]) / 1024,
                        vram_free_gb=float(parts[3]) / 1024,
                        vram_used_gb=float(parts[4]) / 1024,
                        compute_capability=parts[5],
                        driver_version=parts[6],
                        temperature_c=float(parts[7]) if len(parts) > 7 and parts[7] != "[N/A]" else None,
                        power_draw_w=float(parts[8]) if len(parts) > 8 and parts[8] != "[N/A]" else None,
                        power_limit_w=float(parts[9]) if len(parts) > 9 and parts[9] != "[N/A]" else None,
                        utilization_percent=float(parts[10]) if len(parts) > 10 and parts[10] != "[N/A]" else 0.0,
                    )
                    devices.append(device)
                except (ValueError, IndexError) as e:
                    log.warning(f"Error parsing NVIDIA GPU info: {e}")

            # Get additional info like CUDA cores
            self._enrich_device_info(devices)

        except Exception as e:
            log.warning(f"NVIDIA detection error: {e}")

        return devices

    def _enrich_device_info(self, devices: List[GPUDevice]):
        """Add additional device information."""
        # CUDA core counts by architecture (approximate)
        cuda_cores_by_arch = {
            "8.9": {"sm_count_multiplier": 128},  # Ada Lovelace
            "8.6": {"sm_count_multiplier": 128},  # Ampere (consumer)
            "8.0": {"sm_count_multiplier": 64},   # Ampere (datacenter)
            "7.5": {"sm_count_multiplier": 64},   # Turing
            "7.0": {"sm_count_multiplier": 64},   # Volta
            "6.1": {"sm_count_multiplier": 128},  # Pascal (consumer)
            "6.0": {"sm_count_multiplier": 64},   # Pascal (datacenter)
        }

        for device in devices:
            # Estimate memory bandwidth based on common GPU configs
            if "4090" in device.name:
                device.memory_bandwidth_gbps = 1008
                device.cuda_cores = 16384
                device.tensor_cores = 512
            elif "4080" in device.name:
                device.memory_bandwidth_gbps = 717
                device.cuda_cores = 9728
                device.tensor_cores = 304
            elif "4070" in device.name:
                device.memory_bandwidth_gbps = 504
                device.cuda_cores = 5888
                device.tensor_cores = 184
            elif "3090" in device.name:
                device.memory_bandwidth_gbps = 936
                device.cuda_cores = 10496
                device.tensor_cores = 328
            elif "3080" in device.name:
                device.memory_bandwidth_gbps = 760
                device.cuda_cores = 8704
                device.tensor_cores = 272
            elif "3070" in device.name:
                device.memory_bandwidth_gbps = 448
                device.cuda_cores = 5888
                device.tensor_cores = 184
            elif "A100" in device.name:
                device.memory_bandwidth_gbps = 2039
                device.cuda_cores = 6912
                device.tensor_cores = 432
            elif "A10" in device.name:
                device.memory_bandwidth_gbps = 600
                device.cuda_cores = 9216
                device.tensor_cores = 288

    def check_cuda_graphs_support(self) -> bool:
        """Check if CUDA graphs are supported (requires CUDA 10.0+)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip().split(".")[0]
                return int(version) >= 450  # CUDA 11+ for good graph support
        except:
            pass
        return False

    def check_tensorrt_available(self) -> bool:
        """Check if TensorRT is available."""
        try:
            result = subprocess.run(
                ["python3", "-c", "import tensorrt"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False


class AMDDetector:
    """Detects AMD GPU capabilities using ROCm."""

    @staticmethod
    def is_available() -> bool:
        """Check if ROCm tools are available."""
        try:
            result = subprocess.run(
                ["rocm-smi"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def detect_devices(self) -> List[GPUDevice]:
        """Detect all AMD GPU devices."""
        devices = []

        if not self.is_available():
            return devices

        try:
            # Get GPU list
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return devices

            # Parse GPU names
            gpu_lines = []
            for line in result.stdout.split("\n"):
                if "GPU" in line and ":" in line:
                    gpu_lines.append(line)

            # Get memory info
            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True
            )

            vram_info = self._parse_vram_info(mem_result.stdout if mem_result.returncode == 0 else "")

            for idx, line in enumerate(gpu_lines):
                name_match = re.search(r":\s*(.+)$", line)
                name = name_match.group(1).strip() if name_match else f"AMD GPU {idx}"

                vram_total = vram_info.get(idx, {}).get("total", 8.0)
                vram_used = vram_info.get(idx, {}).get("used", 0.0)

                device = GPUDevice(
                    index=idx,
                    name=name,
                    vendor=GPUVendor.AMD,
                    vram_total_gb=vram_total,
                    vram_free_gb=vram_total - vram_used,
                    vram_used_gb=vram_used,
                    compute_capability="ROCm",
                    driver_version=self._get_rocm_version(),
                )
                devices.append(device)

        except Exception as e:
            log.warning(f"AMD detection error: {e}")

        return devices

    def _parse_vram_info(self, output: str) -> Dict[int, Dict[str, float]]:
        """Parse ROCm VRAM info output."""
        info = {}
        current_gpu = -1

        for line in output.split("\n"):
            gpu_match = re.search(r"GPU\[(\d+)\]", line)
            if gpu_match:
                current_gpu = int(gpu_match.group(1))
                info[current_gpu] = {"total": 0.0, "used": 0.0}

            if current_gpu >= 0:
                total_match = re.search(r"Total Memory \(B\):\s*(\d+)", line)
                if total_match:
                    info[current_gpu]["total"] = int(total_match.group(1)) / 1024 / 1024 / 1024

                used_match = re.search(r"Used Memory \(B\):\s*(\d+)", line)
                if used_match:
                    info[current_gpu]["used"] = int(used_match.group(1)) / 1024 / 1024 / 1024

        return info

    def _get_rocm_version(self) -> str:
        """Get ROCm version."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showversion"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
        except:
            pass
        return "unknown"


class AppleSiliconDetector:
    """Detects Apple Silicon GPU capabilities."""

    @staticmethod
    def is_available() -> bool:
        """Check if running on Apple Silicon."""
        return (
            platform.system().lower() == "darwin" and
            platform.machine().lower() in ("arm64", "aarch64")
        )

    def detect_devices(self) -> List[GPUDevice]:
        """Detect Apple Silicon GPU."""
        devices = []

        if not self.is_available():
            return devices

        try:
            # Get chip model
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            chip_name = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"

            # Get total memory (unified architecture)
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True
            )
            total_mem_gb = 8.0
            if result.returncode == 0:
                total_mem_gb = int(result.stdout.strip()) / 1024 / 1024 / 1024

            # Estimate GPU-available memory (unified memory, ~75% available to GPU)
            # This varies by chip - M1 Max/Ultra can dedicate more
            gpu_memory_ratio = self._get_gpu_memory_ratio(chip_name)
            gpu_vram = total_mem_gb * gpu_memory_ratio

            # Get Metal version
            metal_version = self._get_metal_version()

            device = GPUDevice(
                index=0,
                name=f"{chip_name} (Metal)",
                vendor=GPUVendor.APPLE,
                vram_total_gb=gpu_vram,
                vram_free_gb=gpu_vram * 0.8,  # Approximate
                vram_used_gb=gpu_vram * 0.2,
                compute_capability=f"Metal {metal_version}",
                driver_version=metal_version,
                is_integrated=True,
            )

            # Estimate GPU cores based on chip
            self._set_apple_gpu_specs(device, chip_name)

            devices.append(device)

        except Exception as e:
            log.warning(f"Apple Silicon detection error: {e}")

        return devices

    def _get_gpu_memory_ratio(self, chip_name: str) -> float:
        """Get GPU memory ratio based on chip type."""
        chip_lower = chip_name.lower()

        if "ultra" in chip_lower:
            return 0.80
        elif "max" in chip_lower:
            return 0.75
        elif "pro" in chip_lower:
            return 0.70
        else:
            return 0.65  # Base M1/M2/M3

    def _get_metal_version(self) -> str:
        """Get Metal API version."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Look for Metal version info
                match = re.search(r"Metal.*?:\s*([\w\s.]+)", result.stdout)
                if match:
                    return match.group(1).strip()
        except:
            pass
        return "3"  # Assume Metal 3 for recent chips

    def _set_apple_gpu_specs(self, device: GPUDevice, chip_name: str):
        """Set GPU specs based on Apple chip."""
        chip_lower = chip_name.lower()

        # GPU core counts by chip
        if "m3 ultra" in chip_lower:
            device.cuda_cores = 76  # GPU cores
            device.memory_bandwidth_gbps = 800
        elif "m3 max" in chip_lower:
            device.cuda_cores = 40
            device.memory_bandwidth_gbps = 400
        elif "m3 pro" in chip_lower:
            device.cuda_cores = 18
            device.memory_bandwidth_gbps = 200
        elif "m3" in chip_lower:
            device.cuda_cores = 10
            device.memory_bandwidth_gbps = 100
        elif "m2 ultra" in chip_lower:
            device.cuda_cores = 76
            device.memory_bandwidth_gbps = 800
        elif "m2 max" in chip_lower:
            device.cuda_cores = 38
            device.memory_bandwidth_gbps = 400
        elif "m2 pro" in chip_lower:
            device.cuda_cores = 19
            device.memory_bandwidth_gbps = 200
        elif "m2" in chip_lower:
            device.cuda_cores = 10
            device.memory_bandwidth_gbps = 100
        elif "m1 ultra" in chip_lower:
            device.cuda_cores = 64
            device.memory_bandwidth_gbps = 800
        elif "m1 max" in chip_lower:
            device.cuda_cores = 32
            device.memory_bandwidth_gbps = 400
        elif "m1 pro" in chip_lower:
            device.cuda_cores = 16
            device.memory_bandwidth_gbps = 200
        else:
            device.cuda_cores = 8
            device.memory_bandwidth_gbps = 68


class GPUDetector:
    """Unified GPU detection across all platforms."""

    def __init__(self):
        self.nvidia_detector = NVIDIADetector()
        self.amd_detector = AMDDetector()
        self.apple_detector = AppleSiliconDetector()

    def detect(self) -> GPUCapabilities:
        """Detect all GPU capabilities."""
        devices: List[GPUDevice] = []
        acceleration_type = AccelerationType.CPU

        # Try NVIDIA first (most common for ML)
        nvidia_devices = self.nvidia_detector.detect_devices()
        if nvidia_devices:
            devices.extend(nvidia_devices)
            acceleration_type = AccelerationType.CUDA

        # Try AMD
        amd_devices = self.amd_detector.detect_devices()
        if amd_devices:
            devices.extend(amd_devices)
            if acceleration_type == AccelerationType.CPU:
                acceleration_type = AccelerationType.ROCM

        # Try Apple Silicon
        apple_devices = self.apple_detector.detect_devices()
        if apple_devices:
            devices.extend(apple_devices)
            if acceleration_type == AccelerationType.CPU:
                acceleration_type = AccelerationType.METAL

        # Calculate totals
        total_vram = sum(d.vram_total_gb for d in devices)
        free_vram = sum(d.vram_free_gb for d in devices)

        # Determine capabilities
        supports_flash_attention = self._check_flash_attention_support(devices, acceleration_type)
        supports_fp16 = len(devices) > 0
        supports_bf16 = self._check_bf16_support(devices)
        supports_int8 = len(devices) > 0
        supports_int4 = len(devices) > 0

        # Calculate recommended batch size based on VRAM
        max_batch_size = self._calculate_max_batch_size(free_vram)

        # Calculate recommended GPU layers
        recommended_layers = self._calculate_recommended_layers(free_vram)

        # Check additional features
        cuda_graphs = (
            acceleration_type == AccelerationType.CUDA and
            self.nvidia_detector.check_cuda_graphs_support()
        )
        tensorrt = (
            acceleration_type == AccelerationType.CUDA and
            self.nvidia_detector.check_tensorrt_available()
        )

        # Get Metal version for Apple
        metal_version = None
        if acceleration_type == AccelerationType.METAL and apple_devices:
            metal_version = apple_devices[0].compute_capability

        return GPUCapabilities(
            has_gpu=len(devices) > 0,
            acceleration_type=acceleration_type,
            devices=devices,
            total_vram_gb=total_vram,
            free_vram_gb=free_vram,
            supports_flash_attention=supports_flash_attention,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            supports_int8=supports_int8,
            supports_int4=supports_int4,
            max_batch_size=max_batch_size,
            recommended_layers=recommended_layers,
            cuda_graphs_supported=cuda_graphs,
            tensorrt_available=tensorrt,
            metal_version=metal_version,
        )

    def _check_flash_attention_support(
        self,
        devices: List[GPUDevice],
        accel_type: AccelerationType
    ) -> bool:
        """Check if flash attention is supported."""
        if not devices:
            return False

        if accel_type == AccelerationType.CUDA:
            # Flash attention requires compute capability 7.0+ (Volta or newer)
            for device in devices:
                try:
                    major = int(device.compute_capability.split(".")[0])
                    if major >= 7:
                        return True
                except:
                    pass
            return False

        if accel_type == AccelerationType.METAL:
            # Metal supports flash attention on M1 and later
            return True

        if accel_type == AccelerationType.ROCM:
            # ROCm supports flash attention on recent GPUs
            return True

        return False

    def _check_bf16_support(self, devices: List[GPUDevice]) -> bool:
        """Check if BF16 is supported."""
        for device in devices:
            if device.vendor == GPUVendor.NVIDIA:
                try:
                    major = int(device.compute_capability.split(".")[0])
                    if major >= 8:  # Ampere or newer
                        return True
                except:
                    pass
            elif device.vendor == GPUVendor.APPLE:
                return True  # M1+ supports BF16
        return False

    def _calculate_max_batch_size(self, free_vram_gb: float) -> int:
        """Calculate maximum batch size based on available VRAM."""
        if free_vram_gb >= 24:
            return 2048
        elif free_vram_gb >= 16:
            return 1024
        elif free_vram_gb >= 8:
            return 512
        elif free_vram_gb >= 4:
            return 256
        else:
            return 128

    def _calculate_recommended_layers(self, free_vram_gb: float) -> int:
        """Calculate recommended GPU layers based on VRAM."""
        # Approximate: each layer needs ~0.1-0.2 GB for 7B model
        # More for larger models
        if free_vram_gb >= 24:
            return 99  # Full offload
        elif free_vram_gb >= 16:
            return 50
        elif free_vram_gb >= 8:
            return 35
        elif free_vram_gb >= 4:
            return 20
        elif free_vram_gb >= 2:
            return 10
        else:
            return 0


class VRAMCalculator:
    """Calculates VRAM requirements for models."""

    # Model size estimates in GB at different quantizations
    MODEL_SIZES = {
        "1B": {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.6, "q4_k_m": 0.7},
        "3B": {"f16": 6.0, "q8_0": 3.0, "q4_0": 1.8, "q4_k_m": 2.0},
        "7B": {"f16": 14.0, "q8_0": 7.0, "q4_0": 4.0, "q4_k_m": 4.5},
        "8B": {"f16": 16.0, "q8_0": 8.0, "q4_0": 4.5, "q4_k_m": 5.0},
        "13B": {"f16": 26.0, "q8_0": 13.0, "q4_0": 7.5, "q4_k_m": 8.5},
        "14B": {"f16": 28.0, "q8_0": 14.0, "q4_0": 8.0, "q4_k_m": 9.0},
        "33B": {"f16": 66.0, "q8_0": 33.0, "q4_0": 19.0, "q4_k_m": 21.0},
        "70B": {"f16": 140.0, "q8_0": 70.0, "q4_0": 40.0, "q4_k_m": 45.0},
    }

    # Layer counts by model size (approximate)
    MODEL_LAYERS = {
        "1B": 22,
        "3B": 26,
        "7B": 32,
        "8B": 32,
        "13B": 40,
        "14B": 40,
        "33B": 60,
        "70B": 80,
    }

    def __init__(self, available_vram_gb: float):
        self.available_vram = available_vram_gb

    def estimate(
        self,
        model_name: str,
        parameter_count: str,
        quantization: str = "q4_k_m",
        context_length: int = 4096
    ) -> VRAMEstimate:
        """Estimate VRAM requirements for a model."""
        # Get base model size
        param_key = self._normalize_param_count(parameter_count)
        quant_key = self._normalize_quantization(quantization)

        if param_key not in self.MODEL_SIZES:
            log.warning(f"Unknown parameter count: {parameter_count}, using 7B estimate")
            param_key = "7B"

        sizes = self.MODEL_SIZES[param_key]
        model_size = sizes.get(quant_key, sizes.get("q4_k_m", 4.5))

        # Calculate context buffer (KV cache)
        # KV cache size = 2 * num_layers * context_length * hidden_dim * dtype_size
        # Approximate: ~0.5MB per 1K context for 7B model at fp16
        num_layers = self.MODEL_LAYERS.get(param_key, 32)
        context_factor = context_length / 4096  # Normalized to 4K context
        kv_cache_gb = 0.5 * (num_layers / 32) * context_factor

        # Overhead for activations, gradients, etc.
        overhead_gb = model_size * 0.1  # ~10% overhead

        # Total required
        total_required = model_size + kv_cache_gb + overhead_gb

        # Calculate max layers that fit
        total_layers = self.MODEL_LAYERS.get(param_key, 32)
        vram_per_layer = model_size / total_layers

        # Reserve VRAM for KV cache and overhead
        usable_vram = max(0, self.available_vram - kv_cache_gb - overhead_gb - 0.5)  # 0.5GB safety margin
        max_gpu_layers = min(total_layers, int(usable_vram / vram_per_layer))

        # Recommended layers (leave some headroom)
        recommended_layers = max(0, int(max_gpu_layers * 0.9))

        return VRAMEstimate(
            model_name=model_name,
            parameter_count=parameter_count,
            quantization=quantization,
            model_size_gb=model_size,
            context_buffer_gb=kv_cache_gb,
            kv_cache_gb=kv_cache_gb,
            overhead_gb=overhead_gb,
            total_required_gb=total_required,
            fits_in_vram=total_required <= self.available_vram,
            max_gpu_layers=max_gpu_layers,
            recommended_gpu_layers=recommended_layers,
        )

    def _normalize_param_count(self, param_count: str) -> str:
        """Normalize parameter count string."""
        param_count = param_count.upper().replace(" ", "")
        # Extract numeric part
        match = re.search(r"(\d+\.?\d*)", param_count)
        if match:
            num = float(match.group(1))
            if num < 2:
                return "1B"
            elif num < 5:
                return "3B"
            elif num < 7.5:
                return "7B"
            elif num < 10:
                return "8B"
            elif num < 14:
                return "13B"
            elif num < 20:
                return "14B"
            elif num < 50:
                return "33B"
            else:
                return "70B"
        return "7B"

    def _normalize_quantization(self, quant: str) -> str:
        """Normalize quantization string."""
        quant = quant.lower().replace("-", "_")
        if "f16" in quant or "fp16" in quant:
            return "f16"
        elif "q8" in quant:
            return "q8_0"
        elif "q4_k" in quant:
            return "q4_k_m"
        elif "q4" in quant:
            return "q4_0"
        return "q4_k_m"


class GPUBenchmark:
    """Benchmarks GPU vs CPU inference performance."""

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host

    def run_benchmark(
        self,
        model_name: str = "llama3.2:1b",
        prompt: str = "Write a short poem about computers.",
        gpu_layers: int = 99
    ) -> Tuple[Optional[BenchmarkResult], Optional[BenchmarkResult]]:
        """Run benchmark comparing GPU and CPU inference."""
        import urllib.request
        import urllib.error

        # Check if Ollama is running
        if not self._check_ollama():
            log.error("Ollama is not running. Start Ollama first.")
            return None, None

        # Check if model is available
        if not self._check_model(model_name):
            log.warning(f"Model {model_name} not available. Attempting to pull...")
            if not self._pull_model(model_name):
                log.error(f"Could not pull model {model_name}")
                return None, None

        # Run GPU benchmark
        log.info(f"Running GPU benchmark with {gpu_layers} layers...")
        gpu_result = self._run_inference(model_name, prompt, gpu_layers, "gpu")

        # Run CPU benchmark
        log.info("Running CPU benchmark (0 GPU layers)...")
        cpu_result = self._run_inference(model_name, prompt, 0, "cpu")

        return gpu_result, cpu_result

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except:
            return False

    def _check_model(self, model_name: str) -> bool:
        """Check if model is available."""
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(model_name in m for m in models)
        except:
            return False

    def _pull_model(self, model_name: str) -> bool:
        """Pull model using Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=600
            )
            return result.returncode == 0
        except:
            return False

    def _run_inference(
        self,
        model_name: str,
        prompt: str,
        gpu_layers: int,
        mode: str
    ) -> Optional[BenchmarkResult]:
        """Run inference and measure performance."""
        import urllib.request

        try:
            # Set GPU layers via modelfile options
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_gpu": gpu_layers,
                }
            }

            data = json.dumps(request_data).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            start_time = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            total_time = time.time() - start_time

            # Extract metrics
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 1) / 1e9  # Convert to seconds
            prompt_eval_count = result.get("prompt_eval_count", 0)
            prompt_eval_duration = result.get("prompt_eval_duration", 1) / 1e9

            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0
            prompt_tokens_per_second = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0

            # First token latency (approximation)
            first_token_latency = (prompt_eval_duration * 1000) if prompt_eval_duration > 0 else 0

            return BenchmarkResult(
                mode=mode,
                model_name=model_name,
                tokens_per_second=tokens_per_second,
                prompt_eval_tokens_per_second=prompt_tokens_per_second,
                total_time_seconds=total_time,
                first_token_latency_ms=first_token_latency,
                memory_used_gb=0.0,  # Would need to query separately
                gpu_layers=gpu_layers,
            )

        except Exception as e:
            log.error(f"Benchmark error ({mode}): {e}")
            return None


class GPUOptimizer:
    """Main GPU optimization class."""

    def __init__(self):
        self.detector = GPUDetector()
        self.capabilities: Optional[GPUCapabilities] = None

    def detect(self) -> GPUCapabilities:
        """Detect GPU capabilities."""
        self.capabilities = self.detector.detect()
        return self.capabilities

    def optimize(
        self,
        profile: str = "auto",
        model_size: str = "8B",
        context_length: int = 4096
    ) -> GPUOptimizationConfig:
        """Generate optimized GPU configuration."""
        if self.capabilities is None:
            self.detect()

        caps = self.capabilities

        # Load profile
        profile_config = self._load_profile(profile)

        # Determine if GPU should be enabled
        enabled = caps.has_gpu and profile_config.get("enabled", True)

        # Calculate GPU layers
        if not enabled:
            gpu_layers = 0
        elif profile == "full_offload":
            gpu_layers = 99
        elif profile == "cpu_only":
            gpu_layers = 0
        elif profile == "hybrid":
            # Calculate based on VRAM
            calculator = VRAMCalculator(caps.free_vram_gb)
            estimate = calculator.estimate("model", model_size, "q4_k_m", context_length)
            gpu_layers = estimate.recommended_gpu_layers
        else:  # auto
            if caps.free_vram_gb >= 16:
                gpu_layers = 99
            elif caps.free_vram_gb >= 8:
                gpu_layers = 50
            elif caps.free_vram_gb >= 4:
                gpu_layers = 25
            else:
                gpu_layers = caps.recommended_layers

        # Flash attention
        flash_attention = caps.supports_flash_attention and enabled

        # CUDA graphs (only for NVIDIA)
        cuda_graphs = caps.cuda_graphs_supported and enabled

        # KV cache type
        if caps.supports_bf16:
            kv_cache_type = "f16"  # BF16 for compute, FP16 for KV cache
        elif caps.supports_fp16:
            kv_cache_type = "f16"
        else:
            kv_cache_type = "f32"

        # Batch size
        batch_size = min(caps.max_batch_size, profile_config.get("batch_size", 512))

        # Number of GPUs to use
        num_gpu = len(caps.devices) if enabled else 0

        # Tensor split for multi-GPU
        tensor_split = []
        if num_gpu > 1:
            total_vram = sum(d.vram_total_gb for d in caps.devices)
            tensor_split = [d.vram_total_gb / total_vram for d in caps.devices]

        # Low VRAM mode
        low_vram_mode = caps.free_vram_gb < 4 and enabled

        # Memory settings
        mmap = True  # Good for USB loading
        mlock = caps.free_vram_gb >= 8 and profile not in ("minimal", "cpu_only")
        numa = False  # Usually not needed for single-socket systems

        # Build environment variables
        env_vars = self._build_environment(
            caps, enabled, gpu_layers, flash_attention, cuda_graphs, batch_size
        )

        return GPUOptimizationConfig(
            enabled=enabled,
            acceleration_type=caps.acceleration_type if enabled else AccelerationType.CPU,
            gpu_layers=gpu_layers,
            main_gpu=0,
            flash_attention=flash_attention,
            cuda_graphs=cuda_graphs,
            kv_cache_type=kv_cache_type,
            batch_size=batch_size,
            num_gpu=num_gpu,
            tensor_split=tensor_split,
            low_vram_mode=low_vram_mode,
            mmap=mmap,
            mlock=mlock,
            numa=numa,
            environment_vars=env_vars,
        )

    def _load_profile(self, profile: str) -> Dict[str, Any]:
        """Load GPU profile configuration."""
        profiles = {
            "full_offload": {
                "enabled": True,
                "gpu_layers": 99,
                "batch_size": 1024,
                "description": "All layers on GPU for maximum performance"
            },
            "hybrid": {
                "enabled": True,
                "gpu_layers": "auto",
                "batch_size": 512,
                "description": "Split CPU/GPU based on VRAM"
            },
            "cpu_only": {
                "enabled": False,
                "gpu_layers": 0,
                "batch_size": 256,
                "description": "No GPU, optimized CPU inference"
            },
            "auto": {
                "enabled": True,
                "gpu_layers": "auto",
                "batch_size": 512,
                "description": "Automatically detect and configure"
            }
        }

        # Try to load from YAML file
        yaml_path = self._find_profiles_yaml()
        if yaml_path and HAS_YAML:
            try:
                with open(yaml_path) as f:
                    file_profiles = yaml.safe_load(f)
                    if file_profiles and "profiles" in file_profiles:
                        profiles.update(file_profiles["profiles"])
            except Exception as e:
                log.warning(f"Could not load GPU profiles YAML: {e}")

        return profiles.get(profile, profiles["auto"])

    def _find_profiles_yaml(self) -> Optional[Path]:
        """Find GPU profiles YAML file."""
        search_paths = [
            Path(__file__).parent.parent.parent / "modules" / "config" / "gpu_profiles.yaml",
            Path.cwd() / "modules" / "config" / "gpu_profiles.yaml",
        ]
        for path in search_paths:
            if path.exists():
                return path
        return None

    def _build_environment(
        self,
        caps: GPUCapabilities,
        enabled: bool,
        gpu_layers: int,
        flash_attention: bool,
        cuda_graphs: bool,
        batch_size: int
    ) -> Dict[str, str]:
        """Build environment variables for Ollama."""
        env = {}

        # Basic Ollama settings
        env["OLLAMA_NUM_GPU"] = str(len(caps.devices)) if enabled else "0"
        env["OLLAMA_GPU_LAYERS"] = str(gpu_layers)
        env["OLLAMA_FLASH_ATTENTION"] = "1" if flash_attention else "0"

        # Platform-specific settings
        if caps.acceleration_type == AccelerationType.CUDA:
            # NVIDIA CUDA settings
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d.index) for d in caps.devices)

            if cuda_graphs:
                env["OLLAMA_CUDA_GRAPHS"] = "1"

            # TensorRT hints (if available)
            if caps.tensorrt_available:
                env["OLLAMA_USE_TENSORRT"] = "1"

            # Memory management
            env["CUDA_LAUNCH_BLOCKING"] = "0"  # Async for performance

        elif caps.acceleration_type == AccelerationType.ROCM:
            # AMD ROCm settings
            env["HIP_VISIBLE_DEVICES"] = ",".join(str(d.index) for d in caps.devices)
            env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"  # Compatibility

        elif caps.acceleration_type == AccelerationType.METAL:
            # Apple Metal settings
            env["OLLAMA_METAL"] = "1"
            env["OLLAMA_METAL_ASYNC"] = "1"  # Async Metal for performance

        else:
            # CPU fallback - optimize for AVX
            env["OLLAMA_NUM_GPU"] = "0"
            # Check for AVX support
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                    if "avx512" in cpuinfo.lower():
                        env["OLLAMA_AVX"] = "512"
                    elif "avx2" in cpuinfo.lower():
                        env["OLLAMA_AVX"] = "2"
            except:
                pass

        return env


def find_root() -> Path:
    """Locate USB-AI root directory."""
    script_dir = Path(__file__).parent.resolve()

    if (script_dir.parent.parent / "modules").exists():
        return script_dir.parent.parent
    if (script_dir.parent / "modules").exists():
        return script_dir.parent
    if (script_dir / "modules").exists():
        return script_dir

    return Path.cwd()


def print_capabilities(caps: GPUCapabilities):
    """Print GPU capabilities report."""
    print("")
    print("=" * 70)
    print("                    GPU ACCELERATION ANALYSIS")
    print("=" * 70)
    print("")

    if caps.has_gpu:
        print(f"  GPU Detected:       Yes")
        print(f"  Acceleration Type:  {caps.acceleration_type.value.upper()}")
        print(f"  Total VRAM:         {caps.total_vram_gb:.1f} GB")
        print(f"  Free VRAM:          {caps.free_vram_gb:.1f} GB")
        print("")

        for device in caps.devices:
            print(f"  Device {device.index}: {device.name}")
            print(f"    Vendor:           {device.vendor.value}")
            print(f"    VRAM:             {device.vram_total_gb:.1f} GB ({device.vram_free_gb:.1f} GB free)")
            print(f"    Compute:          {device.compute_capability}")
            if device.driver_version:
                print(f"    Driver:           {device.driver_version}")
            if device.temperature_c:
                print(f"    Temperature:      {device.temperature_c:.0f}C")
            if device.memory_bandwidth_gbps > 0:
                print(f"    Memory BW:        {device.memory_bandwidth_gbps:.0f} GB/s")
            print("")
    else:
        print("  GPU Detected:       No")
        print("  Acceleration Type:  CPU")
        print("")

    print("  Capabilities:")
    print(f"    Flash Attention:  {'Yes' if caps.supports_flash_attention else 'No'}")
    print(f"    FP16:             {'Yes' if caps.supports_fp16 else 'No'}")
    print(f"    BF16:             {'Yes' if caps.supports_bf16 else 'No'}")
    print(f"    INT8:             {'Yes' if caps.supports_int8 else 'No'}")
    print(f"    INT4:             {'Yes' if caps.supports_int4 else 'No'}")
    if caps.cuda_graphs_supported:
        print(f"    CUDA Graphs:      Yes")
    if caps.tensorrt_available:
        print(f"    TensorRT:         Yes")
    if caps.metal_version:
        print(f"    Metal:            {caps.metal_version}")
    print("")

    print(f"  Recommended:")
    print(f"    Max Batch Size:   {caps.max_batch_size}")
    print(f"    GPU Layers:       {caps.recommended_layers}")
    print("")


def print_vram_estimate(estimate: VRAMEstimate, available_vram: float):
    """Print VRAM estimate."""
    print("")
    print(f"  VRAM Estimate for {estimate.model_name} ({estimate.parameter_count}):")
    print(f"    Quantization:     {estimate.quantization}")
    print(f"    Model Size:       {estimate.model_size_gb:.1f} GB")
    print(f"    KV Cache:         {estimate.kv_cache_gb:.1f} GB")
    print(f"    Overhead:         {estimate.overhead_gb:.1f} GB")
    print(f"    Total Required:   {estimate.total_required_gb:.1f} GB")
    print(f"    Available VRAM:   {available_vram:.1f} GB")
    print("")

    if estimate.fits_in_vram:
        print(f"    Status:           FITS (full GPU offload possible)")
    else:
        print(f"    Status:           PARTIAL (hybrid CPU/GPU)")
    print(f"    Max GPU Layers:   {estimate.max_gpu_layers}")
    print(f"    Recommended:      {estimate.recommended_gpu_layers} layers")
    print("")


def print_benchmark_results(
    gpu_result: Optional[BenchmarkResult],
    cpu_result: Optional[BenchmarkResult]
):
    """Print benchmark comparison."""
    print("")
    print("=" * 70)
    print("                    BENCHMARK RESULTS")
    print("=" * 70)
    print("")

    if gpu_result:
        print("  GPU Inference:")
        print(f"    GPU Layers:           {gpu_result.gpu_layers}")
        print(f"    Tokens/sec:           {gpu_result.tokens_per_second:.1f}")
        print(f"    Prompt Tokens/sec:    {gpu_result.prompt_eval_tokens_per_second:.1f}")
        print(f"    First Token Latency:  {gpu_result.first_token_latency_ms:.0f} ms")
        print(f"    Total Time:           {gpu_result.total_time_seconds:.2f} s")
        print("")
    else:
        print("  GPU Inference:          Failed or unavailable")
        print("")

    if cpu_result:
        print("  CPU Inference:")
        print(f"    GPU Layers:           {cpu_result.gpu_layers}")
        print(f"    Tokens/sec:           {cpu_result.tokens_per_second:.1f}")
        print(f"    Prompt Tokens/sec:    {cpu_result.prompt_eval_tokens_per_second:.1f}")
        print(f"    First Token Latency:  {cpu_result.first_token_latency_ms:.0f} ms")
        print(f"    Total Time:           {cpu_result.total_time_seconds:.2f} s")
        print("")
    else:
        print("  CPU Inference:          Failed")
        print("")

    if gpu_result and cpu_result and cpu_result.tokens_per_second > 0:
        speedup = gpu_result.tokens_per_second / cpu_result.tokens_per_second
        print(f"  GPU Speedup:            {speedup:.1f}x faster")
        print("")


def print_config(config: GPUOptimizationConfig):
    """Print optimization configuration."""
    print("")
    print("=" * 70)
    print("                    GPU OPTIMIZATION CONFIG")
    print("=" * 70)
    print("")

    print(f"  Enabled:            {config.enabled}")
    print(f"  Acceleration:       {config.acceleration_type.value.upper()}")
    print(f"  GPU Layers:         {config.gpu_layers}")
    print(f"  Main GPU:           {config.main_gpu}")
    print(f"  Num GPUs:           {config.num_gpu}")
    print(f"  Flash Attention:    {config.flash_attention}")
    print(f"  CUDA Graphs:        {config.cuda_graphs}")
    print(f"  KV Cache Type:      {config.kv_cache_type}")
    print(f"  Batch Size:         {config.batch_size}")
    print(f"  Low VRAM Mode:      {config.low_vram_mode}")
    print(f"  MMAP:               {config.mmap}")
    print(f"  MLOCK:              {config.mlock}")

    if config.tensor_split:
        print(f"  Tensor Split:       {[f'{x:.2f}' for x in config.tensor_split]}")

    print("")
    print("  Environment Variables:")
    for key, value in config.environment_vars.items():
        print(f"    {key}={value}")
    print("")


def main() -> int:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="USB-AI GPU Acceleration Optimizer"
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "full_offload", "hybrid", "cpu_only"],
        default="auto",
        help="GPU profile to use (default: auto)"
    )
    parser.add_argument(
        "--model-size",
        default="8B",
        help="Model size for VRAM calculation (default: 8B)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help="Context length for VRAM calculation (default: 4096)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run GPU vs CPU benchmark"
    )
    parser.add_argument(
        "--benchmark-model",
        default="llama3.2:1b",
        help="Model to use for benchmarking (default: llama3.2:1b)"
    )
    parser.add_argument(
        "--vram-estimate",
        action="store_true",
        help="Show VRAM estimate for model"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write configuration files"
    )

    args = parser.parse_args()

    # Detect GPU
    optimizer = GPUOptimizer()
    caps = optimizer.detect()

    if args.json:
        output = {
            "capabilities": {
                "has_gpu": caps.has_gpu,
                "acceleration_type": caps.acceleration_type.value,
                "total_vram_gb": caps.total_vram_gb,
                "free_vram_gb": caps.free_vram_gb,
                "supports_flash_attention": caps.supports_flash_attention,
                "devices": [asdict(d) for d in caps.devices],
            }
        }

        # Generate optimization config
        config = optimizer.optimize(args.profile, args.model_size, args.context)
        output["optimization"] = {
            "enabled": config.enabled,
            "acceleration_type": config.acceleration_type.value,
            "gpu_layers": config.gpu_layers,
            "flash_attention": config.flash_attention,
            "cuda_graphs": config.cuda_graphs,
            "batch_size": config.batch_size,
            "environment": config.environment_vars,
        }

        # VRAM estimate
        if caps.has_gpu:
            calculator = VRAMCalculator(caps.free_vram_gb)
            estimate = calculator.estimate("model", args.model_size, "q4_k_m", args.context)
            output["vram_estimate"] = asdict(estimate)

        print(json.dumps(output, indent=2, default=str))
        return 0

    # Print capabilities
    print_capabilities(caps)

    # VRAM estimate
    if args.vram_estimate or caps.has_gpu:
        calculator = VRAMCalculator(caps.free_vram_gb if caps.has_gpu else 0)
        estimate = calculator.estimate("model", args.model_size, "q4_k_m", args.context)
        print_vram_estimate(estimate, caps.free_vram_gb)

    # Generate and print optimization config
    config = optimizer.optimize(args.profile, args.model_size, args.context)
    print_config(config)

    # Run benchmark if requested
    if args.benchmark:
        print("Running benchmark (this may take a minute)...")
        benchmark = GPUBenchmark()
        gpu_result, cpu_result = benchmark.run_benchmark(
            args.benchmark_model,
            gpu_layers=config.gpu_layers
        )
        print_benchmark_results(gpu_result, cpu_result)

    # Write configuration if requested
    if args.write_config:
        root = find_root()
        config_dir = root / "modules" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Write GPU config JSON
        config_path = config_dir / "gpu_optimization.json"
        with open(config_path, "w") as f:
            json.dump({
                "capabilities": {
                    "has_gpu": caps.has_gpu,
                    "acceleration_type": caps.acceleration_type.value,
                    "total_vram_gb": caps.total_vram_gb,
                    "free_vram_gb": caps.free_vram_gb,
                    "devices": [{
                        "index": d.index,
                        "name": d.name,
                        "vendor": d.vendor.value,
                        "vram_gb": d.vram_total_gb,
                    } for d in caps.devices],
                },
                "optimization": {
                    "enabled": config.enabled,
                    "acceleration_type": config.acceleration_type.value,
                    "gpu_layers": config.gpu_layers,
                    "flash_attention": config.flash_attention,
                    "cuda_graphs": config.cuda_graphs,
                    "batch_size": config.batch_size,
                    "low_vram_mode": config.low_vram_mode,
                },
                "environment": config.environment_vars,
            }, f, indent=2)
        print(f"Written: {config_path}")

        # Write environment script
        scripts_dir = root / "scripts" / "performance"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        if platform.system().lower() == "windows":
            script_path = scripts_dir / "set_gpu.bat"
            lines = ["@echo off", "REM USB-AI GPU Settings"]
            for key, value in config.environment_vars.items():
                lines.append(f"set {key}={value}")
            content = "\r\n".join(lines)
        else:
            script_path = scripts_dir / "set_gpu.sh"
            lines = ["#!/bin/bash", "# USB-AI GPU Settings"]
            for key, value in config.environment_vars.items():
                lines.append(f'export {key}="{value}"')
            content = "\n".join(lines)

        with open(script_path, "w") as f:
            f.write(content)

        if platform.system().lower() != "windows":
            os.chmod(script_path, 0o755)

        print(f"Written: {script_path}")

    print("=" * 70)
    print("           GPU Optimization Complete")
    print("=" * 70)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
