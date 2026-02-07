#!/usr/bin/env python3
"""
quantize_models.py

Model quantization and optimization infrastructure for USB-AI.
Supports GGUF quantization levels with automatic selection based on USB size.

Quantization Levels:
    - Q4_K_M: 4-bit quantization, medium quality (smallest, fastest)
    - Q5_K_M: 5-bit quantization, medium quality (balanced)
    - Q8_0: 8-bit quantization (highest quality, largest)

Usage:
    python quantize_models.py --model dolphin-llama3:8b --level Q4_K_M
    python quantize_models.py --usb-size 128  # Auto-select based on USB size
    python quantize_models.py --analyze  # Show storage analysis only
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


class QuantizationLevel(Enum):
    """GGUF quantization levels with metadata."""
    Q4_K_M = ("Q4_K_M", 4, 0.45, 0.92, "4-bit medium quantization")
    Q5_K_M = ("Q5_K_M", 5, 0.55, 0.95, "5-bit medium quantization")
    Q8_0 = ("Q8_0", 8, 0.85, 0.99, "8-bit quantization")

    def __init__(self, code: str, bits: int, size_ratio: float,
                 quality_ratio: float, description: str):
        self.code = code
        self.bits = bits
        self.size_ratio = size_ratio  # Ratio compared to FP16
        self.quality_ratio = quality_ratio  # Quality retention
        self.description = description


@dataclass
class ModelSpec:
    """Model specification with quantization details."""
    name: str
    base_size_gb: float  # FP16 size
    parameters: str  # e.g., "8B", "14B"
    architecture: str
    context_length: int = 8192

    def get_quantized_size(self, level: QuantizationLevel) -> float:
        """Calculate estimated size for given quantization level."""
        return round(self.base_size_gb * level.size_ratio, 2)

    def get_memory_requirement(self, level: QuantizationLevel) -> float:
        """
        Estimate runtime memory requirement.
        Formula: quantized_size + context_memory + overhead
        """
        quantized_size = self.get_quantized_size(level)
        # Context memory: ~1MB per 1K tokens for 8B model
        context_memory_gb = (self.context_length / 1000) * 0.001 * self._param_multiplier()
        # Overhead: ~10% for KV cache and runtime buffers
        overhead = quantized_size * 0.10
        return round(quantized_size + context_memory_gb + overhead, 2)

    def _param_multiplier(self) -> float:
        """Get multiplier based on parameter count."""
        param_map = {
            "1B": 1, "3B": 3, "7B": 7, "8B": 8,
            "13B": 13, "14B": 14, "34B": 34, "70B": 70
        }
        return param_map.get(self.parameters.upper(), 8)


@dataclass
class QuantizationConfig:
    """Configuration for quantization process."""
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    default_level: QuantizationLevel = QuantizationLevel.Q4_K_M
    usb_size_gb: int = 128
    reserved_space_gb: int = 10  # For VeraCrypt, launchers, etc.

    @property
    def available_space_gb(self) -> float:
        """Space available for models after reserved space."""
        return self.usb_size_gb - self.reserved_space_gb


# Default model specifications
DEFAULT_MODELS = {
    "dolphin-llama3:8b": ModelSpec(
        name="dolphin-llama3:8b",
        base_size_gb=8.5,  # FP16 base
        parameters="8B",
        architecture="llama3",
        context_length=8192
    ),
    "llama3.2:8b": ModelSpec(
        name="llama3.2:8b",
        base_size_gb=8.5,
        parameters="8B",
        architecture="llama3.2",
        context_length=8192
    ),
    "llama3.2:latest": ModelSpec(
        name="llama3.2:latest",
        base_size_gb=4.0,  # 3B model
        parameters="3B",
        architecture="llama3.2",
        context_length=8192
    ),
    "qwen2.5:14b": ModelSpec(
        name="qwen2.5:14b",
        base_size_gb=15.0,
        parameters="14B",
        architecture="qwen2.5",
        context_length=32768
    ),
    "qwen2.5:7b": ModelSpec(
        name="qwen2.5:7b",
        base_size_gb=8.0,
        parameters="7B",
        architecture="qwen2.5",
        context_length=32768
    ),
    "mistral:7b": ModelSpec(
        name="mistral:7b",
        base_size_gb=8.0,
        parameters="7B",
        architecture="mistral",
        context_length=32768
    ),
    "codellama:7b": ModelSpec(
        name="codellama:7b",
        base_size_gb=8.0,
        parameters="7B",
        architecture="llama2",
        context_length=16384
    ),
}


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def load_quantization_config(config_path: Optional[Path] = None) -> QuantizationConfig:
    """Load quantization configuration from YAML file."""
    if config_path is None:
        config_path = get_root_path() / "modules" / "models" / "quantization_config.yaml"

    config = QuantizationConfig(models=DEFAULT_MODELS.copy())

    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                if "usb_size_gb" in yaml_config:
                    config.usb_size_gb = yaml_config["usb_size_gb"]
                if "reserved_space_gb" in yaml_config:
                    config.reserved_space_gb = yaml_config["reserved_space_gb"]
                if "default_level" in yaml_config:
                    level_name = yaml_config["default_level"]
                    for level in QuantizationLevel:
                        if level.code == level_name:
                            config.default_level = level
                            break

                # Load model specs
                if "models" in yaml_config:
                    for model_name, model_data in yaml_config["models"].items():
                        config.models[model_name] = ModelSpec(
                            name=model_name,
                            base_size_gb=model_data.get("base_size_gb", 8.0),
                            parameters=model_data.get("parameters", "8B"),
                            architecture=model_data.get("architecture", "unknown"),
                            context_length=model_data.get("context_length", 8192)
                        )

            log.info(f"Loaded config from {config_path}")
        except Exception as e:
            log.warning(f"Failed to load config: {e}, using defaults")

    return config


def calculate_storage_requirements(
    models: List[str],
    level: QuantizationLevel,
    config: QuantizationConfig
) -> Dict[str, Dict]:
    """Calculate storage requirements for models at given quantization level."""
    results = {}

    for model_name in models:
        if model_name in config.models:
            spec = config.models[model_name]
            results[model_name] = {
                "base_size_gb": spec.base_size_gb,
                "quantized_size_gb": spec.get_quantized_size(level),
                "memory_required_gb": spec.get_memory_requirement(level),
                "quality_retention": f"{level.quality_ratio * 100:.1f}%",
                "quantization_level": level.code,
                "parameters": spec.parameters,
                "architecture": spec.architecture
            }
        else:
            log.warning(f"Unknown model: {model_name}")

    return results


def recommend_quantization(
    models: List[str],
    usb_size_gb: int,
    config: QuantizationConfig
) -> Tuple[QuantizationLevel, Dict[str, Dict]]:
    """
    Recommend optimal quantization level based on USB size and models.

    Strategy:
    - 128GB USB: Prefer Q4_K_M for smaller models, may need to skip largest
    - 256GB USB: Can use Q5_K_M or even Q8_0 for important models
    """
    config.usb_size_gb = usb_size_gb
    available = config.available_space_gb

    log.info(f"USB size: {usb_size_gb}GB, Available for models: {available}GB")

    # Try each level from highest quality to lowest
    levels_to_try = [QuantizationLevel.Q8_0, QuantizationLevel.Q5_K_M, QuantizationLevel.Q4_K_M]

    for level in levels_to_try:
        requirements = calculate_storage_requirements(models, level, config)
        total_size = sum(r["quantized_size_gb"] for r in requirements.values())

        if total_size <= available:
            log.info(f"Recommended level: {level.code} (total: {total_size:.1f}GB)")
            return level, requirements

    # If nothing fits, return Q4_K_M with warning
    log.warning("Models may not fit even with Q4_K_M quantization")
    requirements = calculate_storage_requirements(models, QuantizationLevel.Q4_K_M, config)
    return QuantizationLevel.Q4_K_M, requirements


def auto_select_quantization(
    usb_size_gb: int,
    config: QuantizationConfig
) -> Dict[str, QuantizationLevel]:
    """
    Auto-select optimal quantization for each model based on USB size.

    Priority:
    1. dolphin-llama3:8b (PRIMARY) - best possible quality
    2. llama3.2 - good quality
    3. qwen2.5:14b - fit if space allows
    """
    config.usb_size_gb = usb_size_gb
    available = config.available_space_gb

    # Model priority order
    priority_order = [
        "dolphin-llama3:8b",
        "llama3.2:8b",
        "llama3.2:latest",
        "qwen2.5:14b"
    ]

    selections = {}
    remaining_space = available

    for model_name in priority_order:
        if model_name not in config.models:
            continue

        spec = config.models[model_name]

        # Try levels from best to worst quality
        for level in [QuantizationLevel.Q8_0, QuantizationLevel.Q5_K_M, QuantizationLevel.Q4_K_M]:
            size = spec.get_quantized_size(level)

            if size <= remaining_space:
                selections[model_name] = level
                remaining_space -= size
                log.info(f"{model_name}: {level.code} ({size:.1f}GB), remaining: {remaining_space:.1f}GB")
                break
        else:
            log.warning(f"{model_name}: Cannot fit even with Q4_K_M")

    return selections


def get_ollama_binary() -> Optional[Path]:
    """Find Ollama binary."""
    root = get_root_path()
    system = platform.system().lower()
    machine = platform.machine().lower()

    arch_map = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}
    arch = arch_map.get(machine, "amd64")

    if system == "darwin":
        local = root / "modules" / "ollama-portable" / "bin" / f"darwin-{arch}" / "ollama"
    elif system == "windows":
        local = root / "modules" / "ollama-portable" / "bin" / "windows-amd64" / "ollama.exe"
    else:
        local = root / "modules" / "ollama-portable" / "bin" / "linux-amd64" / "ollama"

    if local.exists():
        return local

    # Try system ollama
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    return None


def get_llama_cpp_quantize() -> Optional[Path]:
    """Find or download llama.cpp quantize binary."""
    root = get_root_path()
    tools_dir = root / "modules" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()

    if system == "linux":
        binary_name = "llama-quantize"
    elif system == "darwin":
        binary_name = "llama-quantize"
    else:
        binary_name = "llama-quantize.exe"

    binary_path = tools_dir / binary_name

    if binary_path.exists():
        return binary_path

    # Note: In production, this would download the binary
    log.warning(f"llama.cpp quantize binary not found at {binary_path}")
    log.info("For GGUF re-quantization, install llama.cpp tools")

    return None


def quantize_gguf_model(
    input_path: Path,
    output_path: Path,
    level: QuantizationLevel,
    quantize_binary: Path
) -> bool:
    """
    Quantize a GGUF model file using llama.cpp quantize.

    Note: Ollama models are already quantized. This is for custom models
    or for re-quantizing to a different level.
    """
    log.info(f"Quantizing {input_path.name} to {level.code}")

    try:
        cmd = [
            str(quantize_binary),
            str(input_path),
            str(output_path),
            level.code
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            log.info(f"Successfully quantized to {output_path}")
            return True
        else:
            log.error(f"Quantization failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        log.error("Quantization timed out")
        return False
    except Exception as e:
        log.error(f"Quantization error: {e}")
        return False


def pull_quantized_model(
    model_name: str,
    level: QuantizationLevel,
    ollama_binary: Path,
    models_path: Path
) -> bool:
    """
    Pull a specific quantization variant from Ollama.

    Ollama supports tags like:
    - dolphin-llama3:8b-q4_K_M
    - dolphin-llama3:8b-q8_0
    """
    # Construct quantized model name
    base_name = model_name.split(":")[0]
    tag = model_name.split(":")[-1] if ":" in model_name else "latest"

    # Some models have quantization variants as tags
    quantized_tag = f"{tag}-{level.code.lower()}"
    quantized_name = f"{base_name}:{quantized_tag}"

    log.info(f"Attempting to pull {quantized_name}")

    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(models_path)

    try:
        result = subprocess.run(
            [str(ollama_binary), "pull", quantized_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode == 0:
            log.info(f"Successfully pulled {quantized_name}")
            return True
        else:
            log.warning(f"Quantized variant not available: {quantized_name}")
            return False

    except Exception as e:
        log.error(f"Pull failed: {e}")
        return False


def print_analysis(
    config: QuantizationConfig,
    models: List[str],
    usb_size: int
):
    """Print storage and memory analysis."""
    print("\n" + "=" * 70)
    print("         USB-AI Model Quantization Analysis")
    print("=" * 70)
    print(f"\nUSB Size: {usb_size}GB | Available for models: {usb_size - config.reserved_space_gb}GB")
    print(f"Reserved space: {config.reserved_space_gb}GB (VeraCrypt, launchers, etc.)")

    print("\n" + "-" * 70)
    print("Quantization Level Comparison:")
    print("-" * 70)

    for level in QuantizationLevel:
        print(f"\n  {level.code} ({level.bits}-bit):")
        print(f"    Quality retention: {level.quality_ratio * 100:.0f}%")
        print(f"    Size ratio: {level.size_ratio * 100:.0f}% of FP16")
        print(f"    Description: {level.description}")

    print("\n" + "-" * 70)
    print("Per-Model Storage Requirements:")
    print("-" * 70)

    header = f"{'Model':<25} {'Params':<8} {'FP16':<8} {'Q4_K_M':<8} {'Q5_K_M':<8} {'Q8_0':<8} {'RAM (Q4)':<10}"
    print(f"\n{header}")
    print("-" * 70)

    for model_name in models:
        if model_name not in config.models:
            continue

        spec = config.models[model_name]
        q4_size = spec.get_quantized_size(QuantizationLevel.Q4_K_M)
        q5_size = spec.get_quantized_size(QuantizationLevel.Q5_K_M)
        q8_size = spec.get_quantized_size(QuantizationLevel.Q8_0)
        ram = spec.get_memory_requirement(QuantizationLevel.Q4_K_M)

        print(f"{model_name:<25} {spec.parameters:<8} {spec.base_size_gb:<8.1f} "
              f"{q4_size:<8.1f} {q5_size:<8.1f} {q8_size:<8.1f} {ram:<10.1f}")

    # Calculate totals
    print("-" * 70)

    for level in QuantizationLevel:
        reqs = calculate_storage_requirements(models, level, config)
        total = sum(r["quantized_size_gb"] for r in reqs.values())
        fits = "YES" if total <= (usb_size - config.reserved_space_gb) else "NO"
        print(f"Total with {level.code}: {total:.1f}GB | Fits on {usb_size}GB USB: {fits}")

    # Auto-selection recommendation
    print("\n" + "-" * 70)
    print("Auto-Selected Quantization Levels:")
    print("-" * 70)

    selections = auto_select_quantization(usb_size, config)
    total_selected = 0

    for model_name, level in selections.items():
        spec = config.models[model_name]
        size = spec.get_quantized_size(level)
        total_selected += size
        print(f"  {model_name}: {level.code} ({size:.1f}GB)")

    remaining = (usb_size - config.reserved_space_gb) - total_selected
    print(f"\nTotal selected: {total_selected:.1f}GB | Remaining: {remaining:.1f}GB")

    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="USB-AI Model Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model", "-m",
        help="Model to quantize (e.g., dolphin-llama3:8b)"
    )

    parser.add_argument(
        "--level", "-l",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
        default="Q4_K_M",
        help="Quantization level (default: Q4_K_M)"
    )

    parser.add_argument(
        "--usb-size", "-u",
        type=int,
        choices=[128, 256],
        default=128,
        help="USB drive size in GB (default: 128)"
    )

    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Show storage analysis only, don't quantize"
    )

    parser.add_argument(
        "--auto", "-A",
        action="store_true",
        help="Auto-select optimal quantization for all models"
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to quantization config YAML"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for quantized models"
    )

    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull pre-quantized models from Ollama if available"
    )

    args = parser.parse_args()

    log.info("USB-AI Model Quantization Tool")
    log.info(f"Version: {__version__}")

    # Load configuration
    config = load_quantization_config(args.config)
    config.usb_size_gb = args.usb_size

    # Get list of models
    models_to_process = list(config.models.keys())
    if args.model:
        models_to_process = [args.model]

    # Analysis mode
    if args.analyze:
        print_analysis(config, models_to_process, args.usb_size)
        return 0

    # Auto mode
    if args.auto:
        print_analysis(config, models_to_process, args.usb_size)

        selections = auto_select_quantization(args.usb_size, config)

        if args.pull:
            ollama = get_ollama_binary()
            if not ollama:
                log.error("Ollama not found")
                return 1

            models_path = get_root_path() / "modules" / "models"

            for model_name, level in selections.items():
                pull_quantized_model(model_name, level, ollama, models_path)

        # Save selections to config
        selections_file = get_root_path() / "modules" / "models" / "config" / "quantization_selections.json"
        selections_file.parent.mkdir(parents=True, exist_ok=True)

        selections_data = {
            "usb_size_gb": args.usb_size,
            "selections": {name: level.code for name, level in selections.items()}
        }

        with open(selections_file, "w") as f:
            json.dump(selections_data, f, indent=2)

        log.info(f"Saved selections to {selections_file}")
        return 0

    # Single model quantization
    if args.model:
        level = QuantizationLevel.Q4_K_M
        for l in QuantizationLevel:
            if l.code == args.level:
                level = l
                break

        requirements = calculate_storage_requirements([args.model], level, config)

        print(f"\nQuantization Plan for {args.model}:")
        print(f"  Level: {level.code}")
        print(f"  Quality: {level.quality_ratio * 100:.1f}%")

        if args.model in requirements:
            req = requirements[args.model]
            print(f"  Estimated size: {req['quantized_size_gb']:.1f}GB")
            print(f"  Memory required: {req['memory_required_gb']:.1f}GB")

        if args.pull:
            ollama = get_ollama_binary()
            if ollama:
                models_path = get_root_path() / "modules" / "models"
                pull_quantized_model(args.model, level, ollama, models_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
