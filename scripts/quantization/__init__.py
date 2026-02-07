"""
USB-AI Model Quantization Module

This module provides tools for:
- Model quantization level selection
- Storage requirement calculations
- Memory usage estimation
- Inference benchmarking
- Quality metrics

Quantization Levels:
    Q4_K_M: 4-bit, ~45% of FP16 size, 92% quality retention
    Q5_K_M: 5-bit, ~55% of FP16 size, 95% quality retention
    Q8_0:   8-bit, ~85% of FP16 size, 99% quality retention

Usage:
    from scripts.quantization import quantize_models, benchmark

    # Analyze storage requirements
    python -m scripts.quantization.quantize_models --analyze --usb-size 128

    # Run benchmarks
    python -m scripts.quantization.benchmark --all
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ["quantize_models", "benchmark"]

MODULE_PATH = Path(__file__).parent
ROOT_PATH = MODULE_PATH.parent.parent
