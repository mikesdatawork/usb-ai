"""
USB-AI Performance Optimization Module

This module provides tools for optimizing USB-AI performance
based on detected hardware capabilities.

Components:
    - optimize.py: Generates optimal Ollama configuration
    - system_check.py: Analyzes system hardware and compatibility
    - gpu_optimizer.py: GPU acceleration with VRAM calculator and benchmarks
    - model_warmup.py: Eliminates cold start latency with model preloading
    - context_optimizer.py: Context window and KV cache optimization
    - ollama_tuner.py: Ollama environment variable optimizer for max speed
    - ollama_benchmark.py: Comprehensive inference benchmarking tool
    - inference_optimizer.py: Full inference pipeline optimization
    - fast_inference.py: Minimal overhead inference client
    - benchmark_profiles.py: Profile benchmarking and comparison
    - performance_profiles.yaml: Profile definitions (in modules/config/)
    - inference_profiles.yaml: Inference profile definitions (in modules/config/)
    - gpu_profiles.yaml: GPU profile definitions (in modules/config/)
    - context_profiles.yaml: Context window profiles (in modules/config/)

Usage:
    # Check system capabilities
    python -m scripts.performance.system_check

    # Generate optimized configuration
    python -m scripts.performance.optimize --profile balanced

    # Dry run (show without writing)
    python -m scripts.performance.optimize --dry-run --json

    # GPU Optimization
    python -m scripts.performance.gpu_optimizer              # Auto-detect GPU
    python -m scripts.performance.gpu_optimizer --profile full_offload
    python -m scripts.performance.gpu_optimizer --profile hybrid
    python -m scripts.performance.gpu_optimizer --profile cpu_only
    python -m scripts.performance.gpu_optimizer --vram-estimate --model-size 8B
    python -m scripts.performance.gpu_optimizer --benchmark  # GPU vs CPU benchmark
    python -m scripts.performance.gpu_optimizer --json       # JSON output
    python -m scripts.performance.gpu_optimizer --write-config  # Save config

    # Model Warmup Manager
    python -m scripts.performance.model_warmup --daemon      # Run daemon
    python -m scripts.performance.model_warmup --status      # Check status
    python -m scripts.performance.model_warmup --warmup MODEL  # Warm up model
    python -m scripts.performance.model_warmup --benchmark   # Run benchmark

    # Context Optimization
    python -m scripts.performance.context_optimizer --priority speed
    python -m scripts.performance.context_optimizer --profile quick_response
    python -m scripts.performance.context_optimizer --estimate-tokens "text"

    # Ollama Tuner (Speed Optimization)
    python -m scripts.performance.ollama_tuner --preset speed
    python -m scripts.performance.ollama_tuner --preset balanced --benchmark
    python -m scripts.performance.ollama_tuner --apply --benchmark

    # Ollama Benchmark (Detailed Performance Testing)
    python -m scripts.performance.ollama_benchmark
    python -m scripts.performance.ollama_benchmark -m llama3.2:latest --prompt long
    python -m scripts.performance.ollama_benchmark --save baseline.json
    python -m scripts.performance.ollama_benchmark --compare baseline.json

    # Inference Pipeline Optimization (NEW)
    python -m scripts.performance.inference_optimizer --benchmark
    python -m scripts.performance.inference_optimizer --profile realtime --prompt "Hello"
    python -m scripts.performance.inference_optimizer --show-profiles

    # Fast Inference Client (NEW)
    python -m scripts.performance.fast_inference "What is 2+2?"
    python -m scripts.performance.fast_inference --profile turbo --benchmark
    python -m scripts.performance.fast_inference --metrics "Explain recursion"

    # Benchmark Profiles (NEW)
    python -m scripts.performance.benchmark_profiles
    python -m scripts.performance.benchmark_profiles --iterations 10
    python -m scripts.performance.benchmark_profiles --profile realtime --quick
"""

__version__ = "1.5.0"

# Import key classes for convenient access
from scripts.performance.model_warmup import (
    WarmupManager,
    WarmupDaemon,
    Benchmarker,
    OllamaClient,
    MemoryMonitor,
    ModelUsagePredictor,
)

from scripts.performance.context_optimizer import (
    ContextOptimizer,
    ContextConfig,
    ContextProfile,
    TokenEstimator,
    MemoryAnalyzer,
    SlidingWindowManager,
    ContextProfileManager,
)

from scripts.performance.gpu_optimizer import (
    GPUOptimizer,
    GPUDetector,
    GPUCapabilities,
    GPUDevice,
    GPUVendor,
    AccelerationType,
    VRAMCalculator,
    VRAMEstimate,
    GPUOptimizationConfig,
    GPUBenchmark,
    BenchmarkResult,
    NVIDIADetector,
    AMDDetector,
    AppleSiliconDetector,
)

# Inference Pipeline Optimization
from scripts.performance.inference_optimizer import (
    InferenceOptimizer,
    FastInferenceClient,
    AsyncInferenceClient,
    RequestBatcher,
    MetricsCollector,
    ConnectionPool,
    RequestTemplate,
    ProfileConfig,
    InferenceProfile,
    PROFILE_CONFIGS,
    get_profile,
)

from scripts.performance.fast_inference import (
    FastInference,
    InferenceConfig,
    RequestMetrics,
    quick_generate,
    quick_stream,
    PROFILES as INFERENCE_PROFILES,
)
