#!/usr/bin/env python3
"""
ollama_benchmark.py

Comprehensive Ollama Inference Benchmark Tool.

Measures:
- Time to First Token (TTFT)
- Tokens per Second (TPS)
- Cold start vs Warm start times
- Memory usage during inference
- Prompt evaluation time
- Generation time breakdown

Supports comparison between different configurations and models.
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
# Benchmark Metrics
# ==============================================================================

@dataclass
class InferenceMetrics:
    """Detailed metrics from a single inference run."""
    # Timing (milliseconds)
    time_to_first_token_ms: float
    prompt_eval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

    # Token counts
    prompt_tokens: int
    generated_tokens: int

    # Rates
    prompt_eval_rate: float  # tokens/sec for prompt processing
    generation_rate: float  # tokens/sec for generation (TPS)

    # Memory (MB)
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float

    # Context
    is_cold_start: bool
    model: str
    prompt_length: str


@dataclass
class ModelBenchmark:
    """Complete benchmark results for a model."""
    model: str
    timestamp: str

    # Cold start metrics
    cold_start: InferenceMetrics

    # Warm start metrics (averaged)
    warm_runs: List[InferenceMetrics]
    warm_ttft_mean_ms: float
    warm_ttft_std_ms: float
    warm_tps_mean: float
    warm_tps_std: float

    # Peak memory
    peak_memory_mb: float

    # Configuration
    env_vars: Dict[str, str]
    ollama_version: str


@dataclass
class ComparisonResult:
    """Comparison between two benchmark runs."""
    baseline: ModelBenchmark
    optimized: ModelBenchmark

    # Improvements
    ttft_improvement_pct: float
    tps_improvement_pct: float
    memory_change_pct: float


# ==============================================================================
# Benchmark Engine
# ==============================================================================

class OllamaBenchmarkEngine:
    """Engine for running Ollama benchmarks."""

    OLLAMA_API = "http://localhost:11434"

    # Test prompts of varying complexity
    PROMPTS = {
        "minimal": "Hi",
        "short": "What is 2+2? Answer in one word.",
        "medium": "Explain the concept of recursion in programming in exactly 3 sentences.",
        "long": """Write a detailed technical explanation of how transformer neural networks work.
Cover the following topics:
1. Self-attention mechanism
2. Multi-head attention
3. Positional encoding
4. Feed-forward layers
5. Layer normalization
Use technical terminology and provide specific examples.""",
        "code": """Write a Python function that implements a binary search tree with the following methods:
- insert(value)
- search(value) -> bool
- delete(value)
- inorder_traversal() -> list
Include docstrings and type hints."""
    }

    def __init__(self):
        self.system = platform.system().lower()

    def check_server(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.OLLAMA_API}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except:
            return False

    def get_ollama_version(self) -> str:
        """Get Ollama version."""
        try:
            req = urllib.request.Request(f"{self.OLLAMA_API}/api/version")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data.get("version", "unknown")
        except:
            return "unknown"

    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with details."""
        try:
            req = urllib.request.Request(f"{self.OLLAMA_API}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("models", [])
        except Exception as e:
            log.warning(f"Failed to list models: {e}")
            return []

    def get_memory_usage(self) -> float:
        """Get current Ollama process memory usage in MB."""
        try:
            if self.system == "windows":
                result = subprocess.run(
                    ["wmic", "process", "where", "name='ollama.exe'",
                     "get", "WorkingSetSize"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        return int(lines[1].strip()) / 1024 / 1024
            else:
                # Linux/macOS
                result = subprocess.run(
                    ["pgrep", "-f", "ollama serve"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    pids = result.stdout.strip().split("\n")
                    total_mem = 0
                    for pid in pids:
                        if pid:
                            try:
                                if self.system == "linux":
                                    with open(f"/proc/{pid}/status") as f:
                                        status = f.read()
                                        rss = re.search(r"VmRSS:\s+(\d+)\s+kB", status)
                                        if rss:
                                            total_mem += int(rss.group(1)) / 1024
                                elif self.system == "darwin":
                                    ps_result = subprocess.run(
                                        ["ps", "-o", "rss=", "-p", pid],
                                        capture_output=True, text=True
                                    )
                                    if ps_result.returncode == 0:
                                        total_mem += int(ps_result.stdout.strip()) / 1024
                            except:
                                pass
                    return total_mem
        except:
            pass
        return 0

    def unload_model(self, model: str) -> bool:
        """Unload model from memory."""
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

            with urllib.request.urlopen(req, timeout=60) as resp:
                # Consume response
                for line in resp:
                    pass
            return True
        except:
            return False

    def run_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100
    ) -> Tuple[Dict[str, Any], str]:
        """
        Run inference and collect detailed metrics.

        Returns: (metrics_dict, response_text)
        """
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens
            }
        }).encode()

        req = urllib.request.Request(
            f"{self.OLLAMA_API}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        start_time = time.perf_counter()
        first_token_time = None
        response_text = ""
        final_data = {}

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode())

                        if "response" in data and data["response"]:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            response_text += data["response"]

                        if data.get("done", False):
                            final_data = data
                            break
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            log.error(f"Inference error: {e}")
            return {}, ""

        end_time = time.perf_counter()

        # Calculate metrics
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_ms = (end_time - start_time) * 1000

        # Extract Ollama-provided metrics
        prompt_eval_count = final_data.get("prompt_eval_count", 0)
        eval_count = final_data.get("eval_count", 0)
        prompt_eval_duration = final_data.get("prompt_eval_duration", 0) / 1e6  # ns to ms
        eval_duration = final_data.get("eval_duration", 0) / 1e6  # ns to ms

        # Calculate rates
        prompt_rate = (prompt_eval_count / (prompt_eval_duration / 1000)) if prompt_eval_duration > 0 else 0
        gen_rate = (eval_count / (eval_duration / 1000)) if eval_duration > 0 else 0

        metrics = {
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "prompt_eval_ms": prompt_eval_duration,
            "generation_ms": eval_duration,
            "prompt_tokens": prompt_eval_count,
            "generated_tokens": eval_count,
            "prompt_rate": prompt_rate,
            "generation_rate": gen_rate
        }

        return metrics, response_text

    def benchmark_model(
        self,
        model: str,
        prompt_type: str = "medium",
        warm_runs: int = 3,
        max_tokens: int = 100,
        verbose: bool = True
    ) -> Optional[ModelBenchmark]:
        """
        Run complete benchmark on a model.

        Args:
            model: Model name
            prompt_type: Prompt complexity ("minimal", "short", "medium", "long", "code")
            warm_runs: Number of warm runs
            max_tokens: Maximum tokens to generate
            verbose: Print progress

        Returns:
            ModelBenchmark with all metrics
        """
        if not self.check_server():
            log.error("Ollama server not running")
            return None

        prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["medium"])

        if verbose:
            log.info(f"Benchmarking: {model}")
            log.info(f"Prompt type: {prompt_type} ({len(prompt)} chars)")

        # Get environment variables
        env_vars = {}
        for key in os.environ:
            if key.startswith("OLLAMA_"):
                env_vars[key] = os.environ[key]

        ollama_version = self.get_ollama_version()
        peak_memory = 0.0

        # === Cold Start ===
        if verbose:
            log.info("Running cold start test...")

        # Unload model first
        self.unload_model(model)
        time.sleep(2)

        mem_before = self.get_memory_usage()
        cold_metrics, _ = self.run_inference(model, prompt, max_tokens)
        mem_after = self.get_memory_usage()

        peak_memory = max(peak_memory, mem_after)

        cold_start = InferenceMetrics(
            time_to_first_token_ms=cold_metrics.get("ttft_ms", 0),
            prompt_eval_time_ms=cold_metrics.get("prompt_eval_ms", 0),
            generation_time_ms=cold_metrics.get("generation_ms", 0),
            total_time_ms=cold_metrics.get("total_ms", 0),
            prompt_tokens=cold_metrics.get("prompt_tokens", 0),
            generated_tokens=cold_metrics.get("generated_tokens", 0),
            prompt_eval_rate=cold_metrics.get("prompt_rate", 0),
            generation_rate=cold_metrics.get("generation_rate", 0),
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=mem_after - mem_before,
            is_cold_start=True,
            model=model,
            prompt_length=prompt_type
        )

        if verbose:
            log.info(f"  TTFT: {cold_start.time_to_first_token_ms:.0f}ms, "
                    f"TPS: {cold_start.generation_rate:.1f}")

        # === Warm Runs ===
        if verbose:
            log.info(f"Running {warm_runs} warm start tests...")

        warm_results = []
        for i in range(warm_runs):
            mem_before = self.get_memory_usage()
            metrics, _ = self.run_inference(model, prompt, max_tokens)
            mem_after = self.get_memory_usage()

            peak_memory = max(peak_memory, mem_after)

            warm_run = InferenceMetrics(
                time_to_first_token_ms=metrics.get("ttft_ms", 0),
                prompt_eval_time_ms=metrics.get("prompt_eval_ms", 0),
                generation_time_ms=metrics.get("generation_ms", 0),
                total_time_ms=metrics.get("total_ms", 0),
                prompt_tokens=metrics.get("prompt_tokens", 0),
                generated_tokens=metrics.get("generated_tokens", 0),
                prompt_eval_rate=metrics.get("prompt_rate", 0),
                generation_rate=metrics.get("generation_rate", 0),
                memory_before_mb=mem_before,
                memory_after_mb=mem_after,
                memory_delta_mb=mem_after - mem_before,
                is_cold_start=False,
                model=model,
                prompt_length=prompt_type
            )
            warm_results.append(warm_run)

            if verbose:
                log.info(f"  Run {i+1}: TTFT={warm_run.time_to_first_token_ms:.0f}ms, "
                        f"TPS={warm_run.generation_rate:.1f}")

        # Calculate warm averages
        warm_ttfts = [r.time_to_first_token_ms for r in warm_results]
        warm_tps = [r.generation_rate for r in warm_results]

        warm_ttft_mean = statistics.mean(warm_ttfts) if warm_ttfts else 0
        warm_ttft_std = statistics.stdev(warm_ttfts) if len(warm_ttfts) > 1 else 0
        warm_tps_mean = statistics.mean(warm_tps) if warm_tps else 0
        warm_tps_std = statistics.stdev(warm_tps) if len(warm_tps) > 1 else 0

        return ModelBenchmark(
            model=model,
            timestamp=datetime.now().isoformat(),
            cold_start=cold_start,
            warm_runs=warm_results,
            warm_ttft_mean_ms=warm_ttft_mean,
            warm_ttft_std_ms=warm_ttft_std,
            warm_tps_mean=warm_tps_mean,
            warm_tps_std=warm_tps_std,
            peak_memory_mb=peak_memory,
            env_vars=env_vars,
            ollama_version=ollama_version
        )

    def compare_benchmarks(
        self,
        baseline: ModelBenchmark,
        optimized: ModelBenchmark
    ) -> ComparisonResult:
        """Compare two benchmark results."""
        # TTFT improvement (lower is better, so positive = improvement)
        if baseline.warm_ttft_mean_ms > 0:
            ttft_improvement = ((baseline.warm_ttft_mean_ms - optimized.warm_ttft_mean_ms)
                               / baseline.warm_ttft_mean_ms) * 100
        else:
            ttft_improvement = 0

        # TPS improvement (higher is better)
        if baseline.warm_tps_mean > 0:
            tps_improvement = ((optimized.warm_tps_mean - baseline.warm_tps_mean)
                              / baseline.warm_tps_mean) * 100
        else:
            tps_improvement = 0

        # Memory change (negative = reduction)
        if baseline.peak_memory_mb > 0:
            memory_change = ((optimized.peak_memory_mb - baseline.peak_memory_mb)
                            / baseline.peak_memory_mb) * 100
        else:
            memory_change = 0

        return ComparisonResult(
            baseline=baseline,
            optimized=optimized,
            ttft_improvement_pct=ttft_improvement,
            tps_improvement_pct=tps_improvement,
            memory_change_pct=memory_change
        )


# ==============================================================================
# Reporting
# ==============================================================================

def print_benchmark_report(benchmark: ModelBenchmark):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print(f"  BENCHMARK REPORT: {benchmark.model}")
    print("=" * 70)

    print(f"\n  Timestamp:       {benchmark.timestamp}")
    print(f"  Ollama Version:  {benchmark.ollama_version}")

    # Cold start
    cold = benchmark.cold_start
    print("\n  COLD START (model not loaded)")
    print("  " + "-" * 40)
    print(f"    Time to First Token:  {cold.time_to_first_token_ms:,.0f} ms")
    print(f"    Prompt Eval Time:     {cold.prompt_eval_time_ms:,.0f} ms")
    print(f"    Generation Time:      {cold.generation_time_ms:,.0f} ms")
    print(f"    Total Time:           {cold.total_time_ms:,.0f} ms")
    print(f"    Tokens Generated:     {cold.generated_tokens}")
    print(f"    Generation Rate:      {cold.generation_rate:.1f} tok/sec")
    print(f"    Memory Delta:         {cold.memory_delta_mb:,.0f} MB")

    # Warm starts
    print(f"\n  WARM STARTS ({len(benchmark.warm_runs)} runs)")
    print("  " + "-" * 40)
    print(f"    Time to First Token:  {benchmark.warm_ttft_mean_ms:.0f} ms "
          f"(+/- {benchmark.warm_ttft_std_ms:.0f})")
    print(f"    Tokens per Second:    {benchmark.warm_tps_mean:.1f} "
          f"(+/- {benchmark.warm_tps_std:.1f})")

    # Compare cold vs warm
    if cold.time_to_first_token_ms > 0 and benchmark.warm_ttft_mean_ms > 0:
        speedup = cold.time_to_first_token_ms / benchmark.warm_ttft_mean_ms
        print(f"\n    Warm vs Cold Speedup: {speedup:.1f}x faster TTFT")

    # Memory
    print(f"\n  MEMORY")
    print("  " + "-" * 40)
    print(f"    Peak Usage:           {benchmark.peak_memory_mb:,.0f} MB")

    # Environment
    if benchmark.env_vars:
        print(f"\n  ENVIRONMENT VARIABLES")
        print("  " + "-" * 40)
        for key, value in sorted(benchmark.env_vars.items()):
            print(f"    {key}={value}")

    print()


def print_comparison_report(comparison: ComparisonResult):
    """Print benchmark comparison report."""
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON")
    print("=" * 70)

    print(f"\n  Model: {comparison.baseline.model}")

    print("\n  BASELINE")
    print("  " + "-" * 40)
    print(f"    TTFT (warm avg):  {comparison.baseline.warm_ttft_mean_ms:.0f} ms")
    print(f"    TPS (warm avg):   {comparison.baseline.warm_tps_mean:.1f}")
    print(f"    Peak Memory:      {comparison.baseline.peak_memory_mb:.0f} MB")

    print("\n  OPTIMIZED")
    print("  " + "-" * 40)
    print(f"    TTFT (warm avg):  {comparison.optimized.warm_ttft_mean_ms:.0f} ms")
    print(f"    TPS (warm avg):   {comparison.optimized.warm_tps_mean:.1f}")
    print(f"    Peak Memory:      {comparison.optimized.peak_memory_mb:.0f} MB")

    print("\n  IMPROVEMENTS")
    print("  " + "-" * 40)

    # TTFT (positive = faster)
    ttft_color = "better" if comparison.ttft_improvement_pct > 0 else "worse"
    print(f"    TTFT:    {comparison.ttft_improvement_pct:+.1f}% ({ttft_color})")

    # TPS (positive = faster)
    tps_color = "better" if comparison.tps_improvement_pct > 0 else "worse"
    print(f"    TPS:     {comparison.tps_improvement_pct:+.1f}% ({tps_color})")

    # Memory (negative = lower = better)
    mem_color = "better" if comparison.memory_change_pct < 0 else "higher"
    print(f"    Memory:  {comparison.memory_change_pct:+.1f}% ({mem_color})")

    print()


def save_benchmark_json(benchmark: ModelBenchmark, path: Path):
    """Save benchmark results to JSON file."""
    # Convert to serializable format
    data = {
        "model": benchmark.model,
        "timestamp": benchmark.timestamp,
        "ollama_version": benchmark.ollama_version,
        "cold_start": asdict(benchmark.cold_start),
        "warm_runs": [asdict(r) for r in benchmark.warm_runs],
        "warm_ttft_mean_ms": benchmark.warm_ttft_mean_ms,
        "warm_ttft_std_ms": benchmark.warm_ttft_std_ms,
        "warm_tps_mean": benchmark.warm_tps_mean,
        "warm_tps_std": benchmark.warm_tps_std,
        "peak_memory_mb": benchmark.peak_memory_mb,
        "env_vars": benchmark.env_vars
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved benchmark results to: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Ollama Inference Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt Types:
  minimal    Very short prompt (1 word)
  short      Short question with brief answer expected
  medium     Multi-sentence explanation (default)
  long       Detailed technical explanation
  code       Code generation task

Examples:
  %(prog)s                          # Benchmark all models with defaults
  %(prog)s -m llama3.2:latest       # Benchmark specific model
  %(prog)s -m dolphin-llama3:8b --prompt long --runs 5
  %(prog)s --compare baseline.json  # Compare with saved baseline
        """
    )

    parser.add_argument(
        "--model", "-m",
        help="Model to benchmark (default: all available)"
    )

    parser.add_argument(
        "--prompt", "-p",
        choices=["minimal", "short", "medium", "long", "code"],
        default="medium",
        help="Prompt complexity (default: medium)"
    )

    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of warm runs (default: 3)"
    )

    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )

    parser.add_argument(
        "--save", "-s",
        type=Path,
        help="Save results to JSON file"
    )

    parser.add_argument(
        "--compare", "-c",
        type=Path,
        help="Compare with baseline JSON file"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    engine = OllamaBenchmarkEngine()

    # Check server
    if not engine.check_server():
        log.error("Ollama server is not running. Start it with: ollama serve")
        return 1

    # Get models to benchmark
    if args.model:
        models = [args.model]
    else:
        available = engine.list_models()
        if not available:
            log.error("No models available. Pull a model with: ollama pull <model>")
            return 1
        models = [m["name"] for m in available]
        if not args.quiet:
            log.info(f"Found {len(models)} models to benchmark")

    # Run benchmarks
    results = []
    for model in models:
        result = engine.benchmark_model(
            model=model,
            prompt_type=args.prompt,
            warm_runs=args.runs,
            max_tokens=args.max_tokens,
            verbose=not args.quiet
        )

        if result:
            results.append(result)

            if not args.json and not args.quiet:
                print_benchmark_report(result)

            if args.save:
                # Append model name if multiple models
                if len(models) > 1:
                    safe_name = model.replace(":", "_").replace("/", "_")
                    save_path = args.save.with_stem(f"{args.save.stem}_{safe_name}")
                else:
                    save_path = args.save
                save_benchmark_json(result, save_path)

    # Handle comparison
    if args.compare and results:
        try:
            with open(args.compare) as f:
                baseline_data = json.load(f)

            # Find matching model in results
            for result in results:
                if result.model == baseline_data.get("model"):
                    # Reconstruct baseline from JSON
                    baseline_cold = InferenceMetrics(**baseline_data["cold_start"])
                    baseline_warm = [InferenceMetrics(**r) for r in baseline_data["warm_runs"]]

                    baseline = ModelBenchmark(
                        model=baseline_data["model"],
                        timestamp=baseline_data["timestamp"],
                        cold_start=baseline_cold,
                        warm_runs=baseline_warm,
                        warm_ttft_mean_ms=baseline_data["warm_ttft_mean_ms"],
                        warm_ttft_std_ms=baseline_data["warm_ttft_std_ms"],
                        warm_tps_mean=baseline_data["warm_tps_mean"],
                        warm_tps_std=baseline_data["warm_tps_std"],
                        peak_memory_mb=baseline_data["peak_memory_mb"],
                        env_vars=baseline_data.get("env_vars", {}),
                        ollama_version=baseline_data.get("ollama_version", "unknown")
                    )

                    comparison = engine.compare_benchmarks(baseline, result)

                    if not args.json:
                        print_comparison_report(comparison)
                    break

        except Exception as e:
            log.error(f"Failed to load comparison baseline: {e}")

    # JSON output
    if args.json:
        output = []
        for result in results:
            output.append({
                "model": result.model,
                "timestamp": result.timestamp,
                "cold_ttft_ms": result.cold_start.time_to_first_token_ms,
                "cold_tps": result.cold_start.generation_rate,
                "warm_ttft_mean_ms": result.warm_ttft_mean_ms,
                "warm_ttft_std_ms": result.warm_ttft_std_ms,
                "warm_tps_mean": result.warm_tps_mean,
                "warm_tps_std": result.warm_tps_std,
                "peak_memory_mb": result.peak_memory_mb,
                "env_vars": result.env_vars
            })
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
