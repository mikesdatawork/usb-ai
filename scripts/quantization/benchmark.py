#!/usr/bin/env python3
"""
benchmark.py

Benchmarks inference speed at different quantization levels for USB-AI models.
Measures tokens/second, first token latency, and quality metrics.

Usage:
    python benchmark.py --model dolphin-llama3:8b
    python benchmark.py --all --iterations 5
    python benchmark.py --compare Q4_K_M Q8_0
"""

import argparse
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Benchmark prompts with expected complexity
BENCHMARK_PROMPTS = [
    {
        "id": "simple",
        "prompt": "What is 2 + 2?",
        "expected_tokens": 20,
        "category": "math_simple"
    },
    {
        "id": "medium",
        "prompt": "Explain the concept of recursion in programming with a simple example.",
        "expected_tokens": 150,
        "category": "explanation"
    },
    {
        "id": "long",
        "prompt": "Write a detailed Python function that implements binary search with comprehensive docstrings and error handling. Include edge cases.",
        "expected_tokens": 400,
        "category": "code_generation"
    },
    {
        "id": "reasoning",
        "prompt": "If a train leaves Station A at 9:00 AM traveling at 60 mph, and another train leaves Station B at 10:00 AM traveling at 80 mph towards Station A. If the stations are 280 miles apart, when do the trains meet?",
        "expected_tokens": 200,
        "category": "reasoning"
    },
    {
        "id": "creative",
        "prompt": "Write a short poem about artificial intelligence and its impact on humanity.",
        "expected_tokens": 100,
        "category": "creative"
    }
]

# Quality check prompts with expected answers
QUALITY_PROMPTS = [
    {
        "id": "math",
        "prompt": "Calculate: (15 * 4) + (28 / 4) = ?",
        "expected_answer": "67",
        "tolerance": "exact"
    },
    {
        "id": "logic",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer: Yes or No",
        "expected_answer": "No",
        "tolerance": "contains"
    },
    {
        "id": "knowledge",
        "prompt": "What is the capital of France? Answer with just the city name.",
        "expected_answer": "Paris",
        "tolerance": "contains"
    }
]


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    model: str
    prompt_id: str
    prompt_tokens: int
    completion_tokens: int
    first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_id": self.prompt_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "first_token_ms": round(self.first_token_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "timestamp": self.timestamp
        }


@dataclass
class QualityResult:
    """Quality check result."""
    model: str
    prompt_id: str
    expected: str
    actual: str
    passed: bool
    response_time_ms: float


@dataclass
class ModelBenchmark:
    """Aggregated benchmark results for a model."""
    model: str
    quantization: str
    results: List[BenchmarkResult] = field(default_factory=list)
    quality_results: List[QualityResult] = field(default_factory=list)

    @property
    def avg_tokens_per_second(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.tokens_per_second for r in self.results)

    @property
    def avg_first_token_ms(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.first_token_ms for r in self.results)

    @property
    def quality_score(self) -> float:
        if not self.quality_results:
            return 0.0
        passed = sum(1 for r in self.quality_results if r.passed)
        return (passed / len(self.quality_results)) * 100

    def summary(self) -> dict:
        return {
            "model": self.model,
            "quantization": self.quantization,
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "avg_first_token_ms": round(self.avg_first_token_ms, 2),
            "quality_score_percent": round(self.quality_score, 1),
            "total_tests": len(self.results),
            "quality_tests": len(self.quality_results)
        }


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


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

    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    return None


def check_ollama_server(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Check if Ollama server is running."""
    try:
        url = f"http://{host}:{port}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def start_ollama_server(binary: Path, models_path: Path) -> Optional[subprocess.Popen]:
    """Start Ollama server if not running."""
    if check_ollama_server():
        log.info("Ollama server already running")
        return None

    log.info("Starting Ollama server...")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_MODELS"] = str(models_path)

    try:
        process = subprocess.Popen(
            [str(binary), "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(30):
            if check_ollama_server():
                log.info("Ollama server started")
                return process
            time.sleep(1)

        log.error("Ollama server failed to start")
        process.terminate()
        return None

    except Exception as e:
        log.error(f"Failed to start Ollama: {e}")
        return None


def stop_ollama_server(process: Optional[subprocess.Popen]):
    """Stop Ollama server if we started it."""
    if process:
        log.info("Stopping Ollama server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def list_models(host: str = "127.0.0.1", port: int = 11434) -> List[str]:
    """List available models."""
    try:
        url = f"http://{host}:{port}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        log.error(f"Failed to list models: {e}")
        return []


def generate_completion(
    model: str,
    prompt: str,
    host: str = "127.0.0.1",
    port: int = 11434,
    stream: bool = True
) -> Tuple[str, int, int, float, float]:
    """
    Generate completion and measure timing.

    Returns: (response, prompt_tokens, completion_tokens, first_token_ms, total_ms)
    """
    url = f"http://{host}:{port}/api/generate"

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "num_predict": 512,
            "temperature": 0.1  # Low temperature for consistent benchmarks
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start_time = time.perf_counter()
    first_token_time = None
    response_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        response_text += data["response"]
                    if "prompt_eval_count" in data:
                        prompt_tokens = data["prompt_eval_count"]
                    if "eval_count" in data:
                        completion_tokens = data["eval_count"]
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        log.error(f"Generation error: {e}")
        return "", 0, 0, 0.0, 0.0

    end_time = time.perf_counter()

    total_ms = (end_time - start_time) * 1000
    first_token_ms = ((first_token_time or end_time) - start_time) * 1000

    return response_text, prompt_tokens, completion_tokens, first_token_ms, total_ms


def run_benchmark(
    model: str,
    iterations: int = 3,
    prompts: Optional[List[dict]] = None
) -> ModelBenchmark:
    """Run benchmark on a model."""
    log.info(f"Benchmarking: {model}")

    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    # Detect quantization from model name
    quantization = "default"
    for level in ["q4_k_m", "q5_k_m", "q8_0"]:
        if level in model.lower():
            quantization = level.upper()
            break

    benchmark = ModelBenchmark(model=model, quantization=quantization)

    # Warm-up run
    log.info("Warm-up run...")
    generate_completion(model, "Hello")
    time.sleep(1)

    # Benchmark runs
    for i in range(iterations):
        log.info(f"Iteration {i + 1}/{iterations}")

        for prompt_config in prompts:
            response, p_tokens, c_tokens, first_ms, total_ms = generate_completion(
                model, prompt_config["prompt"]
            )

            if c_tokens > 0 and total_ms > 0:
                tps = (c_tokens / total_ms) * 1000

                result = BenchmarkResult(
                    model=model,
                    prompt_id=prompt_config["id"],
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    first_token_ms=first_ms,
                    total_time_ms=total_ms,
                    tokens_per_second=tps
                )
                benchmark.results.append(result)

                log.info(f"  {prompt_config['id']}: {tps:.1f} tok/s, {first_ms:.0f}ms first token")

            time.sleep(0.5)  # Brief pause between runs

    return benchmark


def run_quality_check(model: str) -> List[QualityResult]:
    """Run quality checks on a model."""
    log.info(f"Quality check: {model}")
    results = []

    for check in QUALITY_PROMPTS:
        start = time.perf_counter()
        response, _, _, _, _ = generate_completion(model, check["prompt"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Check answer
        response_clean = response.strip().lower()
        expected_clean = check["expected_answer"].lower()

        if check["tolerance"] == "exact":
            passed = expected_clean in response_clean
        else:  # contains
            passed = expected_clean in response_clean

        result = QualityResult(
            model=model,
            prompt_id=check["id"],
            expected=check["expected_answer"],
            actual=response.strip()[:100],
            passed=passed,
            response_time_ms=elapsed_ms
        )
        results.append(result)

        status = "PASS" if passed else "FAIL"
        log.info(f"  {check['id']}: {status}")

    return results


def compare_models(
    models: List[str],
    iterations: int = 3
) -> Dict[str, ModelBenchmark]:
    """Compare multiple models."""
    results = {}

    for model in models:
        benchmark = run_benchmark(model, iterations)
        benchmark.quality_results = run_quality_check(model)
        results[model] = benchmark

    return results


def print_benchmark_report(benchmarks: Dict[str, ModelBenchmark]):
    """Print formatted benchmark report."""
    print("\n" + "=" * 80)
    print("                    USB-AI Model Benchmark Report")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # Summary table
    print("\n" + "-" * 80)
    print("Performance Summary:")
    print("-" * 80)

    header = f"{'Model':<35} {'Quant':<10} {'Tok/s':<10} {'First (ms)':<12} {'Quality':<10}"
    print(f"\n{header}")
    print("-" * 80)

    for model, benchmark in benchmarks.items():
        summary = benchmark.summary()
        print(f"{model:<35} {summary['quantization']:<10} "
              f"{summary['avg_tokens_per_second']:<10.1f} "
              f"{summary['avg_first_token_ms']:<12.0f} "
              f"{summary['quality_score_percent']:<10.1f}%")

    # Detailed results
    print("\n" + "-" * 80)
    print("Detailed Results by Prompt Type:")
    print("-" * 80)

    for model, benchmark in benchmarks.items():
        print(f"\n{model}:")

        by_prompt = {}
        for result in benchmark.results:
            if result.prompt_id not in by_prompt:
                by_prompt[result.prompt_id] = []
            by_prompt[result.prompt_id].append(result)

        for prompt_id, results in by_prompt.items():
            avg_tps = statistics.mean(r.tokens_per_second for r in results)
            avg_first = statistics.mean(r.first_token_ms for r in results)
            avg_tokens = statistics.mean(r.completion_tokens for r in results)

            print(f"  {prompt_id:<15} {avg_tps:>6.1f} tok/s  "
                  f"{avg_first:>6.0f}ms first  {avg_tokens:>4.0f} tokens avg")

    # Quality details
    print("\n" + "-" * 80)
    print("Quality Check Results:")
    print("-" * 80)

    for model, benchmark in benchmarks.items():
        print(f"\n{model}:")
        for qr in benchmark.quality_results:
            status = "PASS" if qr.passed else "FAIL"
            print(f"  {qr.prompt_id:<15} [{status}] Expected: {qr.expected}, "
                  f"Got: {qr.actual[:50]}...")

    # Recommendations
    print("\n" + "-" * 80)
    print("Recommendations:")
    print("-" * 80)

    if benchmarks:
        fastest = max(benchmarks.items(), key=lambda x: x[1].avg_tokens_per_second)
        best_quality = max(benchmarks.items(), key=lambda x: x[1].quality_score)
        lowest_latency = min(benchmarks.items(), key=lambda x: x[1].avg_first_token_ms)

        print(f"\n  Fastest inference:     {fastest[0]} ({fastest[1].avg_tokens_per_second:.1f} tok/s)")
        print(f"  Best quality:          {best_quality[0]} ({best_quality[1].quality_score:.0f}%)")
        print(f"  Lowest first-token:    {lowest_latency[0]} ({lowest_latency[1].avg_first_token_ms:.0f}ms)")

    print("\n" + "=" * 80)


def save_benchmark_results(
    benchmarks: Dict[str, ModelBenchmark],
    output_path: Path
):
    """Save benchmark results to JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "benchmarks": {}
    }

    for model, benchmark in benchmarks.items():
        data["benchmarks"][model] = {
            "summary": benchmark.summary(),
            "results": [r.to_dict() for r in benchmark.results],
            "quality_results": [
                {
                    "prompt_id": qr.prompt_id,
                    "expected": qr.expected,
                    "actual": qr.actual,
                    "passed": qr.passed,
                    "response_time_ms": round(qr.response_time_ms, 2)
                }
                for qr in benchmark.quality_results
            ]
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Results saved to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="USB-AI Model Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model", "-m",
        help="Model to benchmark (e.g., dolphin-llama3:8b)"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Benchmark all available models"
    )

    parser.add_argument(
        "--compare", "-c",
        nargs="+",
        help="Compare specific models"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of iterations per prompt (default: 3)"
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick benchmark (1 iteration, fewer prompts)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON file for results"
    )

    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Run quality checks only"
    )

    args = parser.parse_args()

    log.info("USB-AI Model Benchmark Tool")
    log.info(f"Version: {__version__}")

    # Find Ollama
    ollama = get_ollama_binary()
    if not ollama:
        log.error("Ollama not found")
        return 1

    # Start server if needed
    root = get_root_path()
    models_path = root / "modules" / "models"
    server_process = start_ollama_server(ollama, models_path)

    if not check_ollama_server():
        log.error("Ollama server not available")
        return 1

    try:
        # Get models to benchmark
        available_models = list_models()
        log.info(f"Available models: {available_models}")

        models_to_benchmark = []

        if args.model:
            if args.model not in available_models:
                log.error(f"Model not found: {args.model}")
                log.info(f"Available: {available_models}")
                return 1
            models_to_benchmark = [args.model]
        elif args.compare:
            models_to_benchmark = [m for m in args.compare if m in available_models]
        elif args.all:
            models_to_benchmark = available_models
        else:
            # Default: benchmark first available model
            if available_models:
                models_to_benchmark = [available_models[0]]

        if not models_to_benchmark:
            log.error("No models to benchmark")
            return 1

        # Configure iterations
        iterations = 1 if args.quick else args.iterations

        # Configure prompts
        prompts = BENCHMARK_PROMPTS
        if args.quick:
            prompts = [p for p in BENCHMARK_PROMPTS if p["id"] in ["simple", "medium"]]

        # Run benchmarks
        benchmarks = {}

        for model in models_to_benchmark:
            if args.quality_only:
                benchmark = ModelBenchmark(model=model, quantization="unknown")
                benchmark.quality_results = run_quality_check(model)
            else:
                benchmark = run_benchmark(model, iterations, prompts)
                benchmark.quality_results = run_quality_check(model)

            benchmarks[model] = benchmark

        # Print report
        print_benchmark_report(benchmarks)

        # Save results if requested
        if args.output:
            save_benchmark_results(benchmarks, args.output)
        else:
            # Default output location
            output_path = root / "modules" / "models" / "config" / "benchmark_results.json"
            save_benchmark_results(benchmarks, output_path)

    finally:
        stop_ollama_server(server_process)

    return 0


if __name__ == "__main__":
    sys.exit(main())
