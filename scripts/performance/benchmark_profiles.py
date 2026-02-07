#!/usr/bin/env python3
"""
benchmark_profiles.py

Benchmarks inference profiles and documents speed vs quality tradeoffs for USB-AI.

This script:
1. Tests each profile (realtime, throughput, quality, turbo)
2. Measures latency percentiles and throughput
3. Evaluates quality metrics (coherence, accuracy)
4. Generates a comprehensive comparison report

Usage:
    python benchmark_profiles.py
    python benchmark_profiles.py --iterations 10 --output results.json
    python benchmark_profiles.py --profile realtime --quick
"""

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Profile Definitions
# =============================================================================


PROFILES = {
    "realtime": {
        "description": "Minimal latency for interactive chat",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
        "num_predict": 256,
        "num_ctx": 2048,
        "stream": True,
    },
    "throughput": {
        "description": "Batch optimized for API backends",
        "temperature": 0.5,
        "top_p": 0.85,
        "top_k": 30,
        "repeat_penalty": 1.0,
        "num_predict": 512,
        "num_ctx": 4096,
        "stream": False,
    },
    "quality": {
        "description": "Full sampling for highest quality",
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 100,
        "repeat_penalty": 1.1,
        "num_predict": 1024,
        "num_ctx": 8192,
        "stream": True,
    },
    "turbo": {
        "description": "Maximum speed with quality tradeoffs",
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "num_predict": 128,
        "num_ctx": 1024,
        "stream": True,
    },
}


# =============================================================================
# Benchmark Prompts
# =============================================================================


BENCHMARK_PROMPTS = [
    {
        "id": "simple",
        "prompt": "What is 2 + 2?",
        "category": "math_simple",
        "expected_length": "short",
    },
    {
        "id": "factual",
        "prompt": "What is the capital of France?",
        "category": "knowledge",
        "expected_length": "short",
    },
    {
        "id": "explanation",
        "prompt": "Explain what recursion is in programming in 2-3 sentences.",
        "category": "explanation",
        "expected_length": "medium",
    },
    {
        "id": "code_simple",
        "prompt": "Write a Python function that returns the sum of two numbers.",
        "category": "code",
        "expected_length": "medium",
    },
    {
        "id": "reasoning",
        "prompt": "If a train leaves at 9 AM going 60 mph and another at 10 AM going 80 mph, when do they meet if they're 200 miles apart?",
        "category": "reasoning",
        "expected_length": "medium",
    },
]


QUALITY_PROMPTS = [
    {
        "id": "math_accuracy",
        "prompt": "What is 15 * 8? Reply with just the number.",
        "expected": "120",
        "type": "exact",
    },
    {
        "id": "capital_accuracy",
        "prompt": "What is the capital of Japan? Reply with just the city name.",
        "expected": "Tokyo",
        "type": "contains",
    },
    {
        "id": "logic",
        "prompt": "Is the statement 'All cats are animals, therefore all animals are cats' logically valid? Answer Yes or No.",
        "expected": "No",
        "type": "contains",
    },
]


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""

    profile: str
    prompt_id: str
    first_token_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    response_length: int


@dataclass
class QualityResult:
    """Result for a quality check."""

    profile: str
    prompt_id: str
    expected: str
    actual: str
    passed: bool
    response_time_ms: float


@dataclass
class ProfileBenchmark:
    """Aggregated benchmark results for a profile."""

    profile: str
    config: Dict[str, Any]
    results: List[BenchmarkResult] = field(default_factory=list)
    quality_results: List[QualityResult] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.first_token_ms for r in self.results)

    @property
    def p50_ttft_ms(self) -> float:
        if not self.results:
            return 0.0
        return statistics.median([r.first_token_ms for r in self.results])

    @property
    def p95_ttft_ms(self) -> float:
        if not self.results:
            return 0.0
        values = sorted([r.first_token_ms for r in self.results])
        idx = min(int(len(values) * 0.95), len(values) - 1)
        return values[idx]

    @property
    def avg_tps(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.tokens_per_second for r in self.results)

    @property
    def max_tps(self) -> float:
        if not self.results:
            return 0.0
        return max(r.tokens_per_second for r in self.results)

    @property
    def quality_score(self) -> float:
        if not self.quality_results:
            return 0.0
        passed = sum(1 for r in self.quality_results if r.passed)
        return (passed / len(self.quality_results)) * 100


# =============================================================================
# Ollama Client
# =============================================================================


def check_ollama(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Check if Ollama is running."""
    try:
        url = f"http://{host}:{port}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


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


def generate(
    model: str,
    prompt: str,
    profile_config: Dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 11434,
) -> Tuple[str, int, float, float]:
    """
    Generate completion with timing.

    Returns: (response, tokens, first_token_ms, total_ms)
    """
    url = f"http://{host}:{port}/api/generate"

    options = {
        "temperature": profile_config["temperature"],
        "top_p": profile_config["top_p"],
        "top_k": profile_config["top_k"],
        "num_predict": profile_config["num_predict"],
        "num_ctx": profile_config["num_ctx"],
    }

    if profile_config.get("repeat_penalty", 1.0) != 1.0:
        options["repeat_penalty"] = profile_config["repeat_penalty"]

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": profile_config.get("stream", True),
        "options": options,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.perf_counter()
    first_token_time = None
    response_text = ""
    tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        response_text += data["response"]
                    if "eval_count" in data:
                        tokens = data["eval_count"]
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        log.error(f"Generation error: {e}")
        return "", 0, 0.0, 0.0

    end_time = time.perf_counter()

    first_token_ms = ((first_token_time or end_time) - start_time) * 1000
    total_ms = (end_time - start_time) * 1000

    return response_text, tokens, first_token_ms, total_ms


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_profile(
    profile_name: str,
    model: str,
    iterations: int = 3,
    host: str = "127.0.0.1",
    port: int = 11434,
) -> ProfileBenchmark:
    """Benchmark a single profile."""
    config = PROFILES[profile_name]
    benchmark = ProfileBenchmark(profile=profile_name, config=config)

    log.info(f"Benchmarking profile: {profile_name}")
    log.info(f"  Config: temp={config['temperature']}, top_p={config['top_p']}, "
             f"top_k={config['top_k']}, ctx={config['num_ctx']}")

    # Warm up
    log.info("  Warming up...")
    generate(model, "Hello", config, host, port)
    time.sleep(0.5)

    # Benchmark iterations
    for i in range(iterations):
        log.info(f"  Iteration {i + 1}/{iterations}")

        for prompt_config in BENCHMARK_PROMPTS:
            response, tokens, first_ms, total_ms = generate(
                model, prompt_config["prompt"], config, host, port
            )

            if tokens > 0 and total_ms > 0:
                tps = (tokens / total_ms) * 1000

                result = BenchmarkResult(
                    profile=profile_name,
                    prompt_id=prompt_config["id"],
                    first_token_ms=first_ms,
                    total_time_ms=total_ms,
                    tokens_generated=tokens,
                    tokens_per_second=tps,
                    response_length=len(response),
                )
                benchmark.results.append(result)

            time.sleep(0.2)

    # Quality checks
    log.info("  Running quality checks...")
    for quality_config in QUALITY_PROMPTS:
        response, _, _, response_ms = generate(
            model, quality_config["prompt"], config, host, port
        )

        response_clean = response.strip().lower()
        expected_clean = quality_config["expected"].lower()

        if quality_config["type"] == "exact":
            passed = expected_clean in response_clean
        else:
            passed = expected_clean in response_clean

        quality_result = QualityResult(
            profile=profile_name,
            prompt_id=quality_config["id"],
            expected=quality_config["expected"],
            actual=response.strip()[:100],
            passed=passed,
            response_time_ms=response_ms,
        )
        benchmark.quality_results.append(quality_result)

    return benchmark


def benchmark_all_profiles(
    model: str,
    iterations: int = 3,
    host: str = "127.0.0.1",
    port: int = 11434,
) -> Dict[str, ProfileBenchmark]:
    """Benchmark all profiles."""
    results = {}

    for profile_name in PROFILES.keys():
        results[profile_name] = benchmark_profile(
            profile_name, model, iterations, host, port
        )
        time.sleep(1)  # Brief pause between profiles

    return results


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(benchmarks: Dict[str, ProfileBenchmark], model: str) -> str:
    """Generate formatted benchmark report."""
    lines = []

    lines.append("")
    lines.append("=" * 80)
    lines.append("          USB-AI INFERENCE PROFILE BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Platform:  {platform.system()} {platform.machine()}")
    lines.append(f"Model:     {model}")
    lines.append("")

    # Summary table
    lines.append("-" * 80)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 80)
    lines.append("")

    header = f"{'Profile':<12} {'TTFT p50':<10} {'TTFT p95':<10} {'TPS Avg':<10} {'TPS Max':<10} {'Quality':<10}"
    lines.append(header)
    lines.append("-" * 80)

    for profile, benchmark in benchmarks.items():
        lines.append(
            f"{profile:<12} "
            f"{benchmark.p50_ttft_ms:<10.0f} "
            f"{benchmark.p95_ttft_ms:<10.0f} "
            f"{benchmark.avg_tps:<10.1f} "
            f"{benchmark.max_tps:<10.1f} "
            f"{benchmark.quality_score:<10.0f}%"
        )

    lines.append("")

    # Speed vs Quality Analysis
    lines.append("-" * 80)
    lines.append("SPEED VS QUALITY TRADEOFFS")
    lines.append("-" * 80)
    lines.append("")

    # Sort by speed
    sorted_by_speed = sorted(benchmarks.items(), key=lambda x: x[1].avg_tps, reverse=True)
    lines.append("Ranked by Speed (tokens/second):")
    for i, (profile, benchmark) in enumerate(sorted_by_speed, 1):
        lines.append(f"  {i}. {profile}: {benchmark.avg_tps:.1f} tok/s")

    lines.append("")

    # Sort by latency
    sorted_by_latency = sorted(benchmarks.items(), key=lambda x: x[1].p50_ttft_ms)
    lines.append("Ranked by Latency (time to first token):")
    for i, (profile, benchmark) in enumerate(sorted_by_latency, 1):
        lines.append(f"  {i}. {profile}: {benchmark.p50_ttft_ms:.0f}ms")

    lines.append("")

    # Sort by quality
    sorted_by_quality = sorted(benchmarks.items(), key=lambda x: x[1].quality_score, reverse=True)
    lines.append("Ranked by Quality Score:")
    for i, (profile, benchmark) in enumerate(sorted_by_quality, 1):
        lines.append(f"  {i}. {profile}: {benchmark.quality_score:.0f}%")

    lines.append("")

    # Profile recommendations
    lines.append("-" * 80)
    lines.append("PROFILE RECOMMENDATIONS")
    lines.append("-" * 80)
    lines.append("")

    recommendations = {
        "realtime": [
            "Best for: Interactive chat, voice assistants, live demos",
            "Tradeoffs: May truncate complex answers, smaller context",
            "Speed: Good first-token latency, moderate throughput",
        ],
        "throughput": [
            "Best for: API backends, batch processing, document analysis",
            "Tradeoffs: Higher individual latency, non-streaming",
            "Speed: Highest aggregate throughput for concurrent requests",
        ],
        "quality": [
            "Best for: Creative writing, code generation, research",
            "Tradeoffs: Slower, uses more resources",
            "Speed: Slowest but highest quality output",
        ],
        "turbo": [
            "Best for: Quick answers, simple queries, autocomplete",
            "Tradeoffs: Very focused output, may miss nuances",
            "Speed: Fastest overall, lowest latency",
        ],
    }

    for profile, recs in recommendations.items():
        lines.append(f"{profile.upper()}:")
        for rec in recs:
            lines.append(f"  - {rec}")
        lines.append("")

    # Detailed results by prompt
    lines.append("-" * 80)
    lines.append("DETAILED RESULTS BY PROMPT")
    lines.append("-" * 80)
    lines.append("")

    for profile, benchmark in benchmarks.items():
        lines.append(f"{profile.upper()} Profile:")

        # Group by prompt
        by_prompt: Dict[str, List[BenchmarkResult]] = {}
        for result in benchmark.results:
            if result.prompt_id not in by_prompt:
                by_prompt[result.prompt_id] = []
            by_prompt[result.prompt_id].append(result)

        for prompt_id, results in by_prompt.items():
            avg_ttft = statistics.mean(r.first_token_ms for r in results)
            avg_tps = statistics.mean(r.tokens_per_second for r in results)
            avg_tokens = statistics.mean(r.tokens_generated for r in results)

            lines.append(
                f"  {prompt_id:<15} TTFT: {avg_ttft:>6.0f}ms  "
                f"TPS: {avg_tps:>5.1f}  Tokens: {avg_tokens:>4.0f}"
            )

        lines.append("")

    # Quality check details
    lines.append("-" * 80)
    lines.append("QUALITY CHECK DETAILS")
    lines.append("-" * 80)
    lines.append("")

    for profile, benchmark in benchmarks.items():
        passed = sum(1 for r in benchmark.quality_results if r.passed)
        total = len(benchmark.quality_results)
        lines.append(f"{profile.upper()}: {passed}/{total} passed ({benchmark.quality_score:.0f}%)")

        for qr in benchmark.quality_results:
            status = "PASS" if qr.passed else "FAIL"
            lines.append(f"  [{status}] {qr.prompt_id}: expected '{qr.expected}', got '{qr.actual[:50]}...'")

        lines.append("")

    # Configuration summary
    lines.append("-" * 80)
    lines.append("PROFILE CONFIGURATIONS")
    lines.append("-" * 80)
    lines.append("")

    config_header = f"{'Profile':<12} {'Temp':<6} {'TopP':<6} {'TopK':<6} {'Ctx':<6} {'MaxTok':<8} {'Repeat':<8}"
    lines.append(config_header)
    lines.append("-" * 80)

    for profile, config in PROFILES.items():
        lines.append(
            f"{profile:<12} "
            f"{config['temperature']:<6.1f} "
            f"{config['top_p']:<6.2f} "
            f"{config['top_k']:<6} "
            f"{config['num_ctx']:<6} "
            f"{config['num_predict']:<8} "
            f"{config['repeat_penalty']:<8.1f}"
        )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_results(
    benchmarks: Dict[str, ProfileBenchmark],
    output_path: Path,
    model: str,
):
    """Save results to JSON."""
    data = {
        "version": __version__,
        "generated": datetime.now().isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "model": model,
        "profiles": {},
    }

    for profile, benchmark in benchmarks.items():
        data["profiles"][profile] = {
            "config": benchmark.config,
            "metrics": {
                "avg_ttft_ms": benchmark.avg_ttft_ms,
                "p50_ttft_ms": benchmark.p50_ttft_ms,
                "p95_ttft_ms": benchmark.p95_ttft_ms,
                "avg_tps": benchmark.avg_tps,
                "max_tps": benchmark.max_tps,
                "quality_score": benchmark.quality_score,
            },
            "results": [
                {
                    "prompt_id": r.prompt_id,
                    "first_token_ms": round(r.first_token_ms, 2),
                    "total_time_ms": round(r.total_time_ms, 2),
                    "tokens_generated": r.tokens_generated,
                    "tokens_per_second": round(r.tokens_per_second, 2),
                }
                for r in benchmark.results
            ],
            "quality_results": [
                {
                    "prompt_id": qr.prompt_id,
                    "expected": qr.expected,
                    "actual": qr.actual,
                    "passed": qr.passed,
                }
                for qr in benchmark.quality_results
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Results saved to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark inference profiles for USB-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        "-m",
        default="dolphin-llama3:8b",
        help="Model to benchmark (default: dolphin-llama3:8b)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Ollama host (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Ollama port (default: 11434)",
    )

    parser.add_argument(
        "--profile",
        "-p",
        choices=list(PROFILES.keys()),
        help="Benchmark single profile only",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=3,
        help="Number of iterations per prompt (default: 3)",
    )

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick mode (1 iteration)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )

    args = parser.parse_args()

    # Check Ollama
    if not check_ollama(args.host, args.port):
        log.error("Ollama is not running. Please start it first.")
        return 1

    # Check model
    available = list_models(args.host, args.port)
    if args.model not in available:
        log.error(f"Model not found: {args.model}")
        log.info(f"Available models: {available}")
        return 1

    # Run benchmark
    iterations = 1 if args.quick else args.iterations

    if args.profile:
        benchmarks = {
            args.profile: benchmark_profile(
                args.profile, args.model, iterations, args.host, args.port
            )
        }
    else:
        benchmarks = benchmark_all_profiles(
            args.model, iterations, args.host, args.port
        )

    # Output results
    if args.json:
        data = {
            profile: {
                "avg_ttft_ms": b.avg_ttft_ms,
                "p50_ttft_ms": b.p50_ttft_ms,
                "p95_ttft_ms": b.p95_ttft_ms,
                "avg_tps": b.avg_tps,
                "max_tps": b.max_tps,
                "quality_score": b.quality_score,
            }
            for profile, b in benchmarks.items()
        }
        print(json.dumps(data, indent=2))
    else:
        report = generate_report(benchmarks, args.model)
        print(report)

    # Save results
    if args.output:
        save_results(benchmarks, args.output, args.model)
    else:
        # Default output location
        root = Path(__file__).parent.parent.parent
        output = root / "modules" / "config" / "benchmark_results.json"
        save_results(benchmarks, output, args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
