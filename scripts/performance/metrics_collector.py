#!/usr/bin/env python3
"""
USB-AI Performance Metrics Collector
Monitors and logs LLM response times, accuracy, and system performance.
"""

import json
import time
import threading
import statistics
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "performance"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"


@dataclass
class InferenceMetric:
    """Single inference measurement."""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float
    time_to_first_token_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


@dataclass
class PerformanceReport:
    """Aggregated performance report."""
    session_id: str
    start_time: str
    end_time: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    avg_tokens_per_second: float
    avg_time_to_first_token_ms: float
    models_tested: List[str]
    metrics: List[Dict[str, Any]] = field(default_factory=list)


class MetricsCollector:
    """Collects and analyzes LLM performance metrics."""

    def __init__(self):
        self.metrics: List[InferenceMetric] = []
        self.session_id = f"perf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.start_time = datetime.now()
        self.log_file = LOGS_DIR / f"{self.session_id}.jsonl"
        self._lock = threading.Lock()

    def measure_inference(
        self,
        model: str,
        prompt: str,
        stream: bool = True
    ) -> InferenceMetric:
        """Measure a single inference request."""
        start_time = time.perf_counter()
        first_token_time = None
        completion_tokens = 0
        prompt_tokens = 0
        error = None
        success = True

        try:
            data = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": stream
            }).encode()

            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                if stream:
                    for line in response:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        chunk = json.loads(line.decode())
                        if "response" in chunk:
                            completion_tokens += 1
                        if chunk.get("done"):
                            prompt_tokens = chunk.get("prompt_eval_count", 0)
                            completion_tokens = chunk.get("eval_count", completion_tokens)
                else:
                    first_token_time = time.perf_counter()
                    result = json.loads(response.read().decode())
                    prompt_tokens = result.get("prompt_eval_count", 0)
                    completion_tokens = result.get("eval_count", 0)

        except Exception as e:
            success = False
            error = str(e)
            if first_token_time is None:
                first_token_time = time.perf_counter()

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms

        # Calculate tokens per second (only generation time, not prompt processing)
        gen_time = end_time - first_token_time if first_token_time else end_time - start_time
        tps = completion_tokens / gen_time if gen_time > 0 else 0

        metric = InferenceMetric(
            timestamp=datetime.now().isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_time_ms=round(total_time_ms, 2),
            time_to_first_token_ms=round(ttft_ms, 2),
            tokens_per_second=round(tps, 2),
            success=success,
            error=error
        )

        with self._lock:
            self.metrics.append(metric)
            self._log_metric(metric)

        return metric

    def _log_metric(self, metric: InferenceMetric):
        """Log metric to file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(metric)) + "\n")

    def run_benchmark(
        self,
        models: List[str] = None,
        prompts: List[str] = None,
        iterations: int = 3
    ) -> PerformanceReport:
        """Run a full benchmark suite."""
        if models is None:
            models = ["llama3.2:latest"]

        if prompts is None:
            prompts = [
                "What is 2+2?",
                "Explain quantum computing in one sentence.",
                "Write a haiku about programming.",
                "List 3 prime numbers.",
                "What color is the sky?",
            ]

        print(f"\n{'='*60}")
        print("USB-AI Performance Benchmark")
        print(f"{'='*60}")
        print(f"Models: {', '.join(models)}")
        print(f"Prompts: {len(prompts)}")
        print(f"Iterations: {iterations}")
        print(f"{'='*60}\n")

        for model in models:
            print(f"\nTesting model: {model}")
            print("-" * 40)

            # Warmup
            print("  Warming up...", end=" ", flush=True)
            self.measure_inference(model, "Hello", stream=False)
            print("done")

            for i, prompt in enumerate(prompts):
                for j in range(iterations):
                    print(f"  [{i+1}/{len(prompts)}] Iteration {j+1}/{iterations}...", end=" ", flush=True)
                    metric = self.measure_inference(model, prompt, stream=True)
                    status = "✓" if metric.success else "✗"
                    print(f"{status} {metric.total_time_ms:.0f}ms, {metric.tokens_per_second:.1f} tok/s")

        return self.generate_report()

    def generate_report(self) -> PerformanceReport:
        """Generate performance report from collected metrics."""
        successful = [m for m in self.metrics if m.success]

        if not successful:
            return PerformanceReport(
                session_id=self.session_id,
                start_time=self.start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                total_requests=len(self.metrics),
                successful_requests=0,
                failed_requests=len(self.metrics),
                avg_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                avg_tokens_per_second=0,
                avg_time_to_first_token_ms=0,
                models_tested=list(set(m.model for m in self.metrics))
            )

        response_times = [m.total_time_ms for m in successful]
        tps_values = [m.tokens_per_second for m in successful]
        ttft_values = [m.time_to_first_token_ms for m in successful]

        def percentile(data, p):
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        report = PerformanceReport(
            session_id=self.session_id,
            start_time=self.start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=len(self.metrics) - len(successful),
            avg_response_time_ms=round(statistics.mean(response_times), 2),
            p50_response_time_ms=round(percentile(response_times, 50), 2),
            p95_response_time_ms=round(percentile(response_times, 95), 2),
            p99_response_time_ms=round(percentile(response_times, 99), 2),
            avg_tokens_per_second=round(statistics.mean(tps_values), 2),
            avg_time_to_first_token_ms=round(statistics.mean(ttft_values), 2),
            models_tested=list(set(m.model for m in self.metrics)),
            metrics=[asdict(m) for m in self.metrics]
        )

        # Save report
        report_file = LOGS_DIR / f"{self.session_id}-report.json"
        with open(report_file, "w") as f:
            json.dump(asdict(report), f, indent=2)

        return report

    def print_report(self, report: PerformanceReport):
        """Print formatted report."""
        print(f"\n{'='*60}")
        print("PERFORMANCE REPORT")
        print(f"{'='*60}")
        print(f"Session: {report.session_id}")
        print(f"Duration: {report.start_time} to {report.end_time}")
        print(f"\n{'─'*60}")
        print("SUMMARY")
        print(f"{'─'*60}")
        print(f"  Total Requests:     {report.total_requests}")
        print(f"  Successful:         {report.successful_requests}")
        print(f"  Failed:             {report.failed_requests}")
        print(f"  Success Rate:       {report.successful_requests/report.total_requests*100:.1f}%")
        print(f"\n{'─'*60}")
        print("LATENCY")
        print(f"{'─'*60}")
        print(f"  Avg Response Time:  {report.avg_response_time_ms:.0f} ms")
        print(f"  P50 Response Time:  {report.p50_response_time_ms:.0f} ms")
        print(f"  P95 Response Time:  {report.p95_response_time_ms:.0f} ms")
        print(f"  P99 Response Time:  {report.p99_response_time_ms:.0f} ms")
        print(f"  Avg TTFT:           {report.avg_time_to_first_token_ms:.0f} ms")
        print(f"\n{'─'*60}")
        print("THROUGHPUT")
        print(f"{'─'*60}")
        print(f"  Avg Tokens/sec:     {report.avg_tokens_per_second:.1f}")
        print(f"\n{'─'*60}")
        print("MODELS TESTED")
        print(f"{'─'*60}")
        for model in report.models_tested:
            print(f"  • {model}")
        print(f"\n{'='*60}")
        print(f"Full report: {LOGS_DIR / f'{report.session_id}-report.json'}")
        print(f"Metrics log: {self.log_file}")


def run_quick_test():
    """Run a quick performance test."""
    collector = MetricsCollector()
    report = collector.run_benchmark(
        models=["llama3.2:latest"],
        prompts=[
            "What is 2+2?",
            "Say hello.",
            "Name a color.",
        ],
        iterations=2
    )
    collector.print_report(report)
    return report


def run_full_benchmark():
    """Run full benchmark on all models."""
    collector = MetricsCollector()
    report = collector.run_benchmark(
        models=["llama3.2:latest", "dolphin-llama3:8b"],
        iterations=3
    )
    collector.print_report(report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="USB-AI Performance Metrics Collector")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    parser.add_argument("--model", type=str, help="Test specific model")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per prompt")

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.full:
        run_full_benchmark()
    elif args.model:
        collector = MetricsCollector()
        report = collector.run_benchmark(models=[args.model], iterations=args.iterations)
        collector.print_report(report)
    else:
        print("USB-AI Performance Metrics Collector")
        print("\nUsage:")
        print("  python metrics_collector.py --quick    # Quick test")
        print("  python metrics_collector.py --full     # Full benchmark")
        print("  python metrics_collector.py --model llama3.2:latest")
        print("\nRunning quick test...")
        run_quick_test()
