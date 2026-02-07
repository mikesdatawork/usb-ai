#!/usr/bin/env python3
"""
fast_inference.py

Fast inference wrapper with minimal overhead for USB-AI.
Provides a simple, high-performance interface for Ollama inference.

Features:
- Pre-compiled request templates
- Connection pooling with keep-alive
- Streaming with optimal chunk size
- Latency percentile metrics
- Profile-based configuration

Usage:
    from fast_inference import FastInference

    # Quick usage
    client = FastInference(model="dolphin-llama3:8b")
    response = client.generate("What is 2+2?")
    print(response)

    # Streaming
    for token in client.stream("Tell me a story"):
        print(token, end="", flush=True)

    # With metrics
    print(client.get_metrics())

    # Using profiles
    client = FastInference(model="dolphin-llama3:8b", profile="turbo")
"""

import collections
import http.client
import json
import logging
import queue
import socket
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

__version__ = "1.0.0"

log = logging.getLogger(__name__)


# =============================================================================
# Profile Configurations (Inline for minimal dependencies)
# =============================================================================


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration."""

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.0

    # Generation
    num_predict: int = 256
    num_ctx: int = 2048

    # Streaming
    stream: bool = True
    chunk_buffer_size: int = 1024

    # Connection
    timeout: float = 30.0
    keep_alive: bool = True
    max_connections: int = 4


PROFILES: Dict[str, InferenceConfig] = {
    "realtime": InferenceConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.0,
        num_predict=256,
        num_ctx=2048,
        stream=True,
        chunk_buffer_size=1024,
        timeout=30.0,
        keep_alive=True,
        max_connections=4,
    ),
    "throughput": InferenceConfig(
        temperature=0.5,
        top_p=0.85,
        top_k=30,
        repeat_penalty=1.0,
        num_predict=512,
        num_ctx=4096,
        stream=False,
        chunk_buffer_size=8192,
        timeout=120.0,
        keep_alive=True,
        max_connections=16,
    ),
    "quality": InferenceConfig(
        temperature=0.8,
        top_p=0.95,
        top_k=100,
        repeat_penalty=1.1,
        num_predict=1024,
        num_ctx=8192,
        stream=True,
        chunk_buffer_size=4096,
        timeout=180.0,
        keep_alive=True,
        max_connections=4,
    ),
    "turbo": InferenceConfig(
        temperature=0.3,
        top_p=0.7,
        top_k=20,
        repeat_penalty=1.0,
        num_predict=128,
        num_ctx=1024,
        stream=True,
        chunk_buffer_size=2048,
        timeout=15.0,
        keep_alive=True,
        max_connections=8,
    ),
}


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    @property
    def total_time_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.end_time is None or self.completion_tokens == 0:
            return None
        duration = self.end_time - self.start_time
        return self.completion_tokens / duration if duration > 0 else None


class MetricsCollector:
    """Collects inference metrics with percentile calculations."""

    def __init__(self, max_samples: int = 1000):
        self._metrics: Deque[RequestMetrics] = collections.deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record(self, metrics: RequestMetrics):
        """Record request metrics."""
        with self._lock:
            self._metrics.append(metrics)

    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)
        return sorted_values[idx]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary with percentiles."""
        with self._lock:
            metrics_list = list(self._metrics)

        if not metrics_list:
            return {
                "total_requests": 0,
                "error_rate": 0.0,
                "time_to_first_token_ms": {},
                "total_time_ms": {},
                "tokens_per_second": {},
            }

        ttft_values = [
            m.time_to_first_token_ms
            for m in metrics_list
            if m.time_to_first_token_ms is not None
        ]
        total_time_values = [
            m.total_time_ms for m in metrics_list if m.total_time_ms is not None
        ]
        tps_values = [
            m.tokens_per_second
            for m in metrics_list
            if m.tokens_per_second is not None
        ]
        errors = sum(1 for m in metrics_list if m.error is not None)

        percentiles = [50, 75, 90, 95, 99]

        return {
            "total_requests": len(metrics_list),
            "error_rate": errors / len(metrics_list),
            "time_to_first_token_ms": {
                f"p{p}": self._percentile(ttft_values, p) for p in percentiles
            },
            "total_time_ms": {
                f"p{p}": self._percentile(total_time_values, p) for p in percentiles
            },
            "tokens_per_second": {
                "mean": statistics.mean(tps_values) if tps_values else 0.0,
                "median": statistics.median(tps_values) if tps_values else 0.0,
                "min": min(tps_values) if tps_values else 0.0,
                "max": max(tps_values) if tps_values else 0.0,
            },
            "total_prompt_tokens": sum(m.prompt_tokens for m in metrics_list),
            "total_completion_tokens": sum(m.completion_tokens for m in metrics_list),
        }

    def reset(self):
        """Reset metrics."""
        with self._lock:
            self._metrics.clear()


# =============================================================================
# Connection Pool
# =============================================================================


class ConnectionPool:
    """Thread-safe HTTP connection pool."""

    def __init__(
        self,
        host: str,
        port: int,
        max_connections: int = 10,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self._created = 0
        self._lock = threading.Lock()

    def _create_connection(self) -> http.client.HTTPConnection:
        """Create a new HTTP connection."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return conn

    def _is_connection_alive(self, conn: http.client.HTTPConnection) -> bool:
        """Check if connection is still alive."""
        try:
            if conn.sock is None:
                return False
            # Check if socket is readable (would indicate closed or error)
            conn.sock.setblocking(False)
            try:
                data = conn.sock.recv(1, socket.MSG_PEEK)
                if not data:
                    return False
            except BlockingIOError:
                # No data available, connection is alive
                pass
            except Exception:
                return False
            finally:
                conn.sock.setblocking(True)
            return True
        except Exception:
            return False

    def acquire(self) -> http.client.HTTPConnection:
        """Acquire a connection from the pool."""
        # Try to get from pool
        try:
            conn = self._pool.get_nowait()
            if self._is_connection_alive(conn):
                return conn
            # Connection dead, create new
        except queue.Empty:
            pass

        # Create new connection if under limit
        with self._lock:
            if self._created < self.max_connections:
                self._created += 1
                return self._create_connection()

        # Wait for connection from pool
        conn = self._pool.get(timeout=self.timeout)
        if self._is_connection_alive(conn):
            return conn
        return self._create_connection()

    def release(self, conn: http.client.HTTPConnection):
        """Release connection back to pool."""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()

    def close(self):
        """Close all connections."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


# =============================================================================
# Request Template
# =============================================================================


class RequestTemplate:
    """Pre-compiled request template for minimal overhead."""

    def __init__(self, model: str, config: InferenceConfig):
        self.model = model
        self.config = config

        # Pre-compile options that don't change
        self._base_options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "num_predict": config.num_predict,
            "num_ctx": config.num_ctx,
        }

        # Only add repeat_penalty if not 1.0 (saves computation)
        if config.repeat_penalty != 1.0:
            self._base_options["repeat_penalty"] = config.repeat_penalty

        # Pre-compile headers
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson" if config.stream else "application/json",
            "Connection": "keep-alive" if config.keep_alive else "close",
        }

    def build(
        self,
        prompt: str,
        system: Optional[str] = None,
        **overrides: Any,
    ) -> Tuple[bytes, Dict[str, str]]:
        """Build request payload."""
        options = {**self._base_options}
        if overrides:
            options.update(overrides)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self.config.stream,
            "options": options,
        }

        if system:
            payload["system"] = system

        return json.dumps(payload).encode("utf-8"), self._headers


# =============================================================================
# Fast Inference Client
# =============================================================================


class FastInference:
    """
    High-performance inference client for Ollama.

    Optimized for:
    - Minimal latency with connection pooling
    - Pre-compiled request templates
    - Streaming with optimal chunk size
    - Comprehensive metrics collection

    Example:
        client = FastInference(model="dolphin-llama3:8b", profile="realtime")

        # Synchronous generation
        response = client.generate("What is 2+2?")

        # Streaming
        for token in client.stream("Tell me a story"):
            print(token, end="")

        # Get metrics
        print(client.get_metrics())
    """

    def __init__(
        self,
        model: str = "dolphin-llama3:8b",
        host: str = "127.0.0.1",
        port: int = 11434,
        profile: str = "realtime",
        config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize the fast inference client.

        Args:
            model: Model name (e.g., "dolphin-llama3:8b")
            host: Ollama host
            port: Ollama port
            profile: Profile name (realtime, throughput, quality, turbo)
            config: Optional custom InferenceConfig (overrides profile)
        """
        self.model = model
        self.host = host
        self.port = port
        self.profile_name = profile

        # Get configuration
        if config:
            self.config = config
        elif profile in PROFILES:
            self.config = PROFILES[profile]
        else:
            raise ValueError(f"Unknown profile: {profile}")

        # Initialize components
        self._pool = ConnectionPool(
            host=host,
            port=port,
            max_connections=self.config.max_connections,
            timeout=self.config.timeout,
        )
        self._template = RequestTemplate(model, self.config)
        self._metrics = MetricsCollector()

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **options: Any,
    ) -> Iterator[str]:
        """
        Generate completion with streaming.

        Args:
            prompt: User prompt
            system: Optional system prompt
            **options: Override sampling options

        Yields:
            Generated tokens
        """
        metrics = RequestMetrics(start_time=time.perf_counter())
        conn = None

        try:
            # Build request
            payload, headers = self._template.build(prompt, system, **options)

            # Acquire connection
            conn = self._pool.acquire()

            # Send request
            conn.request("POST", "/api/generate", body=payload, headers=headers)

            # Get response
            response = conn.getresponse()

            if response.status != 200:
                error_body = response.read().decode("utf-8", errors="replace")
                metrics.error = f"HTTP {response.status}: {error_body[:200]}"
                metrics.end_time = time.perf_counter()
                self._metrics.record(metrics)
                raise RuntimeError(metrics.error)

            # Stream response
            buffer = b""
            buffer_size = self.config.chunk_buffer_size

            for chunk in iter(lambda: response.read(buffer_size), b""):
                if not chunk:
                    break

                if metrics.first_token_time is None:
                    metrics.first_token_time = time.perf_counter()

                buffer += chunk

                # Process complete JSON lines
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if "response" in data:
                            yield data["response"]

                        if "prompt_eval_count" in data:
                            metrics.prompt_tokens = data["prompt_eval_count"]

                        if "eval_count" in data:
                            metrics.completion_tokens = data["eval_count"]

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        continue

            metrics.end_time = time.perf_counter()
            self._metrics.record(metrics)

        except Exception as e:
            if metrics.error is None:
                metrics.error = str(e)
            metrics.end_time = time.perf_counter()
            self._metrics.record(metrics)
            raise

        finally:
            if conn:
                self._pool.release(conn)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **options: Any,
    ) -> str:
        """
        Generate completion (synchronous).

        Args:
            prompt: User prompt
            system: Optional system prompt
            **options: Override sampling options

        Returns:
            Complete generated response
        """
        return "".join(self.stream(prompt, system, **options))

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics summary with percentiles."""
        return self._metrics.get_summary()

    def reset_metrics(self):
        """Reset collected metrics."""
        self._metrics.reset()

    def close(self):
        """Close connections and cleanup."""
        self._pool.close()

    def __enter__(self) -> "FastInference":
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_generate(
    prompt: str,
    model: str = "dolphin-llama3:8b",
    host: str = "127.0.0.1",
    port: int = 11434,
    profile: str = "turbo",
) -> str:
    """
    Quick one-shot generation with minimal setup.

    Example:
        response = quick_generate("What is the capital of France?")
    """
    with FastInference(model=model, host=host, port=port, profile=profile) as client:
        return client.generate(prompt)


def quick_stream(
    prompt: str,
    model: str = "dolphin-llama3:8b",
    host: str = "127.0.0.1",
    port: int = 11434,
    profile: str = "realtime",
) -> Iterator[str]:
    """
    Quick streaming generation with minimal setup.

    Example:
        for token in quick_stream("Tell me a story"):
            print(token, end="")
    """
    client = FastInference(model=model, host=host, port=port, profile=profile)
    try:
        for token in client.stream(prompt):
            yield token
    finally:
        client.close()


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fast Inference CLI for USB-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("prompt", nargs="?", help="Prompt to generate from")

    parser.add_argument(
        "--model",
        "-m",
        default="dolphin-llama3:8b",
        help="Model name (default: dolphin-llama3:8b)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Ollama host (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=11434,
        help="Ollama port (default: 11434)",
    )

    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="realtime",
        help="Inference profile (default: realtime)",
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (return full response at once)",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Show metrics after generation",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark with test prompts",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Benchmark iterations (default: 5)",
    )

    args = parser.parse_args()

    if args.benchmark:
        print(f"\nBenchmarking profile: {args.profile}")
        print("-" * 50)

        test_prompts = [
            "What is 2 + 2?",
            "Explain recursion briefly.",
            "Write a hello world function in Python.",
        ]

        client = FastInference(
            model=args.model,
            host=args.host,
            port=args.port,
            profile=args.profile,
        )

        try:
            # Warm up
            client.generate("Hello")

            # Run benchmark
            for _ in range(args.iterations):
                for prompt in test_prompts:
                    try:
                        _ = client.generate(prompt)
                    except Exception as e:
                        print(f"Error: {e}")

            # Print metrics
            metrics = client.get_metrics()
            print(f"\nResults for {args.profile}:")
            print(f"  Requests: {metrics['total_requests']}")
            print(f"  Error rate: {metrics['error_rate']*100:.1f}%")
            print(f"  TTFT p50: {metrics['time_to_first_token_ms'].get('p50', 0):.1f}ms")
            print(f"  TTFT p95: {metrics['time_to_first_token_ms'].get('p95', 0):.1f}ms")
            print(f"  TPS mean: {metrics['tokens_per_second'].get('mean', 0):.1f}")
            print(f"  TPS max: {metrics['tokens_per_second'].get('max', 0):.1f}")

        finally:
            client.close()

        return 0

    if not args.prompt:
        parser.print_help()
        return 0

    client = FastInference(
        model=args.model,
        host=args.host,
        port=args.port,
        profile=args.profile,
    )

    try:
        if args.no_stream:
            response = client.generate(args.prompt)
            print(response)
        else:
            for token in client.stream(args.prompt):
                print(token, end="", flush=True)
            print()

        if args.metrics:
            metrics = client.get_metrics()
            print(f"\n--- Metrics ---")
            print(f"TTFT: {metrics['time_to_first_token_ms'].get('p50', 0):.1f}ms")
            print(f"TPS: {metrics['tokens_per_second'].get('mean', 0):.1f}")

    finally:
        client.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
