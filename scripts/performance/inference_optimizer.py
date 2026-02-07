#!/usr/bin/env python3
"""
inference_optimizer.py

Optimizes the entire inference pipeline from request to response for USB-AI.
Features:
- Request batching for concurrent users
- Streaming chunk size optimization
- Connection reuse and pooling
- Response buffering strategies
- Async request handling
- Speculative decoding hints
- Optimal temperature/top_p for speed
- Metrics collection (latency percentiles)

Usage:
    python inference_optimizer.py --profile realtime
    python inference_optimizer.py --benchmark --iterations 10
    python inference_optimizer.py --server --port 3001
"""

import asyncio
import collections
import http.client
import json
import logging
import os
import platform
import queue
import socket
import statistics
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlparse

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Inference Profiles
# =============================================================================


class InferenceProfile(Enum):
    """Predefined inference profiles for different use cases."""

    REALTIME = "realtime"  # Minimal latency, streaming
    THROUGHPUT = "throughput"  # Batch optimized
    QUALITY = "quality"  # Full sampling parameters
    TURBO = "turbo"  # Speed over quality tradeoffs


@dataclass(frozen=True)
class ProfileConfig:
    """Configuration for an inference profile."""

    name: str
    description: str
    # Sampling parameters
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    # Generation parameters
    num_predict: int
    num_ctx: int
    # Performance parameters
    stream: bool
    stream_chunk_size: int  # Tokens per chunk
    batch_size: int  # For concurrent batching
    # Connection parameters
    connection_timeout: float
    read_timeout: float
    keep_alive: bool
    max_connections: int
    # Buffer settings
    response_buffer_size: int
    prefetch_tokens: int
    # Advanced options
    mirostat: int  # 0=disabled, 1=mirostat1, 2=mirostat2
    mirostat_eta: float
    mirostat_tau: float
    use_flash_attention: bool
    speculative_decoding: bool


# Profile definitions
PROFILE_CONFIGS: Dict[str, ProfileConfig] = {
    "realtime": ProfileConfig(
        name="realtime",
        description="Minimal latency, streaming optimized for interactive chat",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.0,  # Disabled for speed
        num_predict=256,
        num_ctx=2048,
        stream=True,
        stream_chunk_size=1,  # Token by token for lowest latency
        batch_size=1,
        connection_timeout=5.0,
        read_timeout=30.0,
        keep_alive=True,
        max_connections=4,
        response_buffer_size=1024,
        prefetch_tokens=0,
        mirostat=0,
        mirostat_eta=0.1,
        mirostat_tau=5.0,
        use_flash_attention=True,
        speculative_decoding=False,
    ),
    "throughput": ProfileConfig(
        name="throughput",
        description="Batch optimized for processing multiple requests",
        temperature=0.5,
        top_p=0.85,
        top_k=30,
        repeat_penalty=1.0,  # Disabled for speed
        num_predict=512,
        num_ctx=4096,
        stream=False,  # Non-streaming for batching efficiency
        stream_chunk_size=64,
        batch_size=8,  # Process 8 requests in parallel
        connection_timeout=10.0,
        read_timeout=120.0,
        keep_alive=True,
        max_connections=16,
        response_buffer_size=8192,
        prefetch_tokens=32,
        mirostat=0,
        mirostat_eta=0.1,
        mirostat_tau=5.0,
        use_flash_attention=True,
        speculative_decoding=True,
    ),
    "quality": ProfileConfig(
        name="quality",
        description="Full sampling parameters for highest quality output",
        temperature=0.8,
        top_p=0.95,
        top_k=100,
        repeat_penalty=1.1,
        num_predict=1024,
        num_ctx=8192,
        stream=True,
        stream_chunk_size=4,
        batch_size=1,
        connection_timeout=10.0,
        read_timeout=180.0,
        keep_alive=True,
        max_connections=4,
        response_buffer_size=4096,
        prefetch_tokens=16,
        mirostat=2,  # Mirostat 2 for quality
        mirostat_eta=0.1,
        mirostat_tau=5.0,
        use_flash_attention=True,
        speculative_decoding=False,
    ),
    "turbo": ProfileConfig(
        name="turbo",
        description="Maximum speed with quality tradeoffs",
        temperature=0.3,  # Lower for faster convergence
        top_p=0.7,  # Narrower sampling
        top_k=20,  # Very focused
        repeat_penalty=1.0,  # Disabled
        num_predict=128,  # Shorter outputs
        num_ctx=1024,  # Smaller context
        stream=True,
        stream_chunk_size=8,  # Larger chunks for efficiency
        batch_size=4,
        connection_timeout=3.0,
        read_timeout=15.0,
        keep_alive=True,
        max_connections=8,
        response_buffer_size=2048,
        prefetch_tokens=64,
        mirostat=0,
        mirostat_eta=0.1,
        mirostat_tau=5.0,
        use_flash_attention=True,
        speculative_decoding=True,
    ),
}


def get_profile(name: str) -> ProfileConfig:
    """Get a profile configuration by name."""
    if name not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile: {name}. Available: {list(PROFILE_CONFIGS.keys())}")
    return PROFILE_CONFIGS[name]


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
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
        if duration <= 0:
            return None
        return self.completion_tokens / duration


class MetricsCollector:
    """Collects and aggregates inference metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._metrics: Deque[RequestMetrics] = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, metrics: RequestMetrics):
        """Record metrics for a request."""
        with self._lock:
            self._metrics.append(metrics)

    def get_percentiles(
        self, values: List[float], percentiles: List[int] = None
    ) -> Dict[str, float]:
        """Calculate percentiles for a list of values."""
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]

        if not values:
            return {f"p{p}": 0.0 for p in percentiles}

        sorted_values = sorted(values)
        result = {}

        for p in percentiles:
            idx = int(len(sorted_values) * p / 100)
            idx = min(idx, len(sorted_values) - 1)
            result[f"p{p}"] = sorted_values[idx]

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            metrics_list = list(self._metrics)

        if not metrics_list:
            return {"total_requests": 0, "error_rate": 0.0}

        ttft_values = [m.time_to_first_token_ms for m in metrics_list if m.time_to_first_token_ms]
        total_time_values = [m.total_time_ms for m in metrics_list if m.total_time_ms]
        tps_values = [m.tokens_per_second for m in metrics_list if m.tokens_per_second]
        errors = sum(1 for m in metrics_list if m.error is not None)

        return {
            "total_requests": len(metrics_list),
            "error_rate": errors / len(metrics_list) if metrics_list else 0.0,
            "time_to_first_token_ms": self.get_percentiles(ttft_values),
            "total_time_ms": self.get_percentiles(total_time_values),
            "tokens_per_second": {
                "mean": statistics.mean(tps_values) if tps_values else 0.0,
                "median": statistics.median(tps_values) if tps_values else 0.0,
                "min": min(tps_values) if tps_values else 0.0,
                "max": max(tps_values) if tps_values else 0.0,
            },
            "prompt_tokens_total": sum(m.prompt_tokens for m in metrics_list),
            "completion_tokens_total": sum(m.completion_tokens for m in metrics_list),
        }


# =============================================================================
# Connection Pool
# =============================================================================


class ConnectionPool:
    """HTTP connection pool with reuse."""

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
        """Create a new connection."""
        conn = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout
        )
        return conn

    def acquire(self) -> http.client.HTTPConnection:
        """Acquire a connection from the pool."""
        try:
            conn = self._pool.get_nowait()
            # Test if connection is still valid
            try:
                conn.sock.getpeername()
                return conn
            except (AttributeError, OSError):
                # Connection is dead, create new one
                pass
        except queue.Empty:
            pass

        with self._lock:
            if self._created < self.max_connections:
                self._created += 1
                return self._create_connection()

        # Wait for a connection
        conn = self._pool.get(timeout=self.timeout)
        return conn

    def release(self, conn: http.client.HTTPConnection):
        """Release a connection back to the pool."""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()

    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


# =============================================================================
# Request Templates
# =============================================================================


class RequestTemplate:
    """Pre-compiled request template for minimal overhead."""

    def __init__(self, profile: ProfileConfig, model: str):
        self.profile = profile
        self.model = model
        self._base_options = self._compile_options()
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson" if profile.stream else "application/json",
            "Connection": "keep-alive" if profile.keep_alive else "close",
        }

    def _compile_options(self) -> Dict[str, Any]:
        """Compile options from profile."""
        options = {
            "temperature": self.profile.temperature,
            "top_p": self.profile.top_p,
            "top_k": self.profile.top_k,
            "num_predict": self.profile.num_predict,
            "num_ctx": self.profile.num_ctx,
        }

        # Only include repeat_penalty if not 1.0 (for speed)
        if self.profile.repeat_penalty != 1.0:
            options["repeat_penalty"] = self.profile.repeat_penalty

        # Mirostat settings
        if self.profile.mirostat > 0:
            options["mirostat"] = self.profile.mirostat
            options["mirostat_eta"] = self.profile.mirostat_eta
            options["mirostat_tau"] = self.profile.mirostat_tau

        return options

    def build_request(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        **overrides: Any,
    ) -> Tuple[bytes, Dict[str, str]]:
        """Build a request payload."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self.profile.stream,
            "options": {**self._base_options, **overrides},
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        return json.dumps(payload).encode("utf-8"), self._headers


# =============================================================================
# Fast Inference Client
# =============================================================================


class FastInferenceClient:
    """High-performance inference client with minimal overhead."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        profile: Union[str, ProfileConfig] = "realtime",
        model: str = "dolphin-llama3:8b",
    ):
        if isinstance(profile, str):
            self.profile = get_profile(profile)
        else:
            self.profile = profile

        self.host = host
        self.port = port
        self.model = model

        # Connection pool
        self._pool = ConnectionPool(
            host=host,
            port=port,
            max_connections=self.profile.max_connections,
            timeout=self.profile.connection_timeout,
        )

        # Request template
        self._template = RequestTemplate(self.profile, model)

        # Metrics
        self.metrics = MetricsCollector()

        # Request counter for IDs
        self._request_counter = 0
        self._counter_lock = threading.Lock()

    def _next_request_id(self) -> str:
        with self._counter_lock:
            self._request_counter += 1
            return f"req-{self._request_counter}"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        **options: Any,
    ) -> Iterator[str]:
        """Generate completion with streaming."""
        request_id = self._next_request_id()
        metrics = RequestMetrics(request_id=request_id, start_time=time.perf_counter())

        conn = None
        try:
            # Build request
            payload, headers = self._template.build_request(
                prompt=prompt,
                system=system,
                context=context,
                **options,
            )

            # Acquire connection
            conn = self._pool.acquire()

            # Send request
            conn.request("POST", "/api/generate", body=payload, headers=headers)

            # Get response
            response = conn.getresponse()

            if response.status != 200:
                error_msg = response.read().decode("utf-8", errors="replace")
                metrics.error = f"HTTP {response.status}: {error_msg[:200]}"
                metrics.end_time = time.perf_counter()
                self.metrics.record(metrics)
                raise RuntimeError(metrics.error)

            # Stream response
            buffer = b""
            for chunk in iter(lambda: response.read(self.profile.response_buffer_size), b""):
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
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
            self.metrics.record(metrics)

        except Exception as e:
            metrics.error = str(e)
            metrics.end_time = time.perf_counter()
            self.metrics.record(metrics)
            raise

        finally:
            if conn:
                self._pool.release(conn)

    def generate_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        **options: Any,
    ) -> str:
        """Generate completion synchronously (non-streaming)."""
        return "".join(self.generate(prompt, system, context, **options))

    def close(self):
        """Close all connections."""
        self._pool.close_all()


# =============================================================================
# Async Inference Client
# =============================================================================


class AsyncInferenceClient:
    """Async inference client for concurrent request handling."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        profile: Union[str, ProfileConfig] = "realtime",
        model: str = "dolphin-llama3:8b",
    ):
        if isinstance(profile, str):
            self.profile = get_profile(profile)
        else:
            self.profile = profile

        self.host = host
        self.port = port
        self.model = model

        # Request template
        self._template = RequestTemplate(self.profile, model)

        # Metrics
        self.metrics = MetricsCollector()

        # Request counter
        self._request_counter = 0

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        **options: Any,
    ) -> AsyncIterator[str]:
        """Generate completion with async streaming."""
        import asyncio

        self._request_counter += 1
        request_id = f"async-req-{self._request_counter}"
        metrics = RequestMetrics(request_id=request_id, start_time=time.perf_counter())

        try:
            payload, headers = self._template.build_request(
                prompt=prompt,
                system=system,
                context=context,
                **options,
            )

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.profile.connection_timeout,
            )

            # Build HTTP request
            request_line = f"POST /api/generate HTTP/1.1\r\n"
            header_lines = f"Host: {self.host}:{self.port}\r\n"
            for key, value in headers.items():
                header_lines += f"{key}: {value}\r\n"
            header_lines += f"Content-Length: {len(payload)}\r\n\r\n"

            writer.write((request_line + header_lines).encode("utf-8") + payload)
            await writer.drain()

            # Read status line
            status_line = await asyncio.wait_for(
                reader.readline(),
                timeout=self.profile.read_timeout,
            )
            if b"200" not in status_line:
                error = await reader.read()
                metrics.error = error.decode("utf-8", errors="replace")[:200]
                metrics.end_time = time.perf_counter()
                self.metrics.record(metrics)
                writer.close()
                raise RuntimeError(metrics.error)

            # Skip headers
            while True:
                header_line = await reader.readline()
                if header_line == b"\r\n" or not header_line:
                    break

            # Stream response
            while True:
                line = await asyncio.wait_for(
                    reader.readline(),
                    timeout=self.profile.read_timeout,
                )

                if not line:
                    break

                if metrics.first_token_time is None:
                    metrics.first_token_time = time.perf_counter()

                try:
                    data = json.loads(line.decode("utf-8"))

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
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
            self.metrics.record(metrics)

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            metrics.error = str(e)
            metrics.end_time = time.perf_counter()
            self.metrics.record(metrics)
            raise

    async def generate_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        **options: Any,
    ) -> str:
        """Generate completion and return full response."""
        chunks = []
        async for chunk in self.generate(prompt, system, context, **options):
            chunks.append(chunk)
        return "".join(chunks)

    async def batch_generate(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        **options: Any,
    ) -> List[str]:
        """Generate completions for multiple prompts concurrently."""
        import asyncio

        if max_concurrent is None:
            max_concurrent = self.profile.batch_size

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_prompt(prompt: str) -> str:
            async with semaphore:
                return await self.generate_sync(prompt, system, **options)

        tasks = [process_prompt(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Request Batcher
# =============================================================================


@dataclass
class BatchRequest:
    """A request in a batch."""

    prompt: str
    system: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class RequestBatcher:
    """Batches requests for efficient processing."""

    def __init__(
        self,
        client: AsyncInferenceClient,
        batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ):
        self.client = client
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the batcher."""
        self._running = True
        self._task = asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the batcher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(
        self,
        prompt: str,
        system: Optional[str] = None,
        **options: Any,
    ) -> str:
        """Submit a request for batched processing."""
        loop = asyncio.get_event_loop()
        request = BatchRequest(
            prompt=prompt,
            system=system,
            options=options,
            future=loop.create_future(),
        )
        await self._queue.put(request)
        return await request.future

    async def _process_batches(self):
        """Process batched requests."""
        while self._running:
            batch: List[BatchRequest] = []

            # Collect batch
            try:
                # Wait for first request
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                batch.append(request)

                # Collect more requests up to batch size or timeout
                deadline = time.perf_counter() + (self.max_wait_ms / 1000)

                while len(batch) < self.batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break

                    try:
                        request = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=remaining,
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

            except asyncio.TimeoutError:
                continue

            if not batch:
                continue

            # Process batch
            try:
                prompts = [r.prompt for r in batch]
                results = await self.client.batch_generate(
                    prompts,
                    system=batch[0].system,  # Use first request's system prompt
                    **batch[0].options,
                )

                for request, result in zip(batch, results):
                    if isinstance(result, Exception):
                        request.future.set_exception(result)
                    else:
                        request.future.set_result(result)

            except Exception as e:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(e)


# =============================================================================
# Inference Optimizer
# =============================================================================


class InferenceOptimizer:
    """Main class for inference pipeline optimization."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        model: str = "dolphin-llama3:8b",
    ):
        self.host = host
        self.port = port
        self.model = model
        self._clients: Dict[str, FastInferenceClient] = {}

    def get_client(self, profile: str = "realtime") -> FastInferenceClient:
        """Get or create a client for a profile."""
        if profile not in self._clients:
            self._clients[profile] = FastInferenceClient(
                host=self.host,
                port=self.port,
                profile=profile,
                model=self.model,
            )
        return self._clients[profile]

    def benchmark_profiles(
        self,
        prompts: List[str],
        iterations: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all profiles."""
        results = {}

        for profile_name in PROFILE_CONFIGS.keys():
            log.info(f"Benchmarking profile: {profile_name}")
            client = self.get_client(profile_name)

            # Warm up
            try:
                client.generate_sync("Hello")
            except Exception:
                pass

            # Run benchmark
            for _ in range(iterations):
                for prompt in prompts:
                    try:
                        _ = client.generate_sync(prompt)
                    except Exception as e:
                        log.warning(f"Error: {e}")

            results[profile_name] = client.metrics.get_summary()
            log.info(f"  TTFT p50: {results[profile_name]['time_to_first_token_ms'].get('p50', 0):.1f}ms")
            log.info(f"  TPS mean: {results[profile_name]['tokens_per_second'].get('mean', 0):.1f}")

        return results

    def recommend_profile(
        self,
        use_case: str = "chat",
        latency_priority: float = 0.5,
    ) -> str:
        """Recommend a profile based on use case."""
        recommendations = {
            "chat": "realtime",
            "interactive": "realtime",
            "batch": "throughput",
            "api": "throughput",
            "creative": "quality",
            "writing": "quality",
            "coding": "quality",
            "fast": "turbo",
            "quick": "turbo",
        }

        # Adjust based on latency priority
        if latency_priority > 0.7:
            return "realtime" if use_case not in ("batch", "api") else "turbo"
        elif latency_priority < 0.3:
            return "quality" if use_case not in ("batch", "api") else "throughput"

        return recommendations.get(use_case.lower(), "realtime")

    def close(self):
        """Close all clients."""
        for client in self._clients.values():
            client.close()


# =============================================================================
# Benchmark Runner
# =============================================================================


BENCHMARK_PROMPTS = [
    "What is 2 + 2?",
    "Explain the concept of recursion in one sentence.",
    "Write a short Python function to calculate factorial.",
    "What is the capital of France?",
]


def run_benchmark(
    host: str = "127.0.0.1",
    port: int = 11434,
    model: str = "dolphin-llama3:8b",
    iterations: int = 5,
) -> Dict[str, Any]:
    """Run comprehensive benchmark."""
    optimizer = InferenceOptimizer(host=host, port=port, model=model)

    try:
        results = optimizer.benchmark_profiles(BENCHMARK_PROMPTS, iterations)
        return results
    finally:
        optimizer.close()


def print_benchmark_report(results: Dict[str, Dict[str, Any]]):
    """Print formatted benchmark report."""
    print("\n" + "=" * 80)
    print("                  INFERENCE PIPELINE BENCHMARK REPORT")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    print("\n" + "-" * 80)
    print("Profile Comparison:")
    print("-" * 80)

    header = f"{'Profile':<15} {'TTFT p50':<12} {'TTFT p99':<12} {'TPS Mean':<12} {'TPS Max':<12} {'Errors':<10}"
    print(f"\n{header}")
    print("-" * 80)

    for profile, metrics in results.items():
        ttft = metrics.get("time_to_first_token_ms", {})
        tps = metrics.get("tokens_per_second", {})
        error_rate = metrics.get("error_rate", 0) * 100

        print(
            f"{profile:<15} "
            f"{ttft.get('p50', 0):<12.1f} "
            f"{ttft.get('p99', 0):<12.1f} "
            f"{tps.get('mean', 0):<12.1f} "
            f"{tps.get('max', 0):<12.1f} "
            f"{error_rate:<10.1f}%"
        )

    print("\n" + "-" * 80)
    print("Profile Recommendations:")
    print("-" * 80)

    for profile, config in PROFILE_CONFIGS.items():
        print(f"\n{profile.upper()}: {config.description}")
        print(f"  Best for: ", end="")
        if profile == "realtime":
            print("Interactive chat, conversational AI")
        elif profile == "throughput":
            print("Batch processing, API backends")
        elif profile == "quality":
            print("Creative writing, code generation, complex reasoning")
        elif profile == "turbo":
            print("Quick answers, simple queries, high-volume low-latency")

    print("\n" + "=" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="USB-AI Inference Pipeline Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--model",
        default="dolphin-llama3:8b",
        help="Model to use (default: dolphin-llama3:8b)",
    )

    parser.add_argument(
        "--profile",
        choices=list(PROFILE_CONFIGS.keys()),
        default="realtime",
        help="Inference profile to use (default: realtime)",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on all profiles",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Benchmark iterations (default: 3)",
    )

    parser.add_argument(
        "--prompt",
        help="Test prompt to run",
    )

    parser.add_argument(
        "--show-profiles",
        action="store_true",
        help="Show available profiles and exit",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.show_profiles:
        print("\nAvailable Inference Profiles:")
        print("-" * 60)
        for name, config in PROFILE_CONFIGS.items():
            print(f"\n{name.upper()}:")
            print(f"  Description: {config.description}")
            print(f"  Temperature: {config.temperature}")
            print(f"  Top-P: {config.top_p}")
            print(f"  Top-K: {config.top_k}")
            print(f"  Context: {config.num_ctx}")
            print(f"  Max Tokens: {config.num_predict}")
            print(f"  Streaming: {config.stream}")
            print(f"  Batch Size: {config.batch_size}")
        return 0

    if args.benchmark:
        log.info("Running inference pipeline benchmark...")
        results = run_benchmark(
            host=args.host,
            port=args.port,
            model=args.model,
            iterations=args.iterations,
        )

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_benchmark_report(results)

        return 0

    if args.prompt:
        log.info(f"Using profile: {args.profile}")
        client = FastInferenceClient(
            host=args.host,
            port=args.port,
            profile=args.profile,
            model=args.model,
        )

        try:
            print("\nResponse:", end=" ", flush=True)
            for token in client.generate(args.prompt):
                print(token, end="", flush=True)
            print("\n")

            summary = client.metrics.get_summary()
            print(f"\nMetrics:")
            print(f"  Time to first token: {summary['time_to_first_token_ms'].get('p50', 0):.1f}ms")
            print(f"  Tokens per second: {summary['tokens_per_second'].get('mean', 0):.1f}")

        finally:
            client.close()

        return 0

    # Default: show usage
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
