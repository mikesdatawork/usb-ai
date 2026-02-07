#!/usr/bin/env python3
"""
USB-AI WebUI Performance Optimizations

This module provides performance enhancements for the Flask + HTMX chat interface:
- Response streaming optimization with chunked encoding
- Connection pooling for Ollama requests
- Static asset caching with ETags
- GZIP compression for responses
- Optimized HTMX polling configuration

Usage:
    from optimizations import apply_optimizations, get_ollama_session

    # Apply all optimizations to Flask app
    apply_optimizations(app)

    # Use pooled session for Ollama requests
    session = get_ollama_session()
    response = session.get(url)
"""

import functools
import gzip
import hashlib
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import requests
from flask import Flask, Response, g, request, make_response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config" / "performance.yaml"
log = logging.getLogger("webui.optimizations")


# =============================================================================
# Configuration Management
# =============================================================================

@dataclass
class PerformanceConfig:
    """Performance configuration settings."""

    # Streaming settings
    stream_buffer_size: int = 512
    stream_chunk_delay_ms: int = 0
    stream_flush_threshold: int = 256

    # Connection pool settings
    pool_connections: int = 10
    pool_maxsize: int = 20
    pool_block: bool = False
    retry_total: int = 3
    retry_backoff_factor: float = 0.5
    retry_status_forcelist: Tuple[int, ...] = (500, 502, 503, 504)

    # Timeout settings
    connect_timeout: float = 5.0
    read_timeout: float = 300.0
    status_timeout: float = 3.0
    models_timeout: float = 5.0

    # Caching settings
    cache_enabled: bool = True
    cache_max_items: int = 100
    cache_default_ttl_sec: int = 300
    static_cache_max_age: int = 3600
    etag_enabled: bool = True

    # Compression settings
    gzip_enabled: bool = True
    gzip_min_size: int = 500
    gzip_level: int = 6
    gzip_mimetypes: Tuple[str, ...] = (
        "text/html",
        "text/css",
        "text/javascript",
        "application/javascript",
        "application/json",
        "text/event-stream",
    )

    # HTMX polling settings
    htmx_status_poll_interval_sec: int = 10
    htmx_models_poll_interval_sec: int = 30
    htmx_idle_poll_interval_sec: int = 30
    htmx_active_poll_interval_sec: int = 5

    @classmethod
    def load_from_yaml(cls, path: Optional[Path] = None) -> "PerformanceConfig":
        """Load configuration from YAML file."""
        path = path or CONFIG_PATH
        config = cls()

        if path.exists():
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f) or {}

                # Update config from YAML
                for section in ["streaming", "connection_pool", "timeouts",
                               "caching", "compression", "htmx_polling"]:
                    if section in data:
                        for key, value in data[section].items():
                            attr_name = f"{section.replace('_pool', '')}_{key}" if section != "streaming" else f"stream_{key}"
                            if section == "timeouts":
                                attr_name = f"{key}_timeout"
                            elif section == "caching":
                                attr_name = f"cache_{key}" if key not in ("static_max_age", "etag_enabled") else key.replace("static_max_age", "static_cache_max_age")
                            elif section == "compression":
                                attr_name = f"gzip_{key}"
                            elif section == "htmx_polling":
                                attr_name = f"htmx_{key}_sec" if "interval" in key else f"htmx_{key}"

                            if hasattr(config, attr_name):
                                setattr(config, attr_name, value)
                            elif hasattr(config, key):
                                setattr(config, key, value)

                log.info(f"Loaded performance config from {path}")
            except Exception as e:
                log.warning(f"Failed to load config from {path}: {e}")
        else:
            log.info("Using default performance configuration")

        return config


# Global configuration
_config: Optional[PerformanceConfig] = None


def get_config() -> PerformanceConfig:
    """Get the performance configuration."""
    global _config
    if _config is None:
        _config = PerformanceConfig.load_from_yaml()
    return _config


def reload_config():
    """Reload configuration from file."""
    global _config
    _config = PerformanceConfig.load_from_yaml()
    return _config


# =============================================================================
# Connection Pooling
# =============================================================================

class OllamaConnectionPool:
    """
    Connection pool for Ollama API requests.

    Uses requests.Session with connection pooling and automatic retries
    for improved performance and reliability.
    """

    _instance: Optional["OllamaConnectionPool"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        config = get_config()

        # Create session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=config.retry_total,
            backoff_factor=config.retry_backoff_factor,
            status_forcelist=config.retry_status_forcelist,
            allowed_methods=["GET", "POST"],
        )

        # Mount adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=config.pool_connections,
            pool_maxsize=config.pool_maxsize,
            pool_block=config.pool_block,
            max_retries=retry_strategy,
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        # Connection stats
        self._request_count = 0
        self._error_count = 0
        self._last_request_time: Optional[float] = None

        self._initialized = True
        log.info(f"Connection pool initialized: {config.pool_connections} connections, max {config.pool_maxsize}")

    def get(self, url: str, timeout: Optional[float] = None, **kwargs) -> requests.Response:
        """Make a GET request using the connection pool."""
        config = get_config()
        timeout = timeout or (config.connect_timeout, config.read_timeout)

        self._request_count += 1
        self._last_request_time = time.time()

        try:
            return self.session.get(url, timeout=timeout, **kwargs)
        except Exception as e:
            self._error_count += 1
            raise

    def post(self, url: str, timeout: Optional[float] = None, stream: bool = False, **kwargs) -> requests.Response:
        """Make a POST request using the connection pool."""
        config = get_config()

        if stream:
            # For streaming, use longer read timeout
            timeout = timeout or (config.connect_timeout, config.read_timeout)
        else:
            timeout = timeout or (config.connect_timeout, config.read_timeout)

        self._request_count += 1
        self._last_request_time = time.time()

        try:
            return self.session.post(url, timeout=timeout, stream=stream, **kwargs)
        except Exception as e:
            self._error_count += 1
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "last_request_time": self._last_request_time,
        }

    def reset(self):
        """Reset the connection pool."""
        self.session.close()
        self._initialized = False
        OllamaConnectionPool._instance = None


def get_ollama_session() -> OllamaConnectionPool:
    """Get the Ollama connection pool instance."""
    return OllamaConnectionPool()


# =============================================================================
# Response Streaming Optimization
# =============================================================================

class StreamBuffer:
    """
    Optimized buffer for streaming responses.

    Accumulates small chunks and flushes when threshold is reached
    for better network efficiency.
    """

    def __init__(self, flush_threshold: Optional[int] = None):
        config = get_config()
        self.flush_threshold = flush_threshold or config.stream_flush_threshold
        self._buffer = []
        self._size = 0

    def write(self, data: str) -> Optional[str]:
        """
        Write data to buffer.

        Returns flushed content if threshold reached, None otherwise.
        """
        self._buffer.append(data)
        self._size += len(data)

        if self._size >= self.flush_threshold:
            return self.flush()
        return None

    def flush(self) -> str:
        """Flush all buffered content."""
        if not self._buffer:
            return ""
        content = "".join(self._buffer)
        self._buffer = []
        self._size = 0
        return content

    def __len__(self) -> int:
        return self._size


def optimized_sse_stream(generator: Generator[str, None, None]) -> Generator[bytes, None, None]:
    """
    Wrap a generator with optimized SSE streaming.

    Features:
    - Buffered output for network efficiency
    - Proper SSE formatting
    - Optional compression
    """
    config = get_config()
    buffer = StreamBuffer()

    for chunk in generator:
        # Format as SSE
        sse_data = f"data: {chunk}\n\n"

        # Buffer for efficiency
        flushed = buffer.write(sse_data)
        if flushed:
            yield flushed.encode("utf-8")

    # Flush remaining buffer
    remaining = buffer.flush()
    if remaining:
        yield remaining.encode("utf-8")


def create_streaming_response(generator: Generator[str, None, None],
                              content_type: str = "text/event-stream") -> Response:
    """
    Create an optimized streaming response.

    Args:
        generator: Content generator
        content_type: Response content type

    Returns:
        Flask Response with optimized streaming
    """
    config = get_config()

    response = Response(
        optimized_sse_stream(generator),
        mimetype=content_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

    return response


# =============================================================================
# Static Asset Caching
# =============================================================================

class StaticCache:
    """
    In-memory cache for static assets with ETag support.

    Features:
    - LRU eviction
    - ETag generation
    - TTL-based expiration
    """

    def __init__(self, max_items: Optional[int] = None, default_ttl: Optional[int] = None):
        config = get_config()
        self.max_items = max_items or config.cache_max_items
        self.default_ttl = default_ttl or config.cache_default_ttl_sec
        self._cache: OrderedDict[str, Tuple[Any, str, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _generate_etag(self, content: Union[str, bytes]) -> str:
        """Generate ETag for content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def get(self, key: str) -> Optional[Tuple[Any, str]]:
        """
        Get cached item.

        Returns:
            Tuple of (content, etag) or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            content, etag, expires = self._cache[key]

            if time.time() > expires:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return content, etag

    def set(self, key: str, content: Union[str, bytes], ttl: Optional[int] = None) -> str:
        """
        Cache content and return ETag.

        Args:
            key: Cache key
            content: Content to cache
            ttl: Time-to-live in seconds

        Returns:
            Generated ETag
        """
        ttl = ttl or self.default_ttl
        etag = self._generate_etag(content)
        expires = time.time() + ttl

        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_items:
                self._cache.popitem(last=False)

            self._cache[key] = (content, etag, expires)

        return etag

    def invalidate(self, key: str):
        """Remove item from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "items": len(self._cache),
            "max_items": self.max_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, total),
        }


# Global cache instance
_static_cache: Optional[StaticCache] = None


def get_static_cache() -> StaticCache:
    """Get the static cache instance."""
    global _static_cache
    if _static_cache is None:
        _static_cache = StaticCache()
    return _static_cache


def cache_response(ttl: Optional[int] = None, etag: bool = True):
    """
    Decorator for caching route responses.

    Args:
        ttl: Cache TTL in seconds (default from config)
        etag: Enable ETag support
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            config = get_config()

            if not config.cache_enabled:
                return f(*args, **kwargs)

            cache = get_static_cache()
            cache_key = f"{request.path}:{request.query_string.decode()}"

            # Check cache
            cached = cache.get(cache_key)
            if cached:
                content, cached_etag = cached

                # Check If-None-Match
                if etag and config.etag_enabled:
                    if request.headers.get("If-None-Match") == cached_etag:
                        return Response(status=304)

                response = make_response(content)
                if etag and config.etag_enabled:
                    response.headers["ETag"] = cached_etag
                response.headers["X-Cache"] = "HIT"
                return response

            # Generate response
            result = f(*args, **kwargs)

            # Cache the response
            if isinstance(result, str):
                content = result
            elif isinstance(result, Response):
                content = result.get_data(as_text=True)
            else:
                return result

            cached_etag = cache.set(cache_key, content, ttl)

            response = make_response(content)
            if etag and config.etag_enabled:
                response.headers["ETag"] = cached_etag
            response.headers["X-Cache"] = "MISS"
            response.headers["Cache-Control"] = f"max-age={config.static_cache_max_age}"

            return response

        return wrapper
    return decorator


# =============================================================================
# GZIP Compression
# =============================================================================

class GzipMiddleware:
    """
    WSGI middleware for GZIP compression.

    Compresses responses based on configuration and Accept-Encoding header.
    """

    def __init__(self, app):
        self.app = app
        self.config = get_config()

    def __call__(self, environ, start_response):
        # Check if client accepts gzip
        accept_encoding = environ.get("HTTP_ACCEPT_ENCODING", "")
        if "gzip" not in accept_encoding or not self.config.gzip_enabled:
            return self.app(environ, start_response)

        # Capture response
        response_body = []
        response_headers = []
        response_status = []

        def custom_start_response(status, headers, exc_info=None):
            response_status.append(status)
            response_headers.extend(headers)
            return response_body.append

        app_iter = self.app(environ, custom_start_response)

        try:
            for item in app_iter:
                response_body.append(item)
        finally:
            if hasattr(app_iter, "close"):
                app_iter.close()

        body = b"".join(response_body)

        # Check if we should compress
        content_type = None
        for name, value in response_headers:
            if name.lower() == "content-type":
                content_type = value.split(";")[0]
                break

        should_compress = (
            len(body) >= self.config.gzip_min_size and
            content_type in self.config.gzip_mimetypes
        )

        if should_compress:
            # Compress
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb",
                              compresslevel=self.config.gzip_level) as gz:
                gz.write(body)
            compressed = buffer.getvalue()

            # Update headers
            new_headers = []
            for name, value in response_headers:
                if name.lower() != "content-length":
                    new_headers.append((name, value))
            new_headers.append(("Content-Encoding", "gzip"))
            new_headers.append(("Content-Length", str(len(compressed))))
            new_headers.append(("Vary", "Accept-Encoding"))

            start_response(response_status[0], new_headers)
            return [compressed]
        else:
            start_response(response_status[0], response_headers)
            return [body]


def compress_response(response: Response) -> Response:
    """
    Compress a Flask response if appropriate.

    Args:
        response: Flask Response to compress

    Returns:
        Compressed or original response
    """
    config = get_config()

    if not config.gzip_enabled:
        return response

    # Check Accept-Encoding
    accept_encoding = request.headers.get("Accept-Encoding", "")
    if "gzip" not in accept_encoding:
        return response

    # Check content type
    content_type = response.content_type or ""
    mime_type = content_type.split(";")[0]
    if mime_type not in config.gzip_mimetypes:
        return response

    # Check size
    data = response.get_data()
    if len(data) < config.gzip_min_size:
        return response

    # Compress
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb",
                      compresslevel=config.gzip_level) as gz:
        gz.write(data)
    compressed = buffer.getvalue()

    response.set_data(compressed)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = len(compressed)
    response.headers["Vary"] = "Accept-Encoding"

    return response


# =============================================================================
# HTMX Polling Optimization
# =============================================================================

@dataclass
class HTMXPollingConfig:
    """HTMX polling configuration with adaptive intervals."""

    status_interval: int = 10
    models_interval: int = 30
    idle_interval: int = 30
    active_interval: int = 5

    @classmethod
    def from_config(cls) -> "HTMXPollingConfig":
        """Create from performance config."""
        config = get_config()
        return cls(
            status_interval=config.htmx_status_poll_interval_sec,
            models_interval=config.htmx_models_poll_interval_sec,
            idle_interval=config.htmx_idle_poll_interval_sec,
            active_interval=config.htmx_active_poll_interval_sec,
        )

    def get_status_trigger(self) -> str:
        """Get HTMX trigger for status polling."""
        return f"load, every {self.status_interval}s"

    def get_models_trigger(self) -> str:
        """Get HTMX trigger for models polling."""
        return f"load, every {self.models_interval}s"

    def get_adaptive_trigger(self, is_active: bool) -> str:
        """Get adaptive polling trigger based on activity state."""
        interval = self.active_interval if is_active else self.idle_interval
        return f"every {interval}s"


def get_htmx_polling_config() -> HTMXPollingConfig:
    """Get HTMX polling configuration."""
    return HTMXPollingConfig.from_config()


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Collects performance metrics for analysis."""

    request_times: list = field(default_factory=list)
    response_sizes: list = field(default_factory=list)
    compression_ratios: list = field(default_factory=list)
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    pool_stats: Dict[str, Any] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _start_time: float = field(default_factory=time.time, repr=False)

    def record_request(self, duration_ms: float, response_size: int,
                       compressed_size: Optional[int] = None):
        """Record request metrics."""
        with self._lock:
            self.request_times.append(duration_ms)
            self.response_sizes.append(response_size)

            if compressed_size is not None:
                ratio = compressed_size / max(1, response_size)
                self.compression_ratios.append(ratio)

            # Keep only last 1000 entries
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
                self.response_sizes = self.response_sizes[-1000:]
                self.compression_ratios = self.compression_ratios[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            if not self.request_times:
                return {"status": "no data"}

            avg_time = sum(self.request_times) / len(self.request_times)
            avg_size = sum(self.response_sizes) / len(self.response_sizes)
            avg_compression = (
                sum(self.compression_ratios) / len(self.compression_ratios)
                if self.compression_ratios else None
            )

            return {
                "uptime_sec": time.time() - self._start_time,
                "total_requests": len(self.request_times),
                "avg_request_time_ms": round(avg_time, 2),
                "avg_response_size_bytes": round(avg_size),
                "avg_compression_ratio": round(avg_compression, 3) if avg_compression else None,
                "p95_request_time_ms": round(sorted(self.request_times)[int(len(self.request_times) * 0.95)], 2),
                "cache": get_static_cache().get_stats(),
                "pool": get_ollama_session().get_stats(),
            }


# Global metrics
_metrics: Optional[PerformanceMetrics] = None


def get_metrics() -> PerformanceMetrics:
    """Get the performance metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PerformanceMetrics()
    return _metrics


# =============================================================================
# Flask Integration
# =============================================================================

def apply_optimizations(app: Flask, enable_gzip_middleware: bool = True):
    """
    Apply all performance optimizations to a Flask app.

    Args:
        app: Flask application
        enable_gzip_middleware: Enable GZIP middleware (default True)
    """
    config = get_config()

    # Request timing middleware
    @app.before_request
    def before_request():
        g.request_start_time = time.time()

    @app.after_request
    def after_request(response):
        # Record metrics
        if hasattr(g, "request_start_time"):
            duration_ms = (time.time() - g.request_start_time) * 1000
            response_size = len(response.get_data())

            compressed_size = None
            if response.headers.get("Content-Encoding") == "gzip":
                compressed_size = response_size

            get_metrics().record_request(duration_ms, response_size, compressed_size)

        # Apply compression if not streaming
        if (config.gzip_enabled and
            response.content_type and
            "event-stream" not in response.content_type):
            response = compress_response(response)

        return response

    # Add performance endpoint
    @app.route("/api/performance")
    def api_performance():
        """Get performance metrics."""
        metrics = get_metrics().get_summary()
        return Response(
            json.dumps(metrics, indent=2),
            mimetype="application/json"
        )

    # Apply GZIP middleware if enabled
    if enable_gzip_middleware and config.gzip_enabled:
        app.wsgi_app = GzipMiddleware(app.wsgi_app)

    log.info("Performance optimizations applied")
    log.info(f"  - Connection pooling: {config.pool_connections} connections")
    log.info(f"  - GZIP compression: {'enabled' if config.gzip_enabled else 'disabled'}")
    log.info(f"  - Static caching: {'enabled' if config.cache_enabled else 'disabled'}")
    log.info(f"  - HTMX polling: {config.htmx_status_poll_interval_sec}s status, {config.htmx_models_poll_interval_sec}s models")


# =============================================================================
# Optimized Ollama Functions
# =============================================================================

def ollama_models_optimized(ollama_url: str) -> list[str]:
    """
    Get available models from Ollama using connection pool.

    Args:
        ollama_url: Ollama API base URL

    Returns:
        List of available model names
    """
    config = get_config()
    pool = get_ollama_session()

    try:
        r = pool.get(f"{ollama_url}/api/tags", timeout=config.models_timeout)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        log.warning(f"Failed to get models: {e}")
    return []


def ollama_status_optimized(ollama_url: str) -> Tuple[bool, str]:
    """
    Check Ollama connection using connection pool.

    Args:
        ollama_url: Ollama API base URL

    Returns:
        Tuple of (is_online, status_message)
    """
    config = get_config()
    pool = get_ollama_session()

    try:
        r = pool.get(f"{ollama_url}/api/tags", timeout=config.status_timeout)
        if r.ok:
            n = len(r.json().get("models", []))
            return True, f"{n} models"
    except Exception:
        pass
    return False, "Offline"


def ollama_chat_optimized(
    ollama_url: str,
    message: str,
    history: list,
    model: str,
    on_token: Optional[Callable[[str], None]] = None
) -> Generator[str, None, None]:
    """
    Stream chat from Ollama with optimized connection handling.

    Args:
        ollama_url: Ollama API base URL
        message: User message
        history: Chat history
        model: Model name
        on_token: Optional callback for each token

    Yields:
        Response tokens
    """
    config = get_config()
    pool = get_ollama_session()

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    start = time.time()
    tokens = 0
    first_token_time = None

    try:
        r = pool.post(
            f"{ollama_url}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=(config.connect_timeout, config.read_timeout)
        )

        if not r.ok:
            try:
                err_data = r.json()
                error_msg = err_data.get("error", f"HTTP {r.status_code}")
            except:
                error_msg = f"HTTP {r.status_code}"
            yield f"[ERROR]{error_msg}"
            return

        # Use optimized streaming with larger chunks
        for line in r.iter_lines(chunk_size=config.stream_buffer_size):
            if line:
                data = json.loads(line)

                if "error" in data:
                    yield f"[ERROR]{data['error']}"
                    return

                if "message" in data and "content" in data["message"]:
                    tokens += 1
                    content = data["message"]["content"]

                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start
                        log.debug(f"TTFT: {ttft:.2f}s | Model: {model}")

                    if on_token:
                        on_token(content)

                    yield content

                if data.get("done"):
                    break

        elapsed = time.time() - start
        tps = tokens / elapsed if elapsed > 0 else 0
        log.info(f"Chat complete | Model: {model} | Tokens: {tokens} | Time: {elapsed:.1f}s | Speed: {tps:.1f} tok/s")

    except requests.exceptions.Timeout:
        yield "[ERROR]Timeout - model may be hanging"
    except requests.exceptions.ConnectionError:
        yield "[ERROR]Cannot connect to Ollama"
    except Exception as e:
        yield f"[ERROR]{e}"


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "PerformanceConfig",
    "get_config",
    "reload_config",

    # Connection pooling
    "OllamaConnectionPool",
    "get_ollama_session",

    # Streaming
    "StreamBuffer",
    "optimized_sse_stream",
    "create_streaming_response",

    # Caching
    "StaticCache",
    "get_static_cache",
    "cache_response",

    # Compression
    "GzipMiddleware",
    "compress_response",

    # HTMX
    "HTMXPollingConfig",
    "get_htmx_polling_config",

    # Metrics
    "PerformanceMetrics",
    "get_metrics",

    # Flask integration
    "apply_optimizations",

    # Optimized Ollama functions
    "ollama_models_optimized",
    "ollama_status_optimized",
    "ollama_chat_optimized",
]
