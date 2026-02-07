#!/usr/bin/env python3
"""
model_warmup.py

Model Warmup Manager for USB-AI - Eliminates Cold Start Latency

This module provides comprehensive model warmup capabilities to ensure
zero cold start latency for users. It includes:

- Pre-loading the default model on startup
- Periodic keepalive requests to prevent unloading
- Efficient model switching with predictive preloading
- Memory-aware loading with automatic unloading when RAM is critical
- Warmup daemon for continuous operation
- Benchmark tooling for cold vs warm comparison

Usage:
    # Start warmup daemon
    python -m scripts.performance.model_warmup --daemon

    # Warm up a specific model
    python -m scripts.performance.model_warmup --warmup dolphin-llama3:8b

    # Run benchmark
    python -m scripts.performance.model_warmup --benchmark

    # Check warmup status
    python -m scripts.performance.model_warmup --status
"""

import argparse
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Callable
import urllib.request
import urllib.error

__version__ = "1.0.0"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("model_warmup")

# Configuration defaults
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_KEEP_ALIVE = "30m"  # 30 minutes
WARMUP_PROMPT = "Hello"  # Minimal prompt to load model
HEALTH_CHECK_INTERVAL = 30  # seconds
MEMORY_CHECK_INTERVAL = 60  # seconds
PRELOAD_QUEUE_SIZE = 3
MIN_RAM_GB_THRESHOLD = 2.0  # Minimum available RAM before unloading


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    size_gb: float
    loaded_at: datetime
    last_used: datetime
    request_count: int = 0
    average_response_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size_gb": self.size_gb,
            "loaded_at": self.loaded_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "request_count": self.request_count,
            "average_response_ms": self.average_response_ms,
        }


@dataclass
class WarmupStats:
    """Statistics for warmup operations."""
    cold_starts: int = 0
    warm_starts: int = 0
    keepalive_sent: int = 0
    preloads_triggered: int = 0
    memory_unloads: int = 0
    total_warmup_time_ms: float = 0.0
    average_cold_start_ms: float = 0.0
    average_warm_start_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of cold vs warm benchmark."""
    model_name: str
    cold_start_ms: float
    warm_start_ms: float
    improvement_percent: float
    improvement_factor: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, timeout: int = 300):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("models", [])
        except Exception as e:
            log.error(f"Failed to list models: {e}")
            return []

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Get currently loaded/running models."""
        try:
            req = urllib.request.Request(f"{self.host}/api/ps")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("models", [])
        except Exception as e:
            log.debug(f"Failed to get running models: {e}")
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send a generate request to warm up or use a model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": keep_alive,
            "options": {
                "num_predict": 1,  # Minimal generation for warmup
            }
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            start_time = time.perf_counter()
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            result["_elapsed_ms"] = elapsed_ms
            return result

        except Exception as e:
            log.error(f"Generate request failed: {e}")
            return {"error": str(e)}

    def unload_model(self, model: str) -> bool:
        """Unload a model from memory."""
        payload = {
            "model": model,
            "keep_alive": "0",  # Immediate unload
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status == 200

        except Exception as e:
            log.error(f"Failed to unload model {model}: {e}")
            return False


class MemoryMonitor:
    """Monitor system memory and trigger unloading when critical."""

    def __init__(self):
        self.system = platform.system().lower()

    def get_available_ram_gb(self) -> float:
        """Get available RAM in GB."""
        try:
            if self.system == "linux":
                return self._get_linux_ram()
            elif self.system == "darwin":
                return self._get_darwin_ram()
            elif self.system == "windows":
                return self._get_windows_ram()
        except Exception as e:
            log.warning(f"Memory detection error: {e}")
        return 8.0  # Safe default

    def _get_linux_ram(self) -> float:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        match = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
        if match:
            return int(match.group(1)) / 1024 / 1024
        return 8.0

    def _get_darwin_ram(self) -> float:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            free_match = re.search(r"Pages free:\s+(\d+)", result.stdout)
            inactive_match = re.search(r"Pages inactive:\s+(\d+)", result.stdout)
            pages = 0
            if free_match:
                pages += int(free_match.group(1))
            if inactive_match:
                pages += int(inactive_match.group(1))
            return pages * 4096 / 1024 / 1024 / 1024
        return 8.0

    def _get_windows_ram(self) -> float:
        result = subprocess.run(
            ["wmic", "OS", "get", "FreePhysicalMemory"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                return int(lines[1].strip()) / 1024 / 1024
        return 8.0

    def is_memory_critical(self) -> bool:
        """Check if memory is critically low."""
        available = self.get_available_ram_gb()
        return available < MIN_RAM_GB_THRESHOLD


class ModelUsagePredictor:
    """Predicts next likely model based on usage patterns."""

    def __init__(self):
        self.usage_history: List[str] = []
        self.transition_counts: Dict[str, Dict[str, int]] = {}
        self.max_history = 100

    def record_usage(self, model: str):
        """Record a model usage event."""
        if self.usage_history:
            prev_model = self.usage_history[-1]
            if prev_model not in self.transition_counts:
                self.transition_counts[prev_model] = {}
            if model not in self.transition_counts[prev_model]:
                self.transition_counts[prev_model][model] = 0
            self.transition_counts[prev_model][model] += 1

        self.usage_history.append(model)
        if len(self.usage_history) > self.max_history:
            self.usage_history.pop(0)

    def predict_next(self, current_model: str) -> Optional[str]:
        """Predict the next likely model to be used."""
        if current_model not in self.transition_counts:
            return None

        transitions = self.transition_counts[current_model]
        if not transitions:
            return None

        # Return most likely next model
        return max(transitions.items(), key=lambda x: x[1])[0]

    def get_top_models(self, n: int = 3) -> List[str]:
        """Get the top N most used models."""
        if not self.usage_history:
            return []

        counts: Dict[str, int] = {}
        for model in self.usage_history:
            counts[model] = counts.get(model, 0) + 1

        sorted_models = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_models[:n]]


class WarmupManager:
    """
    Main warmup manager that coordinates all warmup operations.

    Features:
    - Pre-loads default model on startup
    - Sends periodic keepalive requests
    - Manages model preloading queue
    - Monitors memory and unloads when critical
    - Tracks warmup statistics
    """

    def __init__(
        self,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        default_model: Optional[str] = None
    ):
        self.client = OllamaClient(ollama_host)
        self.keep_alive = keep_alive
        self.default_model = default_model

        self.memory_monitor = MemoryMonitor()
        self.predictor = ModelUsagePredictor()

        self.loaded_models: Dict[str, ModelInfo] = {}
        self.stats = WarmupStats()
        self.preload_queue: Queue = Queue(maxsize=PRELOAD_QUEUE_SIZE)

        self._running = False
        self._keepalive_thread: Optional[threading.Thread] = None
        self._preload_thread: Optional[threading.Thread] = None
        self._memory_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_model_loaded: Optional[Callable[[str], None]] = None
        self.on_model_unloaded: Optional[Callable[[str], None]] = None
        self.on_memory_critical: Optional[Callable[[], None]] = None

    def _find_default_model(self) -> Optional[str]:
        """Find the default model to use."""
        if self.default_model:
            return self.default_model

        # Check for configured default
        config_path = self._find_config_path()
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if "default_model" in config:
                        return config["default_model"]
            except Exception:
                pass

        # Fall back to first available model
        models = self.client.list_models()
        if models:
            # Prefer dolphin-llama3 if available
            for m in models:
                if "dolphin" in m.get("name", "").lower():
                    return m["name"]
            return models[0]["name"]

        return None

    def _find_config_path(self) -> Optional[Path]:
        """Find the USB-AI config directory."""
        script_dir = Path(__file__).parent.resolve()

        # scripts/performance -> scripts -> root
        root = script_dir.parent.parent
        config_path = root / "modules" / "config" / "system.json"

        if config_path.exists():
            return config_path
        return None

    def warmup_model(
        self,
        model: str,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Warm up a model by sending a minimal request.

        Returns:
            Dict with warmup results including timing information.
        """
        if keep_alive is None:
            keep_alive = self.keep_alive

        log.info(f"Warming up model: {model}")

        # Check if already loaded
        running = self.client.get_running_models()
        was_loaded = any(m.get("name") == model for m in running)

        start_time = time.perf_counter()
        result = self.client.generate(
            model=model,
            prompt=WARMUP_PROMPT,
            keep_alive=keep_alive
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if "error" in result:
            log.error(f"Warmup failed for {model}: {result['error']}")
            return result

        # Update statistics
        if was_loaded:
            self.stats.warm_starts += 1
            # Update rolling average for warm starts
            n = self.stats.warm_starts
            self.stats.average_warm_start_ms = (
                (self.stats.average_warm_start_ms * (n - 1) + elapsed_ms) / n
            )
        else:
            self.stats.cold_starts += 1
            self.stats.total_warmup_time_ms += elapsed_ms
            # Update rolling average for cold starts
            n = self.stats.cold_starts
            self.stats.average_cold_start_ms = (
                (self.stats.average_cold_start_ms * (n - 1) + elapsed_ms) / n
            )

        # Track loaded model
        now = datetime.now()
        if model not in self.loaded_models:
            # Estimate size from model info
            models = self.client.list_models()
            size_gb = 0.0
            for m in models:
                if m.get("name") == model:
                    size_bytes = m.get("size", 0)
                    size_gb = size_bytes / 1024 / 1024 / 1024
                    break

            self.loaded_models[model] = ModelInfo(
                name=model,
                size_gb=size_gb,
                loaded_at=now,
                last_used=now,
                request_count=1,
                average_response_ms=elapsed_ms,
            )

            if self.on_model_loaded:
                self.on_model_loaded(model)
        else:
            info = self.loaded_models[model]
            info.last_used = now
            info.request_count += 1
            # Update rolling average
            n = info.request_count
            info.average_response_ms = (
                (info.average_response_ms * (n - 1) + elapsed_ms) / n
            )

        # Record for prediction
        self.predictor.record_usage(model)

        log.info(
            f"Model {model} warmed up in {elapsed_ms:.0f}ms "
            f"({'warm' if was_loaded else 'cold'} start)"
        )

        return {
            "model": model,
            "elapsed_ms": elapsed_ms,
            "was_loaded": was_loaded,
            "keep_alive": keep_alive,
        }

    def send_keepalive(self, model: str) -> bool:
        """Send a keepalive request to keep model loaded."""
        try:
            result = self.client.generate(
                model=model,
                prompt="",  # Empty prompt for keepalive
                keep_alive=self.keep_alive
            )

            if "error" not in result:
                self.stats.keepalive_sent += 1
                if model in self.loaded_models:
                    self.loaded_models[model].last_used = datetime.now()
                log.debug(f"Keepalive sent for {model}")
                return True

        except Exception as e:
            log.warning(f"Keepalive failed for {model}: {e}")

        return False

    def queue_preload(self, model: str):
        """Add a model to the preload queue."""
        try:
            self.preload_queue.put_nowait(model)
            log.debug(f"Queued preload for {model}")
        except Exception:
            log.debug(f"Preload queue full, skipping {model}")

    def unload_model(self, model: str) -> bool:
        """Unload a model from memory."""
        log.info(f"Unloading model: {model}")

        success = self.client.unload_model(model)

        if success:
            if model in self.loaded_models:
                del self.loaded_models[model]
            self.stats.memory_unloads += 1

            if self.on_model_unloaded:
                self.on_model_unloaded(model)

        return success

    def _keepalive_loop(self):
        """Background loop for sending keepalive requests."""
        # Parse keep_alive duration
        keep_alive_seconds = self._parse_duration(self.keep_alive)
        # Send keepalive at 75% of the keep_alive duration
        interval = max(30, keep_alive_seconds * 0.75)

        while self._running:
            try:
                # Get currently running models
                running = self.client.get_running_models()

                for model_info in running:
                    model_name = model_info.get("name")
                    if model_name:
                        self.send_keepalive(model_name)

                # Sleep in small increments to allow quick shutdown
                for _ in range(int(interval / 5)):
                    if not self._running:
                        break
                    time.sleep(5)

            except Exception as e:
                log.error(f"Keepalive loop error: {e}")
                time.sleep(HEALTH_CHECK_INTERVAL)

    def _preload_loop(self):
        """Background loop for processing preload queue."""
        while self._running:
            try:
                model = self.preload_queue.get(timeout=5)

                # Check if already loaded
                running = self.client.get_running_models()
                if any(m.get("name") == model for m in running):
                    continue

                # Check memory before preloading
                if self.memory_monitor.is_memory_critical():
                    log.warning(f"Memory critical, skipping preload of {model}")
                    continue

                self.warmup_model(model)
                self.stats.preloads_triggered += 1

            except Empty:
                continue
            except Exception as e:
                log.error(f"Preload loop error: {e}")

    def _memory_loop(self):
        """Background loop for memory monitoring."""
        while self._running:
            try:
                if self.memory_monitor.is_memory_critical():
                    log.warning("Memory critical, unloading least recently used model")

                    if self.on_memory_critical:
                        self.on_memory_critical()

                    # Find and unload least recently used model
                    if self.loaded_models:
                        lru_model = min(
                            self.loaded_models.values(),
                            key=lambda m: m.last_used
                        )
                        self.unload_model(lru_model.name)

                # Sleep in small increments
                for _ in range(MEMORY_CHECK_INTERVAL // 5):
                    if not self._running:
                        break
                    time.sleep(5)

            except Exception as e:
                log.error(f"Memory loop error: {e}")
                time.sleep(MEMORY_CHECK_INTERVAL)

    def _parse_duration(self, duration: str) -> int:
        """Parse duration string (e.g., '30m', '1h') to seconds."""
        match = re.match(r"(\d+)([smh])?", duration)
        if not match:
            return 1800  # Default 30 minutes

        value = int(match.group(1))
        unit = match.group(2) or "s"

        if unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        else:
            return value

    def start(self):
        """Start the warmup manager background threads."""
        if self._running:
            log.warning("Warmup manager already running")
            return

        log.info("Starting warmup manager...")
        self._running = True

        # Wait for Ollama to be available
        retries = 0
        while not self.client.is_available() and retries < 30:
            log.info("Waiting for Ollama server...")
            time.sleep(2)
            retries += 1

        if not self.client.is_available():
            log.error("Ollama server not available")
            self._running = False
            return

        # Warm up default model
        default_model = self._find_default_model()
        if default_model:
            log.info(f"Warming up default model: {default_model}")
            self.warmup_model(default_model)

        # Start background threads
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="warmup-keepalive",
            daemon=True
        )
        self._keepalive_thread.start()

        self._preload_thread = threading.Thread(
            target=self._preload_loop,
            name="warmup-preload",
            daemon=True
        )
        self._preload_thread.start()

        self._memory_thread = threading.Thread(
            target=self._memory_loop,
            name="warmup-memory",
            daemon=True
        )
        self._memory_thread.start()

        log.info("Warmup manager started")

    def stop(self):
        """Stop the warmup manager."""
        log.info("Stopping warmup manager...")
        self._running = False

        # Wait for threads to finish
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=10)
        if self._preload_thread and self._preload_thread.is_alive():
            self._preload_thread.join(timeout=10)
        if self._memory_thread and self._memory_thread.is_alive():
            self._memory_thread.join(timeout=10)

        log.info("Warmup manager stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current warmup manager status."""
        running_models = self.client.get_running_models()

        return {
            "running": self._running,
            "ollama_available": self.client.is_available(),
            "default_model": self.default_model or self._find_default_model(),
            "keep_alive": self.keep_alive,
            "loaded_models": [
                m.to_dict() for m in self.loaded_models.values()
            ],
            "running_models": running_models,
            "stats": self.stats.to_dict(),
            "available_ram_gb": self.memory_monitor.get_available_ram_gb(),
            "memory_critical": self.memory_monitor.is_memory_critical(),
            "top_predicted_models": self.predictor.get_top_models(3),
        }


class WarmupDaemon:
    """
    Daemon process for continuous warmup operation.

    Starts with the system and:
    - Monitors Ollama model state
    - Automatically warms up after idle periods
    - Logs warmup statistics
    """

    def __init__(
        self,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        default_model: Optional[str] = None,
        log_path: Optional[Path] = None
    ):
        self.manager = WarmupManager(
            ollama_host=ollama_host,
            keep_alive=keep_alive,
            default_model=default_model
        )

        self.log_path = log_path or self._get_default_log_path()
        self._setup_file_logging()

        self._running = False

    def _get_default_log_path(self) -> Path:
        """Get default log file path."""
        script_dir = Path(__file__).parent.resolve()
        root = script_dir.parent.parent
        log_dir = root / "logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir / "warmup_daemon.log"

    def _setup_file_logging(self):
        """Set up file logging for daemon."""
        if self.log_path:
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            log.addHandler(file_handler)

    def run(self):
        """Run the daemon."""
        log.info("=" * 60)
        log.info("USB-AI Model Warmup Daemon Starting")
        log.info("=" * 60)

        self._running = True

        # Set up signal handlers
        def signal_handler(sig, frame):
            log.info(f"Received signal {sig}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Set up callbacks
        self.manager.on_model_loaded = lambda m: log.info(f"Model loaded: {m}")
        self.manager.on_model_unloaded = lambda m: log.info(f"Model unloaded: {m}")
        self.manager.on_memory_critical = lambda: log.warning("Memory critical!")

        # Start the manager
        self.manager.start()

        # Main daemon loop
        stats_interval = 300  # Log stats every 5 minutes
        last_stats_log = time.time()

        try:
            while self._running:
                time.sleep(1)

                # Periodically log statistics
                if time.time() - last_stats_log >= stats_interval:
                    status = self.manager.get_status()
                    stats = status["stats"]
                    log.info(
                        f"Stats: cold={stats['cold_starts']}, "
                        f"warm={stats['warm_starts']}, "
                        f"keepalives={stats['keepalive_sent']}, "
                        f"preloads={stats['preloads_triggered']}, "
                        f"unloads={stats['memory_unloads']}"
                    )
                    log.info(
                        f"Timing: avg_cold={stats['average_cold_start_ms']:.0f}ms, "
                        f"avg_warm={stats['average_warm_start_ms']:.0f}ms"
                    )
                    last_stats_log = time.time()

        finally:
            self.manager.stop()
            self._save_final_stats()
            log.info("Warmup daemon stopped")

    def _save_final_stats(self):
        """Save final statistics to file."""
        try:
            stats_path = self.log_path.parent / "warmup_stats.json"
            status = self.manager.get_status()

            with open(stats_path, "w") as f:
                json.dump(status, f, indent=2, default=str)

            log.info(f"Final stats saved to {stats_path}")
        except Exception as e:
            log.error(f"Failed to save stats: {e}")


class Benchmarker:
    """Benchmark cold vs warm start times."""

    def __init__(self, ollama_host: str = DEFAULT_OLLAMA_HOST):
        self.client = OllamaClient(ollama_host)

    def run_benchmark(self, model: str, iterations: int = 3) -> BenchmarkResult:
        """
        Run cold vs warm benchmark on a model.

        Args:
            model: Model to benchmark
            iterations: Number of warm iterations to average

        Returns:
            BenchmarkResult with timing comparisons
        """
        log.info(f"Benchmarking model: {model}")
        log.info("=" * 50)

        # Ensure model is unloaded for cold start test
        log.info("Unloading model for cold start test...")
        self.client.unload_model(model)
        time.sleep(2)  # Allow time for cleanup

        # Cold start measurement
        log.info("Measuring cold start time...")
        cold_start = time.perf_counter()
        result = self.client.generate(
            model=model,
            prompt=WARMUP_PROMPT,
            keep_alive="30m"
        )
        cold_end = time.perf_counter()

        if "error" in result:
            log.error(f"Cold start failed: {result['error']}")
            return BenchmarkResult(
                model_name=model,
                cold_start_ms=-1,
                warm_start_ms=-1,
                improvement_percent=0,
                improvement_factor=0,
            )

        cold_start_ms = (cold_end - cold_start) * 1000
        log.info(f"Cold start: {cold_start_ms:.0f}ms")

        # Warm start measurements (average over iterations)
        log.info(f"Measuring warm start time ({iterations} iterations)...")
        warm_times = []

        for i in range(iterations):
            time.sleep(0.5)  # Small delay between requests

            warm_start = time.perf_counter()
            result = self.client.generate(
                model=model,
                prompt=WARMUP_PROMPT,
                keep_alive="30m"
            )
            warm_end = time.perf_counter()

            if "error" not in result:
                warm_ms = (warm_end - warm_start) * 1000
                warm_times.append(warm_ms)
                log.info(f"  Iteration {i+1}: {warm_ms:.0f}ms")

        if not warm_times:
            log.error("All warm start iterations failed")
            return BenchmarkResult(
                model_name=model,
                cold_start_ms=cold_start_ms,
                warm_start_ms=-1,
                improvement_percent=0,
                improvement_factor=0,
            )

        warm_start_ms = sum(warm_times) / len(warm_times)

        # Calculate improvement
        improvement_ms = cold_start_ms - warm_start_ms
        improvement_percent = (improvement_ms / cold_start_ms) * 100
        improvement_factor = cold_start_ms / warm_start_ms if warm_start_ms > 0 else 0

        result = BenchmarkResult(
            model_name=model,
            cold_start_ms=cold_start_ms,
            warm_start_ms=warm_start_ms,
            improvement_percent=improvement_percent,
            improvement_factor=improvement_factor,
        )

        # Print results
        log.info("")
        log.info("=" * 50)
        log.info("BENCHMARK RESULTS")
        log.info("=" * 50)
        log.info(f"Model:              {model}")
        log.info(f"Cold start:         {cold_start_ms:.0f}ms")
        log.info(f"Warm start (avg):   {warm_start_ms:.0f}ms")
        log.info(f"Improvement:        {improvement_ms:.0f}ms ({improvement_percent:.1f}%)")
        log.info(f"Speed factor:       {improvement_factor:.1f}x faster")
        log.info("=" * 50)

        return result

    def save_results(self, results: List[BenchmarkResult], path: Path):
        """Save benchmark results to file."""
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        log.info(f"Results saved to {path}")


def find_root() -> Path:
    """Locate USB-AI root directory."""
    script_dir = Path(__file__).parent.resolve()

    if (script_dir.parent.parent / "modules").exists():
        return script_dir.parent.parent
    if (script_dir.parent / "modules").exists():
        return script_dir.parent
    if (script_dir / "modules").exists():
        return script_dir

    return script_dir.parent.parent


def main() -> int:
    """Entry point for model warmup manager."""
    parser = argparse.ArgumentParser(
        description="USB-AI Model Warmup Manager - Eliminate cold start latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start warmup daemon
  python model_warmup.py --daemon

  # Warm up a specific model
  python model_warmup.py --warmup dolphin-llama3:8b

  # Run benchmark
  python model_warmup.py --benchmark --model dolphin-llama3:8b

  # Check status
  python model_warmup.py --status
"""
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as warmup daemon (continuous operation)"
    )
    parser.add_argument(
        "--warmup",
        metavar="MODEL",
        help="Warm up a specific model"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run cold vs warm benchmark"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current warmup status"
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Model to use for benchmark (default: auto-detect)"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama API host (default: {DEFAULT_OLLAMA_HOST})"
    )
    parser.add_argument(
        "--keep-alive",
        default=DEFAULT_KEEP_ALIVE,
        help=f"Keep-alive duration (default: {DEFAULT_KEEP_ALIVE})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # No action specified
    if not any([args.daemon, args.warmup, args.benchmark, args.status]):
        parser.print_help()
        return 0

    # Create manager
    manager = WarmupManager(
        ollama_host=args.host,
        keep_alive=args.keep_alive,
        default_model=args.model
    )

    # Check Ollama availability
    if not manager.client.is_available():
        log.error(f"Ollama server not available at {args.host}")
        log.error("Please start Ollama first: ollama serve")
        return 1

    # Handle actions
    if args.status:
        status = manager.get_status()

        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print("")
            print("=" * 60)
            print("         USB-AI Model Warmup Status")
            print("=" * 60)
            print("")
            print(f"  Ollama Available:  {'Yes' if status['ollama_available'] else 'No'}")
            print(f"  Default Model:     {status['default_model'] or 'Not set'}")
            print(f"  Keep Alive:        {status['keep_alive']}")
            print(f"  Available RAM:     {status['available_ram_gb']:.1f} GB")
            print(f"  Memory Critical:   {'Yes' if status['memory_critical'] else 'No'}")
            print("")

            if status['running_models']:
                print("  Currently Loaded Models:")
                for m in status['running_models']:
                    print(f"    - {m.get('name', 'Unknown')}")
            else:
                print("  No models currently loaded")

            print("")
            stats = status['stats']
            print("  Statistics:")
            print(f"    Cold starts:       {stats['cold_starts']}")
            print(f"    Warm starts:       {stats['warm_starts']}")
            print(f"    Keepalives sent:   {stats['keepalive_sent']}")
            print(f"    Preloads:          {stats['preloads_triggered']}")
            print(f"    Memory unloads:    {stats['memory_unloads']}")

            if stats['average_cold_start_ms'] > 0:
                print(f"    Avg cold start:    {stats['average_cold_start_ms']:.0f}ms")
            if stats['average_warm_start_ms'] > 0:
                print(f"    Avg warm start:    {stats['average_warm_start_ms']:.0f}ms")

            print("")
            print("=" * 60)

        return 0

    if args.warmup:
        result = manager.warmup_model(args.warmup)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if "error" in result:
                log.error(f"Warmup failed: {result['error']}")
                return 1

            print("")
            print("=" * 60)
            print("         Model Warmup Complete")
            print("=" * 60)
            print("")
            print(f"  Model:         {result['model']}")
            print(f"  Time:          {result['elapsed_ms']:.0f}ms")
            print(f"  Start Type:    {'warm' if result['was_loaded'] else 'cold'}")
            print(f"  Keep Alive:    {result['keep_alive']}")
            print("")
            print("=" * 60)

        return 0

    if args.benchmark:
        model = args.model

        if not model:
            # Auto-detect model
            models = manager.client.list_models()
            if not models:
                log.error("No models available for benchmark")
                return 1

            # Prefer dolphin-llama3
            for m in models:
                if "dolphin" in m.get("name", "").lower():
                    model = m["name"]
                    break

            if not model:
                model = models[0]["name"]

        benchmarker = Benchmarker(args.host)
        result = benchmarker.run_benchmark(model)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))

        # Save results
        root = find_root()
        results_path = root / "logs" / "warmup_benchmark.json"
        results_path.parent.mkdir(exist_ok=True)
        benchmarker.save_results([result], results_path)

        return 0 if result.cold_start_ms > 0 else 1

    if args.daemon:
        daemon = WarmupDaemon(
            ollama_host=args.host,
            keep_alive=args.keep_alive,
            default_model=args.model
        )
        daemon.run()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
