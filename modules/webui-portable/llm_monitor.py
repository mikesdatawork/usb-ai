#!/usr/bin/env python3
"""
LLM Monitor - Robust logging for model state tracking.

Tracks:
- Model loading/unloading states
- Request/response times
- Hang detection
- Health status
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import requests

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "llm_monitor.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("llm_monitor")


class ModelState(Enum):
    """Possible states for an LLM model."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"      # Model exists, not loaded
    LOADING = "loading"          # Currently loading into memory
    READY = "ready"              # Loaded and ready for inference
    PROCESSING = "processing"    # Currently generating response
    HANGING = "hanging"          # No response within timeout
    ERROR = "error"              # Error state
    UNLOADING = "unloading"      # Being unloaded from memory


@dataclass
class ModelStatus:
    """Status information for a model."""
    name: str
    state: ModelState = ModelState.UNKNOWN
    last_checked: Optional[datetime] = None
    last_response_time: Optional[float] = None  # seconds
    avg_response_time: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    is_loaded: bool = False
    memory_usage_mb: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_response_time_sec": self.last_response_time,
            "avg_response_time_sec": round(self.avg_response_time, 2),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "is_loaded": self.is_loaded,
            "memory_usage_mb": self.memory_usage_mb
        }


@dataclass
class RequestTracker:
    """Tracks an in-flight request."""
    model: str
    start_time: float
    prompt_preview: str
    thread_id: int
    timeout: float = 300.0
    completed: bool = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def is_hanging(self) -> bool:
        return not self.completed and self.elapsed > self.timeout


class LLMMonitor:
    """
    Monitors LLM model states and health.

    Usage:
        monitor = LLMMonitor()
        monitor.start()

        # Track a request
        with monitor.track_request("dolphin-llama3:8b", "Hello"):
            response = ollama_chat(...)

        # Get status
        status = monitor.get_model_status("dolphin-llama3:8b")
    """

    HANG_THRESHOLD_SEC = 60.0  # Consider hanging after 60s no response
    HEALTH_CHECK_INTERVAL = 30  # Check health every 30s

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        self.ollama_url = ollama_url
        self.models: dict[str, ModelStatus] = {}
        self.active_requests: dict[int, RequestTracker] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._request_id = 0

    def start(self):
        """Start background monitoring."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        log.info("LLM Monitor started")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        log.info("LLM Monitor stopped")

    def _monitor_loop(self):
        """Background loop for health checks and hang detection."""
        while self._running:
            try:
                self._check_ollama_health()
                self._detect_hanging_requests()
            except Exception as e:
                log.error(f"Monitor loop error: {e}")
            time.sleep(self.HEALTH_CHECK_INTERVAL)

    def _check_ollama_health(self):
        """Check Ollama server and model availability."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                available_models = {m["name"] for m in data.get("models", [])}

                with self._lock:
                    # Update known models
                    for model_name in available_models:
                        if model_name not in self.models:
                            self.models[model_name] = ModelStatus(name=model_name)

                        model = self.models[model_name]
                        model.last_checked = datetime.now()

                        # Update state if not actively processing
                        if model.state not in (ModelState.PROCESSING, ModelState.LOADING):
                            model.state = ModelState.AVAILABLE

                log.debug(f"Health check: {len(available_models)} models available")
            else:
                log.warning(f"Ollama health check failed: HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            log.error("Ollama server not reachable")
            with self._lock:
                for model in self.models.values():
                    model.state = ModelState.ERROR
                    model.last_error = "Ollama server not reachable"
        except Exception as e:
            log.error(f"Health check error: {e}")

    def _detect_hanging_requests(self):
        """Check for requests that have exceeded timeout."""
        with self._lock:
            for req_id, tracker in list(self.active_requests.items()):
                if tracker.is_hanging and not tracker.completed:
                    model = self.models.get(tracker.model)
                    if model:
                        model.state = ModelState.HANGING
                        model.last_error = f"Request hanging for {tracker.elapsed:.1f}s"

                    log.warning(
                        f"HANGING DETECTED - Model: {tracker.model}, "
                        f"Elapsed: {tracker.elapsed:.1f}s, "
                        f"Prompt: {tracker.prompt_preview[:50]}..."
                    )

    def track_request(self, model: str, prompt: str, timeout: float = 300.0):
        """
        Context manager to track a request.

        Usage:
            with monitor.track_request("model", "prompt") as tracker:
                # make request
        """
        return _RequestContext(self, model, prompt, timeout)

    def _start_request(self, model: str, prompt: str, timeout: float) -> int:
        """Internal: Start tracking a request."""
        with self._lock:
            self._request_id += 1
            req_id = self._request_id

            tracker = RequestTracker(
                model=model,
                start_time=time.time(),
                prompt_preview=prompt[:100],
                thread_id=threading.current_thread().ident,
                timeout=timeout
            )
            self.active_requests[req_id] = tracker

            # Update model state
            if model not in self.models:
                self.models[model] = ModelStatus(name=model)
            self.models[model].state = ModelState.PROCESSING
            self.models[model].request_count += 1

            log.info(f"Request started - ID: {req_id}, Model: {model}, Prompt: {prompt[:50]}...")

        return req_id

    def _end_request(self, req_id: int, success: bool = True, error: Optional[str] = None):
        """Internal: End tracking a request."""
        with self._lock:
            tracker = self.active_requests.pop(req_id, None)
            if not tracker:
                return

            tracker.completed = True
            elapsed = tracker.elapsed
            model = self.models.get(tracker.model)

            if model:
                model.last_response_time = elapsed
                model.last_checked = datetime.now()

                # Update average response time
                if model.request_count > 0:
                    model.avg_response_time = (
                        (model.avg_response_time * (model.request_count - 1) + elapsed)
                        / model.request_count
                    )

                if success:
                    model.state = ModelState.READY
                    log.info(
                        f"Request completed - ID: {req_id}, Model: {tracker.model}, "
                        f"Time: {elapsed:.2f}s"
                    )
                else:
                    model.state = ModelState.ERROR
                    model.error_count += 1
                    model.last_error = error
                    log.error(
                        f"Request failed - ID: {req_id}, Model: {tracker.model}, "
                        f"Time: {elapsed:.2f}s, Error: {error}"
                    )

    def get_model_status(self, model: str) -> Optional[ModelStatus]:
        """Get status for a specific model."""
        with self._lock:
            return self.models.get(model)

    def get_all_status(self) -> dict[str, dict]:
        """Get status for all models."""
        with self._lock:
            return {name: status.to_dict() for name, status in self.models.items()}

    def get_active_requests(self) -> list[dict]:
        """Get info about active requests."""
        with self._lock:
            return [
                {
                    "id": req_id,
                    "model": t.model,
                    "elapsed_sec": round(t.elapsed, 1),
                    "is_hanging": t.is_hanging,
                    "prompt_preview": t.prompt_preview[:50]
                }
                for req_id, t in self.active_requests.items()
            ]

    def get_summary(self) -> dict:
        """Get a summary of all model states."""
        with self._lock:
            hanging = [m.name for m in self.models.values() if m.state == ModelState.HANGING]
            errors = [m.name for m in self.models.values() if m.state == ModelState.ERROR]
            ready = [m.name for m in self.models.values() if m.state in (ModelState.READY, ModelState.AVAILABLE)]
            processing = [m.name for m in self.models.values() if m.state == ModelState.PROCESSING]

            return {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.models),
                "ready": ready,
                "processing": processing,
                "hanging": hanging,
                "errors": errors,
                "active_requests": len(self.active_requests),
                "models": {name: status.to_dict() for name, status in self.models.items()}
            }

    def log_summary(self):
        """Log current status summary."""
        summary = self.get_summary()

        log.info("=" * 60)
        log.info("LLM STATUS SUMMARY")
        log.info("=" * 60)
        log.info(f"Total Models: {summary['total_models']}")
        log.info(f"Ready: {summary['ready']}")
        log.info(f"Processing: {summary['processing']}")

        if summary['hanging']:
            log.warning(f"HANGING: {summary['hanging']}")
        if summary['errors']:
            log.error(f"ERRORS: {summary['errors']}")

        log.info(f"Active Requests: {summary['active_requests']}")
        log.info("=" * 60)


class _RequestContext:
    """Context manager for request tracking."""

    def __init__(self, monitor: LLMMonitor, model: str, prompt: str, timeout: float):
        self.monitor = monitor
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.req_id: Optional[int] = None

    def __enter__(self):
        self.req_id = self.monitor._start_request(self.model, self.prompt, self.timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.monitor._end_request(self.req_id, success=success, error=error)
        return False  # Don't suppress exceptions


# Global monitor instance
_monitor: Optional[LLMMonitor] = None


def get_monitor() -> LLMMonitor:
    """Get or create the global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = LLMMonitor()
        _monitor.start()
    return _monitor


def log_model_states():
    """Convenience function to log all model states."""
    get_monitor().log_summary()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Monitor CLI")
    parser.add_argument("--url", default="http://127.0.0.1:11434", help="Ollama URL")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")
    args = parser.parse_args()

    monitor = LLMMonitor(args.url)
    monitor.start()

    print("\nLLM Monitor - Press Ctrl+C to stop\n")

    try:
        while True:
            monitor._check_ollama_health()
            monitor.log_summary()

            if not args.watch:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.stop()
