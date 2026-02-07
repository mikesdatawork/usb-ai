#!/usr/bin/env python3
"""
parallel_builder.py

Orchestrates parallel build tasks for USB-AI.
Manages dependencies between build steps, provides progress reporting,
handles failures gracefully with retry logic, and supports resumable builds.

Usage:
    python scripts/build/parallel_builder.py [options]

Options:
    --resume            Resume from last saved state
    --phase PHASE       Start from specific phase
    --workers N         Number of parallel workers (default: 4)
    --dry-run           Show what would be executed
    --verbose           Enable verbose logging
    --manifest PATH     Custom manifest file
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

__version__ = "1.0.0"


# =============================================================================
# Enums and Constants
# =============================================================================

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class BuildPhase(Enum):
    """Build phases in execution order."""
    ENVIRONMENT = "environment"
    DOWNLOAD_OLLAMA = "download_ollama"
    DOWNLOAD_MODELS = "download_models"
    SETUP_WEBUI = "setup_webui"
    APPLY_THEME = "apply_theme"
    CREATE_LAUNCHERS = "create_launchers"
    VALIDATION = "validation"
    PACKAGING = "packaging"


# Default configuration
DEFAULT_CONFIG = {
    "max_workers": 4,
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "task_timeout_seconds": 3600,  # 1 hour max per task
    "state_file": ".build_state.json",
    "log_file": "build.log",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: str = ""
    error: str = ""
    retries: int = 0
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "output": self.output[:1000],  # Truncate for storage
            "error": self.error,
            "retries": self.retries,
            "artifacts": self.artifacts,
        }


@dataclass
class BuildTask:
    """Represents a single build task."""
    id: str
    name: str
    phase: BuildPhase
    description: str
    script: Optional[str] = None
    function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    parallel_group: Optional[str] = None
    timeout_seconds: int = 3600
    retries: int = 3
    skip_on_failure: bool = False
    artifacts: List[str] = field(default_factory=list)
    checksum_files: List[str] = field(default_factory=list)

    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None

    def can_run(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class BuildState:
    """Persistent build state for resume capability."""
    build_id: str
    started_at: datetime
    last_updated: datetime
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)
    current_phase: Optional[BuildPhase] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "build_id": self.build_id,
            "started_at": self.started_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "task_results": {k: v.to_dict() for k, v in self.task_results.items()},
            "checksums": self.checksums,
            "current_phase": self.current_phase.value if self.current_phase else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BuildState:
        """Create from dictionary."""
        state = cls(
            build_id=data["build_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            completed_tasks=set(data.get("completed_tasks", [])),
            failed_tasks=set(data.get("failed_tasks", [])),
            checksums=data.get("checksums", {}),
        )
        if data.get("current_phase"):
            state.current_phase = BuildPhase(data["current_phase"])
        return state


# =============================================================================
# Logging Setup
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',
    }

    def format(self, record):
        if hasattr(record, 'task_id'):
            record.msg = f"[{record.task_id}] {record.msg}"

        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Only colorize if terminal supports it
        if sys.stdout.isatty():
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging with optional file output."""
    logger = logging.getLogger("parallel_builder")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = ColoredFormatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Progress Reporting
# =============================================================================

class ProgressReporter:
    """Thread-safe progress reporting."""

    def __init__(self, total_tasks: int, logger: logging.Logger):
        self.total_tasks = total_tasks
        self.completed = 0
        self.failed = 0
        self.running: Set[str] = set()
        self.logger = logger
        self.lock = threading.Lock()
        self.start_time = time.time()

    def task_started(self, task_id: str):
        """Mark task as started."""
        with self.lock:
            self.running.add(task_id)
            self._print_status()

    def task_completed(self, task_id: str, success: bool):
        """Mark task as completed."""
        with self.lock:
            self.running.discard(task_id)
            if success:
                self.completed += 1
            else:
                self.failed += 1
            self._print_status()

    def _print_status(self):
        """Print current progress status."""
        elapsed = time.time() - self.start_time
        remaining = self.total_tasks - self.completed - self.failed

        status_parts = [
            f"Progress: {self.completed}/{self.total_tasks}",
            f"Failed: {self.failed}",
            f"Running: {len(self.running)}",
            f"Remaining: {remaining}",
            f"Elapsed: {elapsed:.1f}s",
        ]

        if self.running:
            status_parts.append(f"Active: {', '.join(sorted(self.running))}")

        self.logger.info(" | ".join(status_parts))

    def get_summary(self) -> dict:
        """Get final summary statistics."""
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "total_tasks": self.total_tasks,
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.total_tasks - self.completed - self.failed,
                "elapsed_seconds": elapsed,
                "success_rate": self.completed / self.total_tasks if self.total_tasks > 0 else 0,
            }


# =============================================================================
# Task Executor
# =============================================================================

class TaskExecutor:
    """Executes individual build tasks with retry logic."""

    def __init__(self, root_path: Path, logger: logging.Logger,
                 max_retries: int = 3, retry_delay: int = 5):
        self.root_path = root_path
        self.logger = logger
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute(self, task: BuildTask) -> TaskResult:
        """Execute a task with retry logic."""
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now(),
        )

        last_error = ""
        for attempt in range(task.retries + 1):
            try:
                self.logger.info(
                    f"Executing task: {task.name}" +
                    (f" (attempt {attempt + 1})" if attempt > 0 else ""),
                    extra={"task_id": task.id}
                )

                if task.script:
                    output, error = self._run_script(task)
                elif task.function:
                    output, error = self._run_function(task)
                else:
                    raise ValueError(f"Task {task.id} has no script or function")

                # Check if artifacts were created
                artifacts = self._verify_artifacts(task)

                result.status = TaskStatus.SUCCESS
                result.output = output
                result.artifacts = artifacts
                result.retries = attempt
                break

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"Task failed: {e}",
                    extra={"task_id": task.id}
                )

                if attempt < task.retries:
                    self.logger.info(
                        f"Retrying in {self.retry_delay} seconds...",
                        extra={"task_id": task.id}
                    )
                    time.sleep(self.retry_delay)
                else:
                    result.status = TaskStatus.FAILED
                    result.error = last_error
                    result.retries = attempt

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    def _run_script(self, task: BuildTask) -> Tuple[str, str]:
        """Run a Python script."""
        script_path = self.root_path / task.script

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root_path)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            cwd=self.root_path,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Script failed with exit code {result.returncode}: {result.stderr}"
            )

        return result.stdout, result.stderr

    def _run_function(self, task: BuildTask) -> Tuple[str, str]:
        """Run a Python function."""
        if not callable(task.function):
            raise ValueError(f"Task function is not callable: {task.function}")

        output = task.function(self.root_path)
        return str(output) if output else "", ""

    def _verify_artifacts(self, task: BuildTask) -> List[str]:
        """Verify that expected artifacts were created."""
        found_artifacts = []

        for artifact_pattern in task.artifacts:
            artifact_path = self.root_path / artifact_pattern

            if "*" in artifact_pattern:
                # Handle glob patterns
                matches = list(self.root_path.glob(artifact_pattern))
                found_artifacts.extend(str(m.relative_to(self.root_path)) for m in matches)
            elif artifact_path.exists():
                found_artifacts.append(artifact_pattern)
            else:
                self.logger.warning(
                    f"Expected artifact not found: {artifact_pattern}",
                    extra={"task_id": task.id}
                )

        return found_artifacts


# =============================================================================
# Manifest Parser
# =============================================================================

class ManifestParser:
    """Parses build manifest YAML files."""

    def __init__(self, root_path: Path, logger: logging.Logger):
        self.root_path = root_path
        self.logger = logger

    def parse(self, manifest_path: Path) -> List[BuildTask]:
        """Parse manifest file and return list of tasks."""
        if not manifest_path.exists():
            self.logger.warning(f"Manifest not found: {manifest_path}")
            return self._get_default_tasks()

        if not YAML_AVAILABLE:
            self.logger.warning("PyYAML not available, using default tasks")
            return self._get_default_tasks()

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        tasks = []

        for task_data in manifest.get("tasks", []):
            task = BuildTask(
                id=task_data["id"],
                name=task_data["name"],
                phase=BuildPhase(task_data["phase"]),
                description=task_data.get("description", ""),
                script=task_data.get("script"),
                dependencies=task_data.get("dependencies", []),
                parallel_group=task_data.get("parallel_group"),
                timeout_seconds=task_data.get("timeout", 3600),
                retries=task_data.get("retries", 3),
                skip_on_failure=task_data.get("skip_on_failure", False),
                artifacts=task_data.get("artifacts", []),
                checksum_files=task_data.get("checksum_files", []),
            )
            tasks.append(task)

        self.logger.info(f"Loaded {len(tasks)} tasks from manifest")
        return tasks

    def _get_default_tasks(self) -> List[BuildTask]:
        """Return default task list when no manifest is available."""
        return [
            BuildTask(
                id="setup_environment",
                name="Setup Environment",
                phase=BuildPhase.ENVIRONMENT,
                description="Create directory structure and config files",
                script="scripts/build/s001_setup_environment.py",
                dependencies=[],
                artifacts=[
                    "modules/config/system.json",
                    "modules/config/user.json",
                ],
            ),
            BuildTask(
                id="download_ollama_darwin_arm64",
                name="Download Ollama (macOS ARM64)",
                phase=BuildPhase.DOWNLOAD_OLLAMA,
                description="Download Ollama binary for macOS ARM64",
                script="scripts/build/s002_download_ollama.py",
                dependencies=["setup_environment"],
                parallel_group="ollama_downloads",
                artifacts=[
                    "modules/ollama-portable/bin/darwin-arm64/ollama",
                ],
            ),
            BuildTask(
                id="download_ollama_darwin_amd64",
                name="Download Ollama (macOS AMD64)",
                phase=BuildPhase.DOWNLOAD_OLLAMA,
                description="Download Ollama binary for macOS AMD64",
                script="scripts/build/s002_download_ollama.py",
                dependencies=["setup_environment"],
                parallel_group="ollama_downloads",
                artifacts=[
                    "modules/ollama-portable/bin/darwin-amd64/ollama",
                ],
            ),
            BuildTask(
                id="download_ollama_linux",
                name="Download Ollama (Linux)",
                phase=BuildPhase.DOWNLOAD_OLLAMA,
                description="Download Ollama binary for Linux",
                script="scripts/build/s002_download_ollama.py",
                dependencies=["setup_environment"],
                parallel_group="ollama_downloads",
                artifacts=[
                    "modules/ollama-portable/bin/linux-amd64/ollama",
                ],
            ),
            BuildTask(
                id="download_ollama_windows",
                name="Download Ollama (Windows)",
                phase=BuildPhase.DOWNLOAD_OLLAMA,
                description="Download Ollama binary for Windows",
                script="scripts/build/s002_download_ollama.py",
                dependencies=["setup_environment"],
                parallel_group="ollama_downloads",
                artifacts=[
                    "modules/ollama-portable/bin/windows-amd64/ollama.exe",
                ],
            ),
            BuildTask(
                id="download_models",
                name="Download AI Models",
                phase=BuildPhase.DOWNLOAD_MODELS,
                description="Download Dolphin-LLaMA3 and other models",
                script="scripts/build/s003_download_models.py",
                dependencies=["download_ollama_linux", "download_ollama_darwin_arm64"],
                timeout_seconds=7200,  # 2 hours for large models
                artifacts=[
                    "modules/models/config/models.json",
                ],
            ),
            BuildTask(
                id="setup_webui",
                name="Setup WebUI",
                phase=BuildPhase.SETUP_WEBUI,
                description="Install Open WebUI dependencies",
                script="scripts/build/s004_setup_webui.py",
                dependencies=["setup_environment"],
                artifacts=[
                    "modules/webui-portable/config/webui.json",
                    "modules/webui-portable/start_webui.py",
                ],
            ),
            BuildTask(
                id="apply_theme",
                name="Apply Theme",
                phase=BuildPhase.APPLY_THEME,
                description="Apply custom dark theme to WebUI",
                script="scripts/build/s005_apply_theme.py",
                dependencies=["setup_webui"],
                artifacts=[
                    "modules/webui-portable/static/css/custom-theme.css",
                ],
            ),
            BuildTask(
                id="validate_build",
                name="Validate Build",
                phase=BuildPhase.VALIDATION,
                description="Run validation checks on build output",
                script="scripts/build/validate_build.py",
                dependencies=[
                    "download_models",
                    "apply_theme",
                ],
            ),
        ]


# =============================================================================
# Dependency Graph
# =============================================================================

class DependencyGraph:
    """Manages task dependencies and determines execution order."""

    def __init__(self, tasks: List[BuildTask], logger: logging.Logger):
        self.tasks = {task.id: task for task in tasks}
        self.logger = logger
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Validate that all dependencies exist and there are no cycles."""
        # Check all dependencies exist
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(
                        f"Task '{task.id}' has unknown dependency: '{dep}'"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for dep in self.tasks[task_id].dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError(f"Circular dependency detected involving: {task_id}")

    def get_ready_tasks(self, completed: Set[str], running: Set[str]) -> List[BuildTask]:
        """Get tasks that are ready to run (all dependencies satisfied)."""
        ready = []

        for task in self.tasks.values():
            if task.id in completed or task.id in running:
                continue

            if task.can_run(completed):
                ready.append(task)

        return ready

    def get_parallel_groups(self, tasks: List[BuildTask]) -> Dict[Optional[str], List[BuildTask]]:
        """Group tasks by their parallel group."""
        groups: Dict[Optional[str], List[BuildTask]] = {}

        for task in tasks:
            group = task.parallel_group
            if group not in groups:
                groups[group] = []
            groups[group].append(task)

        return groups

    def get_execution_order(self) -> List[List[BuildTask]]:
        """Get tasks in topological order, grouped by parallel execution potential."""
        completed: Set[str] = set()
        order: List[List[BuildTask]] = []

        while len(completed) < len(self.tasks):
            ready = self.get_ready_tasks(completed, set())

            if not ready:
                remaining = set(self.tasks.keys()) - completed
                raise RuntimeError(f"Deadlock detected. Remaining tasks: {remaining}")

            order.append(ready)
            completed.update(task.id for task in ready)

        return order


# =============================================================================
# Parallel Build Orchestrator
# =============================================================================

class ParallelBuilder:
    """Main build orchestrator with parallel execution support."""

    def __init__(
        self,
        root_path: Path,
        manifest_path: Optional[Path] = None,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        self.root_path = root_path
        self.manifest_path = manifest_path or (root_path / "scripts" / "build" / "build_manifest.yaml")
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose
        self.dry_run = dry_run

        self.state_file = root_path / ".build_state.json"
        self.log_file = root_path / "build.log"

        self.logger = setup_logging(verbose, self.log_file)
        self.executor = TaskExecutor(root_path, self.logger, max_retries, retry_delay)

        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def handler(signum, frame):
            self.logger.warning("Shutdown requested, finishing current tasks...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def load_state(self) -> Optional[BuildState]:
        """Load previous build state for resume."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                data = json.load(f)
            state = BuildState.from_dict(data)
            self.logger.info(f"Loaded build state: {len(state.completed_tasks)} tasks completed")
            return state
        except Exception as e:
            self.logger.warning(f"Failed to load build state: {e}")
            return None

    def save_state(self, state: BuildState):
        """Save build state for resume capability."""
        state.last_updated = datetime.now()

        try:
            with open(self.state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save build state: {e}")

    def clear_state(self):
        """Clear saved build state."""
        if self.state_file.exists():
            self.state_file.unlink()
            self.logger.info("Build state cleared")

    def build(
        self,
        resume: bool = False,
        start_phase: Optional[BuildPhase] = None,
    ) -> bool:
        """Execute the build process."""
        self.logger.info("=" * 60)
        self.logger.info("         USB-AI Parallel Build System")
        self.logger.info("=" * 60)
        self.logger.info(f"Platform: {platform.system()} {platform.machine()}")
        self.logger.info(f"Root: {self.root_path}")
        self.logger.info(f"Workers: {self.max_workers}")
        self.logger.info("")

        # Parse manifest
        parser = ManifestParser(self.root_path, self.logger)
        tasks = parser.parse(self.manifest_path)

        if not tasks:
            self.logger.error("No tasks found")
            return False

        # Load or create state
        state: Optional[BuildState] = None
        if resume:
            state = self.load_state()

        if state is None:
            state = BuildState(
                build_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                started_at=datetime.now(),
                last_updated=datetime.now(),
            )

        # Filter tasks based on start phase
        if start_phase:
            phase_order = list(BuildPhase)
            start_index = phase_order.index(start_phase)
            valid_phases = set(phase_order[start_index:])
            tasks = [t for t in tasks if t.phase in valid_phases]
            self.logger.info(f"Starting from phase: {start_phase.value}")

        # Skip already completed tasks
        tasks = [t for t in tasks if t.id not in state.completed_tasks]

        if not tasks:
            self.logger.info("All tasks already completed")
            return True

        # Create dependency graph
        try:
            graph = DependencyGraph(tasks, self.logger)
        except ValueError as e:
            self.logger.error(f"Dependency error: {e}")
            return False

        # Dry run - just show execution plan
        if self.dry_run:
            return self._show_execution_plan(graph)

        # Execute build
        progress = ProgressReporter(len(tasks), self.logger)
        success = self._execute_build(graph, state, progress)

        # Print summary
        self._print_summary(progress.get_summary(), state)

        return success

    def _show_execution_plan(self, graph: DependencyGraph) -> bool:
        """Show what would be executed without running."""
        self.logger.info("")
        self.logger.info("Execution Plan (dry-run):")
        self.logger.info("-" * 40)

        order = graph.get_execution_order()

        for wave_num, wave in enumerate(order, 1):
            self.logger.info(f"\nWave {wave_num} (parallel):")
            for task in wave:
                deps = ", ".join(task.dependencies) if task.dependencies else "none"
                self.logger.info(f"  - {task.name}")
                self.logger.info(f"    Script: {task.script}")
                self.logger.info(f"    Dependencies: {deps}")

        self.logger.info("")
        return True

    def _execute_build(
        self,
        graph: DependencyGraph,
        state: BuildState,
        progress: ProgressReporter,
    ) -> bool:
        """Execute build with parallel task execution."""
        completed: Set[str] = set(state.completed_tasks)
        failed: Set[str] = set(state.failed_tasks)
        running: Set[str] = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[concurrent.futures.Future, BuildTask] = {}

            while not self._shutdown_requested:
                # Submit ready tasks
                ready_tasks = graph.get_ready_tasks(completed, running)

                for task in ready_tasks:
                    # Skip if any required dependency failed
                    if any(dep in failed for dep in task.dependencies):
                        if task.skip_on_failure:
                            self.logger.warning(f"Skipping {task.name} due to failed dependency")
                            task.status = TaskStatus.SKIPPED
                            completed.add(task.id)
                            progress.task_completed(task.id, True)
                            continue
                        else:
                            self.logger.error(f"Cannot run {task.name} - dependency failed")
                            failed.add(task.id)
                            progress.task_completed(task.id, False)
                            continue

                    running.add(task.id)
                    progress.task_started(task.id)

                    future = executor.submit(self.executor.execute, task)
                    futures[future] = task

                # Wait for any task to complete
                if not futures:
                    if running:
                        time.sleep(0.1)
                        continue
                    break

                done, _ = concurrent.futures.wait(
                    futures,
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    task = futures.pop(future)
                    running.discard(task.id)

                    try:
                        result = future.result()
                        task.result = result
                        state.task_results[task.id] = result

                        if result.status == TaskStatus.SUCCESS:
                            completed.add(task.id)
                            state.completed_tasks.add(task.id)
                            progress.task_completed(task.id, True)
                            self.logger.info(
                                f"Completed: {task.name} ({result.duration_seconds:.1f}s)",
                                extra={"task_id": task.id}
                            )
                        else:
                            failed.add(task.id)
                            state.failed_tasks.add(task.id)
                            progress.task_completed(task.id, False)
                            self.logger.error(
                                f"Failed: {task.name} - {result.error}",
                                extra={"task_id": task.id}
                            )

                    except Exception as e:
                        failed.add(task.id)
                        state.failed_tasks.add(task.id)
                        progress.task_completed(task.id, False)
                        self.logger.error(
                            f"Exception in {task.name}: {e}",
                            extra={"task_id": task.id}
                        )

                # Save state periodically
                self.save_state(state)

        # Handle shutdown
        if self._shutdown_requested:
            self.logger.warning("Build interrupted, state saved for resume")
            self.save_state(state)
            return False

        # Final state save
        self.save_state(state)

        return len(failed) == 0

    def _print_summary(self, stats: dict, state: BuildState):
        """Print build summary."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("                 Build Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Build ID: {state.build_id}")
        self.logger.info(f"Duration: {stats['elapsed_seconds']:.1f} seconds")
        self.logger.info("")
        self.logger.info(f"Tasks Completed: {stats['completed']}/{stats['total_tasks']}")
        self.logger.info(f"Tasks Failed: {stats['failed']}")
        self.logger.info(f"Success Rate: {stats['success_rate']*100:.1f}%")
        self.logger.info("")

        if state.failed_tasks:
            self.logger.info("Failed Tasks:")
            for task_id in state.failed_tasks:
                result = state.task_results.get(task_id)
                error = result.error if result else "Unknown error"
                self.logger.info(f"  - {task_id}: {error[:100]}")
            self.logger.info("")
            self.logger.info("To resume: python scripts/build/parallel_builder.py --resume")
        else:
            self.logger.info("BUILD SUCCESSFUL")
            # Clear state on success
            self.clear_state()

        self.logger.info("=" * 60)


# =============================================================================
# Checksum Utilities
# =============================================================================

def compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute file checksum."""
    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_checksums(root_path: Path, checksums: Dict[str, str]) -> List[str]:
    """Verify file checksums, return list of mismatched files."""
    mismatched = []

    for file_path, expected in checksums.items():
        full_path = root_path / file_path

        if not full_path.exists():
            mismatched.append(f"{file_path} (missing)")
            continue

        actual = compute_checksum(full_path)
        if actual != expected:
            mismatched.append(f"{file_path} (mismatch)")

    return mismatched


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="USB-AI Parallel Build System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full build
    python scripts/build/parallel_builder.py

    # Resume interrupted build
    python scripts/build/parallel_builder.py --resume

    # Start from specific phase
    python scripts/build/parallel_builder.py --phase download_models

    # Show execution plan without running
    python scripts/build/parallel_builder.py --dry-run

    # Use more workers for faster parallel downloads
    python scripts/build/parallel_builder.py --workers 8
        """
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved state",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=[p.value for p in BuildPhase],
        help="Start from specific phase",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to custom manifest file",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear saved state and start fresh",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"parallel_builder {__version__}",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine root path
    script_path = Path(__file__).resolve()
    root_path = script_path.parent.parent.parent

    # Create builder
    builder = ParallelBuilder(
        root_path=root_path,
        manifest_path=Path(args.manifest) if args.manifest else None,
        max_workers=args.workers,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    # Handle clean flag
    if args.clean:
        builder.clear_state()
        if not args.resume:
            print("State cleared. Run again without --clean to start fresh build.")
            return 0

    # Parse start phase
    start_phase = BuildPhase(args.phase) if args.phase else None

    # Run build
    success = builder.build(
        resume=args.resume,
        start_phase=start_phase,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
