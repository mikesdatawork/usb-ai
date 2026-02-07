#!/usr/bin/env python3
"""
USB-AI Agent Logger
Centralized logging system for all agents with real-time monitoring.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "agents"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AGENT_ACTION = "AGENT_ACTION"
    AGENT_RESULT = "AGENT_RESULT"
    PERMISSION = "PERMISSION"
    OPTIMIZATION = "OPTIMIZATION"


@dataclass
class AgentLogEntry:
    """Structured log entry for agent activities."""
    timestamp: str
    agent_id: str
    agent_name: str
    level: str
    category: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class AgentLogger:
    """Centralized logger for USB-AI agents."""

    _instance = None
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

        self._initialized = True
        self.log_queue = queue.Queue()
        self.handlers: Dict[str, logging.Handler] = {}
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self.permissions: Dict[str, List[str]] = {}

        # Set up main log file
        self.main_log = LOGS_DIR / "agents.jsonl"
        self.activity_log = LOGS_DIR / "activity.log"
        self.optimization_log = LOGS_DIR / "optimization.jsonl"
        self.permission_log = LOGS_DIR / "permissions.jsonl"

        # Configure Python logging
        self._setup_logging()

        # Start background writer
        self._writer_thread = threading.Thread(target=self._log_writer, daemon=True)
        self._writer_thread.start()

    def _setup_logging(self):
        """Configure logging handlers."""
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)

        # File handler for activity
        file_handler = logging.FileHandler(self.activity_log)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Root logger
        root = logging.getLogger("usb-ai.agents")
        root.setLevel(logging.DEBUG)
        root.addHandler(console)
        root.addHandler(file_handler)

        self.logger = root

    def _log_writer(self):
        """Background thread to write logs."""
        while True:
            try:
                entry = self.log_queue.get(timeout=1.0)
                if entry is None:
                    break

                # Write to main log
                with open(self.main_log, "a") as f:
                    f.write(entry.to_json() + "\n")

                # Write to specialized logs
                if entry.category == "optimization":
                    with open(self.optimization_log, "a") as f:
                        f.write(entry.to_json() + "\n")
                elif entry.category == "permission":
                    with open(self.permission_log, "a") as f:
                        f.write(entry.to_json() + "\n")

                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Log writer error: {e}", file=sys.stderr)

    def log(
        self,
        agent_id: str,
        agent_name: str,
        level: LogLevel,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        tokens_used: Optional[int] = None
    ):
        """Log an agent activity."""
        entry = AgentLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            agent_id=agent_id,
            agent_name=agent_name,
            level=level.value,
            category=category,
            message=message,
            metadata=metadata,
            duration_ms=duration_ms,
            tokens_used=tokens_used
        )

        # Queue for async writing
        self.log_queue.put(entry)

        # Also log to Python logger
        log_level = getattr(logging, level.value, logging.INFO)
        self.logger.log(log_level, f"[{agent_name}] {message}")

        # Update stats
        self._update_stats(agent_id, agent_name, level, tokens_used)

    def _update_stats(
        self,
        agent_id: str,
        agent_name: str,
        level: LogLevel,
        tokens_used: Optional[int]
    ):
        """Update agent statistics."""
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "name": agent_name,
                "total_logs": 0,
                "errors": 0,
                "warnings": 0,
                "tokens_used": 0,
                "last_activity": None
            }

        stats = self.agent_stats[agent_id]
        stats["total_logs"] += 1
        stats["last_activity"] = datetime.utcnow().isoformat()

        if level == LogLevel.ERROR:
            stats["errors"] += 1
        elif level == LogLevel.WARNING:
            stats["warnings"] += 1

        if tokens_used:
            stats["tokens_used"] += tokens_used

    # Convenience methods
    def info(self, agent_id: str, agent_name: str, message: str, **kwargs):
        self.log(agent_id, agent_name, LogLevel.INFO, "general", message, **kwargs)

    def error(self, agent_id: str, agent_name: str, message: str, **kwargs):
        self.log(agent_id, agent_name, LogLevel.ERROR, "error", message, **kwargs)

    def action(self, agent_id: str, agent_name: str, action: str, **kwargs):
        self.log(agent_id, agent_name, LogLevel.AGENT_ACTION, "action", action, **kwargs)

    def result(self, agent_id: str, agent_name: str, result: str, **kwargs):
        self.log(agent_id, agent_name, LogLevel.AGENT_RESULT, "result", result, **kwargs)

    def optimization(self, agent_id: str, agent_name: str, message: str, **kwargs):
        self.log(agent_id, agent_name, LogLevel.OPTIMIZATION, "optimization", message, **kwargs)

    def permission(self, agent_id: str, agent_name: str, permission: str, granted: bool, **kwargs):
        metadata = kwargs.get("metadata", {})
        metadata["permission"] = permission
        metadata["granted"] = granted
        kwargs["metadata"] = metadata
        self.log(agent_id, agent_name, LogLevel.PERMISSION, "permission",
                 f"Permission '{permission}' {'granted' if granted else 'denied'}", **kwargs)

    # Permission management
    def grant_permission(self, agent_id: str, permission: str):
        """Grant a permission to an agent."""
        if agent_id not in self.permissions:
            self.permissions[agent_id] = []
        if permission not in self.permissions[agent_id]:
            self.permissions[agent_id].append(permission)
        self.permission("system", "PermissionManager", permission, True,
                       metadata={"target_agent": agent_id})

    def revoke_permission(self, agent_id: str, permission: str):
        """Revoke a permission from an agent."""
        if agent_id in self.permissions and permission in self.permissions[agent_id]:
            self.permissions[agent_id].remove(permission)
        self.permission("system", "PermissionManager", permission, False,
                       metadata={"target_agent": agent_id, "action": "revoke"})

    def has_permission(self, agent_id: str, permission: str) -> bool:
        """Check if agent has a permission."""
        return permission in self.permissions.get(agent_id, [])

    def grant_all_llm_permissions(self, agent_id: str):
        """Grant all LLM optimization permissions to an agent."""
        llm_permissions = [
            "llm.quantize",
            "llm.benchmark",
            "llm.configure",
            "llm.download_models",
            "llm.optimize_context",
            "llm.adjust_threads",
            "llm.gpu_offload",
            "llm.memory_limit",
            "llm.batch_size",
            "system.read_config",
            "system.write_config",
            "system.execute_scripts",
            "system.modify_env",
        ]
        for perm in llm_permissions:
            self.grant_permission(agent_id, perm)
        return llm_permissions

    # Monitoring
    def get_stats(self) -> Dict[str, Any]:
        """Get all agent statistics."""
        return {
            "agents": self.agent_stats,
            "permissions": self.permissions,
            "log_files": {
                "main": str(self.main_log),
                "activity": str(self.activity_log),
                "optimization": str(self.optimization_log),
                "permissions": str(self.permission_log)
            }
        }

    def tail_logs(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get last N log entries."""
        entries = []
        if self.main_log.exists():
            with open(self.main_log) as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def watch(self, callback=None):
        """Watch logs in real-time."""
        import subprocess
        proc = subprocess.Popen(
            ["tail", "-f", str(self.main_log)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            for line in proc.stdout:
                entry = json.loads(line.decode())
                if callback:
                    callback(entry)
                else:
                    print(f"[{entry['agent_name']}] {entry['message']}")
        except KeyboardInterrupt:
            proc.terminate()


# Singleton instance
logger = AgentLogger()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="USB-AI Agent Logger")
    parser.add_argument("--stats", action="store_true", help="Show agent statistics")
    parser.add_argument("--tail", type=int, default=0, help="Show last N log entries")
    parser.add_argument("--watch", action="store_true", help="Watch logs in real-time")
    parser.add_argument("--grant-llm", type=str, help="Grant LLM permissions to agent ID")
    parser.add_argument("--list-permissions", type=str, help="List permissions for agent ID")

    args = parser.parse_args()

    if args.stats:
        stats = logger.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.tail > 0:
        entries = logger.tail_logs(args.tail)
        for entry in entries:
            ts = entry.get("timestamp", "")[:19]
            agent = entry.get("agent_name", "unknown")
            msg = entry.get("message", "")
            print(f"{ts} [{agent}] {msg}")

    elif args.watch:
        print("Watching logs (Ctrl+C to stop)...")
        logger.watch()

    elif args.grant_llm:
        perms = logger.grant_all_llm_permissions(args.grant_llm)
        print(f"Granted {len(perms)} LLM permissions to {args.grant_llm}:")
        for p in perms:
            print(f"  ✓ {p}")

    elif args.list_permissions:
        perms = logger.permissions.get(args.list_permissions, [])
        if perms:
            print(f"Permissions for {args.list_permissions}:")
            for p in perms:
                print(f"  • {p}")
        else:
            print(f"No permissions for {args.list_permissions}")

    else:
        # Demo logging
        logger.info("demo-001", "DemoAgent", "Agent logger initialized")
        logger.action("demo-001", "DemoAgent", "Testing action logging")
        logger.optimization("demo-001", "DemoAgent", "Testing optimization logging")

        # Grant permissions demo
        perms = logger.grant_all_llm_permissions("llm-optimizer")
        print(f"\nGranted {len(perms)} permissions to llm-optimizer agent")

        print("\nLog files created in:", LOGS_DIR)
        print("\nUsage:")
        print("  python agent_logger.py --stats          # Show statistics")
        print("  python agent_logger.py --tail 20        # Show last 20 entries")
        print("  python agent_logger.py --watch          # Real-time monitoring")
        print("  python agent_logger.py --grant-llm ID   # Grant LLM permissions")
