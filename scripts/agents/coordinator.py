#!/usr/bin/env python3
"""
coordinator.py

Agent Swarm Coordinator for USB-AI Build System.
Tracks active agents, manages task queues, logs activities,
and provides status dashboard output.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from typing import Dict, List, Optional, Callable, Any
from uuid import uuid4

__version__ = "1.0.0"

# Configure logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "coordinator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("coordinator")


class AgentStatus(Enum):
    """Agent status states."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class TaskStatus(Enum):
    """Task status states."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    RETRY = "RETRY"


class MessageType(Enum):
    """Message types for agent communication."""
    COMMAND = "COMMAND"
    STATUS = "STATUS"
    TASK_ASSIGN = "TASK_ASSIGN"
    TASK_COMPLETE = "TASK_COMPLETE"
    ERROR = "ERROR"
    HANDOFF = "HANDOFF"
    BROADCAST = "BROADCAST"


@dataclass
class Agent:
    """Agent representation."""
    id: str
    name: str
    role: str
    tier: str
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    progress: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "tier": self.tier,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "current_task": self.current_task,
            "progress": self.progress,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "error_count": self.error_count,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed
        }


@dataclass(order=True)
class Task:
    """Task representation with priority ordering."""
    priority: int
    id: str = field(compare=False)
    name: str = field(compare=False)
    task_type: str = field(compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    assigned_agent: Optional[str] = field(default=None, compare=False)
    parameters: dict = field(default_factory=dict, compare=False)
    created_at: datetime = field(default_factory=datetime.now, compare=False)
    started_at: Optional[datetime] = field(default=None, compare=False)
    completed_at: Optional[datetime] = field(default=None, compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    result: Optional[dict] = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "priority": self.priority,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "result": self.result,
            "error": self.error
        }


@dataclass
class Message:
    """Message for agent communication."""
    id: str
    msg_type: MessageType
    from_agent: str
    to_agent: str
    payload: dict
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.msg_type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


class EventBus:
    """Simple event bus for agent coordination."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event_type: str, handler: Callable) -> None:
        """Register event handler."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def emit(self, event_type: str, data: Any = None) -> None:
        """Emit event to all handlers."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                log.error(f"Event handler error for {event_type}: {e}")


class TaskQueue:
    """Priority-based task queue."""

    def __init__(self):
        self._queue: PriorityQueue = PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, task: Task) -> str:
        """Add task to queue."""
        with self._lock:
            self._queue.put(task)
            self._tasks[task.id] = task
            log.info(f"Task queued: {task.name} (priority={task.priority})")
        return task.id

    def get_next(self) -> Optional[Task]:
        """Get next task from queue."""
        with self._lock:
            if self._queue.empty():
                return None
            task = self._queue.get_nowait()
            return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, **updates) -> bool:
        """Update task fields."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            return True

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_pending_tasks(self) -> List[Task]:
        """Get pending tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def get_running_tasks(self) -> List[Task]:
        """Get running tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]


class AgentRegistry:
    """Registry for tracking agents."""

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._lock = threading.Lock()

    def register(self, agent: Agent) -> str:
        """Register an agent."""
        with self._lock:
            self._agents[agent.id] = agent
            log.info(f"Agent registered: {agent.name} ({agent.id})")
        return agent.id

    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent."""
        with self._lock:
            if agent_id in self._agents:
                agent = self._agents.pop(agent_id)
                log.info(f"Agent unregistered: {agent.name}")
                return True
            return False

    def get(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_all(self) -> List[Agent]:
        """Get all agents."""
        return list(self._agents.values())

    def get_by_status(self, status: AgentStatus) -> List[Agent]:
        """Get agents by status."""
        return [a for a in self._agents.values() if a.status == status]

    def get_by_capability(self, capability: str) -> List[Agent]:
        """Get agents with capability."""
        return [a for a in self._agents.values() if capability in a.capabilities]

    def update(self, agent_id: str, **updates) -> bool:
        """Update agent fields."""
        with self._lock:
            if agent_id not in self._agents:
                return False
            agent = self._agents[agent_id]
            for key, value in updates.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            return True


class ActivityLog:
    """Log agent activities."""

    def __init__(self, log_file: Path):
        self._log_file = log_file
        self._lock = threading.Lock()

    def log(self, event_type: str, actor: str, action: str,
            target: str = None, details: dict = None, outcome: str = "success"):
        """Log an activity."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "target": target,
            "details": details or {},
            "outcome": outcome
        }

        with self._lock:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def get_recent(self, count: int = 50) -> List[dict]:
        """Get recent log entries."""
        entries = []
        try:
            with open(self._log_file, "r") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except FileNotFoundError:
            pass
        return entries[-count:]


class Coordinator:
    """Main coordinator for agent swarm."""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session-{uuid4().hex[:8]}"
        self.started_at = datetime.now()

        self.registry = AgentRegistry()
        self.task_queue = TaskQueue()
        self.event_bus = EventBus()
        self.activity_log = ActivityLog(LOG_DIR / "activity.jsonl")

        self._running = False
        self._lock = threading.Lock()

        self._setup_event_handlers()
        self._register_default_agents()

        log.info(f"Coordinator initialized: {self.session_id}")

    def _setup_event_handlers(self):
        """Setup default event handlers."""
        self.event_bus.on("AGENT_REGISTERED", self._on_agent_registered)
        self.event_bus.on("TASK_COMPLETED", self._on_task_completed)
        self.event_bus.on("TASK_FAILED", self._on_task_failed)
        self.event_bus.on("AGENT_ERROR", self._on_agent_error)

    def _on_agent_registered(self, agent: Agent):
        """Handle agent registration."""
        self.activity_log.log(
            "AGENT_REGISTERED",
            "coordinator",
            "register",
            agent.id,
            {"name": agent.name, "capabilities": agent.capabilities}
        )

    def _on_task_completed(self, data: dict):
        """Handle task completion."""
        task_id = data.get("task_id")
        agent_id = data.get("agent_id")
        self.activity_log.log(
            "TASK_COMPLETED",
            agent_id,
            "complete",
            task_id,
            data.get("result")
        )

    def _on_task_failed(self, data: dict):
        """Handle task failure."""
        task_id = data.get("task_id")
        agent_id = data.get("agent_id")
        self.activity_log.log(
            "TASK_FAILED",
            agent_id,
            "fail",
            task_id,
            {"error": data.get("error")},
            outcome="failure"
        )

    def _on_agent_error(self, data: dict):
        """Handle agent error."""
        agent_id = data.get("agent_id")
        self.activity_log.log(
            "AGENT_ERROR",
            agent_id,
            "error",
            details={"error": data.get("error")},
            outcome="error"
        )

    def _register_default_agents(self):
        """Register default agents based on AGENTS.md configuration."""
        default_agents = [
            Agent(
                id="lead-001",
                name="BuildOrchestrator",
                role="Team Lead",
                tier="Command",
                capabilities=["coordinate", "assign", "monitor"]
            ),
            Agent(
                id="plan-001",
                name="ProactiveBuilder",
                role="Anticipatory Prep",
                tier="Planning",
                capabilities=["prefetch", "optimize", "schedule"]
            ),
            Agent(
                id="plan-002",
                name="Optimizer",
                role="Self-Improvement",
                tier="Planning",
                capabilities=["analyze", "improve", "report"]
            ),
            Agent(
                id="exec-001",
                name="DownloadManager",
                role="File Downloads",
                tier="Execution",
                capabilities=["download", "verify", "resume"]
            ),
            Agent(
                id="exec-002",
                name="FlaskBuilder",
                role="WebUI Development",
                tier="Execution",
                capabilities=["flask", "api", "streaming"]
            ),
            Agent(
                id="exec-003",
                name="CryptoManager",
                role="Encryption Ops",
                tier="Execution",
                capabilities=["encrypt", "decrypt", "mount"]
            ),
            Agent(
                id="exec-004",
                name="InterfaceDesigner",
                role="UI/Theme",
                tier="Execution",
                capabilities=["css", "theme", "layout"]
            ),
            Agent(
                id="valid-001",
                name="QualityGate",
                role="Build Validation",
                tier="Validation",
                capabilities=["validate", "test", "verify"]
            ),
            Agent(
                id="supp-001",
                name="ResourceFetcher",
                role="Web Resources",
                tier="Support",
                capabilities=["fetch", "cache", "update"]
            ),
            Agent(
                id="mon-001",
                name="StatusReporter",
                role="Status Updates",
                tier="Monitoring",
                capabilities=["monitor", "report", "alert"]
            )
        ]

        for agent in default_agents:
            self.registry.register(agent)
            self.event_bus.emit("AGENT_REGISTERED", agent)

    def assign_task(self, task: Task, agent_id: str = None) -> Optional[str]:
        """Assign a task to an agent."""
        if agent_id:
            agent = self.registry.get(agent_id)
            if not agent:
                log.error(f"Agent not found: {agent_id}")
                return None
        else:
            agent = self._select_best_agent(task)
            if not agent:
                log.warning(f"No available agent for task: {task.name}")
                self.task_queue.add(task)
                return None

        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent.id
        task.started_at = datetime.now()

        self.task_queue.add(task)
        self.registry.update(
            agent.id,
            status=AgentStatus.RUNNING,
            current_task=task.id,
            progress=0
        )

        self.activity_log.log(
            "TASK_ASSIGN",
            "coordinator",
            "assign",
            agent.id,
            {"task_id": task.id, "task_name": task.name}
        )

        log.info(f"Task assigned: {task.name} -> {agent.name}")
        return task.id

    def _select_best_agent(self, task: Task) -> Optional[Agent]:
        """Select best available agent for task."""
        capable_agents = self.registry.get_by_capability(task.task_type)
        available_agents = [
            a for a in capable_agents
            if a.status == AgentStatus.IDLE
        ]

        if not available_agents:
            return None

        available_agents.sort(key=lambda a: (a.tasks_completed, a.error_count))
        return available_agents[0]

    def complete_task(self, task_id: str, result: dict = None) -> bool:
        """Mark task as complete."""
        task = self.task_queue.get_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETE
        task.completed_at = datetime.now()
        task.result = result

        if task.assigned_agent:
            agent = self.registry.get(task.assigned_agent)
            if agent:
                self.registry.update(
                    agent.id,
                    status=AgentStatus.IDLE,
                    current_task=None,
                    progress=0,
                    tasks_completed=agent.tasks_completed + 1
                )

        self.event_bus.emit("TASK_COMPLETED", {
            "task_id": task_id,
            "agent_id": task.assigned_agent,
            "result": result
        })

        log.info(f"Task completed: {task.name}")
        return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        task = self.task_queue.get_task(task_id)
        if not task:
            return False

        task.retry_count += 1

        if task.retry_count < task.max_retries:
            task.status = TaskStatus.RETRY
            task.error = error
            log.warning(f"Task retry {task.retry_count}/{task.max_retries}: {task.name}")
            return self._retry_task(task)

        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error = error

        if task.assigned_agent:
            agent = self.registry.get(task.assigned_agent)
            if agent:
                self.registry.update(
                    agent.id,
                    status=AgentStatus.IDLE,
                    current_task=None,
                    progress=0,
                    tasks_failed=agent.tasks_failed + 1,
                    error_count=agent.error_count + 1
                )

        self.event_bus.emit("TASK_FAILED", {
            "task_id": task_id,
            "agent_id": task.assigned_agent,
            "error": error
        })

        log.error(f"Task failed: {task.name} - {error}")
        return True

    def _retry_task(self, task: Task) -> bool:
        """Retry a failed task."""
        task.status = TaskStatus.PENDING
        task.assigned_agent = None

        if task.assigned_agent:
            self.registry.update(
                task.assigned_agent,
                status=AgentStatus.IDLE,
                current_task=None
            )

        return self.assign_task(task) is not None

    def update_progress(self, task_id: str, progress: int) -> bool:
        """Update task progress."""
        task = self.task_queue.get_task(task_id)
        if not task or not task.assigned_agent:
            return False

        self.registry.update(task.assigned_agent, progress=progress)
        return True

    def get_build_status(self) -> dict:
        """Get overall build status."""
        agents = self.registry.get_all()
        tasks = self.task_queue.get_all_tasks()

        active_agents = len([a for a in agents if a.status == AgentStatus.RUNNING])
        idle_agents = len([a for a in agents if a.status == AgentStatus.IDLE])
        error_agents = len([a for a in agents if a.status == AgentStatus.ERROR])

        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETE])
        running_tasks = len([t for t in tasks if t.status == TaskStatus.RUNNING])
        pending_tasks = len([t for t in tasks if t.status == TaskStatus.PENDING])
        failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])

        elapsed = (datetime.now() - self.started_at).total_seconds()

        return {
            "session_id": self.session_id,
            "started": self.started_at.isoformat(),
            "elapsed_seconds": elapsed,
            "agents": {
                "total": len(agents),
                "active": active_agents,
                "idle": idle_agents,
                "error": error_agents
            },
            "tasks": {
                "total": total_tasks,
                "completed": completed_tasks,
                "running": running_tasks,
                "pending": pending_tasks,
                "failed": failed_tasks
            },
            "health": self._calculate_health(error_agents, failed_tasks)
        }

    def _calculate_health(self, error_agents: int, failed_tasks: int) -> str:
        """Calculate overall health status."""
        if error_agents > 0 or failed_tasks > 2:
            return "CRITICAL"
        elif failed_tasks > 0:
            return "DEGRADED"
        return "HEALTHY"

    def print_dashboard(self):
        """Print status dashboard to console."""
        status = self.get_build_status()
        agents = self.registry.get_all()
        pending_tasks = self.task_queue.get_pending_tasks()

        width = 80
        elapsed = status["elapsed_seconds"]
        elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"

        print("=" * width)
        print(" " * 20 + "USB-AI Build Status Dashboard")
        print("=" * width)
        print(f"Session: {status['session_id']:<35} Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Health: {status['health']:<37} Elapsed: {elapsed_str}")
        print("-" * width)
        print("")

        print(f"{'AGENTS':<25} {'STATUS':<12} {'TASK':<25} {'PROGRESS':<10}")
        print("-" * width)

        for agent in agents:
            status_str = agent.status.value
            task_str = agent.current_task or "-"
            if len(task_str) > 23:
                task_str = task_str[:20] + "..."
            progress_str = f"{agent.progress}%" if agent.current_task else "-"
            print(f"{agent.name:<25} {status_str:<12} {task_str:<25} {progress_str:<10}")

        print("")
        print("-" * width)
        print(f"{'PENDING TASKS':<50} {'PRIORITY':<12} {'TYPE':<15}")
        print("-" * width)

        for task in pending_tasks[:5]:
            print(f"{task.name:<50} {task.priority:<12} {task.task_type:<15}")

        if len(pending_tasks) > 5:
            print(f"... and {len(pending_tasks) - 5} more pending tasks")

        print("")
        print("-" * width)
        print("SUMMARY")
        print("-" * width)
        tasks_info = status["tasks"]
        agents_info = status["agents"]

        success_rate = 0
        if tasks_info["completed"] + tasks_info["failed"] > 0:
            success_rate = tasks_info["completed"] / (tasks_info["completed"] + tasks_info["failed"]) * 100

        print(f"Agents: {agents_info['active']} active, {agents_info['idle']} idle, {agents_info['error']} error")
        print(f"Tasks: {tasks_info['completed']}/{tasks_info['total']} complete, "
              f"{tasks_info['running']} running, {tasks_info['pending']} pending")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * width)

    def save_state(self, file_path: Path = None):
        """Save coordinator state to file."""
        if file_path is None:
            file_path = LOG_DIR / f"state_{self.session_id}.json"

        state = {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "agents": [a.to_dict() for a in self.registry.get_all()],
            "tasks": [t.to_dict() for t in self.task_queue.get_all_tasks()],
            "build_status": self.get_build_status()
        }

        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)

        log.info(f"State saved: {file_path}")

    def load_state(self, file_path: Path) -> bool:
        """Load coordinator state from file."""
        try:
            with open(file_path, "r") as f:
                state = json.load(f)

            self.session_id = state["session_id"]
            self.started_at = datetime.fromisoformat(state["started_at"])

            log.info(f"State loaded: {file_path}")
            return True
        except Exception as e:
            log.error(f"Failed to load state: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="USB-AI Agent Swarm Coordinator"
    )
    parser.add_argument(
        "--mode",
        choices=["status", "dashboard", "daemon"],
        default="status",
        help="Run mode"
    )
    parser.add_argument(
        "--session",
        help="Session ID for state persistence"
    )
    parser.add_argument(
        "--assign-task",
        help="Task type to assign"
    )
    parser.add_argument(
        "--agent",
        help="Target agent ID for task assignment"
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save coordinator state"
    )
    parser.add_argument(
        "--load-state",
        help="Load coordinator state from file"
    )

    args = parser.parse_args()

    coordinator = Coordinator(session_id=args.session)

    if args.load_state:
        coordinator.load_state(Path(args.load_state))

    if args.assign_task:
        task = Task(
            id=f"task-{uuid4().hex[:8]}",
            name=f"Manual: {args.assign_task}",
            task_type=args.assign_task,
            priority=5
        )
        coordinator.assign_task(task, args.agent)

    if args.mode == "dashboard":
        coordinator.print_dashboard()
    elif args.mode == "status":
        status = coordinator.get_build_status()
        print(json.dumps(status, indent=2))
    elif args.mode == "daemon":
        print(f"Coordinator running: {coordinator.session_id}")
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(5)
                coordinator.print_dashboard()
                print("\n" + "=" * 80 + "\n")
        except KeyboardInterrupt:
            print("\nShutting down...")

    if args.save_state:
        coordinator.save_state()

    return 0


if __name__ == "__main__":
    sys.exit(main())
