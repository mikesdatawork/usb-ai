# Agent Swarm Coordination System
## USB-AI Multi-Agent Build Orchestration

This document defines the agent swarm coordination infrastructure for the USB-AI build system.

---

## Agent Hierarchy

```
                     +---------------------------+
                     |        TEAM LEAD          |
                     |    (Build Orchestrator)   |
                     +---------------------------+
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
+---------v---------+   +---------v---------+   +---------v---------+
|   PLANNING TIER   |   |  EXECUTION TIER   |   |  VALIDATION TIER  |
+-------------------+   +-------------------+   +-------------------+
| - Proactive Agent |   | - Download Agent  |   | - Validation Agent|
| - Self-Improve    |   | - Flask Agent     |   | - Quality Gate    |
|                   |   | - Encryption Agent|   |                   |
|                   |   | - UI Agent        |   |                   |
+-------------------+   +-------------------+   +-------------------+
          |                       |                       |
          +----------+------------+------------+----------+
                     |                         |
          +----------v----------+   +----------v----------+
          |   SUPPORT TIER      |   |   MONITORING TIER   |
          +---------------------+   +---------------------+
          | - Browser Agent     |   | - Status Reporter   |
          | - Resource Fetcher  |   | - Metrics Collector |
          +---------------------+   +---------------------+
```

---

## Agent Registry

| Agent ID | Name | Role | Tier | Priority |
|----------|------|------|------|----------|
| `lead-001` | BuildOrchestrator | Team Lead | Command | Critical |
| `plan-001` | ProactiveBuilder | Anticipatory Prep | Planning | High |
| `plan-002` | Optimizer | Self-Improvement | Planning | Normal |
| `exec-001` | DownloadManager | File Downloads | Execution | Critical |
| `exec-002` | FlaskBuilder | WebUI Development | Execution | High |
| `exec-003` | CryptoManager | Encryption Ops | Execution | Critical |
| `exec-004` | InterfaceDesigner | UI/Theme | Execution | Normal |
| `valid-001` | QualityGate | Build Validation | Validation | Critical |
| `supp-001` | ResourceFetcher | Web Resources | Support | Normal |
| `mon-001` | StatusReporter | Status Updates | Monitoring | High |

---

## Communication Protocols

### Message Types

```yaml
message_types:
  COMMAND:
    description: "Direct instruction from Team Lead"
    direction: "lead -> agent"
    requires_ack: true
    priority: "immediate"

  STATUS:
    description: "Agent status update"
    direction: "agent -> lead"
    requires_ack: false
    priority: "normal"

  TASK_ASSIGN:
    description: "Task assignment to agent"
    direction: "lead -> agent"
    requires_ack: true
    priority: "high"

  TASK_COMPLETE:
    description: "Task completion notification"
    direction: "agent -> lead"
    requires_ack: true
    priority: "high"

  ERROR:
    description: "Error report"
    direction: "agent -> lead"
    requires_ack: true
    priority: "critical"

  HANDOFF:
    description: "Task handoff between agents"
    direction: "agent -> agent"
    requires_ack: true
    priority: "high"

  BROADCAST:
    description: "System-wide announcement"
    direction: "lead -> all"
    requires_ack: false
    priority: "normal"
```

### Message Format

```json
{
  "id": "msg-uuid-001",
  "timestamp": "2026-02-06T10:30:00Z",
  "type": "TASK_ASSIGN",
  "from": {
    "agent_id": "lead-001",
    "name": "BuildOrchestrator"
  },
  "to": {
    "agent_id": "exec-001",
    "name": "DownloadManager"
  },
  "payload": {
    "task_id": "task-001",
    "task_type": "download",
    "parameters": {
      "url": "https://ollama.com/download/...",
      "destination": "modules/ollama-portable/bin/",
      "checksum": "sha256:abc123..."
    },
    "timeout": 300,
    "priority": "high"
  },
  "metadata": {
    "phase": "phase-3",
    "retry_count": 0,
    "correlation_id": "build-session-001"
  }
}
```

### Communication Channels

| Channel | Purpose | Participants | Protocol |
|---------|---------|--------------|----------|
| `cmd` | Commands | Lead -> Agents | Sync |
| `status` | Status updates | Agents -> Lead | Async |
| `tasks` | Task queue | Lead <-> Agents | Queue |
| `errors` | Error reports | Agents -> Lead | Immediate |
| `handoff` | Agent-to-agent | Agents <-> Agents | Sync |
| `broadcast` | Announcements | Lead -> All | Pub/Sub |

---

## Task Assignment Workflow

### Task Lifecycle

```
+----------+    +------------+    +-----------+    +-----------+
| PENDING  | -> | ASSIGNED   | -> | RUNNING   | -> | COMPLETE  |
+----------+    +------------+    +-----------+    +-----------+
     |               |                 |                |
     |               v                 v                v
     |          +----------+     +----------+     +----------+
     +--------> | REJECTED |     | FAILED   |     | VERIFIED |
                +----------+     +----------+     +----------+
                                      |
                                      v
                                 +----------+
                                 | RETRY    |
                                 +----------+
```

### Assignment Algorithm

```python
def assign_task(task: Task) -> Agent:
    """
    Task assignment algorithm:
    1. Filter agents by capability
    2. Filter by availability
    3. Sort by priority and load
    4. Select best match
    """

    # Step 1: Filter by capability
    capable_agents = [
        agent for agent in agents
        if task.type in agent.capabilities
    ]

    # Step 2: Filter by availability
    available_agents = [
        agent for agent in capable_agents
        if agent.status == "IDLE" or agent.can_accept_task()
    ]

    # Step 3: Sort by criteria
    sorted_agents = sorted(
        available_agents,
        key=lambda a: (
            -a.priority_for_task(task),  # Higher priority first
            a.current_load,               # Lower load first
            a.last_task_time              # Least recently used
        )
    )

    # Step 4: Select
    if sorted_agents:
        return sorted_agents[0]

    # No available agent - queue task
    task_queue.add(task, priority=task.priority)
    return None
```

### Task Priority Matrix

| Task Type | Base Priority | Deadline Modifier | Dependency Modifier |
|-----------|--------------|-------------------|---------------------|
| download | 5 | +2 if <1hr | +1 if blocking |
| build | 4 | +2 if <30min | +2 if blocking |
| validation | 6 | +1 if <15min | +3 if blocking |
| documentation | 2 | +0 | +0 |
| encryption | 7 | +3 if <5min | +3 if blocking |

---

## Status Reporting Format

### Agent Status Report

```yaml
agent_status:
  agent_id: "exec-001"
  name: "DownloadManager"

  state:
    status: "RUNNING"  # IDLE | RUNNING | PAUSED | ERROR | STOPPED
    current_task: "task-001"
    progress: 75

  health:
    last_heartbeat: "2026-02-06T10:30:00Z"
    uptime_seconds: 3600
    error_count: 0

  resources:
    memory_mb: 256
    cpu_percent: 15
    active_connections: 3

  queue:
    pending_tasks: 2
    completed_tasks: 15
    failed_tasks: 1

  metrics:
    avg_task_duration: 45.2
    success_rate: 0.94
    throughput: 12.5  # tasks/hour
```

### Build Status Report

```yaml
build_status:
  session_id: "build-session-001"
  started: "2026-02-06T08:00:00Z"

  phase:
    current: "phase-4"
    name: "Model Downloads"
    progress: 60

  agents:
    active: 5
    idle: 2
    error: 0

  tasks:
    total: 45
    completed: 28
    running: 3
    pending: 14
    failed: 0

  timeline:
    elapsed_minutes: 150
    estimated_remaining: 30

  health:
    status: "HEALTHY"  # HEALTHY | DEGRADED | CRITICAL
    issues: []
```

### Dashboard Output Format

```
================================================================================
                         USB-AI Build Status Dashboard
================================================================================
Session: build-session-001                        Time: 2026-02-06 10:30:00
Phase: 4/8 - Model Downloads                      Progress: [########--] 60%
--------------------------------------------------------------------------------

AGENTS                          STATUS          TASK                    PROGRESS
--------------------------------------------------------------------------------
BuildOrchestrator               RUNNING         Coordinating            -
ProactiveBuilder                RUNNING         Pre-fetching qwen2.5    45%
DownloadManager                 RUNNING         dolphin-llama3:8b       78%
FlaskBuilder                    IDLE            -                       -
CryptoManager                   IDLE            -                       -
QualityGate                     IDLE            -                       -
ResourceFetcher                 RUNNING         Fetching checksums      100%

--------------------------------------------------------------------------------
TASK QUEUE                                       PRIORITY        ETA
--------------------------------------------------------------------------------
download:llama3.2:8b                            HIGH            5 min
download:qwen2.5:14b                            NORMAL          20 min
validate:all_models                             HIGH            Waiting

--------------------------------------------------------------------------------
METRICS
--------------------------------------------------------------------------------
Elapsed: 2h 30m          Remaining: ~30m          Success Rate: 100%
Tasks Complete: 28/45    Downloads: 1.2 GB/s      Errors: 0

================================================================================
```

---

## Conflict Resolution Procedures

### Conflict Types

| Conflict | Description | Resolution |
|----------|-------------|------------|
| RESOURCE | Multiple agents need same resource | Priority queue |
| DEPENDENCY | Task blocked by incomplete dependency | Wait/escalate |
| TIMING | Deadline conflicts | Reschedule lower priority |
| DATA | Conflicting state modifications | Last-write-wins with version |
| AGENT | Agent failure during task | Reassign to backup |

### Resolution Workflow

```
+------------------+
| Conflict Detected|
+------------------+
         |
         v
+------------------+
| Classify Type    |
+------------------+
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
 RSRC  DEP  TIME DATA AGENT
    |    |    |    |    |
    v    v    v    v    v
+------+ +------+ +------+ +------+ +------+
|Queue | |Wait  | |Resch | |Merge | |Reassn|
|by Pri| |/Escal| |Lower | |w/Ver | |Task  |
+------+ +------+ +------+ +------+ +------+
         |
         v
+------------------+
| Log Resolution   |
+------------------+
         |
         v
+------------------+
| Resume Execution |
+------------------+
```

### Resource Conflict Resolution

```python
def resolve_resource_conflict(requests: List[ResourceRequest]) -> Resolution:
    """
    Resolve resource conflicts by priority and timestamp.
    """
    # Sort by priority (higher first), then timestamp (earlier first)
    sorted_requests = sorted(
        requests,
        key=lambda r: (-r.priority, r.timestamp)
    )

    winner = sorted_requests[0]
    losers = sorted_requests[1:]

    resolution = Resolution(
        winner=winner,
        action="grant",
        losers=[
            LoserAction(
                request=r,
                action="queue" if r.can_wait else "reject",
                retry_after=calculate_retry_time(winner)
            )
            for r in losers
        ]
    )

    log_resolution(resolution)
    return resolution
```

### Dependency Conflict Resolution

```python
def resolve_dependency_conflict(task: Task, dependency: Task) -> Resolution:
    """
    Handle blocked task waiting on dependency.
    """
    if dependency.status == "RUNNING":
        # Wait for completion
        return Resolution(
            action="wait",
            timeout=dependency.estimated_completion,
            on_timeout="escalate"
        )

    elif dependency.status == "FAILED":
        # Check if retryable
        if dependency.retry_count < MAX_RETRIES:
            return Resolution(
                action="retry_dependency",
                target=dependency
            )
        else:
            return Resolution(
                action="escalate",
                reason="dependency_failed",
                suggested="manual_intervention"
            )

    elif dependency.status == "PENDING":
        # Increase priority
        return Resolution(
            action="boost_priority",
            target=dependency,
            new_priority=task.priority + 1
        )
```

### Agent Failure Recovery

```python
def handle_agent_failure(agent: Agent, task: Task) -> Resolution:
    """
    Handle agent failure during task execution.
    """
    # Log failure
    log_agent_failure(agent, task)

    # Find backup agent
    backup_agents = find_capable_agents(task.type)
    backup_agents = [a for a in backup_agents if a.id != agent.id]

    if not backup_agents:
        return Resolution(
            action="escalate",
            reason="no_backup_agent",
            task_state="suspended"
        )

    # Select backup
    backup = select_best_agent(backup_agents, task)

    # Transfer task state
    task_state = agent.get_task_state(task)

    return Resolution(
        action="reassign",
        from_agent=agent,
        to_agent=backup,
        task_state=task_state,
        resume_from="last_checkpoint"
    )
```

---

## Agent Coordination API

### Team Lead API

```python
class TeamLead:
    """Team Lead coordination interface."""

    def register_agent(self, agent: Agent) -> str:
        """Register a new agent. Returns agent ID."""
        pass

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        pass

    def assign_task(self, task: Task, agent_id: str = None) -> str:
        """Assign task. Auto-assigns if agent_id is None."""
        pass

    def get_agent_status(self, agent_id: str) -> AgentStatus:
        """Get agent status."""
        pass

    def get_all_status(self) -> List[AgentStatus]:
        """Get all agent statuses."""
        pass

    def broadcast(self, message: str, priority: str = "normal") -> None:
        """Broadcast message to all agents."""
        pass

    def pause_agent(self, agent_id: str) -> bool:
        """Pause agent execution."""
        pass

    def resume_agent(self, agent_id: str) -> bool:
        """Resume agent execution."""
        pass

    def get_build_status(self) -> BuildStatus:
        """Get overall build status."""
        pass
```

### Agent API

```python
class Agent:
    """Base agent interface."""

    def accept_task(self, task: Task) -> bool:
        """Accept or reject task assignment."""
        pass

    def execute_task(self, task: Task) -> TaskResult:
        """Execute assigned task."""
        pass

    def report_status(self) -> AgentStatus:
        """Report current status."""
        pass

    def report_progress(self, task_id: str, progress: int) -> None:
        """Report task progress."""
        pass

    def report_error(self, error: Error) -> None:
        """Report error to Team Lead."""
        pass

    def request_handoff(self, task: Task, target_agent: str) -> bool:
        """Request task handoff to another agent."""
        pass

    def checkpoint(self, task: Task, state: dict) -> None:
        """Save task checkpoint for recovery."""
        pass
```

---

## Event Handling

### Event Types

```yaml
events:
  AGENT_REGISTERED:
    trigger: "New agent joins swarm"
    handler: "update_agent_registry"

  AGENT_FAILED:
    trigger: "Agent becomes unresponsive"
    handler: "initiate_failover"

  TASK_COMPLETED:
    trigger: "Task finishes successfully"
    handler: "update_progress, assign_next"

  TASK_FAILED:
    trigger: "Task fails after retries"
    handler: "log_error, escalate"

  PHASE_COMPLETE:
    trigger: "All phase tasks done"
    handler: "validate_phase, start_next"

  BUILD_COMPLETE:
    trigger: "All phases done"
    handler: "generate_report, cleanup"

  RESOURCE_EXHAUSTED:
    trigger: "Disk/memory critical"
    handler: "pause_downloads, alert"
```

### Event Handler Registration

```python
coordinator.on("AGENT_REGISTERED", handle_agent_registered)
coordinator.on("TASK_COMPLETED", handle_task_completed)
coordinator.on("TASK_FAILED", handle_task_failed)
coordinator.on("PHASE_COMPLETE", handle_phase_complete)
```

---

## Logging and Auditing

### Log Levels

| Level | Purpose | Retention |
|-------|---------|-----------|
| DEBUG | Detailed execution traces | 1 day |
| INFO | Normal operations | 7 days |
| WARN | Potential issues | 30 days |
| ERROR | Failures requiring attention | 90 days |
| AUDIT | Coordination decisions | Permanent |

### Audit Log Format

```json
{
  "timestamp": "2026-02-06T10:30:00Z",
  "event_type": "TASK_ASSIGN",
  "actor": "lead-001",
  "action": "assign_task",
  "target": "exec-001",
  "details": {
    "task_id": "task-001",
    "task_type": "download",
    "reason": "best_available_agent"
  },
  "outcome": "success"
}
```

---

## Quick Reference

### Start Coordinator

```bash
python scripts/agents/coordinator.py --mode daemon
```

### Check Status

```bash
python scripts/agents/coordinator.py --status
```

### Assign Task

```bash
python scripts/agents/coordinator.py --assign-task download --agent exec-001
```

### View Dashboard

```bash
python scripts/agents/coordinator.py --dashboard
```

---

**See AGENTS.md for individual agent configurations and SKILLS_TASKS_MCP.md for task execution details.**
