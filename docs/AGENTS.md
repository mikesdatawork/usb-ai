# Agents Configuration
## USB-AI Build System Agents

This document defines all agents used in the USB-AI build orchestration system.

---

## Agent Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Build Orchestrator                        │
│                      (Primary Agent)                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Proactive   │  │    Self      │  │   Browser    │      │
│  │    Agent     │  │ Improvement  │  │    Agent     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Validation  │  │ Encryption   │  │   Download   │      │
│  │    Agent     │  │   Agent      │  │    Agent     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Build Orchestrator Agent (Primary)

### Purpose
Central coordinator for all build operations. Manages state, delegates tasks, handles errors.

### Configuration

```yaml
agent:
  name: "BuildOrchestrator"
  type: "primary"
  version: "1.0.0"
  
  identity:
    role: "Build System Coordinator"
    expertise: 
      - "Project management"
      - "Task delegation"
      - "Error handling"
      - "State management"
    
  state_machine:
    initial: "IDLE"
    states:
      IDLE:
        transitions:
          - trigger: "start_build"
            target: "INITIALIZING"
      INITIALIZING:
        on_enter: "run_initialization_checks"
        transitions:
          - trigger: "init_complete"
            target: "DOWNLOADING"
          - trigger: "init_failed"
            target: "ERROR"
      DOWNLOADING:
        on_enter: "start_downloads"
        transitions:
          - trigger: "downloads_complete"
            target: "BUILDING"
          - trigger: "download_failed"
            target: "ERROR"
      BUILDING:
        on_enter: "start_build_process"
        transitions:
          - trigger: "build_complete"
            target: "TESTING"
          - trigger: "build_failed"
            target: "ERROR"
      TESTING:
        on_enter: "run_validation"
        transitions:
          - trigger: "tests_passed"
            target: "PACKAGING"
          - trigger: "tests_failed"
            target: "ERROR"
      PACKAGING:
        on_enter: "create_release"
        transitions:
          - trigger: "packaging_complete"
            target: "COMPLETE"
      COMPLETE:
        on_enter: "generate_report"
      ERROR:
        on_enter: "handle_error"
        transitions:
          - trigger: "retry"
            target: "IDLE"
          - trigger: "abort"
            target: "ABORTED"
            
  behaviors:
    - coordinate_sub_agents
    - track_progress
    - handle_interrupts
    - manage_resources
    
  communication:
    protocol: "event_driven"
    channels:
      - "agent_commands"
      - "status_updates"
      - "error_reports"
```

### Interaction Commands

```python
# Start build
orchestrator.trigger("start_build", config={
    "usb_size": "128GB",
    "models": ["dolphin-llama3", "llama3.2", "qwen2.5"],
    "target_os": ["macos", "windows", "linux"]
})

# Check status
status = orchestrator.get_state()

# Handle interrupt
orchestrator.trigger("pause")
orchestrator.trigger("resume")
```

---

## 2. Proactive Agent

### Purpose
Anticipates needs, prepares resources before they're requested, optimizes workflow.

### Configuration

```yaml
agent:
  name: "ProactiveBuilder"
  type: "sub_agent"
  version: "1.0.0"
  
  identity:
    role: "Anticipatory Resource Manager"
    expertise:
      - "Predictive preparation"
      - "Resource optimization"
      - "Bottleneck prevention"
      
  behaviors:
    pre_fetch:
      description: "Download resources before explicitly needed"
      triggers:
        - "phase_about_to_start"
        - "idle_detected"
      actions:
        - check_next_phase_requirements
        - start_background_downloads
        - warm_up_caches
        
    space_monitor:
      description: "Monitor disk space proactively"
      interval: "30s"
      thresholds:
        warn: "10GB"
        critical: "5GB"
      actions:
        - alert_user
        - suggest_cleanup
        - pause_downloads
        
    dependency_check:
      description: "Verify dependencies before phases"
      timing: "before_each_phase"
      actions:
        - scan_requirements
        - install_missing
        - verify_versions
        
    network_monitor:
      description: "Monitor network for download optimization"
      triggers:
        - "download_started"
      actions:
        - measure_bandwidth
        - prioritize_downloads
        - retry_failed_downloads
        
  rules:
    - name: "pre_download_models"
      condition: "phase == 'BUILDING' AND models_not_downloaded"
      action: "start_model_download_in_background"
      
    - name: "prepare_encryption"
      condition: "phase == 'DOWNLOADING' AND veracrypt_available"
      action: "verify_encryption_tools"
      
    - name: "optimize_sequence"
      condition: "large_download_queued"
      action: "reorder_for_parallel_execution"
      
  outputs:
    log_file: "logs/proactive_agent.log"
    metrics:
      - downloads_prepped
      - time_saved
      - errors_prevented
```

### Proactive Actions Matrix

| Trigger | Condition | Action | Benefit |
|---------|-----------|--------|---------|
| Phase start | Next phase has downloads | Pre-fetch resources | Reduce wait time |
| Idle 30s | Pending tasks exist | Execute low-priority tasks | Efficient use of time |
| Download starts | Bandwidth available | Parallel download | Faster completion |
| Error detected | Known fix exists | Auto-remediate | Reduce manual intervention |
| Space low | < 10GB free | Alert + suggest cleanup | Prevent failures |

---

## 3. Self-Improvement Agent

### Purpose
Analyzes build performance, identifies optimizations, suggests and implements improvements.

### Configuration

```yaml
agent:
  name: "Optimizer"
  type: "sub_agent"
  version: "1.0.0"
  
  identity:
    role: "Continuous Improvement Manager"
    expertise:
      - "Performance analysis"
      - "Pattern recognition"
      - "Process optimization"
      
  metrics_collection:
    timing:
      - phase_duration
      - task_duration
      - download_speeds
      - error_frequency
    resources:
      - memory_usage
      - disk_io
      - network_bandwidth
    quality:
      - error_count
      - retry_count
      - success_rate
      
  analysis:
    frequency: "after_each_phase"
    methods:
      - statistical_analysis
      - trend_detection
      - anomaly_identification
      
  improvement_types:
    parallelization:
      description: "Identify tasks that can run concurrently"
      detection: "sequential_tasks_no_dependency"
      action: "suggest_parallel_execution"
      
    caching:
      description: "Identify repeated operations"
      detection: "same_operation_multiple_times"
      action: "implement_caching"
      
    error_prevention:
      description: "Learn from errors to prevent recurrence"
      detection: "error_pattern_identified"
      action: "add_pre_check"
      
    resource_optimization:
      description: "Optimize resource usage"
      detection: "resource_underutilized"
      action: "adjust_allocation"
      
  learning:
    method: "rule_extraction"
    storage: "improvement_rules.yaml"
    retention: "permanent"
    
  outputs:
    improvement_log: "logs/improvements.md"
    metrics_report: "reports/performance_metrics.json"
    suggestions: "reports/optimization_suggestions.md"
```

### Improvement Log Format

```markdown
## Improvement Log - USB-AI Build

### Session: 2026-02-03

#### Observation
Phase 4 (Model Downloads) took 45 minutes. Network was underutilized.

#### Analysis
- Dolphin-LLaMA3 downloaded sequentially
- Llama 3.2 waited for Dolphin to complete
- Bandwidth usage: 30% of available

#### Improvement Implemented
Changed to parallel downloads with 3 concurrent connections.

#### Result
Phase 4 duration reduced to 20 minutes (55% improvement).

---

### Metrics Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total build time | 120 min | 85 min | -29% |
| Download phase | 45 min | 20 min | -55% |
| Error count | 3 | 1 | -67% |
```

---

## 4. Browser Agent

### Purpose
Fetches external resources, documentation, and verifies online information.

### Configuration

```yaml
agent:
  name: "ResourceFetcher"
  type: "sub_agent"
  version: "1.0.0"
  
  identity:
    role: "External Resource Manager"
    expertise:
      - "Web scraping"
      - "API interaction"
      - "Documentation retrieval"
      
  capabilities:
    web_fetch:
      allowed_domains:
        - "ollama.com"
        - "github.com"
        - "huggingface.co"
        - "veracrypt.io"
        - "launchpad.net"
      blocked_domains:
        - "*.onion"
        - "*.ru"
      timeout: "30s"
      retries: 3
      
    api_calls:
      - name: "ollama_api"
        base_url: "https://ollama.com/api"
        endpoints:
          - "/tags"
          - "/library"
      - name: "github_api"
        base_url: "https://api.github.com"
        endpoints:
          - "/repos/*/releases"
          
  tasks:
    fetch_latest_releases:
      description: "Get latest release versions"
      targets:
        - ollama
        - open_webui
        - veracrypt
      frequency: "on_demand"
      
    verify_checksums:
      description: "Fetch and verify file checksums"
      sources:
        - github_releases
        - official_sites
        
    update_documentation:
      description: "Fetch latest documentation"
      sources:
        - ollama_docs
        - veracrypt_docs
        
  rate_limiting:
    requests_per_minute: 10
    respect_robots_txt: true
    user_agent: "USB-AI-Builder/1.0"
    
  caching:
    enabled: true
    ttl: "1h"
    storage: "cache/web_cache.db"
    
  outputs:
    downloaded_files: "downloads/"
    checksums: "downloads/checksums.json"
    version_info: "config/latest_versions.json"
```

### Resource Fetch Matrix

| Resource | URL | Purpose | Frequency |
|----------|-----|---------|-----------|
| Ollama releases | ollama.com/download | Get latest binary | Per build |
| Flask | pypi.org/flask | Get latest release | Per build |
| VeraCrypt | veracrypt.io | Get latest installer | Per build |
| Model info | ollama.com/library | Verify model specs | On demand |

---

## 5. Validation Agent

### Purpose
Verifies build outputs, runs tests, ensures quality gates are met.

### Configuration

```yaml
agent:
  name: "QualityGate"
  type: "sub_agent"
  version: "1.0.0"
  
  identity:
    role: "Quality Assurance Manager"
    expertise:
      - "Testing"
      - "Verification"
      - "Compliance checking"
      
  validation_suites:
    file_integrity:
      description: "Verify all required files exist"
      checks:
        - file_exists
        - file_size_minimum
        - file_permissions
      targets:
        - encrypted.vc: "size > 1GB"
        - launchers/*: "executable"
        - README.txt: "exists"
        
    model_validation:
      description: "Verify AI models are functional"
      checks:
        - model_loads
        - model_responds
        - model_format_valid
      targets:
        - dolphin-llama3
        - llama3.2
        - qwen2.5
        
    encryption_validation:
      description: "Verify encryption is working"
      checks:
        - container_mounts
        - password_required
        - encryption_algorithm
      expected:
        algorithm: "AES-256"
        
    cross_platform:
      description: "Verify scripts work on all platforms"
      checks:
        - syntax_valid
        - paths_portable
        - no_hardcoded_paths
      targets:
        - "*.sh"
        - "*.bat"
        - "*.command"
        
  test_execution:
    mode: "sequential"
    stop_on_failure: false
    generate_report: true
    
  reporting:
    format: "markdown"
    output: "reports/validation_report.md"
    include:
      - test_results
      - error_details
      - recommendations
      
  quality_gates:
    must_pass:
      - file_integrity
      - encryption_validation
    should_pass:
      - model_validation
      - cross_platform
```

### Validation Test Cases

```yaml
test_cases:
  - id: "T001"
    name: "Encrypted container exists"
    type: "file_check"
    target: "encrypted.vc"
    assertion: "exists AND size > 50GB"
    
  - id: "T002"
    name: "Container requires password"
    type: "encryption_check"
    target: "encrypted.vc"
    assertion: "mount_fails_without_password"
    
  - id: "T003"
    name: "Ollama binary executable"
    type: "executable_check"
    target: "ollama/linux/ollama"
    assertion: "is_executable"
    
  - id: "T004"
    name: "Model responds to prompt"
    type: "functional_test"
    target: "dolphin-llama3"
    input: "Say 'hello'"
    assertion: "response_contains_text"
    
  - id: "T005"
    name: "Flask chat UI starts successfully"
    type: "service_check"
    target: "localhost:3000"
    assertion: "http_200_within_10s"
```

---

## 6. Encryption Agent

### Purpose
Handles all encryption-related operations securely.

### Configuration

```yaml
agent:
  name: "CryptoManager"
  type: "sub_agent"
  version: "1.0.0"
  
  identity:
    role: "Encryption Specialist"
    expertise:
      - "VeraCrypt operations"
      - "Key management"
      - "Secure storage"
      
  operations:
    create_container:
      parameters:
        - size: "required"
        - encryption: "AES-256"
        - hash: "SHA-512"
        - filesystem: "exFAT"
      security:
        - never_log_password
        - clear_memory_after_use
        
    mount_container:
      parameters:
        - container_path: "required"
        - mount_point: "required"
      security:
        - password_prompt_only
        - no_password_storage
        
    unmount_container:
      parameters:
        - mount_point: "required"
      security:
        - verify_no_open_files
        - force_option_available
        
  security_policies:
    password_handling:
      - never_store
      - never_log
      - prompt_securely
      - clear_from_memory
      
    container_handling:
      - verify_integrity_before_mount
      - check_corruption_after_unmount
      
  logging:
    enabled: true
    exclude:
      - passwords
      - key_material
    include:
      - operations_performed
      - success_status
      - error_codes
```

---

## 7. Download Agent

### Purpose
Manages all file downloads efficiently and reliably.

### Configuration

```yaml
agent:
  name: "DownloadManager"
  type: "sub_agent"
  version: "1.0.0"
  
  identity:
    role: "Download Specialist"
    expertise:
      - "Efficient downloading"
      - "Resume capability"
      - "Checksum verification"
      
  download_config:
    concurrent_downloads: 3
    chunk_size: "8MB"
    retry_attempts: 5
    retry_delay: "5s"
    timeout: "120s"
    
  features:
    resume_support:
      enabled: true
      method: "range_headers"
      
    progress_reporting:
      enabled: true
      interval: "1s"
      format: "percentage_and_speed"
      
    checksum_verification:
      enabled: true
      algorithms:
        - sha256
        - md5
      auto_redownload_on_mismatch: true
      
  queue_management:
    priority_levels:
      - critical: "required for build"
      - high: "improves quality"
      - normal: "optional"
    ordering: "priority_then_fifo"
    
  bandwidth_management:
    limit: "none"
    throttle_during:
      - other_critical_operations
    
  outputs:
    download_dir: "downloads/"
    progress_log: "logs/download_progress.log"
    checksum_file: "downloads/checksums.sha256"
```

---

## Agent Communication Protocol

### Event Types

```yaml
events:
  - type: "TASK_ASSIGNED"
    from: "orchestrator"
    to: "sub_agent"
    payload:
      task_id: "string"
      task_type: "string"
      parameters: "object"
      
  - type: "TASK_COMPLETE"
    from: "sub_agent"
    to: "orchestrator"
    payload:
      task_id: "string"
      status: "success|failure"
      result: "object"
      
  - type: "STATUS_UPDATE"
    from: "any_agent"
    to: "orchestrator"
    payload:
      agent_name: "string"
      status: "string"
      progress: "number"
      
  - type: "ERROR_REPORT"
    from: "any_agent"
    to: "orchestrator"
    payload:
      error_type: "string"
      message: "string"
      recoverable: "boolean"
      suggested_action: "string"
```

### Message Format

```json
{
  "id": "msg_001",
  "timestamp": "2026-02-03T10:30:00Z",
  "from": "ProactiveBuilder",
  "to": "BuildOrchestrator",
  "type": "STATUS_UPDATE",
  "payload": {
    "agent_name": "ProactiveBuilder",
    "status": "active",
    "progress": 75,
    "message": "Pre-fetching model files"
  }
}
```

---

## Agent Instantiation

### Python Implementation

```python
# agents/base.py
class BaseAgent:
    def __init__(self, config):
        self.name = config['name']
        self.role = config['identity']['role']
        self.state = 'IDLE'
        
    def start(self):
        self.state = 'RUNNING'
        
    def stop(self):
        self.state = 'STOPPED'
        
    def handle_event(self, event):
        raise NotImplementedError

# agents/orchestrator.py
class BuildOrchestrator(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.sub_agents = {}
        
    def register_agent(self, agent):
        self.sub_agents[agent.name] = agent
        
    def delegate_task(self, task, agent_name):
        agent = self.sub_agents[agent_name]
        return agent.execute(task)

# Usage
orchestrator = BuildOrchestrator(load_config('orchestrator.yaml'))
proactive = ProactiveAgent(load_config('proactive.yaml'))
orchestrator.register_agent(proactive)
orchestrator.start()
```

---

## Agent Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  CREATED │ ──▶ │ STARTING │ ──▶ │  RUNNING │
└──────────┘     └──────────┘     └──────────┘
                                       │
                 ┌──────────┐          │
                 │  PAUSED  │ ◀────────┤
                 └──────────┘          │
                      │                │
                      ▼                ▼
                 ┌──────────┐     ┌──────────┐
                 │ STOPPING │ ◀── │  ERROR   │
                 └──────────┘     └──────────┘
                      │
                      ▼
                 ┌──────────┐
                 │ STOPPED  │
                 └──────────┘
```

---

**Next: See PLAN_MODE.md for detailed plan mode instructions.**
