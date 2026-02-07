# CLAUDE.md
## Claude Max Build Orchestration for USB-AI

This document provides Claude Max with all necessary context, instructions, and configurations to build the USB-AI system autonomously.

---

## Project Context

**Project**: USB-AI - Portable Offline Encrypted AI System  
**Repository**: https://github.com/mikesdatawork/usb-ai  
**Target**: 128GB USB drive (expandable to 256GB)  
**Primary Model**: Dolphin-LLaMA3 with additional options  

---

## Memory Persistence Configuration

### Project Memory (Critical)

Enable Project Memory to persist across sessions:

```yaml
project_memory:
  enabled: true
  scope: "usb-ai-build"
  persist_on_compact: true
  key_facts:
    - "Target USB: 128GB minimum"
    - "Primary model: Dolphin-LLaMA3 8B"
    - "Encryption: VeraCrypt AES-256"
    - "UI: Flask + HTMX"
    - "Repo: github.com/mikesdatawork/usb-ai"
```

### Memory Items to Persist

| Key | Value | Priority |
|-----|-------|----------|
| `usb_target_size` | 128GB | Critical |
| `primary_model` | dolphin-llama3 | Critical |
| `encryption_method` | veracrypt | Critical |
| `ui_choice` | flask-htmx | Critical |
| `github_repo` | mikesdatawork/usb-ai | Critical |
| `build_status` | in_progress | Dynamic |
| `current_phase` | varies | Dynamic |

---

## Git Worktree Configuration

### Initialize Worktrees

```bash
# From repository root
cd usb-ai

# Create build worktree
git branch build 2>/dev/null || true
git worktree add ../usb-ai-build build

# Create release worktree  
git branch release 2>/dev/null || true
git worktree add ../usb-ai-release release

# Verify worktrees
git worktree list
```

### Worktree Purposes

| Worktree | Branch | Purpose |
|----------|--------|---------|
| `usb-ai/` | main | Documentation, source scripts |
| `usb-ai-build/` | build | Compilation, downloaded artifacts |
| `usb-ai-release/` | release | Final packaged USB images |

---

## Agent Configurations

### Multi-Agent Swarm Architecture

```
                    ┌─────────────────┐
                    │   TEAM LEAD     │
                    │  (Coordinator)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ LLM Engineer  │  │  Performance  │  │    Build      │
│    Agent      │  │    Agent      │  │    Agent      │
└───────────────┘  └───────────────┘  └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Quantization  │  │   WebUI       │  │  Validation   │
│  Specialist   │  │  Optimizer    │  │    Agent      │
└───────────────┘  └───────────────┘  └───────────────┘
```

### 0. Team Lead Agent (Coordinator)

```yaml
agent_team_lead:
  name: "TeamLead"
  role: "Coordinate all sub-agents and orchestrate parallel workloads"
  responsibilities:
    - spawn_and_manage_sub_agents
    - distribute_tasks_across_agents
    - aggregate_results
    - resolve_conflicts
    - report_overall_progress
  coordination:
    max_parallel_agents: 5
    communication: "async_message_queue"
    status_file: "docs/AGENT_SWARM.md"
  scripts:
    - scripts/agents/coordinator.py
```

### 1. LLM Engineer Agent

```yaml
agent_llm_engineer:
  name: "LLMEngineer"
  role: "Optimize and configure LLM models for portable operation"
  responsibilities:
    - model_selection_optimization
    - quantization_strategy
    - context_length_tuning
    - inference_optimization
  sub_agents:
    - QuantizationSpecialist
  outputs:
    - scripts/quantization/quantize_models.py
    - modules/models/quantization_config.yaml
```

### 2. Quantization Specialist Agent

```yaml
agent_quantization:
  name: "QuantizationSpecialist"
  role: "Handle model quantization for optimal size/quality tradeoff"
  capabilities:
    - gguf_quantization: [Q4_K_M, Q5_K_M, Q6_K, Q8_0]
    - calculate_storage_requirements
    - memory_estimation
    - auto_select_optimal_quantization
  usb_profiles:
    128GB:
      max_model_size: "8B"
      preferred_quant: "Q4_K_M"
    256GB:
      max_model_size: "14B"
      preferred_quant: "Q5_K_M"
  scripts:
    - scripts/quantization/quantize_models.py
    - scripts/quantization/benchmark.py
```

### 3. Performance Optimization Agent

```yaml
agent_performance:
  name: "PerformanceOptimizer"
  role: "Optimize system performance for USB operation"
  responsibilities:
    - ollama_configuration
    - thread_optimization
    - memory_management
    - gpu_offloading
    - startup_optimization
  profiles:
    minimal:
      threads: 2
      memory_limit: "4GB"
      batch_size: 128
    balanced:
      threads: 4
      memory_limit: "8GB"
      batch_size: 256
    performance:
      threads: 8
      memory_limit: "16GB"
      batch_size: 512
    max:
      threads: "auto"
      memory_limit: "auto"
      batch_size: 1024
  scripts:
    - scripts/performance/optimize.py
    - scripts/performance/system_check.py
```

### 4. Build System Agent

```yaml
agent_build:
  name: "BuildMaster"
  role: "Orchestrate parallel build operations"
  responsibilities:
    - parallel_task_execution
    - dependency_management
    - progress_reporting
    - failure_recovery
    - resumable_builds
  state_machine:
    states: [IDLE, INITIALIZING, BUILDING, TESTING, PACKAGING, COMPLETE, ERROR]
  scripts:
    - scripts/build/parallel_builder.py
    - scripts/build/build_manifest.yaml
    - scripts/build/validate_build.py
```

### 5. WebUI Optimizer Agent

```yaml
agent_webui:
  name: "WebUIOptimizer"
  role: "Optimize Flask + HTMX chat interface performance"
  responsibilities:
    - response_streaming_optimization
    - connection_pooling
    - static_asset_caching
    - gzip_compression
    - htmx_polling_optimization
  outputs:
    - modules/webui-portable/optimizations.py
    - modules/webui-portable/config/performance.yaml
    - docs/WEBUI_PERFORMANCE.md
```

### 6. Validation Agent

```yaml
agent_validation:
  name: "QualityGate"
  role: "Verify build outputs and ensure quality"
  checks:
    - file_integrity
    - model_checksums
    - encryption_verification
    - cross_platform_scripts
    - performance_benchmarks
  report_format: "markdown"
  outputs:
    - docs/VALIDATION_REPORT.md
```

### 7. Proactive Agent

```yaml
agent_proactive:
  name: "ProactiveBuilder"
  role: "Anticipate needs and prepare resources"
  behaviors:
    - pre_download_models_during_idle
    - check_disk_space_before_operations
    - validate_dependencies_early
    - suggest_optimizations
  triggers:
    - on_phase_complete
    - on_idle_30s
    - on_error
```

### 8. Browser Agent

```yaml
agent_browser:
  name: "ResourceFetcher"
  role: "Fetch external resources and documentation"
  capabilities:
    - fetch_latest_ollama_releases
    - verify_model_checksums
    - check_veracrypt_updates
    - download_documentation
  rate_limit: "10 requests/minute"
```

### Agent Communication Protocol

```yaml
agent_communication:
  protocol: "async_message_queue"
  message_format:
    type: "[TASK|STATUS|RESULT|ERROR]"
    from_agent: "agent_name"
    to_agent: "agent_name|broadcast"
    payload: {}
    timestamp: "ISO8601"
  status_updates:
    frequency: "on_milestone"
    aggregate_to: "docs/AGENT_STATUS.md"
```

### Parallel Execution Rules

```yaml
parallel_execution:
  enabled: true
  max_concurrent: 5
  task_distribution:
    independent_tasks: "parallel"
    dependent_tasks: "sequential"
  load_balancing: "round_robin"
  failure_handling:
    strategy: "isolate_and_continue"
    max_retries: 3
    fallback: "sequential_execution"
```

---

## MCP Configurations

### PlayWriter MCP (Token Optimization)

```yaml
mcp_playwriter:
  enabled: true
  mode: "token_efficient"
  strategies:
    - compress_repeated_content
    - cache_static_references
    - batch_similar_operations
    - minimize_verbose_output
  token_budget:
    per_phase: 50000
    total_build: 500000
  alerts:
    warn_at: 80%
    pause_at: 95%
```

### File System MCP

```yaml
mcp_filesystem:
  enabled: true
  root_paths:
    - "/home/claude/usb-ai-docs"
    - "/mnt/user-data/outputs"
  allowed_operations:
    - read
    - write
    - create
    - delete
  excluded_patterns:
    - "*.vc"  # Don't manipulate encrypted containers directly
    - ".git/*"
```

### GitHub MCP

```yaml
mcp_github:
  enabled: true
  repository: "mikesdatawork/usb-ai"
  operations:
    - push
    - pull
    - branch
    - worktree
  auto_commit:
    enabled: true
    message_prefix: "[claude-max-build]"
```

---

## Skills Configuration

### GSD (Get Stuff Done) Methodology

```yaml
skill_gsd:
  enabled: true
  approach:
    - define_clear_outcome
    - break_into_tasks
    - execute_sequentially
    - validate_each_step
    - iterate_on_failure
  task_format:
    status: "[TODO|DOING|DONE|BLOCKED]"
    owner: "agent_name"
    deadline: "phase_end"
```

### Claude Tasks

```yaml
skill_tasks:
  enabled: true
  task_types:
    - build_task
    - download_task
    - verification_task
    - documentation_task
  tracking:
    method: "markdown_checklist"
    location: "docs/TASK_TRACKER.md"
```

---

## Build Phase Instructions

### Phase 1: Initialization

```markdown
## Phase 1: Initialization

### Objectives
- Verify all prerequisites
- Initialize git worktrees
- Set up directory structure

### Commands
```bash
# Check prerequisites
which git || echo "ERROR: git not found"
which python3 || echo "ERROR: python3 not found"

# Initialize worktrees
git worktree add ../usb-ai-build build 2>/dev/null || git worktree prune
git worktree add ../usb-ai-release release 2>/dev/null || git worktree prune

# Create directory structure
mkdir -p ../usb-ai-build/{ollama,models,webui,veracrypt}
mkdir -p ../usb-ai-release/usb-image
```

### Validation
- [ ] Git worktrees exist
- [ ] Directory structure created
- [ ] Dependencies available
```

### Phase 2: Encryption Setup

```markdown
## Phase 2: Encryption Setup

### Objectives
- Download VeraCrypt portable binaries
- Create encrypted container
- Document unlock procedures

### Resources
- macOS: https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_1.26.14.dmg
- Windows: https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_Setup_x64_1.26.14.exe
- Linux: https://launchpad.net/veracrypt/trunk/1.26.14/+download/veracrypt-1.26.14-setup.tar.bz2

### Container Specs
- Size: 100GB (128GB USB) or 200GB (256GB USB)
- Encryption: AES-256
- Hash: SHA-512
- Filesystem: exFAT (cross-platform)
```

### Phase 3: Ollama Installation

```markdown
## Phase 3: Ollama Installation

### Objectives
- Download Ollama binaries for all platforms
- Configure portable mode
- Set environment variables

### Downloads
- macOS: https://ollama.com/download/Ollama-darwin.zip
- Windows: https://ollama.com/download/OllamaSetup.exe
- Linux: https://ollama.com/download/ollama-linux-amd64

### Portable Configuration
Set OLLAMA_MODELS to point to USB path:
- Windows: set OLLAMA_MODELS=%USB_DRIVE%\encrypted\ollama\models
- macOS/Linux: export OLLAMA_MODELS=/Volumes/USBAI/encrypted/ollama/models
```

### Phase 4: Model Downloads

```markdown
## Phase 4: Model Downloads

### Objectives
- Download specified models via Ollama
- Verify checksums
- Organize model storage

### Models (Priority Order)
1. dolphin-llama3:8b (PRIMARY)
   - Command: ollama pull dolphin-llama3:8b
   - Size: ~4.7GB

2. llama3.2:8b
   - Command: ollama pull llama3.2:8b
   - Size: ~4.7GB

3. qwen2.5:14b
   - Command: ollama pull qwen2.5:14b
   - Size: ~8.9GB

### User Selection Implementation
Create model_selector.sh that:
1. Lists available models
2. Prompts user selection
3. Sets default model
4. Updates Ollama config
```

### Phase 5: UI Setup

```markdown
## Phase 5: UI Setup

### Objectives
- Set up Flask + HTMX chat interface
- Configure for portable operation
- Apply dark theme with orange accent

### Why Flask + HTMX
| Factor | Flask + HTMX | Open WebUI |
|--------|--------------|------------|
| Startup time | Fast (<2s) | Slow (30s+) |
| Dependencies | Minimal | Heavy |
| Size | ~10MB | ~1GB |
| Customization | Full control | Limited |
| Reliability | High | Variable |

### Installation
```bash
# Using pip (portable)
pip install flask requests --target ./webui/app

# Single-file chat UI included in project
# modules/webui-portable/chat_ui.py
```

### Configuration
- Default port: 3000
- Ollama URL: http://localhost:11434
- Theme: Dark (#1a1a1a) with orange accent (#ffa222)
```

### Phase 6: Launcher Scripts

```markdown
## Phase 6: Launcher Scripts

### Objectives
- Create OS-specific launchers
- Handle mount/unmount
- Auto-start services

### Scripts to Create
1. start_macos.command
2. start_windows.bat
3. start_linux.sh
4. stop_all.sh (graceful shutdown)

### Launcher Flow
1. Detect USB mount point
2. Prompt for encryption password
3. Mount VeraCrypt container
4. Set environment variables
5. Start Ollama server
6. Start Open WebUI
7. Open browser to localhost:3000
```

### Phase 7: Testing

```markdown
## Phase 7: Testing

### Objectives
- Validate all components
- Test on each OS
- Document issues

### Test Cases
| ID | Test | Expected |
|----|------|----------|
| T01 | Mount encrypted volume | Password prompt, successful mount |
| T02 | Start Ollama | Server on port 11434 |
| T03 | Load model | Model responds to prompt |
| T04 | Chat UI | Browser shows chat interface |
| T05 | Send message | AI responds correctly |
| T06 | Switch model | New model loads and responds |
| T07 | Graceful shutdown | All services stop, volume unmounts |
```

### Phase 8: Packaging

```markdown
## Phase 8: Packaging

### Objectives
- Create final USB image
- Document for end users
- Commit to release branch

### Final Structure
```
USB_ROOT/
├── veracrypt/
├── encrypted.vc
├── launchers/
├── README.txt
└── LICENSE
```

### End-User README Content
- Quick start guide
- Password requirements
- Troubleshooting
- Model selection
- Resource requirements
```

---

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `Model too large` | Insufficient USB space | Use smaller quantization |
| `Mount failed` | Wrong password | Retry with correct password |
| `Port in use` | Service already running | Kill existing process |
| `Out of memory` | Model too large for RAM | Use smaller model |

### Recovery Procedures

```yaml
recovery:
  on_error:
    - log_error_details
    - save_current_state
    - attempt_rollback
    - notify_user
  max_retries: 3
  retry_delay: "5s"
```

---

## Reporting

### Progress Updates

```yaml
reporting:
  frequency: "per_phase"
  format: "markdown"
  include:
    - phase_name
    - status
    - duration
    - issues
    - next_steps
  output: "docs/BUILD_LOG.md"
```

### Final Report

Generate comprehensive build report including:
- Total build time
- Component versions
- Model checksums
- Test results
- Known issues

---

## Quick Reference Commands

```bash
# Start build
./scripts/s001_full_build.sh

# Check status
cat docs/BUILD_LOG.md

# Resume from phase
./scripts/resume_build.sh --phase 4

# Validate build
./scripts/validate_all.sh

# Package release
./scripts/package_release.sh
```

---

## Contact and Support

- GitHub Issues: https://github.com/mikesdatawork/usb-ai/issues
- Documentation: See /docs folder

---

**This document is the primary reference for Claude Max during build operations.**
