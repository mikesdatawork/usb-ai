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
    - "UI: Open WebUI"
    - "Repo: github.com/mikesdatawork/usb-ai"
```

### Memory Items to Persist

| Key | Value | Priority |
|-----|-------|----------|
| `usb_target_size` | 128GB | Critical |
| `primary_model` | dolphin-llama3 | Critical |
| `encryption_method` | veracrypt | Critical |
| `ui_choice` | open-webui | Critical |
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

### 1. Proactive Agent

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

### 2. Self-Improvement Agent

```yaml
agent_self_improvement:
  name: "Optimizer"
  role: "Analyze and improve build process"
  behaviors:
    - track_build_times
    - identify_bottlenecks
    - suggest_parallelization
    - log_improvements
  outputs:
    - improvement_log.md
    - performance_metrics.json
```

### 3. Browser Agent

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

### 4. Build Orchestrator Agent

```yaml
agent_build:
  name: "BuildMaster"
  role: "Primary build orchestration"
  responsibilities:
    - execute_build_phases
    - coordinate_sub_agents
    - handle_errors
    - report_progress
  state_machine:
    states:
      - IDLE
      - INITIALIZING
      - BUILDING
      - TESTING
      - PACKAGING
      - COMPLETE
      - ERROR
```

### 5. Validation Agent

```yaml
agent_validation:
  name: "QualityGate"
  role: "Verify build outputs"
  checks:
    - file_integrity
    - model_checksums
    - encryption_verification
    - cross_platform_scripts
  report_format: "markdown"
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
- Install Open WebUI
- Configure for portable operation
- Set up auto-launch

### Why Open WebUI over AnythingLLM
| Factor | Open WebUI | AnythingLLM |
|--------|------------|-------------|
| Stability | High | Medium |
| Speed | Fast | Medium |
| Size | ~1GB | ~2GB |
| Features | Comprehensive | Comprehensive |
| Docker-free option | Yes | Partial |

### Installation
```bash
# Using pip (portable)
pip install open-webui --target ./webui

# Or standalone binary
# Download from: https://github.com/open-webui/open-webui/releases
```

### Configuration
- Default port: 3000
- Data directory: ./data
- Ollama URL: http://localhost:11434
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
| T04 | Open WebUI | Browser shows chat interface |
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
