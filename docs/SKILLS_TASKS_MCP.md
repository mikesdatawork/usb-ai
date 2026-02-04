# Skills, Tasks, and MCP Configuration
## USB-AI Build System Integration Guide

This document specifies all skills, tasks, and MCP configurations for the USB-AI build system.

---

## Skills Configuration

### GSD (Get Stuff Done) Methodology

The GSD skill provides structured task execution with clear outcomes.

```yaml
skill:
  name: "GSD"
  version: "1.0"
  
  principles:
    - "Define clear, measurable outcomes"
    - "Break complex tasks into atomic steps"
    - "Execute sequentially with validation"
    - "Iterate on failure, don't abandon"
    - "Document everything"
    
  task_structure:
    outcome:
      description: "What success looks like"
      required: true
      
    steps:
      description: "Ordered list of actions"
      required: true
      max_complexity: "5 sub-steps per step"
      
    validation:
      description: "How to verify success"
      required: true
      
    fallback:
      description: "What to do on failure"
      required: false
      
  execution_rules:
    - "Complete one step before starting next"
    - "Validate each step before proceeding"
    - "Log all outcomes (success and failure)"
    - "Never skip validation"
    - "Retry failed steps before escalating"
    
  example:
    outcome: "Ollama server running on port 11434"
    steps:
      - "Source environment configuration"
      - "Start ollama serve in background"
      - "Wait 5 seconds for startup"
      - "Test API endpoint"
    validation:
      - "curl http://localhost:11434/api/tags returns JSON"
    fallback:
      - "Check if port is in use"
      - "Kill existing process"
      - "Retry startup"
```

### Task Tracking Skill

```yaml
skill:
  name: "TaskTracker"
  version: "1.0"
  
  format:
    statuses:
      - "TODO"      # Not started
      - "DOING"     # In progress
      - "BLOCKED"   # Waiting on dependency
      - "DONE"      # Completed successfully
      - "FAILED"    # Completed with failure
      
  markdown_format: |
    ## Task: ${task_name}
    
    - **ID**: ${task_id}
    - **Status**: ${status}
    - **Started**: ${start_time}
    - **Duration**: ${duration}
    - **Owner**: ${agent_name}
    
    ### Steps
    - [x] Step 1 completed
    - [ ] Step 2 pending
    
    ### Notes
    ${notes}
    
  tracking_file: "docs/TASK_TRACKER.md"
  
  auto_update:
    on_status_change: true
    on_step_complete: true
    on_error: true
```

### Code Generation Skill

```yaml
skill:
  name: "CodeGenerator"
  version: "1.0"
  
  languages:
    - bash
    - python
    - batch
    
  templates:
    bash_script:
      header: |
        #!/bin/bash
        # ${description}
        # Generated: ${date}
        # Version: ${version}
        
        set -e  # Exit on error
        
      error_handling: |
        trap 'echo "Error on line $LINENO"; exit 1' ERR
        
    python_script:
      header: |
        #!/usr/bin/env python3
        """${description}"""
        
        import sys
        import os
        
    batch_script:
      header: |
        @echo off
        REM ${description}
        REM Generated: ${date}
        
        setlocal enabledelayedexpansion
        
  conventions:
    naming: "s{NNN}_{descriptive_name}.{ext}"
    documentation: "inline comments required"
    error_handling: "mandatory"
```

---

## Claude Tasks Configuration

### Task Types

```yaml
task_types:
  build_task:
    description: "Execute build operations"
    properties:
      - commands
      - success_criteria
      - timeout
      - retry_count
    example:
      name: "Download Ollama"
      type: "build_task"
      commands:
        - "curl -L -o ollama.zip https://ollama.com/download"
      success_criteria:
        - "file exists"
        - "file size > 50MB"
      timeout: "300s"
      retry_count: 3
      
  download_task:
    description: "Download files from URLs"
    properties:
      - url
      - destination
      - checksum
      - resume_support
    example:
      name: "Download Model"
      type: "download_task"
      url: "https://ollama.com/library/dolphin-llama3"
      destination: "models/"
      resume_support: true
      
  verification_task:
    description: "Verify conditions are met"
    properties:
      - checks
      - pass_threshold
      - report_format
    example:
      name: "Verify Installation"
      type: "verification_task"
      checks:
        - "ollama --version"
        - "ls models/"
      pass_threshold: "100%"
      
  documentation_task:
    description: "Generate documentation"
    properties:
      - template
      - data_sources
      - output_format
    example:
      name: "Generate README"
      type: "documentation_task"
      template: "templates/readme.md"
      output_format: "markdown"
```

### Task Queue Management

```yaml
task_queue:
  name: "usb_ai_build_queue"
  
  priority_levels:
    1: "CRITICAL"   # Must complete for build to succeed
    2: "HIGH"       # Should complete, but can skip
    3: "NORMAL"     # Nice to have
    4: "LOW"        # Optional enhancements
    
  scheduling:
    mode: "priority_fifo"
    parallel_limit: 3
    timeout_default: "600s"
    
  dependencies:
    resolution: "topological_sort"
    circular_handling: "error"
    
  persistence:
    enabled: true
    storage: "task_queue.json"
    recover_on_restart: true
```

---

## MCP Configurations

### PlayWriter MCP (Token Optimization)

```yaml
mcp:
  name: "PlayWriter"
  type: "token_management"
  version: "1.0"
  
  purpose: "Optimize token usage without sacrificing quality"
  
  strategies:
    content_compression:
      description: "Reduce redundant content"
      techniques:
        - reference_previous: "Use 'as shown above' instead of repeating"
        - summarize_large_blocks: "Condense long outputs to key points"
        - code_diff_only: "Show only changed lines in code updates"
        
    output_batching:
      description: "Combine multiple small outputs"
      rules:
        - batch_size: 5
        - batch_trigger: "similar_content_type"
        - format: "combined_list"
        
    progressive_disclosure:
      description: "Start brief, expand on request"
      levels:
        1: "One-line summary"
        2: "Key points (3-5 bullets)"
        3: "Full detail"
      default_level: 2
      
    lazy_evaluation:
      description: "Don't generate until needed"
      applies_to:
        - optional_documentation
        - verbose_logs
        - alternative_approaches
        
  token_tracking:
    budget:
      per_message: 4000
      per_phase: 50000
      per_build: 500000
      
    alerts:
      warning: 80%
      critical: 95%
      
    actions:
      on_warning: "switch to compressed mode"
      on_critical: "pause and summarize"
      
  reporting:
    frequency: "per_phase"
    metrics:
      - tokens_used
      - tokens_saved
      - compression_ratio
      - efficiency_score
      
  integration:
    hooks:
      before_response: "apply_compression"
      after_phase: "report_usage"
```

### File System MCP

```yaml
mcp:
  name: "FileSystem"
  type: "file_operations"
  version: "1.0"
  
  capabilities:
    read:
      - text_files
      - binary_files
      - directory_listings
      
    write:
      - create_file
      - append_file
      - overwrite_file
      
    manage:
      - create_directory
      - delete_file
      - move_file
      - copy_file
      
  paths:
    allowed:
      - "/home/claude/usb-ai-docs"
      - "/mnt/user-data/outputs"
      - "${USB_PATH}"
      - "${ENCRYPTED_PATH}"
      
    blocked:
      - "/etc"
      - "/usr"
      - "/var"
      
  safety:
    confirm_destructive: true
    backup_before_overwrite: true
    max_file_size: "100MB"
```

### GitHub MCP

```yaml
mcp:
  name: "GitHub"
  type: "version_control"
  version: "1.0"
  
  repository:
    url: "https://github.com/mikesdatawork/usb-ai"
    default_branch: "main"
    
  operations:
    read:
      - clone
      - fetch
      - status
      - log
      - diff
      
    write:
      - commit
      - push
      - branch
      - merge
      
    worktrees:
      - add
      - remove
      - list
      
  automation:
    auto_commit:
      enabled: true
      triggers:
        - phase_complete
        - documentation_update
      message_template: "[claude-max] ${action}: ${description}"
      
    branch_strategy:
      main: "documentation and source"
      build: "build artifacts"
      release: "release packages"
```

### Project Memory MCP

```yaml
mcp:
  name: "ProjectMemory"
  type: "state_persistence"
  version: "1.0"
  
  purpose: "Persist project state across sessions"
  
  storage:
    type: "key_value"
    persistence: "permanent"
    
  keys:
    project_state:
      current_phase: "string"
      last_completed_task: "string"
      error_count: "integer"
      
    environment:
      usb_path: "string"
      encrypted_path: "string"
      build_path: "string"
      
    progress:
      tasks_completed: "list"
      tasks_pending: "list"
      models_downloaded: "list"
      
    metrics:
      start_time: "datetime"
      phase_durations: "object"
      token_usage: "object"
      
  operations:
    save:
      triggers:
        - phase_complete
        - error_occurred
        - explicit_save
      data: "current_state"
      
    load:
      triggers:
        - session_start
        - explicit_load
      fallback: "default_state"
      
    clear:
      triggers:
        - build_complete
        - explicit_clear
      archive: true
      
  recovery:
    enabled: true
    strategy: "resume_from_last_checkpoint"
    max_retries: 3
```

---

## Plugin Configuration

### Code Simplifier Plugin

```yaml
plugin:
  name: "CodeSimplifier"
  type: "code_optimization"
  version: "1.0"
  
  purpose: "Simplify and optimize generated code"
  
  applicability:
    languages:
      - bash
      - python
      - batch
    file_patterns:
      - "*.sh"
      - "*.py"
      - "*.bat"
      
  simplification_rules:
    remove_redundancy:
      - duplicate_variable_declarations
      - unused_imports
      - dead_code
      
    optimize_logic:
      - simplify_conditionals
      - combine_similar_operations
      - use_built_in_functions
      
    improve_readability:
      - consistent_naming
      - logical_grouping
      - appropriate_comments
      
  quality_checks:
    - syntax_validation
    - shellcheck (for bash)
    - pylint (for python)
    
  integration:
    apply_on:
      - file_create
      - file_update
    report_changes: true
    
  example_simplification:
    before: |
      x=1
      y=2
      z=3
      if [ $x -eq 1 ]; then
        if [ $y -eq 2 ]; then
          echo "yes"
        fi
      fi
    after: |
      x=1; y=2; z=3
      [[ $x -eq 1 && $y -eq 2 ]] && echo "yes"
```

---

## Integration Example

### Complete Phase Execution with All Components

```python
# Pseudo-code showing integration

from skills import GSD, TaskTracker, CodeGenerator
from mcp import PlayWriter, FileSystem, GitHub, ProjectMemory
from plugins import CodeSimplifier

# Initialize components
gsd = GSD()
tracker = TaskTracker()
codegen = CodeGenerator()
playwriter = PlayWriter(budget=50000)
fs = FileSystem()
git = GitHub(repo="mikesdatawork/usb-ai")
memory = ProjectMemory()

# Load previous state
state = memory.load()
if state.current_phase:
    print(f"Resuming from phase: {state.current_phase}")

# Execute phase with GSD methodology
phase = gsd.create_phase(
    name="Download Models",
    outcome="All models downloaded and verified",
    steps=[
        "Start Ollama server",
        "Download dolphin-llama3",
        "Download llama3.2",
        "Download qwen2.5",
        "Verify all models"
    ]
)

for step in phase.steps:
    # Track task
    task = tracker.create_task(step.name, status="DOING")
    
    # Optimize tokens
    playwriter.start_tracking()
    
    try:
        # Execute step
        result = step.execute()
        
        # Validate
        if step.validate(result):
            task.status = "DONE"
            
            # Generate code if needed
            if step.generates_code:
                code = codegen.generate(step.code_spec)
                code = CodeSimplifier.simplify(code)
                fs.write(step.output_path, code)
                
            # Commit to git
            git.commit(f"Completed: {step.name}")
            
        else:
            task.status = "FAILED"
            
    except Exception as e:
        task.status = "FAILED"
        task.error = str(e)
        
    finally:
        # Save state
        memory.save({
            "current_phase": phase.name,
            "last_task": task.name,
            "last_status": task.status
        })
        
        # Report token usage
        playwriter.report()

# Phase complete
memory.save({"phase_complete": phase.name})
git.commit(f"Phase complete: {phase.name}")
```

---

## Quick Reference Commands

### Enable Skills

```yaml
# In CLAUDE.md or conversation
skills:
  - GSD: enabled
  - TaskTracker: enabled
  - CodeGenerator: enabled
```

### Configure MCPs

```yaml
mcps:
  PlayWriter:
    budget: 50000
    mode: "token_efficient"
  FileSystem:
    root: "/home/claude/usb-ai-docs"
  GitHub:
    repo: "mikesdatawork/usb-ai"
    auto_commit: true
  ProjectMemory:
    persist: true
```

### Enable Plugins

```yaml
plugins:
  CodeSimplifier:
    enabled: true
    languages: ["bash", "python"]
```

---

**See individual component documentation for detailed configuration options.**
