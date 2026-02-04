# Plan Mode Instructions
## USB-AI Build System - AI Orchestration Plans

This document contains comprehensive plan-mode instruction sets for Claude Max to execute the USB-AI build process.

---

## Plan Mode Overview

Plan mode enables Claude Max to:
1. Understand complex multi-step processes
2. Execute tasks in correct order
3. Handle dependencies and failures
4. Maintain state across sessions
5. Optimize token usage

---

## Master Build Plan

### Plan ID: `USB_AI_MASTER_001`

```yaml
plan:
  id: "USB_AI_MASTER_001"
  name: "USB-AI Complete Build"
  version: "1.0.0"
  description: "Build portable offline encrypted AI system"
  
  metadata:
    author: "Claude Max"
    created: "2026-02-03"
    estimated_duration: "90-120 minutes"
    token_budget: 500000
    
  prerequisites:
    - git_installed
    - python3_10_plus
    - internet_connection
    - usb_drive_128gb_plus
    - admin_access
    
  phases:
    - phase_init
    - phase_download
    - phase_encryption
    - phase_ollama
    - phase_models
    - phase_webui
    - phase_launchers
    - phase_validation
    - phase_packaging
    
  error_handling:
    strategy: "retry_then_escalate"
    max_retries: 3
    escalation: "pause_and_notify"
    
  memory_persistence:
    enabled: true
    key: "usb_ai_build_state"
    persist_on:
      - phase_complete
      - error
      - pause
```

---

## Phase Plans

### Phase 1: Initialization

```yaml
phase:
  id: "phase_init"
  name: "Environment Initialization"
  order: 1
  
  objectives:
    - "Verify all prerequisites"
    - "Initialize git worktrees"
    - "Create directory structure"
    - "Configure build environment"
    
  tasks:
    - task_id: "init_001"
      name: "Check Prerequisites"
      type: "verification"
      commands:
        - "which git && git --version"
        - "which python3 && python3 --version"
        - "which pip3 && pip3 --version"
      success_criteria:
        - "git version 2.30+"
        - "python version 3.10+"
      on_failure: "ABORT with prerequisite instructions"
      
    - task_id: "init_002"
      name: "Clone Repository"
      type: "command"
      condition: "NOT directory_exists('usb-ai')"
      commands:
        - "git clone https://github.com/mikesdatawork/usb-ai.git"
        - "cd usb-ai"
      success_criteria:
        - "directory 'usb-ai' exists"
        - ".git directory present"
      on_failure: "RETRY 3 times, then ABORT"
      
    - task_id: "init_003"
      name: "Create Worktrees"
      type: "command"
      commands:
        - "git branch build 2>/dev/null || true"
        - "git branch release 2>/dev/null || true"
        - "git worktree add ../usb-ai-build build"
        - "git worktree add ../usb-ai-release release"
      success_criteria:
        - "git worktree list shows 3 entries"
      on_failure: "git worktree prune && RETRY"
      
    - task_id: "init_004"
      name: "Create Build Directories"
      type: "command"
      commands:
        - "cd ../usb-ai-build"
        - "mkdir -p downloads/{macos,windows,linux}"
        - "mkdir -p staging/{veracrypt,ollama,webui,models}"
        - "mkdir -p output"
      success_criteria:
        - "all directories exist"
        
    - task_id: "init_005"
      name: "Install Python Dependencies"
      type: "command"
      commands:
        - "pip3 install requests tqdm pyyaml --quiet"
      success_criteria:
        - "pip3 list | grep requests"
        
  outputs:
    - "usb-ai/ (main repo)"
    - "usb-ai-build/ (build worktree)"
    - "usb-ai-release/ (release worktree)"
    
  persist_state:
    - "repo_path"
    - "build_path"
    - "release_path"
```

### Phase 2: Downloads

```yaml
phase:
  id: "phase_download"
  name: "Resource Downloads"
  order: 2
  depends_on: "phase_init"
  
  objectives:
    - "Download VeraCrypt for all platforms"
    - "Download Ollama for all platforms"
    - "Verify checksums"
    
  parallel_execution: true
  
  tasks:
    - task_id: "dl_001"
      name: "Download VeraCrypt macOS"
      type: "download"
      url: "https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_1.26.14.dmg"
      destination: "downloads/macos/VeraCrypt.dmg"
      expected_size: "~35MB"
      checksum_url: "https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_1.26.14.dmg.sig"
      
    - task_id: "dl_002"
      name: "Download VeraCrypt Windows"
      type: "download"
      url: "https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_Portable_1.26.14.exe"
      destination: "downloads/windows/VeraCrypt_Portable.exe"
      expected_size: "~35MB"
      
    - task_id: "dl_003"
      name: "Download VeraCrypt Linux"
      type: "download"
      url: "https://launchpad.net/veracrypt/trunk/1.26.14/+download/veracrypt-1.26.14-setup.tar.bz2"
      destination: "downloads/linux/veracrypt-setup.tar.bz2"
      expected_size: "~25MB"
      
    - task_id: "dl_004"
      name: "Download Ollama macOS"
      type: "download"
      url: "https://ollama.com/download/Ollama-darwin.zip"
      destination: "downloads/macos/Ollama-darwin.zip"
      expected_size: "~60MB"
      
    - task_id: "dl_005"
      name: "Download Ollama Windows"
      type: "download"
      url: "https://ollama.com/download/OllamaSetup.exe"
      destination: "downloads/windows/OllamaSetup.exe"
      expected_size: "~100MB"
      
    - task_id: "dl_006"
      name: "Download Ollama Linux"
      type: "download"
      url: "https://ollama.com/download/ollama-linux-amd64"
      destination: "downloads/linux/ollama"
      expected_size: "~100MB"
      post_action: "chmod +x downloads/linux/ollama"
      
  verification:
    - "all files exist"
    - "file sizes match expected"
    - "checksums valid (where available)"
    
  outputs:
    - "downloads/checksums.sha256"
    
  token_optimization:
    strategy: "batch_status_updates"
    update_frequency: "per_download_complete"
```

### Phase 3: Encryption Setup

```yaml
phase:
  id: "phase_encryption"
  name: "Encryption Configuration"
  order: 3
  depends_on: "phase_download"
  
  objectives:
    - "Prepare USB drive"
    - "Create encrypted container"
    - "Set up portable VeraCrypt"
    
  user_interaction_required: true
  
  tasks:
    - task_id: "enc_001"
      name: "Identify USB Drive"
      type: "interactive"
      prompt: |
        Please identify your USB drive:
        
        macOS: Run 'diskutil list' and identify USB disk (e.g., disk2)
        Linux: Run 'lsblk' and identify USB device (e.g., sdb)
        
        Enter the device identifier:
      validation:
        - "device exists"
        - "device is removable"
        - "device size >= 128GB"
      warning: "ALL DATA ON THIS DEVICE WILL BE ERASED"
      
    - task_id: "enc_002"
      name: "Format USB Drive"
      type: "command"
      requires_confirmation: true
      commands_macos:
        - "diskutil unmountDisk /dev/${USB_DEVICE}"
        - "diskutil eraseDisk exFAT USBAI GPT /dev/${USB_DEVICE}"
      commands_linux:
        - "sudo umount /dev/${USB_DEVICE}* || true"
        - "sudo parted /dev/${USB_DEVICE} --script mklabel gpt"
        - "sudo parted /dev/${USB_DEVICE} --script mkpart primary 0% 100%"
        - "sudo mkfs.exfat -n USBAI /dev/${USB_DEVICE}1"
      success_criteria:
        - "USB formatted as exFAT"
        - "Label is USBAI"
        
    - task_id: "enc_003"
      name: "Mount USB Drive"
      type: "command"
      commands_macos:
        - "# Usually auto-mounts to /Volumes/USBAI"
        - "export USB_PATH=/Volumes/USBAI"
      commands_linux:
        - "sudo mkdir -p /mnt/usbai"
        - "sudo mount /dev/${USB_DEVICE}1 /mnt/usbai"
        - "export USB_PATH=/mnt/usbai"
        
    - task_id: "enc_004"
      name: "Create USB Structure"
      type: "command"
      commands:
        - "mkdir -p ${USB_PATH}/veracrypt"
        - "mkdir -p ${USB_PATH}/launchers"
        
    - task_id: "enc_005"
      name: "Create Encrypted Container"
      type: "interactive"
      description: |
        Create a VeraCrypt encrypted container.
        
        Parameters:
        - Size: 100GB (for 128GB USB) or 200GB (for 256GB USB)
        - Encryption: AES
        - Hash: SHA-512
        - Filesystem: exFAT
        
        IMPORTANT: Remember your password!
      commands:
        - |
          veracrypt --text --create "${USB_PATH}/encrypted.vc" \
            --size=100G \
            --encryption=AES \
            --hash=SHA-512 \
            --filesystem=exFAT \
            --pim=0 \
            --random-source=/dev/urandom
      success_criteria:
        - "encrypted.vc exists"
        - "file size ~100GB"
        
    - task_id: "enc_006"
      name: "Mount Encrypted Container"
      type: "command"
      commands:
        - "sudo mkdir -p /mnt/encrypted"
        - "veracrypt --text ${USB_PATH}/encrypted.vc /mnt/encrypted"
        - "export ENCRYPTED_PATH=/mnt/encrypted"
        
    - task_id: "enc_007"
      name: "Create Internal Structure"
      type: "command"
      commands:
        - "mkdir -p ${ENCRYPTED_PATH}/{ollama,models,webui,data,config}"
        
  persist_state:
    - "USB_PATH"
    - "ENCRYPTED_PATH"
    - "encryption_password_hint: 'User defined'"
```

### Phase 4: Ollama Installation

```yaml
phase:
  id: "phase_ollama"
  name: "Ollama Setup"
  order: 4
  depends_on: "phase_encryption"
  
  objectives:
    - "Install Ollama binaries for all platforms"
    - "Configure portable mode"
    - "Create environment scripts"
    
  tasks:
    - task_id: "oll_001"
      name: "Extract macOS Ollama"
      type: "command"
      commands:
        - "unzip -o downloads/macos/Ollama-darwin.zip -d ${ENCRYPTED_PATH}/ollama/macos/"
        
    - task_id: "oll_002"
      name: "Copy Windows Ollama"
      type: "command"
      commands:
        - "mkdir -p ${ENCRYPTED_PATH}/ollama/windows"
        - "cp downloads/windows/OllamaSetup.exe ${ENCRYPTED_PATH}/ollama/windows/"
      note: "Windows users will need to extract/install on first use"
      
    - task_id: "oll_003"
      name: "Copy Linux Ollama"
      type: "command"
      commands:
        - "mkdir -p ${ENCRYPTED_PATH}/ollama/linux"
        - "cp downloads/linux/ollama ${ENCRYPTED_PATH}/ollama/linux/"
        - "chmod +x ${ENCRYPTED_PATH}/ollama/linux/ollama"
        
    - task_id: "oll_004"
      name: "Create Environment Config"
      type: "file_create"
      path: "${ENCRYPTED_PATH}/config/ollama_env.sh"
      content: |
        #!/bin/bash
        # Ollama environment configuration
        
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        
        case "$(uname -s)" in
            Darwin)
                export OLLAMA_MODELS="${SCRIPT_DIR}/../models"
                export OLLAMA_HOST="127.0.0.1:11434"
                ;;
            Linux)
                export OLLAMA_MODELS="${SCRIPT_DIR}/../models"
                export OLLAMA_HOST="127.0.0.1:11434"
                ;;
        esac
        
        echo "OLLAMA_MODELS: $OLLAMA_MODELS"
      permissions: "755"
```

### Phase 5: Model Downloads

```yaml
phase:
  id: "phase_models"
  name: "AI Model Downloads"
  order: 5
  depends_on: "phase_ollama"
  
  objectives:
    - "Download specified LLM models"
    - "Verify model integrity"
    - "Create model selector"
    
  estimated_duration: "30-60 minutes"
  
  tasks:
    - task_id: "mod_001"
      name: "Start Temporary Ollama Server"
      type: "command"
      commands:
        - "source ${ENCRYPTED_PATH}/config/ollama_env.sh"
        - "${ENCRYPTED_PATH}/ollama/linux/ollama serve &"
        - "sleep 10"
        - "curl -s http://localhost:11434/api/tags"
      success_criteria:
        - "API responds with JSON"
        
    - task_id: "mod_002"
      name: "Download Dolphin-LLaMA3"
      type: "command"
      priority: "critical"
      commands:
        - "${ENCRYPTED_PATH}/ollama/linux/ollama pull dolphin-llama3:8b"
      expected_size: "4.7GB"
      expected_duration: "10-20 minutes"
      progress_tracking: true
      
    - task_id: "mod_003"
      name: "Download Llama 3.2"
      type: "command"
      priority: "high"
      commands:
        - "${ENCRYPTED_PATH}/ollama/linux/ollama pull llama3.2:8b"
      expected_size: "4.7GB"
      expected_duration: "10-20 minutes"
      
    - task_id: "mod_004"
      name: "Download Qwen2.5 14B"
      type: "command"
      priority: "normal"
      condition: "USB_SIZE >= 128GB"
      commands:
        - "${ENCRYPTED_PATH}/ollama/linux/ollama pull qwen2.5:14b"
      expected_size: "8.9GB"
      expected_duration: "15-30 minutes"
      
    - task_id: "mod_005"
      name: "Verify Models"
      type: "command"
      commands:
        - "${ENCRYPTED_PATH}/ollama/linux/ollama list"
      success_criteria:
        - "dolphin-llama3:8b listed"
        - "llama3.2:8b listed"
        - "qwen2.5:14b listed (if downloaded)"
        
    - task_id: "mod_006"
      name: "Create Model Selector"
      type: "file_create"
      path: "${ENCRYPTED_PATH}/config/select_model.sh"
      content: |
        #!/bin/bash
        echo "========================================"
        echo "       USB-AI Model Selection"
        echo "========================================"
        echo ""
        echo "Available Models:"
        echo "  1) Dolphin-LLaMA3 8B  (General/Uncensored)"
        echo "  2) Llama 3.2 8B       (General Purpose)"
        echo "  3) Qwen2.5 14B        (High Quality)"
        echo ""
        read -p "Select model [1-3]: " choice
        
        case $choice in
            1) MODEL="dolphin-llama3:8b" ;;
            2) MODEL="llama3.2:8b" ;;
            3) MODEL="qwen2.5:14b" ;;
            *) MODEL="dolphin-llama3:8b" ;;
        esac
        
        echo "Selected: $MODEL"
        export SELECTED_MODEL="$MODEL"
      permissions: "755"
      
    - task_id: "mod_007"
      name: "Set Default Model"
      type: "command"
      commands:
        - "echo 'dolphin-llama3:8b' > ${ENCRYPTED_PATH}/config/default_model.txt"
        
    - task_id: "mod_008"
      name: "Stop Temporary Server"
      type: "command"
      commands:
        - "pkill -f 'ollama serve' || true"
        
  persist_state:
    - "models_downloaded"
    - "total_model_size"
```

### Phase 6: WebUI Setup

```yaml
phase:
  id: "phase_webui"
  name: "Open WebUI Installation"
  order: 6
  depends_on: "phase_models"
  
  objectives:
    - "Install Open WebUI"
    - "Configure for portable operation"
    - "Create startup script"
    
  tasks:
    - task_id: "web_001"
      name: "Install Open WebUI"
      type: "command"
      commands:
        - "pip3 install open-webui --target ${ENCRYPTED_PATH}/webui/python"
      expected_duration: "5-10 minutes"
      
    - task_id: "web_002"
      name: "Create WebUI Start Script"
      type: "file_create"
      path: "${ENCRYPTED_PATH}/webui/start_webui.sh"
      content: |
        #!/bin/bash
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        
        export PYTHONPATH="$SCRIPT_DIR/python:$PYTHONPATH"
        export DATA_DIR="$SCRIPT_DIR/../data/webui"
        export OLLAMA_BASE_URL="http://127.0.0.1:11434"
        
        mkdir -p "$DATA_DIR"
        
        python3 -m open_webui.main --port 3000 --host 127.0.0.1
      permissions: "755"
      
    - task_id: "web_003"
      name: "Create WebUI Config"
      type: "file_create"
      path: "${ENCRYPTED_PATH}/config/webui_config.json"
      content: |
        {
          "ollama_base_url": "http://127.0.0.1:11434",
          "enable_signup": false,
          "default_user_role": "admin"
        }
      
    - task_id: "web_004"
      name: "Create Data Directory"
      type: "command"
      commands:
        - "mkdir -p ${ENCRYPTED_PATH}/data/webui"
```

### Phase 7: Launcher Creation

```yaml
phase:
  id: "phase_launchers"
  name: "Launcher Scripts"
  order: 7
  depends_on: "phase_webui"
  
  objectives:
    - "Create OS-specific launchers"
    - "Create shutdown script"
    - "Copy VeraCrypt portable"
    
  tasks:
    - task_id: "launch_001"
      name: "Create macOS Launcher"
      type: "file_create"
      path: "${USB_PATH}/launchers/start_macos.command"
      content: |
        #!/bin/bash
        # Full macOS launcher script
        # (See BUILD_PROCESS.md for complete content)
        
        echo "USB-AI Starting..."
        # ... (full script content)
      permissions: "755"
      
    - task_id: "launch_002"
      name: "Create Windows Launcher"
      type: "file_create"
      path: "${USB_PATH}/launchers/start_windows.bat"
      # ... (Windows launcher content)
      
    - task_id: "launch_003"
      name: "Create Linux Launcher"
      type: "file_create"
      path: "${USB_PATH}/launchers/start_linux.sh"
      permissions: "755"
      # ... (Linux launcher content)
      
    - task_id: "launch_004"
      name: "Create Shutdown Script"
      type: "file_create"
      path: "${USB_PATH}/launchers/stop_all.sh"
      permissions: "755"
      
    - task_id: "launch_005"
      name: "Copy VeraCrypt Portable"
      type: "command"
      commands:
        - "cp downloads/macos/VeraCrypt.dmg ${USB_PATH}/veracrypt/"
        - "cp downloads/windows/VeraCrypt_Portable.exe ${USB_PATH}/veracrypt/"
        - "cp downloads/linux/veracrypt-setup.tar.bz2 ${USB_PATH}/veracrypt/"
```

### Phase 8: Validation

```yaml
phase:
  id: "phase_validation"
  name: "Build Validation"
  order: 8
  depends_on: "phase_launchers"
  
  objectives:
    - "Verify all components"
    - "Test functionality"
    - "Generate validation report"
    
  tasks:
    - task_id: "val_001"
      name: "File Integrity Check"
      type: "verification"
      checks:
        - path: "${USB_PATH}/encrypted.vc"
          condition: "exists AND size > 50GB"
        - path: "${USB_PATH}/launchers/start_macos.command"
          condition: "exists AND executable"
        - path: "${USB_PATH}/launchers/start_windows.bat"
          condition: "exists"
        - path: "${USB_PATH}/launchers/start_linux.sh"
          condition: "exists AND executable"
        - path: "${USB_PATH}/README.txt"
          condition: "exists"
          
    - task_id: "val_002"
      name: "Encryption Test"
      type: "test"
      commands:
        - "veracrypt --text -l ${USB_PATH}/encrypted.vc"
      success_criteria:
        - "Volume mounted successfully"
        
    - task_id: "val_003"
      name: "Model Test"
      type: "test"
      commands:
        - "source ${ENCRYPTED_PATH}/config/ollama_env.sh"
        - "${ENCRYPTED_PATH}/ollama/linux/ollama serve &"
        - "sleep 5"
        - "curl -X POST http://localhost:11434/api/generate -d '{\"model\":\"dolphin-llama3:8b\",\"prompt\":\"Say hello\",\"stream\":false}'"
      success_criteria:
        - "Response contains text"
        
    - task_id: "val_004"
      name: "Generate Report"
      type: "file_create"
      path: "reports/validation_report.md"
      content_template: |
        # USB-AI Validation Report
        
        Date: ${DATE}
        Build: ${BUILD_ID}
        
        ## Results
        
        | Check | Status |
        |-------|--------|
        | File Integrity | ${FILE_CHECK_RESULT} |
        | Encryption | ${ENC_CHECK_RESULT} |
        | Models | ${MODEL_CHECK_RESULT} |
        
        ## Summary
        
        Build Status: ${OVERALL_STATUS}
```

### Phase 9: Packaging

```yaml
phase:
  id: "phase_packaging"
  name: "Final Packaging"
  order: 9
  depends_on: "phase_validation"
  
  objectives:
    - "Create end-user documentation"
    - "Clean up build artifacts"
    - "Prepare for release"
    
  tasks:
    - task_id: "pkg_001"
      name: "Create User README"
      type: "file_create"
      path: "${USB_PATH}/README.txt"
      # ... (User documentation content)
      
    - task_id: "pkg_002"
      name: "Unmount Encrypted Container"
      type: "command"
      commands:
        - "veracrypt -d ${ENCRYPTED_PATH}"
        
    - task_id: "pkg_003"
      name: "Calculate Final Size"
      type: "command"
      commands:
        - "du -sh ${USB_PATH}"
        
    - task_id: "pkg_004"
      name: "Generate Build Summary"
      type: "file_create"
      path: "reports/build_summary.md"
      
    - task_id: "pkg_005"
      name: "Commit to Release Branch"
      type: "command"
      condition: "ALL_VALIDATIONS_PASSED"
      commands:
        - "cd ../usb-ai-release"
        - "cp -r ../usb-ai-build/reports ."
        - "git add ."
        - "git commit -m '[release] USB-AI build ${BUILD_ID}'"
```

---

## Token Optimization Strategies

### PlayWriter MCP Integration

```yaml
token_optimization:
  mcp: "PlayWriter"
  
  strategies:
    batch_output:
      description: "Combine multiple outputs into single response"
      trigger: "multiple_small_updates"
      action: "buffer_and_batch"
      
    progressive_detail:
      description: "Start brief, add detail on request"
      trigger: "complex_explanation_needed"
      action: "summarize_first"
      
    reference_compression:
      description: "Reference previous content instead of repeating"
      trigger: "content_already_shown"
      action: "use_reference"
      
    code_folding:
      description: "Show only changed portions of code"
      trigger: "minor_code_update"
      action: "show_diff_only"
      
  budget_tracking:
    per_phase_limit: 50000
    warning_threshold: 80%
    pause_threshold: 95%
    
  reporting:
    frequency: "per_phase"
    metrics:
      - tokens_used
      - tokens_remaining
      - efficiency_score
```

---

## State Persistence

### Project Memory Configuration

```yaml
project_memory:
  enabled: true
  
  auto_save:
    triggers:
      - phase_complete
      - error_occurred
      - user_interrupt
      
  data_to_persist:
    critical:
      - current_phase
      - usb_path
      - encrypted_path
      - models_downloaded
      
    important:
      - task_completion_status
      - error_log
      - timing_data
      
    optional:
      - optimization_suggestions
      - performance_metrics
      
  recovery:
    on_session_start: "load_last_state"
    resume_from: "last_incomplete_phase"
    
  cleanup:
    on_build_complete: "archive_state"
    retention: "30_days"
```

---

## Execution Commands

### Start Build

```
Claude Max: Execute USB_AI_MASTER_001 with parameters:
- usb_size: 128GB
- models: [dolphin-llama3, llama3.2, qwen2.5]
- target_os: [macos, windows, linux]
```

### Resume Build

```
Claude Max: Resume USB_AI_MASTER_001 from last checkpoint
```

### Status Check

```
Claude Max: Report status of USB_AI_MASTER_001
```

### Cancel Build

```
Claude Max: Cancel USB_AI_MASTER_001 and cleanup
```

---

**This document is the primary reference for Claude Max plan-mode execution.**
