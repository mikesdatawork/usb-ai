# Product Requirements Document (PRD)
## USB-AI: Portable Offline Encrypted AI System

**Version**: 1.0.0  
**Date**: 2026-02-03  
**Author**: Claude Max Build System  
**Status**: Active Development

---

## 1. Executive Summary

### 1.1 Product Vision
Create a fully portable, encrypted, offline AI assistant that runs from a USB drive without requiring internet connectivity, cloud services, or installation on the host machine.

### 1.2 Problem Statement
Current AI assistants require:
- Internet connectivity
- Cloud data transmission
- Trust in third-party data handling
- Subscription fees
- Installation permissions

Users needing privacy-first AI have no simple portable solution.

### 1.3 Solution
A pre-configured USB drive containing:
- Encrypted storage container
- Local LLM runtime (Ollama)
- Pre-downloaded AI models
- Web-based chat interface
- Cross-platform launch scripts

---

## 2. Goals and Objectives

### 2.1 Primary Goals
| Goal | Success Metric |
|------|----------------|
| Complete offline operation | Zero network calls after build |
| Cross-platform support | Works on macOS, Windows, Linux |
| Data encryption | AES-256 at rest |
| User selection of models | Minimum 3 model choices |
| Sub-60 second startup | From USB insert to chat ready |

### 2.2 Non-Goals
- Cloud synchronization
- Multi-user concurrent access
- Model fine-tuning on device
- GPU passthrough (CPU inference acceptable)

---

## 3. User Personas

### 3.1 Privacy-Conscious Professional
- **Name**: Alex
- **Role**: Lawyer / Healthcare worker
- **Need**: Process sensitive documents without cloud exposure
- **Tech Level**: Moderate

### 3.2 Traveling Consultant
- **Name**: Jordan
- **Role**: Business consultant
- **Need**: AI assistance on flights/remote locations
- **Tech Level**: Low to Moderate

### 3.3 Security Researcher
- **Name**: Sam
- **Role**: Penetration tester / Security analyst
- **Need**: Isolated AI for sensitive analysis
- **Tech Level**: High

---

## 4. Functional Requirements

### 4.1 Encryption System

| ID | Requirement | Priority |
|----|-------------|----------|
| ENC-001 | VeraCrypt encrypted container | P0 |
| ENC-002 | AES-256 encryption standard | P0 |
| ENC-003 | Password-based unlock | P0 |
| ENC-004 | Portable VeraCrypt binaries | P0 |
| ENC-005 | Optional keyfile support | P2 |

### 4.2 AI Runtime

| ID | Requirement | Priority |
|----|-------------|----------|
| AI-001 | Ollama runtime (portable) | P0 |
| AI-002 | Minimum 3 selectable models | P0 |
| AI-003 | Model hot-swap capability | P1 |
| AI-004 | Quantized models (Q4/Q5) | P0 |
| AI-005 | Context length 4096+ tokens | P1 |

### 4.3 User Interface

| ID | Requirement | Priority |
|----|-------------|----------|
| UI-001 | Flask + HTMX chat interface | P0 |
| UI-002 | Browser-based access | P0 |
| UI-003 | Chat history in session | P1 |
| UI-004 | Model switching in UI | P1 |
| UI-005 | Dark theme with orange accent | P0 |

### 4.4 Cross-Platform Support

| ID | Requirement | Priority |
|----|-------------|----------|
| CP-001 | macOS 12+ support | P0 |
| CP-002 | Windows 10/11 support | P0 |
| CP-003 | Ubuntu 22.04+ support | P1 |
| CP-004 | Auto-detect OS on launch | P0 |
| CP-005 | Native binary per platform | P0 |

### 4.5 Launch System

| ID | Requirement | Priority |
|----|-------------|----------|
| LS-001 | One-click launch per OS | P0 |
| LS-002 | Auto-mount encrypted volume | P1 |
| LS-003 | Auto-start Ollama server | P0 |
| LS-004 | Auto-open browser to UI | P1 |
| LS-005 | Graceful shutdown script | P1 |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Target |
|--------|--------|
| Cold start time | < 60 seconds |
| Token generation | > 10 tokens/sec (8B model) |
| Memory usage | < 12GB RAM |
| USB read speed | 400MB/s minimum |

### 5.2 Storage

| Component | Size (128GB) | Size (256GB) |
|-----------|--------------|--------------|
| VeraCrypt overhead | 2GB | 2GB |
| Ollama runtime | 500MB | 500MB |
| Chat UI (Flask) | 50MB | 50MB |
| Models | 25GB | 60GB |
| Reserved/Free | 99.5GB | 192.5GB |

### 5.3 Security

- Encryption: AES-256-XTS
- Key derivation: PBKDF2-RIPEMD160 (500,000 iterations)
- No plaintext credentials stored
- Memory cleared on unmount

---

## 6. Model Specifications

### 6.1 Required Models

| Model | Source | Size | Quantization |
|-------|--------|------|--------------|
| Dolphin-LLaMA3 8B | ollama.com/library/dolphin-llama3 | 4.7GB | Q4_K_M |
| Llama 3.2 8B | ollama.com/library/llama3.2 | 4.7GB | Q4_K_M |
| Qwen2.5 14B | ollama.com/library/qwen2.5:14b | 8.9GB | Q4_K_M |

### 6.2 Optional Models (256GB builds)

| Model | Source | Size | Quantization |
|-------|--------|------|--------------|
| Mistral 7B v0.3 | ollama.com/library/mistral | 4.1GB | Q4_K_M |
| DeepSeek-Coder 6.7B | ollama.com/library/deepseek-coder | 3.8GB | Q4_K_M |
| Llama 3.3 70B | ollama.com/library/llama3.3 | 40GB | Q4_K_M |

---

## 7. Technical Architecture

### 7.1 Component Stack

```
┌─────────────────────────────────────────────┐
│              User Browser                    │
├─────────────────────────────────────────────┤
│       Flask + HTMX UI (localhost:3000)       │
├─────────────────────────────────────────────┤
│         Ollama API (localhost:11434)         │
├─────────────────────────────────────────────┤
│              LLM Models (GGUF)               │
├─────────────────────────────────────────────┤
│         VeraCrypt Encrypted Volume           │
├─────────────────────────────────────────────┤
│              USB 3.1 Drive                   │
└─────────────────────────────────────────────┘
```

### 7.2 Data Flow

```
User Input → Browser → Flask UI → Ollama API → Model Inference → Response
     ↑                                                                  │
     └──────────────────────────────────────────────────────────────────┘
```

---

## 8. Build Process Overview

### 8.1 Phases

| Phase | Description | Duration |
|-------|-------------|----------|
| 1. Setup | Initialize environment, worktrees | 5 min |
| 2. Encryption | Create VeraCrypt container | 15 min |
| 3. Runtime | Install Ollama binaries | 10 min |
| 4. Models | Download and verify models | 30-60 min |
| 5. UI | Configure Flask chat UI | 5 min |
| 6. Launchers | Create OS-specific scripts | 5 min |
| 7. Testing | Validate all components | 15 min |
| 8. Package | Final USB preparation | 10 min |

### 8.2 Automation

All phases automated via shell scripts with Claude Max orchestration.

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model too large for USB | Medium | High | Use quantized versions |
| VeraCrypt incompatibility | Low | High | Test all target OS versions |
| Slow inference on CPU | High | Medium | Recommend 16GB+ RAM |
| Encryption password forgotten | Medium | Critical | Document recovery process |

---

## 10. Success Criteria

### 10.1 Acceptance Tests

| Test | Pass Criteria |
|------|---------------|
| Cold boot | Chat ready in < 60 seconds |
| Encryption | Volume mounts with correct password only |
| Model switch | Can change model without restart |
| Cross-platform | Works on macOS, Windows, Linux |
| Offline | Functions with network disabled |

### 10.2 Quality Gates

- All scripts exit 0 on success
- No hardcoded paths (use relative/env vars)
- Documentation complete for all components
- GitHub repo updated with all artifacts

---

## 11. Timeline

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| PRD Complete | 2026-02-03 | ✅ |
| Build Scripts | 2026-02-03 | 🔄 |
| First USB Image | 2026-02-04 | ⏳ |
| Testing Complete | 2026-02-05 | ⏳ |
| Release v1.0 | 2026-02-06 | ⏳ |

---

## 12. Appendices

### 12.1 Glossary

| Term | Definition |
|------|------------|
| GGUF | GPT-Generated Unified Format (model format) |
| LLM | Large Language Model |
| MCP | Model Context Protocol |
| Ollama | Local LLM runtime tool |
| Quantization | Model compression technique |
| VeraCrypt | Open-source disk encryption |

### 12.2 References

- Ollama Documentation: https://ollama.com/docs
- Flask: https://flask.palletsprojects.com
- HTMX: https://htmx.org
- VeraCrypt: https://veracrypt.io/en/Documentation.html
- Hugging Face Models: https://huggingface.co/models

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Claude Max | Initial PRD |
