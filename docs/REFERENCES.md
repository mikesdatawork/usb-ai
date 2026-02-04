# References and Resources
## USB-AI Build System External Dependencies

This document contains all external links, download URLs, and documentation references.

---

## Download Links

### VeraCrypt

| Platform | URL | Version | Size |
|----------|-----|---------|------|
| macOS | https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_1.26.14.dmg | 1.26.14 | ~35MB |
| Windows (Portable) | https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_Portable_1.26.14.exe | 1.26.14 | ~35MB |
| Windows (Installer) | https://launchpad.net/veracrypt/trunk/1.26.14/+download/VeraCrypt_Setup_x64_1.26.14.exe | 1.26.14 | ~35MB |
| Linux | https://launchpad.net/veracrypt/trunk/1.26.14/+download/veracrypt-1.26.14-setup.tar.bz2 | 1.26.14 | ~25MB |

**Checksums**: https://launchpad.net/veracrypt/trunk/1.26.14/+download/

### Ollama

| Platform | URL | Size |
|----------|-----|------|
| macOS (Universal) | https://ollama.com/download/Ollama-darwin.zip | ~60MB |
| Windows | https://ollama.com/download/OllamaSetup.exe | ~100MB |
| Linux (AMD64) | https://ollama.com/download/ollama-linux-amd64 | ~100MB |
| Linux (ARM64) | https://ollama.com/download/ollama-linux-arm64 | ~100MB |

**Official Site**: https://ollama.com/download

### AI Models (via Ollama)

| Model | Command | Size | Parameters |
|-------|---------|------|------------|
| Dolphin-LLaMA3 8B | `ollama pull dolphin-llama3:8b` | 4.7GB | 8B |
| Llama 3.2 8B | `ollama pull llama3.2:8b` | 4.7GB | 8B |
| Llama 3.2 3B | `ollama pull llama3.2:3b` | 2.0GB | 3B |
| Qwen2.5 14B | `ollama pull qwen2.5:14b` | 8.9GB | 14B |
| Qwen2.5 7B | `ollama pull qwen2.5:7b` | 4.7GB | 7B |
| Mistral 7B | `ollama pull mistral` | 4.1GB | 7B |
| DeepSeek-Coder 6.7B | `ollama pull deepseek-coder:6.7b` | 3.8GB | 6.7B |
| Llama 3.3 70B | `ollama pull llama3.3:70b` | 40GB | 70B |

**Model Library**: https://ollama.com/library

### Flask Chat UI

| Method | URL/Command |
|--------|-------------|
| Flask | `pip install flask requests` |
| HTMX CDN | https://unpkg.com/htmx.org@1.9.10 |

**Flask**: https://flask.palletsprojects.com
**HTMX**: https://htmx.org

---

## Documentation Links

### Official Documentation

| Resource | URL |
|----------|-----|
| Ollama Documentation | https://ollama.com/docs |
| Ollama API Reference | https://github.com/ollama/ollama/blob/main/docs/api.md |
| VeraCrypt Documentation | https://veracrypt.io/en/Documentation.html |
| VeraCrypt Command Line | https://veracrypt.io/en/Command%20Line%20Usage.html |
| Flask Documentation | https://flask.palletsprojects.com/en/stable/ |
| HTMX Documentation | https://htmx.org/docs/ |

### Model Documentation

| Model | Documentation |
|-------|---------------|
| Dolphin Models | https://huggingface.co/cognitivecomputations |
| Llama 3 | https://llama.meta.com/ |
| Qwen | https://qwenlm.github.io/ |
| Mistral | https://docs.mistral.ai/ |
| DeepSeek | https://www.deepseek.com/ |

### Hugging Face Resources

| Resource | URL |
|----------|-----|
| Model Hub | https://huggingface.co/models |
| Dolphin-LLaMA3 | https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b |
| GGUF Models | https://huggingface.co/TheBloke |

---

## GitHub Repositories

### Core Projects

| Project | Repository | Purpose |
|---------|------------|---------|
| Ollama | https://github.com/ollama/ollama | LLM runtime |
| Flask | https://github.com/pallets/flask | Web framework |
| HTMX | https://github.com/bigskysoftware/htmx | Dynamic HTML |
| VeraCrypt | https://github.com/veracrypt/VeraCrypt | Encryption |

### Related Projects

| Project | Repository | Purpose |
|---------|------------|---------|
| llama.cpp | https://github.com/ggerganov/llama.cpp | CPU inference |
| GGUF | https://github.com/ggerganov/ggml | Model format |
| LM Studio | https://github.com/lmstudio-ai | Alternative UI |
| AnythingLLM | https://github.com/Mintplex-Labs/anything-llm | Alternative UI |

### Utility Projects

| Project | Repository | Purpose |
|---------|------------|---------|
| vCrypt2Go | https://github.com/wandersick/vcrypt2go | VeraCrypt portable helper |
| PlugMind AI | https://github.com/shete7/PLUGMIND-AI | USB AI reference |

---

## API Endpoints

### Ollama API

**Base URL**: `http://localhost:11434`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate` | POST | Generate text completion |
| `/api/chat` | POST | Chat completion |
| `/api/tags` | GET | List installed models |
| `/api/pull` | POST | Download model |
| `/api/delete` | DELETE | Remove model |
| `/api/show` | POST | Show model info |

**Example Request**:
```bash
curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"dolphin-llama3:8b","prompt":"Hello"}'
```

### Open WebUI API

**Base URL**: `http://localhost:3000`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/chats` | GET | List chats |
| `/api/v1/models` | GET | List available models |
| `/ollama/*` | * | Proxy to Ollama |

---

## System Requirements Reference

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| RAM | 8GB (16GB recommended) |
| Storage | 128GB USB 3.0+ |
| CPU | x86_64 or ARM64 |
| OS | macOS 12+, Windows 10+, Ubuntu 22.04+ |

### Recommended Specifications

| Component | Specification |
|-----------|---------------|
| RAM | 32GB |
| Storage | 256GB USB 3.1+ |
| CPU | Apple M1+ or Intel i7+ |
| USB Speed | 400MB/s+ read |

### Model RAM Requirements

| Model Size | Minimum RAM | Recommended RAM |
|------------|-------------|-----------------|
| 3B | 4GB | 8GB |
| 7-8B | 8GB | 16GB |
| 14B | 16GB | 24GB |
| 70B | 48GB | 64GB |

---

## Troubleshooting Resources

### Common Issues

| Issue | Resource |
|-------|----------|
| Ollama not starting | https://github.com/ollama/ollama/issues |
| VeraCrypt mount fails | https://veracrypt.io/en/FAQ.html |
| Model too slow | https://github.com/ollama/ollama/blob/main/docs/faq.md |
| Flask/HTMX issues | https://github.com/pallets/flask/issues |

### Community Forums

| Platform | URL |
|----------|-----|
| Ollama Discord | https://discord.gg/ollama |
| Reddit r/LocalLLaMA | https://www.reddit.com/r/LocalLLaMA/ |
| Hugging Face Forums | https://discuss.huggingface.co/ |

---

## Version Information

### Current Versions (as of 2026-02-03)

| Component | Version | Release Date |
|-----------|---------|--------------|
| VeraCrypt | 1.26.14 | 2024-12 |
| Ollama | Latest | Rolling |
| Flask | 3.0+ | Stable |
| HTMX | 1.9.10 | Stable |
| Dolphin-LLaMA3 | 2.9 | 2024-04 |
| Llama 3.2 | 3.2 | 2024-09 |
| Qwen2.5 | 2.5 | 2024-09 |

### Checking Versions

```bash
# VeraCrypt
veracrypt --version

# Ollama
ollama --version

# Flask
pip show flask

# Installed Models
ollama list
```

---

## License Information

| Component | License |
|-----------|---------|
| Ollama | MIT |
| Flask | BSD-3 |
| HTMX | BSD-2 |
| VeraCrypt | Apache 2.0 + TrueCrypt |
| Llama Models | Meta Llama License |
| Qwen Models | Tongyi Qianwen License |
| Dolphin Models | Apache 2.0 |

---

## Security Resources

### Encryption Standards

| Standard | Reference |
|----------|-----------|
| AES-256 | https://csrc.nist.gov/publications/detail/fips/197/final |
| SHA-512 | https://csrc.nist.gov/publications/detail/fips/180-4/final |
| PBKDF2 | https://tools.ietf.org/html/rfc2898 |

### Security Audits

| Project | Audit |
|---------|-------|
| VeraCrypt | https://ostif.org/the-veracrypt-audit-results/ |

---

## Build System References

### Git Worktrees

| Resource | URL |
|----------|-----|
| Git Worktree Docs | https://git-scm.com/docs/git-worktree |
| Tutorial | https://git-scm.com/book/en/v2/Git-Tools-Worktrees |

### Shell Scripting

| Resource | URL |
|----------|-----|
| Bash Reference | https://www.gnu.org/software/bash/manual/ |
| ShellCheck | https://www.shellcheck.net/ |
| Batch Reference | https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/windows-commands |

---

## Quick Reference Card

### Essential Commands

```bash
# Clone project
git clone https://github.com/mikesdatawork/usb-ai.git

# Install Ollama (macOS)
brew install ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull dolphin-llama3:8b

# Run model
ollama run dolphin-llama3:8b

# Mount VeraCrypt container
veracrypt --text /path/to/container.vc /mount/point

# Unmount VeraCrypt container
veracrypt -d /mount/point

# Start Flask Chat UI
python chat_ui.py --port 3000
```

### Key URLs

| Purpose | URL |
|---------|-----|
| Project Repository | https://github.com/mikesdatawork/usb-ai |
| Ollama Downloads | https://ollama.com/download |
| Model Library | https://ollama.com/library |
| VeraCrypt Downloads | https://veracrypt.io/en/Downloads.html |
| Flask Docs | https://flask.palletsprojects.com |
| HTMX Docs | https://htmx.org |

---

**Last Updated**: 2026-02-03  
**Maintained By**: USB-AI Build System
