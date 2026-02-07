# USB-AI

Portable offline AI assistant that runs entirely from a USB drive.

No cloud. No internet required. No data leaves your machine.

---

## What This Is

USB-AI packages a complete local AI system onto a USB drive. Plug it into any computer with 16GB RAM, run the launcher, and you have a private AI assistant with a chat interface.

The system uses Ollama for model inference and a lightweight Flask + HTMX chat interface. Everything runs locally on the host machine. The USB drive contains the runtime, models, and configuration.

---

## Why This Exists

Cloud AI services require internet connectivity and send your data to external servers. This creates problems for:

- Sensitive document analysis
- Air-gapped environments
- Privacy-conscious users
- Locations without reliable internet
- Compliance-restricted industries

USB-AI solves these by running completely offline after initial setup.

---

## Architecture

The system follows modular design principles. Each component operates independently and communicates through defined interfaces.

```
usb-ai/
├── modules/
│   ├── ollama-portable/     # LLM inference runtime (cross-platform binaries)
│   ├── webui-portable/      # Flask + HTMX chat interface
│   ├── models/              # AI model files (GGUF format)
│   ├── launchers/           # Cross-platform start scripts
│   └── config/              # Shared settings (JSON)
├── resources/               # Static assets (images, etc.)
├── scripts/                 # Build automation
└── docs/                    # Documentation
```

### Module Independence

Each module can be updated or replaced without affecting others:

- **ollama-portable**: Replace binaries to update Ollama version
- **webui-portable**: Swap chat UI version independently
- **models**: Add or remove models without touching runtime
- **launchers**: Modify startup behavior in isolation
- **config**: Centralized settings read by all modules

---

## Technology Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| Inference | LLM Engine | Ollama (Go) |
| Models | Format | GGUF via llama.cpp |
| Interface | Web UI | Flask + HTMX (Python) |
| Frontend | Chat | HTMX (minimal JS) |
| Automation | Scripts | Python 3.10+ |
| Storage | Database | SQLite |

### Why These Choices

**Ollama**: Single binary, no dependencies, cross-platform, simple API. Handles model management and inference.

**Flask + HTMX**: Lightweight chat interface, minimal dependencies, fast startup, full control over UI/UX.

**Python for scripts**: Cross-platform without separate implementations per OS. Standard library covers most needs.

**GGUF models**: Quantized format runs on CPU. No GPU required. Reasonable performance on modern hardware.

---

## Included Models

| Model | Size | Parameters | Purpose |
|-------|------|------------|---------|
| Dolphin-LLaMA3 8B | 4.7GB | 8B | General use, default |
| Llama 3.2 8B | 4.7GB | 8B | General purpose |
| Qwen2.5 14B | 8.9GB | 14B | Higher quality, slower |

User selects the active model at runtime through the web interface.

---

## System Requirements

### Build Machine

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| Storage | 50GB free | 100GB free |
| Python | 3.10+ | 3.11+ |
| Internet | Required for downloads | Fast connection |

### Runtime (Target Machine)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| USB Port | USB 3.0 | USB 3.1+ |
| OS | macOS 12+, Windows 10+, Ubuntu 22.04+ | Latest |

### USB Drive

| Capacity | Models Supported |
|----------|------------------|
| 128GB | 2-3 models |
| 256GB | 4-5 models |

USB 3.1 or faster recommended for acceptable model load times.

---

## Quick Start

### Build

```bash
git clone https://github.com/mikesdatawork/usb-ai.git
cd usb-ai

python scripts/build/s001_setup_environment.py
python scripts/build/s002_download_ollama.py
python scripts/build/s003_download_models.py
python scripts/build/s004_setup_webui.py
python scripts/build/s005_apply_theme.py
```

### Run

```bash
python scripts/launchers/start.py
```

Browser opens to http://127.0.0.1:3000

### Stop

```bash
python scripts/launchers/stop.py
```

---

## UI Design

The chat interface uses a minimal dark flat theme:

- **Font**: Arial, Helvetica
- **Background**: Dark (#1a1a1a)
- **Accent**: Orange (#ffa222) for headers and list markers
- **Text weight**: Normal only, no bold
- **Style**: No gradients, no shadows, flat design

![USB-AI Chat Interface](resources/images/screenshot_example.png)

See docs/UI_THEME.md for complete CSS specifications.

---

## Documentation

| Document | Description |
|----------|-------------|
| CLAUDE.md | Claude Max build orchestration |
| docs/PRD.md | Product requirements |
| docs/BUILD_PROCESS.md | Detailed build steps |
| docs/STYLE_GUIDES.md | Code and component standards |
| docs/AGENTS.md | AI agent configurations |
| docs/PLAN_MODE.md | Plan-mode instructions |
| docs/SKILLS_TASKS_MCP.md | Skills and MCP setup |
| docs/MODELS.md | Model specifications |
| docs/UI_THEME.md | Interface theming |
| docs/REFERENCES.md | External links |

---

## Claude Max Integration

This project is designed for automated building using Claude Max with:

- **Plan Mode**: Structured multi-phase build process
- **MCPs**: PlayWriter (token optimization), Project Memory (state persistence)
- **Tasks**: GSD methodology for execution tracking

### Build Agents

| Agent | Role | Responsibility |
|-------|------|----------------|
| Build Orchestrator | Primary Coordinator | Manages state, delegates tasks, handles errors |
| Proactive Agent | Resource Manager | Anticipates needs, pre-fetches resources, monitors space |
| Self-Improvement Agent | Optimizer | Analyzes performance, identifies bottlenecks, suggests improvements |
| Browser Agent | Resource Fetcher | Fetches external resources, verifies checksums, updates docs |
| Validation Agent | Quality Gate | Verifies build outputs, runs tests, ensures quality |
| Encryption Agent | Crypto Manager | Handles VeraCrypt operations, secure key handling |
| Download Agent | Download Manager | Manages downloads, resume support, checksum verification |
| Flask Agent | WebUI Builder | Builds Flask chat application, API endpoints, SSE streaming |
| UI Agent | Interface Designer | Applies themes, CSS styling, frontend layout, HTMX integration |

The documentation provides complete instruction sets for Claude Max to execute the build autonomously.

### LLM Performance Optimization Agents

Specialized agents focused on maximizing inference speed and minimizing latency:

| Agent | Role | Optimization Focus |
|-------|------|-------------------|
| Ollama Config Optimizer | Configuration Tuner | Thread count, memory limits, GPU layers, keep-alive settings |
| Context Window Optimizer | Context Manager | Dynamic context sizing, KV cache optimization, sliding windows |
| Model Warmup Manager | Cold Start Eliminator | Pre-loads models, keepalive daemon, 3.3x speedup achieved |
| GPU Acceleration Optimizer | Hardware Accelerator | CUDA/ROCm/Metal detection, VRAM calculation, layer offloading |
| Inference Pipeline Optimizer | Request Handler | Connection pooling, streaming optimization, request batching |
| Log Analysis Agent | Performance Monitor | Analyzes metrics, identifies bottlenecks, provides recommendations |

#### Performance Results

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Cold Start | 31,124 ms | 9,439 ms | 3.3x faster |
| Warm Response | 19,525 ms | 923 ms | 21x faster |
| Time to First Token | 15,837 ms | 434 ms | 36x faster |

#### Optimization Scripts

```bash
# Apply Ollama optimizations
source scripts/performance/ollama_optimized.sh

# Start warmup daemon (keeps models loaded)
python -m scripts.performance.model_warmup --daemon

# Run performance benchmark
python -m scripts.performance.metrics_collector --quick

# Check system capabilities
python -m scripts.performance.system_check
```

#### Configuration Profiles

| Profile | Use Case | Context | Threads | Memory |
|---------|----------|---------|---------|--------|
| `speed` | Fast responses | 2048 | 90% cores | 70% RAM |
| `balanced` | General use | 4096 | 70% cores | 50% RAM |
| `quality` | Best output | 8192 | 50% cores | 80% RAM |
| `memory_saver` | Low RAM systems | 1024 | 50% cores | 30% RAM |

See `docs/AGENT_SWARM.md` for complete agent architecture and communication protocols.

---

## Privacy

- Runs 100% offline after build
- No telemetry or analytics
- No network calls during operation
- Chat history stored locally only
- Models run entirely on local hardware

---

## Project Structure

```
usb-ai/
├── CLAUDE.md                              # Claude Max build instructions
├── README.md                              # This file
├── LICENSE                                # MIT License
├── docs/
│   ├── AGENTS.md                          # Agent definitions
│   ├── BUILD_PROCESS.md                   # Build guide
│   ├── MODELS.md                          # Model specifications
│   ├── PLAN_MODE.md                       # AI plan-mode instructions
│   ├── PRD.md                             # Product requirements
│   ├── REFERENCES.md                      # Links and resources
│   ├── SKILLS_TASKS_MCP.md                # Skills and MCP config
│   ├── STYLE_GUIDES.md                    # Component standards
│   └── UI_THEME.md                        # Theme specifications
├── resources/
│   └── images/
│       └── screenshot_example.png         # UI screenshot
├── modules/
│   ├── config/
│   │   ├── launcher.json                  # Launcher settings
│   │   ├── system.json                    # System configuration
│   │   └── user.json                      # User preferences
│   ├── launchers/
│   │   ├── start_linux.sh                 # Linux launcher
│   │   ├── start_macos.command            # macOS launcher
│   │   ├── start_windows.bat              # Windows launcher
│   │   └── stop_all.sh                    # Shutdown script
│   ├── models/
│   │   ├── blobs/                         # Model binary data
│   │   ├── config/
│   │   │   └── models.json                # Model configuration
│   │   └── manifests/                     # Ollama model manifests
│   │       └── registry.ollama.ai/library/
│   │           ├── dolphin-llama3/8b      # Dolphin-LLaMA3 8B
│   │           ├── llama3.2/latest        # Llama 3.2 8B
│   │           └── qwen2.5/14b            # Qwen 2.5 14B
│   ├── ollama-portable/
│   │   ├── bin/
│   │   │   ├── darwin-amd64/              # macOS Intel binaries
│   │   │   ├── darwin-arm64/              # macOS Apple Silicon
│   │   │   ├── linux-amd64/               # Linux x64 binaries
│   │   │   └── windows-amd64/             # Windows x64 binaries
│   │   └── config/                        # Ollama configuration
│   └── webui-portable/
│       ├── chat_ui.py                     # Flask + HTMX chat app
│       ├── llm_monitor.py                 # LLM monitoring module
│       ├── app/                           # Python dependencies
│       ├── data/                          # Runtime data
│       ├── logs/                          # Application logs
│       └── static/css/                    # Custom stylesheets
└── scripts/
    ├── agents/                            # Agent coordination system
    │   ├── coordinator.py                 # Multi-agent orchestrator
    │   ├── agent_logger.py                # Centralized logging
    │   └── permissions.yaml               # Agent permissions config
    ├── build/                             # Build automation
    │   ├── parallel_builder.py            # Parallel build orchestrator
    │   ├── build_manifest.yaml            # Build task definitions
    │   └── validate_build.py              # Build validation
    ├── launchers/                         # Runtime scripts
    ├── performance/                       # LLM optimization scripts
    │   ├── ollama_tuner.py                # Ollama configuration optimizer
    │   ├── context_optimizer.py           # Context window management
    │   ├── model_warmup.py                # Cold start elimination
    │   ├── gpu_optimizer.py               # GPU acceleration
    │   ├── inference_optimizer.py         # Request pipeline optimization
    │   ├── metrics_collector.py           # Performance monitoring
    │   └── system_check.py                # Hardware capability detection
    └── quantization/                      # Model quantization
        ├── quantize_models.py             # GGUF quantization manager
        └── benchmark.py                   # Quantization benchmarking
```

---

## How It Works

1. **Build phase**: Download Ollama binaries, pull AI models, set up Flask chat UI, apply theme
2. **Copy to USB**: Transfer the built system to a USB drive
3. **Run anywhere**: Plug USB into target machine, run start.py
4. **Chat locally**: Browser opens to local web interface
5. **Stop cleanly**: Run stop.py before removing USB

The launcher scripts detect the operating system and use the appropriate binaries. All paths are relative to the USB mount point.

---

## Modular Design

The architecture follows these principles:

1. **Independence**: Each module works standalone
2. **Loose coupling**: Modules communicate via defined interfaces (HTTP APIs, config files, environment variables)
3. **Single responsibility**: One module, one purpose
4. **Replaceable**: Swap any module without breaking others

This means you can:

- Update Ollama without touching the UI
- Change the chat interface without affecting models
- Add models without modifying any code
- Customize launchers for specific environments

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the style guides in docs/STYLE_GUIDES.md
4. Submit a pull request

---

## License

MIT License

---

## Acknowledgments

- Ollama - https://ollama.com
- Flask - https://flask.palletsprojects.com
- HTMX - https://htmx.org
- Eric Hartford / Cognitive Computations - Dolphin models
- Meta AI - Llama models
- Alibaba - Qwen models
