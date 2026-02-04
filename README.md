# USB-AI

Portable offline AI assistant that runs entirely from a USB drive.

No cloud. No internet required. No data leaves your machine.

---

## What This Is

USB-AI packages a complete local AI system onto a USB drive. Plug it into any computer with 16GB RAM, run the launcher, and you have a private AI assistant with a chat interface.

The system uses Ollama for model inference and Open WebUI for the browser-based chat interface. Everything runs locally on the host machine. The USB drive contains the runtime, models, and configuration.

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
│   ├── ollama-portable/     # LLM inference runtime
│   ├── webui-portable/      # Browser chat interface
│   ├── models/              # AI model files
│   ├── launchers/           # Cross-platform start scripts
│   └── config/              # Shared settings
├── scripts/                 # Build automation
└── docs/                    # Documentation
```

### Module Independence

Each module can be updated or replaced without affecting others:

- **ollama-portable**: Replace binaries to update Ollama version
- **webui-portable**: Swap Open WebUI version independently
- **models**: Add or remove models without touching runtime
- **launchers**: Modify startup behavior in isolation
- **config**: Centralized settings read by all modules

---

## Technology Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| Inference | LLM Engine | Ollama (Go) |
| Models | Format | GGUF via llama.cpp |
| Interface | Web UI | Open WebUI (Python/FastAPI) |
| Frontend | Chat | Svelte |
| Automation | Scripts | Python 3.10+ |
| Storage | Database | SQLite |

### Why These Choices

**Ollama**: Single binary, no dependencies, cross-platform, simple API. Handles model management and inference.

**Open WebUI**: Full-featured chat interface, works with Ollama out of the box, active development, good UX.

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
- **Agents**: Proactive, Self-Improvement, Browser, Build, Validation
- **MCPs**: PlayWriter (token optimization), Project Memory (state persistence)
- **Tasks**: GSD methodology for execution tracking

The documentation provides complete instruction sets for Claude Max to execute the build autonomously.

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
.
├── CLAUDE.md                              # Claude Max instructions
├── README.md                              # This file
├── docs/
│   ├── AGENTS.md                          # Agent definitions
│   ├── BUILD_PROCESS.md                   # Build guide
│   ├── MODELS.md                          # Model specs
│   ├── PLAN_MODE.md                       # AI plan-mode instructions
│   ├── PRD.md                             # Product requirements
│   ├── REFERENCES.md                      # Links and resources
│   ├── SKILLS_TASKS_MCP.md                # Skills and MCP config
│   ├── STYLE_GUIDES.md                    # Component standards
│   └── UI_THEME.md                        # Theme specifications
├── modules/
│   ├── config/                            # Shared configuration
│   ├── models/                            # AI model storage
│   ├── ollama-portable/                   # LLM runtime
│   └── webui-portable/                    # Chat interface
│       └── static/css/custom-theme.css    # UI theme
└── scripts/
    ├── build/                             # Build automation
    │   └── s001_setup_environment.py
    └── launchers/                         # Runtime scripts
        ├── start.py
        ├── stop.py
        └── status.py
```

---

## How It Works

1. **Build phase**: Download Ollama binaries, pull AI models, install Open WebUI, apply theme
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
- Open WebUI - https://github.com/open-webui/open-webui
- Eric Hartford / Cognitive Computations - Dolphin models
- Meta AI - Llama models
- Alibaba - Qwen models
