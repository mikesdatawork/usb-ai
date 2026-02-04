# Component Style Guides
## USB-AI Opinionated Standards

Each module follows strict conventions. This ensures consistency and independent development.

---

## Modular Design Principles

### Core Tenets

1. **Independence** - Each module works standalone
2. **Loose coupling** - Modules communicate via defined interfaces
3. **Single responsibility** - One module, one purpose
4. **Replaceable** - Swap any module without breaking others

### Module Boundaries

```
usb-ai/
├── modules/
│   ├── ollama-portable/    # LLM runtime (independent)
│   ├── webui-portable/     # Chat interface (independent)
│   ├── models/             # AI models (independent)
│   ├── launchers/          # OS starters (independent)
│   └── config/             # User settings (shared)
└── scripts/                # Build automation
```

### Interface Contracts

Modules communicate through:
- Environment variables
- Configuration files (JSON/YAML)
- HTTP APIs (localhost only)
- File system paths

No direct imports between modules.

---

## Module: ollama-portable

### Purpose
Portable LLM inference runtime.

### Directory Structure

```
ollama-portable/
├── bin/
│   ├── darwin-arm64/
│   │   └── ollama
│   ├── darwin-amd64/
│   │   └── ollama
│   ├── linux-amd64/
│   │   └── ollama
│   └── windows-amd64/
│       └── ollama.exe
├── config/
│   └── ollama.json
└── README.md
```

### Configuration Schema

```json
{
  "host": "127.0.0.1",
  "port": 11434,
  "models_path": "../models",
  "gpu_enabled": false,
  "num_parallel": 1,
  "max_loaded_models": 1
}
```

### Style Rules

| Aspect | Rule |
|--------|------|
| Binary naming | `ollama` (no version suffix) |
| Config format | JSON only |
| Port | Always 11434 |
| Binding | localhost only, never 0.0.0.0 |
| Logging | Minimal, errors only by default |

### Interface

```
Input:  HTTP API on localhost:11434
Output: JSON responses per Ollama API spec
Config: ollama.json in config/
```

### Update Independence

To update Ollama:
1. Replace binaries in `bin/`
2. No other modules affected
3. API contract remains stable

---

## Module: webui-portable

### Purpose
Browser-based chat interface for Ollama using Flask + HTMX.

### Directory Structure

```
webui-portable/
├── app/
│   └── (Flask dependencies)
├── chat_ui.py           # Main chat application
├── static/
│   └── css/
│       └── custom-theme.css
└── README.md
```

### Style Rules

| Aspect | Rule |
|--------|------|
| Framework | Flask + HTMX (Python) |
| Port | Always 3000 |
| Session | Server-side Flask session |
| Theme | Dark flat with #ffa222 accent |
| Auth | None (local use only) |

### Theme Requirements

```css
/* Built into chat_ui.py */
font-family: Arial, sans-serif;
background: #1a1a1a;
accent: #ffa222;
```

### Interface

```
Input:  Browser on localhost:3000
Output: HTML with HTMX dynamic updates
Depends: ollama-portable (HTTP API)
```

### Update Independence

To update chat UI:
1. Replace `chat_ui.py`
2. Update Flask in `app/` if needed
3. No other modules affected

---

## Module: models

### Purpose
Store and manage LLM model files.

### Directory Structure

```
models/
├── manifests/
│   └── registry.ollama.ai/
│       └── library/
│           ├── dolphin-llama3/
│           ├── llama3.2/
│           └── qwen2.5/
├── blobs/
│   └── sha256-*/
├── config/
│   └── models.json
└── README.md
```

### Configuration Schema

```json
{
  "default_model": "dolphin-llama3:8b",
  "available_models": [
    {
      "name": "dolphin-llama3:8b",
      "size_gb": 4.7,
      "parameters": "8B",
      "use_case": "general"
    },
    {
      "name": "llama3.2:8b",
      "size_gb": 4.7,
      "parameters": "8B",
      "use_case": "general"
    },
    {
      "name": "qwen2.5:14b",
      "size_gb": 8.9,
      "parameters": "14B",
      "use_case": "complex"
    }
  ]
}
```

### Style Rules

| Aspect | Rule |
|--------|------|
| Format | GGUF only (Ollama native) |
| Quantization | Q4_K_M default |
| Naming | Ollama library names |
| Metadata | models.json required |

### Interface

```
Input:  File path from ollama-portable
Output: Model files for inference
Config: models.json
```

### Update Independence

To add/remove models:
1. Use `ollama pull` or `ollama rm`
2. Update models.json
3. No other modules affected

---

## Module: launchers

### Purpose
Cross-platform startup scripts.

### Directory Structure

```
launchers/
├── start.py           # Main cross-platform launcher
├── stop.py            # Graceful shutdown
├── status.py          # Health check
├── select_model.py    # Model switcher
├── config/
│   └── launcher.json
└── README.md
```

### Configuration Schema

```json
{
  "auto_open_browser": true,
  "startup_timeout_seconds": 60,
  "health_check_interval": 5,
  "ollama_module": "../ollama-portable",
  "webui_module": "../webui-portable",
  "models_module": "../models"
}
```

### Style Rules

| Aspect | Rule |
|--------|------|
| Language | Python 3.10+ only |
| Dependencies | Standard library preferred |
| OS detection | platform module |
| Path handling | pathlib only |
| Output | Minimal, status updates only |

### Python Standards

```python
#!/usr/bin/env python3
"""
Module docstring. One line description.
"""

import sys
from pathlib import Path

# Constants at top
DEFAULT_TIMEOUT = 60
OLLAMA_PORT = 11434
WEBUI_PORT = 3000

def main():
    """Entry point."""
    pass

if __name__ == "__main__":
    main()
```

### Interface

```
Input:  User execution
Output: Running services
Config: launcher.json
Depends: All other modules (orchestration)
```

### Update Independence

To modify startup behavior:
1. Edit Python scripts
2. Update launcher.json
3. No other modules affected

---

## Module: config

### Purpose
Shared configuration and user preferences.

### Directory Structure

```
config/
├── system.json        # System-wide settings
├── user.json          # User preferences
├── paths.json         # Module paths
└── README.md
```

### system.json Schema

```json
{
  "version": "1.0.0",
  "created": "2026-02-03",
  "modules": {
    "ollama": "../ollama-portable",
    "webui": "../webui-portable",
    "models": "../models",
    "launchers": "../launchers"
  }
}
```

### user.json Schema

```json
{
  "default_model": "dolphin-llama3:8b",
  "auto_start_browser": true,
  "theme": "dark",
  "language": "en"
}
```

### Style Rules

| Aspect | Rule |
|--------|------|
| Format | JSON only |
| Encoding | UTF-8 |
| Naming | lowercase with underscores |
| Comments | Not in JSON (use README) |

### Interface

```
Input:  Read by all modules
Output: Configuration values
Depends: None (base module)
```

---

## Scripts Module

### Purpose
Build and maintenance automation.

### Directory Structure

```
scripts/
├── build/
│   ├── s001_setup_environment.py
│   ├── s002_download_ollama.py
│   ├── s003_download_models.py
│   ├── s004_setup_webui.py
│   └── s005_apply_theme.py
├── utils/
│   ├── downloader.py
│   ├── platform_detect.py
│   └── config_loader.py
├── requirements.txt
└── README.md
```

### Naming Convention

```
s{NNN}_{action}_{target}.py

Examples:
s001_setup_environment.py
s002_download_ollama.py
s003_download_models.py
```

### Python Standards

```python
#!/usr/bin/env python3
"""
s001_setup_environment.py

Sets up the build environment for USB-AI.
Run this first before any other build scripts.
"""

import sys
import logging
from pathlib import Path

# Version for tracking
__version__ = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def main() -> int:
    """
    Main entry point.
    
    Returns:
        0 on success, non-zero on failure
    """
    log.info("Starting environment setup")
    # Implementation
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Style Rules

| Aspect | Rule |
|--------|------|
| Shebang | `#!/usr/bin/env python3` |
| Docstrings | Required for modules and functions |
| Type hints | Encouraged |
| Return codes | 0 = success, non-zero = failure |
| Logging | Use logging module, not print |
| Paths | pathlib.Path only |

---

## Cross-Module Communication

### Environment Variables

```bash
# Set by launchers, read by modules
USB_AI_ROOT=/path/to/usb-ai
USB_AI_CONFIG=/path/to/config
USB_AI_MODELS=/path/to/models
OLLAMA_HOST=127.0.0.1
OLLAMA_MODELS=/path/to/models
```

### Health Check Protocol

Each module exposes status:

```python
# Standard health check response
{
    "module": "ollama-portable",
    "status": "running",  # running | stopped | error
    "version": "0.5.0",
    "uptime_seconds": 120
}
```

### Startup Order

```
1. config/          (load configuration)
2. ollama-portable/ (start LLM server)
3. models/          (verify models available)
4. webui-portable/  (start web interface)
5. launchers/       (open browser)
```

### Shutdown Order

```
1. webui-portable/  (stop web server)
2. ollama-portable/ (stop LLM server)
3. (config and models are stateless)
```

---

## Version Compatibility

### Module Versions

Track independently:

```json
{
  "ollama-portable": "0.5.0",
  "webui-portable": "0.4.5",
  "models": "1.0.0",
  "launchers": "1.0.0",
  "config": "1.0.0"
}
```

### API Contracts

When updating modules:
1. Check API compatibility
2. Update interface version
3. Document breaking changes

---

**Each module is a self-contained unit. Work on one without touching others.**
