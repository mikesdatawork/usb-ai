# LLM Model Specifications
## USB-AI Supported Models

This document details all AI models supported by the USB-AI system.

---

## Model Overview

### Included Models (Default Build)

| Model | Size | Quality | Speed | Best For |
|-------|------|---------|-------|----------|
| Dolphin-LLaMA3 8B | 4.7GB | ★★★★☆ | Fast | General/Uncensored |
| Llama 3.2 8B | 4.7GB | ★★★★☆ | Fast | General Purpose |
| Qwen2.5 14B | 8.9GB | ★★★★★ | Medium | High Quality |

### Optional Models (Extended Build)

| Model | Size | Quality | Speed | Best For |
|-------|------|---------|-------|----------|
| Mistral 7B | 4.1GB | ★★★☆☆ | Very Fast | Quick responses |
| DeepSeek-Coder 6.7B | 3.8GB | ★★★★☆ | Fast | Code generation |
| Llama 3.2 3B | 2.0GB | ★★★☆☆ | Very Fast | Low RAM systems |
| Llama 3.3 70B | 40GB | ★★★★★ | Slow | Maximum quality |

---

## Detailed Model Specifications

### 1. Dolphin-LLaMA3 8B (Primary)

```yaml
model:
  name: "Dolphin-LLaMA3"
  version: "2.9"
  parameters: "8B"
  quantization: "Q4_K_M"
  
  ollama_name: "dolphin-llama3:8b"
  pull_command: "ollama pull dolphin-llama3:8b"
  
  size:
    download: "4.7GB"
    disk: "4.7GB"
    
  requirements:
    ram_minimum: "8GB"
    ram_recommended: "16GB"
    
  performance:
    tokens_per_second: "20-40"
    context_length: "8192"
    
  capabilities:
    - general_conversation
    - creative_writing
    - code_assistance
    - analysis
    - uncensored_responses
    
  source:
    creator: "Eric Hartford / Cognitive Computations"
    base_model: "Meta Llama 3 8B"
    license: "Apache 2.0"
    huggingface: "cognitivecomputations/dolphin-2.9-llama3-8b"
    
  notes: |
    Dolphin is fine-tuned for instruction following with reduced
    safety guardrails. It provides more direct answers but should
    be used responsibly.
```

### 2. Llama 3.2 8B

```yaml
model:
  name: "Llama 3.2"
  version: "3.2"
  parameters: "8B"
  quantization: "Q4_K_M"
  
  ollama_name: "llama3.2:8b"
  pull_command: "ollama pull llama3.2:8b"
  
  size:
    download: "4.7GB"
    disk: "4.7GB"
    
  requirements:
    ram_minimum: "8GB"
    ram_recommended: "16GB"
    
  performance:
    tokens_per_second: "25-45"
    context_length: "128000"
    
  capabilities:
    - general_conversation
    - reasoning
    - code_generation
    - summarization
    - multilingual
    
  source:
    creator: "Meta AI"
    license: "Meta Llama License"
    official: "https://llama.meta.com/"
    
  notes: |
    Latest Llama model with improved reasoning and longer context.
    Well-balanced for general use cases.
```

### 3. Qwen2.5 14B

```yaml
model:
  name: "Qwen2.5"
  version: "2.5"
  parameters: "14B"
  quantization: "Q4_K_M"
  
  ollama_name: "qwen2.5:14b"
  pull_command: "ollama pull qwen2.5:14b"
  
  size:
    download: "8.9GB"
    disk: "8.9GB"
    
  requirements:
    ram_minimum: "16GB"
    ram_recommended: "24GB"
    
  performance:
    tokens_per_second: "15-25"
    context_length: "32768"
    
  capabilities:
    - advanced_reasoning
    - complex_analysis
    - code_generation
    - mathematics
    - multilingual (100+ languages)
    
  source:
    creator: "Alibaba Cloud"
    license: "Tongyi Qianwen License"
    official: "https://qwenlm.github.io/"
    
  notes: |
    Highest quality model in default build. Best for complex tasks
    requiring deeper reasoning. Slower but more accurate.
```

### 4. Mistral 7B (Optional)

```yaml
model:
  name: "Mistral"
  version: "0.3"
  parameters: "7B"
  quantization: "Q4_K_M"
  
  ollama_name: "mistral"
  pull_command: "ollama pull mistral"
  
  size:
    download: "4.1GB"
    disk: "4.1GB"
    
  requirements:
    ram_minimum: "8GB"
    ram_recommended: "12GB"
    
  performance:
    tokens_per_second: "30-50"
    context_length: "32768"
    
  capabilities:
    - fast_responses
    - general_tasks
    - code_assistance
    
  source:
    creator: "Mistral AI"
    license: "Apache 2.0"
    official: "https://mistral.ai/"
```

### 5. DeepSeek-Coder 6.7B (Optional)

```yaml
model:
  name: "DeepSeek-Coder"
  version: "1.0"
  parameters: "6.7B"
  quantization: "Q4_K_M"
  
  ollama_name: "deepseek-coder:6.7b"
  pull_command: "ollama pull deepseek-coder:6.7b"
  
  size:
    download: "3.8GB"
    disk: "3.8GB"
    
  requirements:
    ram_minimum: "8GB"
    ram_recommended: "12GB"
    
  performance:
    tokens_per_second: "25-40"
    context_length: "16384"
    
  capabilities:
    - code_generation
    - code_completion
    - code_explanation
    - debugging
    - multiple_languages
    
  supported_languages:
    - Python
    - JavaScript/TypeScript
    - Java
    - C/C++
    - Go
    - Rust
    - And 80+ more
    
  source:
    creator: "DeepSeek"
    license: "DeepSeek License"
    official: "https://www.deepseek.com/"
    
  notes: |
    Specialized for code tasks. Not suitable for general conversation.
```

---

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| General chat | Dolphin-LLaMA3 8B | Balanced, uncensored |
| Professional work | Llama 3.2 8B | Balanced, safe |
| Complex analysis | Qwen2.5 14B | Best reasoning |
| Quick questions | Mistral 7B | Fastest response |
| Coding tasks | DeepSeek-Coder | Specialized |
| Low RAM (<12GB) | Llama 3.2 3B | Smallest size |

### By Hardware

| RAM Available | Recommended Models |
|---------------|-------------------|
| 8GB | Llama 3.2 3B, Mistral 7B |
| 16GB | Any 7-8B model |
| 24GB+ | Qwen2.5 14B |
| 48GB+ | Llama 3.3 70B |

### By Quality vs Speed

```
Quality                           Speed
   ↑                                ↑
   │  Qwen2.5 14B                   │  Mistral 7B
   │       ↓                        │       ↓
   │  Llama 3.2 8B                  │  Llama 3.2 3B
   │       ↓                        │       ↓
   │  Dolphin-LLaMA3 8B             │  Dolphin-LLaMA3 8B
   │       ↓                        │       ↓
   └──────────────────────────→     └──────────────────────→
```

---

## Model Storage Requirements

### Default Build (128GB USB)

```
Encrypted Container: 100GB
├── Ollama Runtime: 500MB
├── Open WebUI: 1GB
├── Dolphin-LLaMA3 8B: 4.7GB
├── Llama 3.2 8B: 4.7GB
├── Qwen2.5 14B: 8.9GB
├── Data/Config: 1GB
└── Free Space: ~79GB
```

### Extended Build (256GB USB)

```
Encrypted Container: 200GB
├── Ollama Runtime: 500MB
├── Open WebUI: 1GB
├── Dolphin-LLaMA3 8B: 4.7GB
├── Llama 3.2 8B: 4.7GB
├── Qwen2.5 14B: 8.9GB
├── Mistral 7B: 4.1GB
├── DeepSeek-Coder 6.7B: 3.8GB
├── Data/Config: 2GB
└── Free Space: ~170GB
```

---

## Quantization Levels

### Available Quantizations

| Level | Size Reduction | Quality Loss | Use Case |
|-------|---------------|--------------|----------|
| FP16 | None | None | Best quality (large) |
| Q8 | ~50% | Minimal | Good balance |
| Q5_K_M | ~65% | Very low | Recommended |
| Q4_K_M | ~75% | Low | Default choice |
| Q4_0 | ~75% | Moderate | Space constrained |
| Q2_K | ~87% | Noticeable | Extreme space saving |

### Selecting Quantization in Ollama

```bash
# Default (usually Q4_K_M)
ollama pull llama3.2:8b

# Specific quantization
ollama pull llama3.2:8b-q5_k_m
ollama pull llama3.2:8b-q8_0
```

---

## Model Configuration

### Ollama Modelfile Customization

```dockerfile
# Create custom model configuration
# File: Modelfile.dolphin-custom

FROM dolphin-llama3:8b

# Set system prompt
SYSTEM """
You are a helpful AI assistant running locally on the user's device.
All conversations are private and never leave this machine.
Be direct, helpful, and honest in your responses.
"""

# Adjust parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
```

### Creating Custom Model

```bash
# Create from Modelfile
ollama create my-assistant -f Modelfile.dolphin-custom

# Use custom model
ollama run my-assistant
```

---

## Model Testing

### Basic Functionality Test

```bash
# Test model loads
ollama run dolphin-llama3:8b "Say hello"

# Test reasoning
ollama run dolphin-llama3:8b "What is 2+2?"

# Test context
ollama run dolphin-llama3:8b "Remember: my name is Alex. What is my name?"
```

### Performance Benchmark

```bash
# Simple benchmark
time ollama run dolphin-llama3:8b "Write a 100-word story about a cat" --verbose

# Check tokens per second in verbose output
```

### API Test

```bash
# Test via API
curl -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "dolphin-llama3:8b",
    "prompt": "Hello!",
    "stream": false
  }'
```

---

## Troubleshooting Models

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Model won't load | Insufficient RAM | Try smaller model |
| Very slow responses | CPU-only inference | Normal for large models |
| Truncated outputs | Context limit | Reduce input length |
| Repetitive outputs | Temperature too low | Increase to 0.7-0.9 |
| Random outputs | Temperature too high | Reduce to 0.5-0.7 |

### Checking Model Status

```bash
# List installed models
ollama list

# Show model details
ollama show dolphin-llama3:8b

# Check model info
ollama show dolphin-llama3:8b --modelfile
```

### Removing Models

```bash
# Remove a model
ollama rm dolphin-llama3:8b

# Remove all models (caution!)
ollama list | awk 'NR>1 {print $1}' | xargs -I {} ollama rm {}
```

---

## Model Updates

### Checking for Updates

```bash
# Pull latest version
ollama pull dolphin-llama3:8b

# If model is current: "up to date"
# If updated: shows download progress
```

### Update Schedule

| Model | Update Frequency |
|-------|------------------|
| Ollama-hosted | Weekly check recommended |
| Custom models | Manual updates |

---

## Privacy Considerations

All models run 100% locally:
- No data sent to external servers
- No telemetry or tracking
- Chat history stored only on encrypted USB
- Models themselves contain no personal data

---

**For model download commands, see BUILD_PROCESS.md Phase 5.**
