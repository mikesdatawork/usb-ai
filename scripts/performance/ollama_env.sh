#!/bin/bash
# Ollama Optimized Environment Variables
# Generated: 2026-02-06T23:50:27.309355
# Preset: speed
# Hardware: AMD Ryzen 5 5560U with Radeon Graphics (6c), 13GB RAM

export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_MAX_LOADED_MODELS="2"
export OLLAMA_KEEP_ALIVE="30m"
export OLLAMA_FLASH_ATTENTION="0"
export OLLAMA_KV_CACHE_TYPE="q8_0"
export OLLAMA_NUM_THREADS="5"

echo "Ollama environment configured for maximum speed"
echo "  Preset: speed"
echo "  Threads: 5"
echo "  GPU Layers: 0"
echo "  Flash Attention: False"