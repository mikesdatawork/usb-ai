#!/bin/bash
# USB-AI Optimized Ollama Configuration
# Generated: 2026-02-07

# Keep models loaded for 24 hours (eliminates cold start)
export OLLAMA_KEEP_ALIVE="24h"

# Parallel request handling
export OLLAMA_NUM_PARALLEL=4

# Keep up to 2 models loaded simultaneously
export OLLAMA_MAX_LOADED_MODELS=2

# Enable flash attention for speed
export OLLAMA_FLASH_ATTENTION=1

# CPU threads (adjust based on your CPU)
export OLLAMA_NUM_THREADS=8

# KV cache quantization (q8_0 = good speed/quality balance)
export OLLAMA_KV_CACHE_TYPE="q8_0"

# Context length (smaller = faster)
export OLLAMA_NUM_CTX=2048

# GPU settings (if available)
export OLLAMA_GPU_OVERHEAD=0

echo "Ollama optimizations applied:"
echo "  KEEP_ALIVE=$OLLAMA_KEEP_ALIVE"
echo "  NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
echo "  MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"
echo "  FLASH_ATTENTION=$OLLAMA_FLASH_ATTENTION"
echo "  NUM_THREADS=$OLLAMA_NUM_THREADS"
echo "  KV_CACHE_TYPE=$OLLAMA_KV_CACHE_TYPE"
echo "  NUM_CTX=$OLLAMA_NUM_CTX"
