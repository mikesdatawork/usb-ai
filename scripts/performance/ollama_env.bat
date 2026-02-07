@echo off
REM Ollama Optimized Environment Variables
REM Generated: 2026-02-06T23:50:27.309565
REM Preset: speed
REM Hardware: AMD Ryzen 5 5560U with Radeon Graphics (6c), 13GB RAM

set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_KEEP_ALIVE=30m
set OLLAMA_FLASH_ATTENTION=0
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_NUM_THREADS=5

echo Ollama environment configured for maximum speed
echo   Preset: speed
echo   Threads: 5
echo   GPU Layers: 0
echo   Flash Attention: False