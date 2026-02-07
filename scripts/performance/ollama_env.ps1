# Ollama Optimized Environment Variables
# Generated: 2026-02-06T23:50:27.309726
# Preset: speed
# Hardware: AMD Ryzen 5 5560U with Radeon Graphics (6c), 13GB RAM

$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "2"
$env:OLLAMA_KEEP_ALIVE = "30m"
$env:OLLAMA_FLASH_ATTENTION = "0"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_NUM_THREADS = "5"

Write-Host "Ollama environment configured for maximum speed"
Write-Host "  Preset: speed"
Write-Host "  Threads: 5"
Write-Host "  GPU Layers: 0"
Write-Host "  Flash Attention: False"