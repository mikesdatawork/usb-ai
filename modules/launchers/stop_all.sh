#!/bin/bash
# ==========================================================
#                    USB-AI Shutdown
#                   Cross-Platform
# ==========================================================

echo ""
echo "========================================"
echo "          USB-AI Stopping"
echo "========================================"
echo ""

# Stop Open WebUI
echo "Stopping Open WebUI..."
pkill -f "open_webui" 2>/dev/null && echo "  Open WebUI stopped." || echo "  Open WebUI not running."

# Stop Ollama
echo "Stopping Ollama..."
pkill -f "ollama serve" 2>/dev/null && echo "  Ollama stopped." || echo "  Ollama not running."

# Alternative method using port
if command -v lsof &> /dev/null; then
    # Kill anything on port 3000
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    # Kill anything on port 11434
    lsof -ti:11434 | xargs kill -9 2>/dev/null || true
fi

echo ""
echo "========================================"
echo "         USB-AI Stopped"
echo ""
echo "   Safe to remove USB drive."
echo "========================================"
echo ""
