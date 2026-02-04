#!/bin/bash
# ==========================================================
#                    USB-AI Launcher
#                      macOS
# ==========================================================

set -e

echo ""
echo "========================================"
echo "           USB-AI Starting"
echo "========================================"
echo ""

# Get script directory (USB mount point)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USB_ROOT="$(dirname "$SCRIPT_DIR")"
MODULES_DIR="$USB_ROOT"

echo "USB Root: $USB_ROOT"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    OLLAMA_BIN="$MODULES_DIR/ollama-portable/bin/darwin-arm64/ollama"
else
    OLLAMA_BIN="$MODULES_DIR/ollama-portable/bin/darwin-amd64/ollama"
fi

# Check Ollama binary exists
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found at: $OLLAMA_BIN"
    echo "Please run the build scripts first."
    read -p "Press Enter to exit..."
    exit 1
fi

# Set environment
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="$MODULES_DIR/models"

echo "Ollama binary: $OLLAMA_BIN"
echo "Models path: $OLLAMA_MODELS"
echo ""

# Check if Ollama is already running
if lsof -i :11434 >/dev/null 2>&1; then
    echo "Ollama already running on port 11434"
else
    echo "Starting Ollama server..."
    "$OLLAMA_BIN" serve &
    OLLAMA_PID=$!

    # Wait for Ollama to start
    echo "Waiting for Ollama to initialize..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            echo "Ollama is ready!"
            break
        fi
        sleep 1
    done
fi

echo ""

# Start Open WebUI
WEBUI_DIR="$MODULES_DIR/webui-portable"
export PYTHONPATH="$WEBUI_DIR/app:$PYTHONPATH"
export DATA_DIR="$WEBUI_DIR/data"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

echo "Starting Open WebUI..."

# Check if WebUI is already running
if lsof -i :3000 >/dev/null 2>&1; then
    echo "Open WebUI already running on port 3000"
else
    cd "$WEBUI_DIR"
    python3 -m open_webui.main --port 3000 --host 127.0.0.1 &
    WEBUI_PID=$!

    # Wait for WebUI to start
    echo "Waiting for WebUI to initialize..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:3000 >/dev/null 2>&1; then
            echo "Open WebUI is ready!"
            break
        fi
        sleep 1
    done
fi

echo ""

# Open browser
echo "Opening browser..."
open "http://127.0.0.1:3000"

echo ""
echo "========================================"
echo "         USB-AI is running!"
echo ""
echo "   Chat: http://127.0.0.1:3000"
echo ""
echo "   Press Ctrl+C to stop"
echo "========================================"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping USB-AI..."

    # Stop WebUI
    if [ ! -z "$WEBUI_PID" ]; then
        kill $WEBUI_PID 2>/dev/null || true
    fi
    pkill -f "open_webui" 2>/dev/null || true

    # Stop Ollama
    if [ ! -z "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    pkill -f "ollama serve" 2>/dev/null || true

    echo "USB-AI stopped."
    echo "Safe to remove USB drive."
}

trap cleanup EXIT INT TERM

# Keep running
wait
