# Build Process Guide
## USB-AI Complete Build Instructions

This document provides the complete step-by-step process to build a USB-AI system from scratch.

---

## Prerequisites

### Software Requirements

| Software | Version | Purpose | Install Command |
|----------|---------|---------|-----------------|
| Git | 2.30+ | Version control | `brew install git` / `apt install git` |
| Python | 3.10+ | Scripting | `brew install python3` / `apt install python3` |
| Node.js | 18+ | Open WebUI | `brew install node` / `apt install nodejs` |
| Docker | 24+ | Optional WebUI | `brew install docker` |
| VeraCrypt | 1.26+ | Encryption | Manual download |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| USB Drive | 128GB USB 3.0 | 256GB USB 3.1 |
| Build Machine RAM | 16GB | 32GB |
| Build Machine Storage | 50GB free | 100GB free |
| Internet | Required for downloads | Fast connection |

---

## Phase 1: Environment Setup

### 1.1 Clone Repository

```bash
# Clone the main repository
git clone https://github.com/mikesdatawork/usb-ai.git
cd usb-ai

# Verify clone
ls -la
```

### 1.2 Initialize Git Worktrees

```bash
# Create build branch and worktree
git branch build 2>/dev/null || echo "Branch exists"
git worktree add ../usb-ai-build build

# Create release branch and worktree
git branch release 2>/dev/null || echo "Branch exists"
git worktree add ../usb-ai-release release

# Verify worktrees
git worktree list
# Expected output:
# /path/to/usb-ai         <commit> [main]
# /path/to/usb-ai-build   <commit> [build]
# /path/to/usb-ai-release <commit> [release]
```

### 1.3 Create Directory Structure

```bash
# In build worktree
cd ../usb-ai-build

# Create build directories
mkdir -p {downloads,staging,output}
mkdir -p staging/{veracrypt,ollama,webui,models,launchers}
mkdir -p downloads/{macos,windows,linux}

# Verify structure
tree -L 2 .
```

### 1.4 Install Build Dependencies

```bash
# Python dependencies
pip3 install requests tqdm pyyaml

# Verify installations
python3 --version
pip3 list | grep -E "requests|tqdm|pyyaml"
```

---

## Phase 2: Download Components

### 2.1 VeraCrypt Downloads

```bash
# Create download script
cat > download_veracrypt.sh << 'EOF'
#!/bin/bash
set -e

VERACRYPT_VERSION="1.26.14"
DOWNLOAD_DIR="./downloads"

# macOS
echo "Downloading VeraCrypt for macOS..."
curl -L -o "$DOWNLOAD_DIR/macos/VeraCrypt.dmg" \
  "https://launchpad.net/veracrypt/trunk/${VERACRYPT_VERSION}/+download/VeraCrypt_${VERACRYPT_VERSION}.dmg"

# Windows
echo "Downloading VeraCrypt for Windows..."
curl -L -o "$DOWNLOAD_DIR/windows/VeraCrypt_Setup.exe" \
  "https://launchpad.net/veracrypt/trunk/${VERACRYPT_VERSION}/+download/VeraCrypt_Setup_x64_${VERACRYPT_VERSION}.exe"

# Also get portable extractor for Windows
curl -L -o "$DOWNLOAD_DIR/windows/VeraCrypt_Portable.exe" \
  "https://launchpad.net/veracrypt/trunk/${VERACRYPT_VERSION}/+download/VeraCrypt_Portable_${VERACRYPT_VERSION}.exe"

# Linux
echo "Downloading VeraCrypt for Linux..."
curl -L -o "$DOWNLOAD_DIR/linux/veracrypt-setup.tar.bz2" \
  "https://launchpad.net/veracrypt/trunk/${VERACRYPT_VERSION}/+download/veracrypt-${VERACRYPT_VERSION}-setup.tar.bz2"

echo "VeraCrypt downloads complete."
EOF

chmod +x download_veracrypt.sh
./download_veracrypt.sh
```

### 2.2 Ollama Downloads

```bash
# Create Ollama download script
cat > download_ollama.sh << 'EOF'
#!/bin/bash
set -e

DOWNLOAD_DIR="./downloads"

# macOS (Apple Silicon)
echo "Downloading Ollama for macOS..."
curl -L -o "$DOWNLOAD_DIR/macos/Ollama-darwin.zip" \
  "https://ollama.com/download/Ollama-darwin.zip"

# Windows
echo "Downloading Ollama for Windows..."
curl -L -o "$DOWNLOAD_DIR/windows/OllamaSetup.exe" \
  "https://ollama.com/download/OllamaSetup.exe"

# Linux
echo "Downloading Ollama for Linux..."
curl -L -o "$DOWNLOAD_DIR/linux/ollama-linux-amd64" \
  "https://ollama.com/download/ollama-linux-amd64"
chmod +x "$DOWNLOAD_DIR/linux/ollama-linux-amd64"

echo "Ollama downloads complete."
EOF

chmod +x download_ollama.sh
./download_ollama.sh
```

### 2.3 Verify Downloads

```bash
# Check file sizes (minimum expected)
echo "Verifying downloads..."

# VeraCrypt should be 30-50MB each
ls -lh downloads/*/VeraCrypt* 2>/dev/null || echo "VeraCrypt files missing"

# Ollama should be 50-100MB each
ls -lh downloads/*/Ollama* downloads/*/ollama* 2>/dev/null || echo "Ollama files missing"

# Generate checksums for verification
find downloads -type f -exec sha256sum {} \; > downloads/checksums.txt
cat downloads/checksums.txt
```

---

## Phase 3: Prepare USB Drive

### 3.1 Format USB Drive

**WARNING: This will erase all data on the USB drive!**

```bash
# Identify USB drive (be VERY careful here)
# macOS
diskutil list

# Linux
lsblk

# Example: Assuming USB is /dev/disk2 (macOS) or /dev/sdb (Linux)
# DO NOT PROCEED unless you are 100% certain of the device
```

**macOS Formatting:**
```bash
# Replace disk2 with your actual disk
USB_DISK="disk2"

# Unmount
diskutil unmountDisk /dev/$USB_DISK

# Format as exFAT with name USBAI
diskutil eraseDisk exFAT USBAI GPT /dev/$USB_DISK
```

**Linux Formatting:**
```bash
# Replace sdb with your actual device
USB_DEV="/dev/sdb"

# Unmount if mounted
sudo umount ${USB_DEV}* 2>/dev/null || true

# Create partition table and format
sudo parted $USB_DEV --script mklabel gpt
sudo parted $USB_DEV --script mkpart primary 0% 100%
sudo mkfs.exfat -n USBAI ${USB_DEV}1
```

### 3.2 Mount USB Drive

```bash
# macOS - usually auto-mounts to /Volumes/USBAI

# Linux
sudo mkdir -p /mnt/usbai
sudo mount /dev/sdb1 /mnt/usbai

# Set USB path variable
export USB_PATH="/Volumes/USBAI"  # macOS
# export USB_PATH="/mnt/usbai"    # Linux
```

### 3.3 Create USB Directory Structure

```bash
mkdir -p "$USB_PATH"/{veracrypt,launchers}
```

---

## Phase 4: Create Encrypted Container

### 4.1 Install VeraCrypt (Build Machine)

**macOS:**
```bash
# Mount DMG and copy to Applications (or run portable)
hdiutil attach downloads/macos/VeraCrypt.dmg
cp -R "/Volumes/VeraCrypt/VeraCrypt.app" /Applications/
hdiutil detach "/Volumes/VeraCrypt"
```

**Linux:**
```bash
# Extract and install
cd downloads/linux
tar xjf veracrypt-setup.tar.bz2
./veracrypt-*-setup-console-x64
cd ../..
```

### 4.2 Create Encrypted Container

Using VeraCrypt GUI or command line:

**Command Line (Linux/macOS):**
```bash
# Calculate container size (leave 10GB for unencrypted area)
# For 128GB USB: ~100GB container
# For 256GB USB: ~230GB container

CONTAINER_SIZE="100G"  # Adjust based on USB size

# Create container (will prompt for password)
veracrypt --text --create "$USB_PATH/encrypted.vc" \
  --size=$CONTAINER_SIZE \
  --encryption=AES \
  --hash=SHA-512 \
  --filesystem=exFAT \
  --pim=0 \
  --random-source=/dev/urandom

echo "Container created. Remember your password!"
```

**GUI Method:**
1. Open VeraCrypt
2. Click "Create Volume"
3. Select "Create an encrypted file container"
4. Select "Standard VeraCrypt volume"
5. Set location: `USB_PATH/encrypted.vc`
6. Select AES encryption, SHA-512 hash
7. Set size (100GB for 128GB USB)
8. Set strong password
9. Format as exFAT
10. Generate random data by moving mouse
11. Click "Format"

### 4.3 Mount Encrypted Container

```bash
# Create mount point
sudo mkdir -p /mnt/encrypted

# Mount container
veracrypt --text "$USB_PATH/encrypted.vc" /mnt/encrypted

# Set environment variable
export ENCRYPTED_PATH="/mnt/encrypted"

# Create internal structure
mkdir -p "$ENCRYPTED_PATH"/{ollama,models,webui,data,config}
```

---

## Phase 5: Install Ollama (Portable)

### 5.1 Copy Ollama Binaries

```bash
# macOS binary
unzip -o downloads/macos/Ollama-darwin.zip -d "$ENCRYPTED_PATH/ollama/macos/"

# Windows binary (extract from installer or use portable)
# This requires running the installer in extraction mode on Windows
# For now, copy the setup file
cp downloads/windows/OllamaSetup.exe "$ENCRYPTED_PATH/ollama/windows/"

# Linux binary
cp downloads/linux/ollama-linux-amd64 "$ENCRYPTED_PATH/ollama/linux/ollama"
chmod +x "$ENCRYPTED_PATH/ollama/linux/ollama"
```

### 5.2 Configure Ollama Model Path

```bash
# Create config script
cat > "$ENCRYPTED_PATH/config/ollama_env.sh" << 'EOF'
#!/bin/bash
# Ollama environment configuration

# Detect OS and set paths
case "$(uname -s)" in
    Darwin)
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        export OLLAMA_MODELS="${SCRIPT_DIR}/../models"
        export OLLAMA_HOST="127.0.0.1:11434"
        ;;
    Linux)
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        export OLLAMA_MODELS="${SCRIPT_DIR}/../models"
        export OLLAMA_HOST="127.0.0.1:11434"
        ;;
esac

echo "OLLAMA_MODELS set to: $OLLAMA_MODELS"
EOF

chmod +x "$ENCRYPTED_PATH/config/ollama_env.sh"
```

---

## Phase 6: Download AI Models

### 6.1 Set Up Temporary Ollama Environment

```bash
# Source the environment
source "$ENCRYPTED_PATH/config/ollama_env.sh"

# Start Ollama server (in background)
"$ENCRYPTED_PATH/ollama/linux/ollama" serve &
OLLAMA_PID=$!

# Wait for server to start
sleep 5

# Verify server is running
curl -s http://localhost:11434/api/tags || echo "Server not ready"
```

### 6.2 Download Models

```bash
# Primary model: Dolphin-LLaMA3
echo "Downloading Dolphin-LLaMA3 (Primary)..."
"$ENCRYPTED_PATH/ollama/linux/ollama" pull dolphin-llama3:8b

# Secondary model: Llama 3.2
echo "Downloading Llama 3.2..."
"$ENCRYPTED_PATH/ollama/linux/ollama" pull llama3.2:8b

# High-quality model: Qwen2.5 14B (if space permits)
echo "Downloading Qwen2.5 14B..."
"$ENCRYPTED_PATH/ollama/linux/ollama" pull qwen2.5:14b

# List installed models
"$ENCRYPTED_PATH/ollama/linux/ollama" list
```

### 6.3 Create Model Selector Script

```bash
cat > "$ENCRYPTED_PATH/config/select_model.sh" << 'EOF'
#!/bin/bash
# Model Selection Script

echo "========================================"
echo "       USB-AI Model Selection"
echo "========================================"
echo ""
echo "Available Models:"
echo "  1) Dolphin-LLaMA3 8B  (General/Uncensored)"
echo "  2) Llama 3.2 8B       (General Purpose)"
echo "  3) Qwen2.5 14B        (High Quality)"
echo ""
read -p "Select model [1-3]: " choice

case $choice in
    1) MODEL="dolphin-llama3:8b" ;;
    2) MODEL="llama3.2:8b" ;;
    3) MODEL="qwen2.5:14b" ;;
    *) MODEL="dolphin-llama3:8b" ;;
esac

echo "Selected: $MODEL"
echo "$MODEL" > "$ENCRYPTED_PATH/config/default_model.txt"
export SELECTED_MODEL="$MODEL"
EOF

chmod +x "$ENCRYPTED_PATH/config/select_model.sh"

# Set default model
echo "dolphin-llama3:8b" > "$ENCRYPTED_PATH/config/default_model.txt"
```

### 6.4 Stop Temporary Ollama Server

```bash
kill $OLLAMA_PID 2>/dev/null || true
```

---

## Phase 7: Install Open WebUI

### 7.1 Download Open WebUI

```bash
# Method 1: Using pip (recommended for portability)
pip3 install open-webui --target "$ENCRYPTED_PATH/webui/python"

# Method 2: Download release binary
# Check: https://github.com/open-webui/open-webui/releases

# Create start script
cat > "$ENCRYPTED_PATH/webui/start_webui.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Add to Python path
export PYTHONPATH="$SCRIPT_DIR/python:$PYTHONPATH"

# Set data directory
export DATA_DIR="$SCRIPT_DIR/../data/webui"
mkdir -p "$DATA_DIR"

# Set Ollama URL
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

# Start Open WebUI
python3 -m open_webui.main --port 3000 --host 127.0.0.1

EOF

chmod +x "$ENCRYPTED_PATH/webui/start_webui.sh"
```

### 7.2 Configure Open WebUI

```bash
# Create configuration
mkdir -p "$ENCRYPTED_PATH/data/webui"

cat > "$ENCRYPTED_PATH/config/webui_config.json" << 'EOF'
{
  "ollama_base_url": "http://127.0.0.1:11434",
  "enable_signup": false,
  "default_user_role": "admin",
  "enable_community_sharing": false,
  "enable_message_rating": true,
  "enable_model_filter": false
}
EOF
```

---

## Phase 8: Create Launcher Scripts

### 8.1 macOS Launcher

```bash
cat > "$USB_PATH/launchers/start_macos.command" << 'EOF'
#!/bin/bash
# USB-AI Launcher for macOS

echo "========================================"
echo "           USB-AI Starting"
echo "========================================"

# Get USB mount point
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USB_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if VeraCrypt is installed
if ! command -v veracrypt &> /dev/null; then
    echo "VeraCrypt not found. Please install from:"
    echo "  $USB_ROOT/veracrypt/"
    echo ""
    echo "Opening VeraCrypt folder..."
    open "$USB_ROOT/veracrypt/"
    exit 1
fi

# Mount encrypted container
MOUNT_POINT="/Volumes/USBAI_ENCRYPTED"
CONTAINER="$USB_ROOT/encrypted.vc"

if [ ! -d "$MOUNT_POINT" ]; then
    echo "Mounting encrypted container..."
    echo "Enter your encryption password:"
    veracrypt --text "$CONTAINER" "$MOUNT_POINT"
    
    if [ $? -ne 0 ]; then
        echo "Failed to mount encrypted container."
        exit 1
    fi
fi

# Set environment
export OLLAMA_MODELS="$MOUNT_POINT/models"
export OLLAMA_HOST="127.0.0.1:11434"

# Start Ollama
echo "Starting Ollama server..."
"$MOUNT_POINT/ollama/macos/Ollama.app/Contents/MacOS/Ollama" serve &
sleep 5

# Model selection
source "$MOUNT_POINT/config/select_model.sh"

# Start Open WebUI
echo "Starting Open WebUI..."
"$MOUNT_POINT/webui/start_webui.sh" &
sleep 3

# Open browser
echo "Opening browser..."
open "http://localhost:3000"

echo ""
echo "========================================"
echo "USB-AI is running!"
echo "Chat at: http://localhost:3000"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
wait
EOF

chmod +x "$USB_PATH/launchers/start_macos.command"
```

### 8.2 Windows Launcher

```bash
cat > "$USB_PATH/launchers/start_windows.bat" << 'EOF'
@echo off
echo ========================================
echo            USB-AI Starting
echo ========================================

set SCRIPT_DIR=%~dp0
set USB_ROOT=%SCRIPT_DIR%..

:: Check if VeraCrypt is installed
where veracrypt >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo VeraCrypt not found. Please install from:
    echo   %USB_ROOT%\veracrypt\
    start "" "%USB_ROOT%\veracrypt\"
    pause
    exit /b 1
)

:: Mount encrypted container
set MOUNT_POINT=V:
set CONTAINER=%USB_ROOT%\encrypted.vc

echo Mounting encrypted container...
echo Enter your encryption password:
veracrypt /v "%CONTAINER%" /l V /q

if %ERRORLEVEL% neq 0 (
    echo Failed to mount encrypted container.
    pause
    exit /b 1
)

:: Set environment
set OLLAMA_MODELS=%MOUNT_POINT%\models
set OLLAMA_HOST=127.0.0.1:11434

:: Start Ollama
echo Starting Ollama server...
start "" "%MOUNT_POINT%\ollama\windows\ollama.exe" serve

timeout /t 5 /nobreak

:: Start Open WebUI
echo Starting Open WebUI...
start "" python -m open_webui.main --port 3000 --host 127.0.0.1

timeout /t 3 /nobreak

:: Open browser
echo Opening browser...
start "" http://localhost:3000

echo.
echo ========================================
echo USB-AI is running!
echo Chat at: http://localhost:3000
echo ========================================
echo.
echo Close this window to stop services

pause
EOF
```

### 8.3 Linux Launcher

```bash
cat > "$USB_PATH/launchers/start_linux.sh" << 'EOF'
#!/bin/bash
# USB-AI Launcher for Linux

echo "========================================"
echo "           USB-AI Starting"
echo "========================================"

# Get USB mount point
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USB_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if VeraCrypt is installed
if ! command -v veracrypt &> /dev/null; then
    echo "VeraCrypt not found. Please install from:"
    echo "  $USB_ROOT/veracrypt/"
    exit 1
fi

# Mount encrypted container
MOUNT_POINT="/mnt/usbai_encrypted"
CONTAINER="$USB_ROOT/encrypted.vc"

if [ ! -d "$MOUNT_POINT" ]; then
    sudo mkdir -p "$MOUNT_POINT"
fi

if ! mountpoint -q "$MOUNT_POINT"; then
    echo "Mounting encrypted container..."
    echo "Enter your encryption password:"
    sudo veracrypt --text "$CONTAINER" "$MOUNT_POINT"
    
    if [ $? -ne 0 ]; then
        echo "Failed to mount encrypted container."
        exit 1
    fi
fi

# Set environment
export OLLAMA_MODELS="$MOUNT_POINT/models"
export OLLAMA_HOST="127.0.0.1:11434"

# Start Ollama
echo "Starting Ollama server..."
"$MOUNT_POINT/ollama/linux/ollama" serve &
OLLAMA_PID=$!
sleep 5

# Model selection
source "$MOUNT_POINT/config/select_model.sh"

# Start Open WebUI
echo "Starting Open WebUI..."
"$MOUNT_POINT/webui/start_webui.sh" &
WEBUI_PID=$!
sleep 3

# Open browser
echo "Opening browser..."
xdg-open "http://localhost:3000" 2>/dev/null || \
  sensible-browser "http://localhost:3000" 2>/dev/null || \
  echo "Please open http://localhost:3000 in your browser"

echo ""
echo "========================================"
echo "USB-AI is running!"
echo "Chat at: http://localhost:3000"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup on exit
cleanup() {
    echo "Stopping services..."
    kill $OLLAMA_PID $WEBUI_PID 2>/dev/null
    sudo veracrypt -d "$MOUNT_POINT" 2>/dev/null
    echo "Stopped."
}
trap cleanup EXIT INT TERM

# Wait
wait
EOF

chmod +x "$USB_PATH/launchers/start_linux.sh"
```

### 8.4 Shutdown Script

```bash
cat > "$USB_PATH/launchers/stop_all.sh" << 'EOF'
#!/bin/bash
# USB-AI Shutdown Script

echo "Stopping USB-AI services..."

# Stop Ollama
pkill -f "ollama serve" 2>/dev/null
echo "  Ollama stopped."

# Stop Open WebUI
pkill -f "open_webui" 2>/dev/null
echo "  Open WebUI stopped."

# Unmount encrypted container
case "$(uname -s)" in
    Darwin)
        veracrypt -d /Volumes/USBAI_ENCRYPTED 2>/dev/null
        ;;
    Linux)
        sudo veracrypt -d /mnt/usbai_encrypted 2>/dev/null
        ;;
esac
echo "  Encrypted container unmounted."

echo ""
echo "USB-AI shutdown complete."
echo "It is now safe to remove the USB drive."
EOF

chmod +x "$USB_PATH/launchers/stop_all.sh"
```

---

## Phase 9: Copy VeraCrypt Portable

```bash
# Copy VeraCrypt portable binaries to USB
cp -R downloads/macos/VeraCrypt.dmg "$USB_PATH/veracrypt/"
cp -R downloads/windows/VeraCrypt_Portable.exe "$USB_PATH/veracrypt/"
cp -R downloads/linux/veracrypt-setup.tar.bz2 "$USB_PATH/veracrypt/"

# Create README for VeraCrypt
cat > "$USB_PATH/veracrypt/README.txt" << 'EOF'
VeraCrypt Installation
======================

If VeraCrypt is not installed on this computer, install it using
the appropriate file for your operating system:

macOS:    VeraCrypt.dmg
Windows:  VeraCrypt_Portable.exe (run as administrator)
Linux:    veracrypt-setup.tar.bz2 (extract and run installer)

After installation, run the launcher script from the 'launchers' folder.
EOF
```

---

## Phase 10: Create End-User Documentation

```bash
cat > "$USB_PATH/README.txt" << 'EOF'
========================================
        USB-AI Quick Start Guide
========================================

WHAT IS THIS?
-------------
This USB drive contains a fully offline, encrypted AI assistant.
No internet required. No cloud. Complete privacy.


FIRST TIME SETUP
----------------
1. Install VeraCrypt (if not already installed)
   - See the 'veracrypt' folder for installers

2. Run the launcher for your operating system:
   - macOS:   launchers/start_macos.command
   - Windows: launchers/start_windows.bat
   - Linux:   launchers/start_linux.sh

3. Enter your encryption password when prompted

4. Wait for the AI to start (30-60 seconds)

5. Your browser will open to http://localhost:3000


USING THE AI
------------
- Type your message in the chat box
- Press Enter or click Send
- The AI will respond (may take a few seconds)
- You can change AI models in the settings


SHUTDOWN
--------
1. Run launchers/stop_all.sh (or close the launcher window)
2. Wait for "safe to remove" message
3. Eject the USB drive safely


INCLUDED AI MODELS
------------------
- Dolphin-LLaMA3 8B  (Default - General/Uncensored)
- Llama 3.2 8B       (General Purpose)
- Qwen2.5 14B        (High Quality - Slower)


SYSTEM REQUIREMENTS
-------------------
- 16GB RAM minimum (32GB recommended)
- macOS 12+, Windows 10+, or Ubuntu 22.04+
- USB 3.0 port (3.1 recommended for speed)


TROUBLESHOOTING
---------------
Q: "VeraCrypt not found"
A: Install VeraCrypt from the 'veracrypt' folder

Q: AI is very slow
A: Try a smaller model, or ensure 16GB+ RAM available

Q: Browser doesn't open
A: Manually go to http://localhost:3000

Q: Forgot encryption password
A: Data cannot be recovered. This is a security feature.


SUPPORT
-------
GitHub: https://github.com/mikesdatawork/usb-ai


========================================
     Your AI. Your Data. Your Privacy.
========================================
EOF
```

---

## Phase 11: Final Validation

```bash
# Create validation script
cat > validate_build.sh << 'EOF'
#!/bin/bash
echo "Validating USB-AI Build..."
echo ""

USB_PATH="${1:-/Volumes/USBAI}"
ERRORS=0

# Check USB structure
check_file() {
    if [ -e "$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [MISSING] $1"
        ((ERRORS++))
    fi
}

echo "Checking USB structure..."
check_file "$USB_PATH/encrypted.vc"
check_file "$USB_PATH/veracrypt/README.txt"
check_file "$USB_PATH/launchers/start_macos.command"
check_file "$USB_PATH/launchers/start_windows.bat"
check_file "$USB_PATH/launchers/start_linux.sh"
check_file "$USB_PATH/launchers/stop_all.sh"
check_file "$USB_PATH/README.txt"

echo ""
echo "Checking encrypted container size..."
SIZE=$(du -h "$USB_PATH/encrypted.vc" | cut -f1)
echo "  Container size: $SIZE"

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "=========================================="
    echo "  BUILD VALIDATION: PASSED"
    echo "=========================================="
else
    echo "=========================================="
    echo "  BUILD VALIDATION: FAILED ($ERRORS errors)"
    echo "=========================================="
fi
EOF

chmod +x validate_build.sh
./validate_build.sh "$USB_PATH"
```

---

## Phase 12: Unmount and Finalize

```bash
# Unmount encrypted container
veracrypt -d "$ENCRYPTED_PATH"

# Safely eject USB
case "$(uname -s)" in
    Darwin)
        diskutil eject "$USB_PATH"
        ;;
    Linux)
        sudo umount "$USB_PATH"
        ;;
esac

echo ""
echo "========================================"
echo "       USB-AI BUILD COMPLETE"
echo "========================================"
echo ""
echo "The USB drive is ready for use."
echo "Test on each target operating system."
```

---

## Build Checklist

| Step | Status |
|------|--------|
| [ ] Repository cloned | |
| [ ] Worktrees created | |
| [ ] VeraCrypt downloaded | |
| [ ] Ollama downloaded | |
| [ ] USB formatted | |
| [ ] Encrypted container created | |
| [ ] Ollama installed in container | |
| [ ] Models downloaded | |
| [ ] Open WebUI installed | |
| [ ] Launchers created | |
| [ ] Documentation added | |
| [ ] Validation passed | |

---

## Estimated Build Times

| Phase | Duration |
|-------|----------|
| Setup | 5 min |
| Downloads | 15 min |
| USB Prep | 5 min |
| Encryption | 15 min |
| Model Downloads | 30-60 min |
| UI Setup | 10 min |
| Launchers | 5 min |
| Validation | 5 min |
| **Total** | **90-120 min** |

---

**Build process complete. See TESTING.md for validation procedures.**
