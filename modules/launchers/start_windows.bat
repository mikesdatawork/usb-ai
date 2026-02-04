@echo off
REM ==========================================================
REM                    USB-AI Launcher
REM                      Windows
REM ==========================================================

title USB-AI

echo.
echo ========================================
echo            USB-AI Starting
echo ========================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set USB_ROOT=%SCRIPT_DIR%..
set MODULES_DIR=%USB_ROOT%

echo USB Root: %USB_ROOT%

REM Set paths
set OLLAMA_BIN=%MODULES_DIR%\ollama-portable\bin\windows-amd64\ollama.exe
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_MODELS=%MODULES_DIR%\models

echo Ollama binary: %OLLAMA_BIN%
echo Models path: %OLLAMA_MODELS%
echo.

REM Check Ollama binary exists
if not exist "%OLLAMA_BIN%" (
    echo ERROR: Ollama binary not found at: %OLLAMA_BIN%
    echo Please run the build scripts first.
    pause
    exit /b 1
)

REM Start Ollama server
echo Starting Ollama server...
start /B "" "%OLLAMA_BIN%" serve

REM Wait for Ollama to start
echo Waiting for Ollama to initialize...
:wait_ollama
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 goto wait_ollama
echo Ollama is ready!

echo.

REM Start Open WebUI
set WEBUI_DIR=%MODULES_DIR%\webui-portable
set PYTHONPATH=%WEBUI_DIR%\app;%PYTHONPATH%
set DATA_DIR=%WEBUI_DIR%\data
set OLLAMA_BASE_URL=http://127.0.0.1:11434

echo Starting Open WebUI...
cd /d "%WEBUI_DIR%"
start /B "" python -m open_webui.main --port 3000 --host 127.0.0.1

REM Wait for WebUI to start
echo Waiting for WebUI to initialize...
:wait_webui
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:3000 >nul 2>&1
if errorlevel 1 goto wait_webui
echo Open WebUI is ready!

echo.

REM Open browser
echo Opening browser...
start "" http://127.0.0.1:3000

echo.
echo ========================================
echo          USB-AI is running!
echo.
echo    Chat: http://127.0.0.1:3000
echo.
echo    Close this window to stop
echo ========================================
echo.

REM Keep window open
pause
