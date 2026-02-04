#!/usr/bin/env python3
"""
start_webui.py

Starts Open WebUI server.
"""

import os
import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.resolve()

    # Add app directory to Python path
    app_dir = script_dir / "app"
    if app_dir.exists():
        sys.path.insert(0, str(app_dir))

    # Set environment
    os.environ["DATA_DIR"] = str(script_dir / "data")
    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    os.environ["WEBUI_AUTH"] = "False"
    os.environ["ENABLE_SIGNUP"] = "False"

    # Import and run
    try:
        from open_webui.main import app
        import uvicorn

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=3000,
            log_level="warning",
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure Open WebUI is installed in the app/ directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
