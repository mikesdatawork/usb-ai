#!/usr/bin/env python3
"""
s005_apply_theme.py

Applies the custom dark flat theme to Open WebUI.
Theme CSS is copied to the appropriate location.
"""

import logging
import platform
import shutil
import sys
from pathlib import Path

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

THEME_CSS = '''/* USB-AI Dark Flat Theme */
/* Version: 1.0.0 */
/* Minimal. Dark. Flat. Functional. */

:root {
  /* Base colors */
  --bg-primary: #1a1a1a;
  --bg-secondary: #242424;
  --bg-tertiary: #2d2d2d;
  --bg-input: #333333;

  /* Text colors */
  --text-primary: #e5e5e5;
  --text-secondary: #a0a0a0;
  --text-muted: #666666;

  /* Accent color - Orange */
  --accent: #ffa222;
  --accent-hover: #ffb44d;
  --accent-dim: #cc8118;

  /* Borders */
  --border: #3d3d3d;
  --border-subtle: #2a2a2a;

  /* Semantic colors */
  --success: #4ade80;
  --warning: #ffa222;
  --error: #f87171;
  --info: #60a5fa;
}

/* Reset all bold to normal weight */
* {
  font-weight: 400 !important;
}

/* Base body styles */
body {
  font-family: Arial, Helvetica, sans-serif !important;
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-size: 15px;
  line-height: 1.6;
}

/* Remove all bold styling */
strong, b, .font-bold, .font-semibold, .font-medium {
  font-weight: 400 !important;
}

/* Headers - Orange accent, no bold */
h1, h2, h3, h4, h5, h6,
.text-xl, .text-2xl, .text-3xl {
  color: #ffa222 !important;
  font-weight: 400 !important;
}

h1 { font-size: 28px; margin: 0 0 16px 0; }
h2 { font-size: 22px; margin: 24px 0 12px 0; }
h3 { font-size: 18px; margin: 20px 0 10px 0; }
h4 { font-size: 16px; margin: 16px 0 8px 0; }

/* List markers - Orange accent */
ul, ol {
  padding-left: 0;
  margin: 12px 0;
}

li {
  margin: 6px 0;
}

ul li::marker,
ol li::marker {
  color: #ffa222 !important;
}

/* Links */
a {
  color: #ffa222;
  text-decoration: none;
}

a:hover {
  color: #ffb44d;
  text-decoration: underline;
}

/* Buttons - Primary */
button[class*="primary"],
.btn-primary,
button[class*="bg-blue"],
button[class*="bg-primary"] {
  background: #ffa222 !important;
  color: #1a1a1a !important;
  border: none !important;
  border-radius: 6px;
  font-weight: 400 !important;
}

button[class*="primary"]:hover,
.btn-primary:hover {
  background: #ffb44d !important;
}

/* Buttons - Secondary */
button[class*="secondary"],
.btn-secondary {
  background: transparent !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px;
}

button[class*="secondary"]:hover,
.btn-secondary:hover {
  border-color: #ffa222 !important;
  color: #ffa222 !important;
}

/* Input fields */
input, textarea, select {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px;
  color: var(--text-primary) !important;
  font-family: Arial, Helvetica, sans-serif !important;
  font-size: 15px;
  font-weight: 400 !important;
}

input:focus, textarea:focus, select:focus {
  outline: none !important;
  border-color: #ffa222 !important;
}

input::placeholder, textarea::placeholder {
  color: var(--text-muted) !important;
}

/* Chat message container */
.message, [class*="message"] {
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 8px 0;
}

/* User messages */
[class*="user"] .message,
.message-user {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border-subtle);
}

/* Sidebar */
[class*="sidebar"], .sidebar, nav {
  background: var(--bg-primary) !important;
  border-right: 1px solid var(--border-subtle) !important;
}

[class*="sidebar"] button,
.sidebar button {
  color: var(--text-secondary) !important;
}

[class*="sidebar"] button:hover,
.sidebar button:hover {
  background: var(--bg-secondary) !important;
  color: var(--text-primary) !important;
}

[class*="sidebar"] button[class*="active"],
.sidebar button.active {
  background: var(--bg-tertiary) !important;
  color: #ffa222 !important;
}

/* Code blocks */
pre, code {
  font-family: 'SF Mono', Monaco, 'Courier New', monospace !important;
  font-size: 13px;
}

pre {
  background: #0d0d0d !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 6px;
  padding: 16px;
  overflow-x: auto;
  margin: 12px 0;
}

/* Inline code */
:not(pre) > code {
  background: var(--bg-tertiary) !important;
  padding: 2px 6px;
  border-radius: 4px;
}

/* Model selector dropdown */
select, [class*="select"] {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #ffa222;
}

/* Modal/Dialog */
[class*="modal"], [class*="dialog"], .modal, .dialog {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px;
}

/* Cards */
[class*="card"], .card {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 8px;
}

/* Tables */
table {
  border-collapse: collapse;
  width: 100%;
}

th {
  color: #ffa222 !important;
  font-weight: 400 !important;
  text-align: left;
  padding: 12px;
  border-bottom: 1px solid var(--border);
}

td {
  padding: 12px;
  border-bottom: 1px solid var(--border-subtle);
}

tr:hover {
  background: var(--bg-tertiary);
}

/* Tooltips */
[class*="tooltip"] {
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px;
}

/* Icons in sidebar and buttons - orange accent on hover */
svg {
  stroke: currentColor;
}

button:hover svg,
a:hover svg {
  stroke: #ffa222;
}

/* Remove shadows and gradients */
* {
  box-shadow: none !important;
  text-shadow: none !important;
}

/* Loading spinner - orange */
[class*="spinner"], [class*="loading"] {
  border-color: var(--border) !important;
  border-top-color: #ffa222 !important;
}

/* Progress bars - orange */
[class*="progress"] {
  background: var(--bg-tertiary) !important;
}

[class*="progress"] > div,
[class*="progress-bar"] {
  background: #ffa222 !important;
}

/* Tags and badges */
[class*="badge"], [class*="tag"] {
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  font-weight: 400 !important;
}

/* Focus states - orange outline */
*:focus-visible {
  outline: 2px solid #ffa222 !important;
  outline-offset: 2px;
}

/* Checkbox and radio - orange accent */
input[type="checkbox"]:checked,
input[type="radio"]:checked {
  background: #ffa222 !important;
  border-color: #ffa222 !important;
}

/* Toggle switches - orange when active */
[class*="toggle"][class*="active"],
[class*="switch"][class*="active"] {
  background: #ffa222 !important;
}

/* Dropdown menus */
[class*="dropdown"], [class*="menu"] {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px;
}

[class*="dropdown"] button:hover,
[class*="menu"] button:hover {
  background: var(--bg-tertiary) !important;
}

/* Chat input area */
[class*="chat-input"], [class*="message-input"] {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px;
}

[class*="chat-input"]:focus-within,
[class*="message-input"]:focus-within {
  border-color: #ffa222 !important;
}

/* Send button */
button[class*="send"], [class*="submit-button"] {
  background: #ffa222 !important;
  color: #1a1a1a !important;
}

button[class*="send"]:hover {
  background: #ffb44d !important;
}

/* Avatar backgrounds */
[class*="avatar"] {
  background: var(--bg-tertiary) !important;
}

/* Dividers */
hr, [class*="divider"] {
  border-color: var(--border-subtle) !important;
  background: var(--border-subtle) !important;
}
'''


def get_root_path() -> Path:
    """Get USB-AI root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def create_theme_file(css_dir: Path) -> bool:
    """Create the custom theme CSS file."""
    log.info("Creating custom theme CSS...")

    css_dir.mkdir(parents=True, exist_ok=True)

    theme_path = css_dir / "custom-theme.css"

    try:
        with open(theme_path, "w") as f:
            f.write(THEME_CSS)
        log.info(f"Created: {theme_path}")
        return True

    except Exception as e:
        log.error(f"Failed to create theme file: {e}")
        return False


def find_webui_static(webui_path: Path) -> Path:
    """Find Open WebUI static directory."""
    possible_paths = [
        webui_path / "app" / "open_webui" / "static",
        webui_path / "app" / "static",
        webui_path / "static",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return webui_path / "static"


def copy_to_webui(theme_path: Path, webui_path: Path) -> bool:
    """Copy theme to Open WebUI static directory."""
    log.info("Copying theme to Open WebUI...")

    static_dir = find_webui_static(webui_path)
    css_dir = static_dir / "css"
    css_dir.mkdir(parents=True, exist_ok=True)

    dest_path = css_dir / "custom-theme.css"

    try:
        shutil.copy2(theme_path, dest_path)
        log.info(f"Copied to: {dest_path}")
        return True

    except Exception as e:
        log.error(f"Failed to copy theme: {e}")
        return False


def create_theme_readme(css_dir: Path) -> bool:
    """Create README for theme customization."""
    readme_content = '''# USB-AI Custom Theme

This directory contains the custom dark flat theme for Open WebUI.

## Theme Specifications

- Background: Dark (#1a1a1a)
- Accent: Orange (#ffa222)
- Font: Arial, Helvetica, sans-serif
- Weight: Normal only (no bold text)
- Style: Flat design (no gradients, no shadows)

## Files

- custom-theme.css: Main theme file

## Customization

Edit custom-theme.css to modify the theme.
Re-run s005_apply_theme.py after changes.

## Color Reference

| Color | Hex | Usage |
|-------|-----|-------|
| Background Primary | #1a1a1a | Main background |
| Background Secondary | #242424 | Cards, messages |
| Background Tertiary | #2d2d2d | Hover states |
| Text Primary | #e5e5e5 | Main text |
| Text Secondary | #a0a0a0 | Secondary text |
| Accent | #ffa222 | Headers, buttons, links |
| Accent Hover | #ffb44d | Hover states |
'''

    readme_path = css_dir / "README.md"

    try:
        with open(readme_path, "w") as f:
            f.write(readme_content)
        log.info(f"Created: {readme_path}")
        return True

    except Exception as e:
        log.error(f"Failed to create README: {e}")
        return False


def print_summary(success: bool, theme_path: Path):
    """Print theme application summary."""
    print("")
    print("=" * 50)
    print("        Theme Application Complete")
    print("=" * 50)
    print("")

    if success:
        print("Theme applied successfully!")
        print("")
        print("Theme specifications:")
        print("  Background: #1a1a1a (dark)")
        print("  Accent: #ffa222 (orange)")
        print("  Font: Arial, Helvetica")
        print("  Weight: Normal only")
        print("  Style: Flat (no shadows/gradients)")
        print("")
        print(f"Theme file: {theme_path}")
        print("")
        print("Build complete! Run the launcher:")
        print("  python scripts/launchers/start.py")
    else:
        print("Theme application incomplete. Check errors above.")

    print("")
    print("=" * 50)


def main() -> int:
    """Entry point."""
    log.info("USB-AI Theme Application")
    log.info(f"Platform: {platform.system()} {platform.machine()}")
    print("")

    root = get_root_path()
    webui_path = root / "modules" / "webui-portable"
    css_dir = webui_path / "static" / "css"

    log.info(f"Root: {root}")
    log.info(f"CSS dir: {css_dir}")
    print("")

    if not create_theme_file(css_dir):
        return 1

    theme_path = css_dir / "custom-theme.css"

    copy_to_webui(theme_path, webui_path)

    create_theme_readme(css_dir)

    print_summary(True, theme_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
