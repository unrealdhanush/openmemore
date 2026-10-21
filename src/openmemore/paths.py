"""Where desktop apps keep per-user files on each OS.

Claude Desktop and VS Code are Electron apps, which store settings in the platform's
per-user application data directory:
- macOS:   ~/Library/Application Support/<App>
- Windows: %APPDATA%\\<App>  (usually C:\\Users\\<you>\\AppData\\Roaming)
- Linux:   $XDG_CONFIG_HOME/<App>  (usually ~/.config)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_data_dir(user_home: Path, platform: str = sys.platform) -> Path:
    if platform == "darwin":
        return user_home / "Library" / "Application Support"
    if platform == "win32":
        return Path(os.environ.get("APPDATA") or user_home / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or user_home / ".config")
