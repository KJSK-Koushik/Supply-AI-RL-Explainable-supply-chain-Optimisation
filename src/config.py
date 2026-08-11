"""Config loading. Every module reads YAML through here so paths stay
relative to the project root no matter where a script is launched from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a plain dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative_path: str) -> Path:
    """Turn a project-relative path from a config file into an absolute Path."""
    return PROJECT_ROOT / relative_path
