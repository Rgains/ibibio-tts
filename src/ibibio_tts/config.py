"""Local configuration loading. Never write real secrets to `.env.example`."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(project_root: Path) -> None:
    """Populate os.environ from `.env` for keys not already set."""
    env_file = project_root / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value
