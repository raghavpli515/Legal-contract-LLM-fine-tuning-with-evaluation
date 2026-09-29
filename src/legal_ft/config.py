"""Config loading and repo paths. Works the same locally and on Kaggle/Colab."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (``name`` may omit the extension)."""
    path = CONFIG_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())
