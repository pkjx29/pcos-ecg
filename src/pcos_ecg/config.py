"""Loading YAML configs. Relative paths in a config are resolved against the current
working directory, so commands are meant to be run from the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DATA_MODES = {"study", "synthetic_demo"}


class ConfigError(ValueError):
    """Raised for a missing or malformed config value."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    mode = cfg.get("data_mode")
    if mode not in DATA_MODES:
        raise ConfigError(f"data_mode must be one of {sorted(DATA_MODES)}, got {mode!r}")
    return cfg


def resolve(path: str | Path | None, root: Path | None = None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (root or Path.cwd()) / p
