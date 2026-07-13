"""Shared path / config helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment() -> dict:
    return load_yaml(ROOT / "config" / "experiment.yaml")


def ensure_dirs() -> None:
    cfg = load_experiment()
    for key in ("checkpoint_dir", "processed_dir", "figures_dir"):
        (ROOT / cfg[key]).mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
