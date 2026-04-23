"""Shared reporting infrastructure used by all analysis scripts.

Provides:
  - VERSIONS      — library version strings logged alongside every run
  - RNG_SEED / rng — single reproducibility seed
  - output_dir()  — resolve/create the per-script output directory
  - save_json()   — serialise any JSON-able payload next to plots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels
    _sm_version = statsmodels.__version__
except Exception:
    _sm_version = "unavailable"

VERSIONS: dict[str, str] = {
    "python":      sys.version.split()[0],
    "pandas":      pd.__version__,
    "numpy":       np.__version__,
    "statsmodels": _sm_version,
}

RNG_SEED: int = 42
rng = np.random.default_rng(RNG_SEED)


def output_dir(script_name: str, base: Path | str | None = None) -> Path:
    """Return (and create) output/<script_name>/ relative to *base*.

    *base* defaults to the workspace root (two levels above this file).
    """
    root = Path(base) if base is not None else Path(__file__).resolve().parent.parent
    d = root / "output" / script_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(payload: Any, path: Path | str) -> None:
    """Write *payload* as indented JSON to *path*, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")
