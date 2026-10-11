"""Where things live. Everything is resolved from one project root.

The root is the repository folder. Set the environment variable SSP_HOME to
point somewhere else (the tests do this so they never touch real results).
"""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    env = os.environ.get("SSP_HOME")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def configs() -> Path:
    return home() / "configs"


def data_root(explicit: str | None = None) -> Path:
    """Dataset folder: the --data-root argument, else the MNMS_ROOT variable."""
    value = explicit or os.environ.get("MNMS_ROOT")
    if not value:
        raise SystemExit(
            "No dataset folder given. Pass --data-root or set the MNMS_ROOT "
            "environment variable (see README, 'Pointing the code at the data')."
        )
    root = Path(value).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Dataset folder does not exist: {root}")
    return root


def is_mock(root: Path) -> bool:
    """Mock datasets carry a marker file so they can never mix with real results."""
    return (root / ".mock").exists()


def results(root: Path | None = None) -> Path:
    """Real results go to results/, mock-data results to results_mock/."""
    name = "results_mock" if (root is not None and is_mock(root)) else "results"
    path = home() / name
    (path / "per_model").mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(parents=True, exist_ok=True)
    return path


def runs() -> Path:
    """Model weights. Not in git. On Colab, point SSP_RUNS at Google Drive."""
    env = os.environ.get("SSP_RUNS")
    path = Path(env).expanduser().resolve() if env else home() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path
