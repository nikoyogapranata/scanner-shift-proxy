"""Read the YAML/JSON files in configs/."""
from __future__ import annotations

import json
import subprocess

import yaml

from . import paths


def _load(name: str) -> dict:
    with open(paths.configs() / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data() -> dict:
    return _load("data.yaml")


def analysis() -> dict:
    return _load("analysis.yaml")


def severity_grid() -> dict:
    grid = _load("severity_grid.yaml")
    levels = grid["levels"]
    for name, spec in grid["corruptions"].items():
        if len(spec["values"]) != levels:
            raise SystemExit(f"severity_grid.yaml: '{name}' needs exactly {levels} values")
    return grid


def grid_is_frozen() -> bool:
    return str(severity_grid().get("status", "draft")).lower() == "frozen"


def model(model_id: str) -> dict:
    cfg = _load("models.yaml")
    if model_id not in cfg["models"]:
        raise SystemExit(f"Unknown model '{model_id}'. Known: {', '.join(cfg['models'])}")
    return {**cfg["defaults"], **cfg["models"][model_id], "model_id": model_id}


def model_ids() -> list[str]:
    return list(_load("models.yaml")["models"])


def split() -> dict:
    path = paths.configs() / "split.json"
    if not path.exists():
        raise SystemExit("configs/split.json not found. Run:  python -m ssp.split")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def git_commit() -> str:
    """Short hash of the current commit, with '+dirty' if files are uncommitted."""
    try:
        cwd = paths.home()
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"
