"""The results table and the run records.

One CSV per model in results/per_model/ (so two people can push results for
different models without merge conflicts), one row per
model x patient x phase x structure x evaluation condition.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "model_id",    # m01 ...
    "arm",         # clean | sim | real
    "vendor",      # A | B | C | D
    "corruption",  # none | bias | blur | gamma | resolution
    "severity",    # 0 for clean and real, 1..S for sim
    "patient",
    "phase",       # ED | ES
    "structure",   # LV | MYO | RV
    "dice",
    "hd95",        # empty unless evaluation was run with --hd95
    "pred_empty",  # True if the model predicted nothing for this structure
]
ARMS = ("clean", "sim", "real")


def model_file(results_dir: Path, model_id: str) -> Path:
    return results_dir / "per_model" / f"{model_id}.csv"


def write_arm(results_dir: Path, model_id: str, arm: str, rows: pd.DataFrame) -> Path:
    """Replace this model's rows for one arm, keeping its other arms."""
    rows = rows[COLUMNS]
    path = model_file(results_dir, model_id)
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[old["arm"] != arm], rows], ignore_index=True)
    rows.sort_values(["arm", "vendor", "corruption", "severity", "patient", "phase", "structure"]).to_csv(path, index=False)
    return path


def load_all(results_dir: Path) -> pd.DataFrame:
    files = sorted((results_dir / "per_model").glob("*.csv"))
    if not files:
        raise SystemExit(f"No result files in {results_dir / 'per_model'}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Result files are missing columns: {missing}")
    return df


def run_file(results_dir: Path, model_id: str) -> Path:
    return results_dir / "runs" / f"{model_id}.json"


def update_run(results_dir: Path, model_id: str, **fields) -> dict:
    """Merge fields into the model's run record (results/runs/<model>.json)."""
    path = run_file(results_dir, model_id)
    record = json.loads(path.read_text()) if path.exists() else {"model_id": model_id}
    record.update(fields)
    path.write_text(json.dumps(record, indent=1, default=str))
    return record


def load_runs(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((results_dir / "runs").glob("*.json"))]
