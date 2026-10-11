"""Show where the project stands: which models are trained and evaluated.

    python -m ssp.status           # real results
    python -m ssp.status --mock    # mock-data results
"""
from __future__ import annotations

import argparse

from . import config, paths, results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    rdir = paths.home() / ("results_mock" if args.mock else "results")
    runs = {r["model_id"]: r for r in results.load_runs(rdir)} if (rdir / "runs").exists() else {}

    grid = config.severity_grid()
    split_path = paths.configs() / "split.json"
    print(f"Severity grid: {grid.get('status', 'draft').upper()}   |   Split: {'made' if split_path.exists() else 'NOT made'}"
          f"   |   Baseline floor: {config.analysis().get('min_baseline_dice')}")
    print()
    head = f"{'model':6s} {'seed':>4s} {'aug':>3s}  {'trained':10s} {'val':>6s}  {'clean':>6s} {'sim':>6s} {'real':>6s}  commit"
    print(head); print("-" * len(head))
    done = 0
    for mid in config.model_ids():
        cfg, r = config.model(mid), runs.get(mid, {})
        cell = lambda k: f"{r[k]:.3f}" if k in r else "  -  "
        trained = str(r.get("trained_at", ""))[:10] or "  -  "
        print(f"{mid:6s} {cfg['seed']:>4d} {cfg['aug']:>3d}  {trained:10s} {cell('best_val_dice'):>6s}  "
              f"{cell('clean_mean_dice'):>6s} {cell('sim_mean_dice'):>6s} {cell('real_mean_dice'):>6s}  {r.get('train_commit', '')}")
        done += all(f"{a}_mean_dice" in r for a in results.ARMS)
    print()
    print(f"{done} of {len(config.model_ids())} models have all three arms. "
          + ("Ready for:  python -m ssp.analyze" + (" --mock" if args.mock else "") if done >= 4 else "Analysis needs at least 4."))


if __name__ == "__main__":
    main()
