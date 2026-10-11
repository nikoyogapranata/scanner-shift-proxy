"""Make a synthetic results table, to exercise the analysis before any model exists.

    python -m ssp.synth --rho 0.8      # writes results_demo/per_model/*.csv
    python -m ssp.analyze --results-dir results_demo

`rho` is the built-in relation between simulated and real degradation.
These numbers are invented. They never go into results/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths, results

STRUCTS = ["LV", "MYO", "RV"]
CORRS = ["bias", "blur", "gamma", "resolution"]


def make(n_models: int = 12, rho: float = 0.8, spread: float = 0.03, noise: float = 0.03,
         n_test: int = 30, n_real: int = 40, levels: int = 5, seed: int = 0, hd95: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_models)
    rows = []
    for k in range(n_models):
        mid = f"m{k + 1:02d}"
        clean = 0.88 + rng.normal(0, 0.01)
        d_sim, d_real = 0.08 + spread * z[k, 0], 0.12 + spread * z[k, 1]

        def emit(arm, vendor, corr, sev, mean, patients):
            for p in patients:
                p_eff = rng.normal(0, 0.02)
                for ph in ("ED", "ES"):
                    for i, s in enumerate(STRUCTS):
                        dice = float(np.clip(mean + p_eff - 0.02 * (i == 1) + rng.normal(0, noise), 0, 1))
                        rows.append((mid, arm, vendor, corr, sev, p, ph, s, dice,
                                     (1 - dice) * 40 if hd95 else np.nan, False))

        test = [f"B{j:03d}" for j in range(n_test)]
        emit("clean", "B", "none", 0, clean, test)
        for ci, c in enumerate(CORRS):
            for s in range(1, levels + 1):
                factor = (s / ((levels + 1) / 2)) * (0.7 + 0.2 * ci)   # averages to 1 over the grid
                emit("sim", "B", c, s, clean - d_sim * factor, test)
        for v, f in (("A", 1.4), ("C", 0.8), ("D", 0.8)):
            emit("real", v, "none", 0, clean - d_real * f, [f"{v}{j:03d}" for j in range(n_real)])
    return pd.DataFrame(rows, columns=results.COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--models", type=int, default=12)
    ap.add_argument("--spread", type=float, default=0.03, help="how much models differ in degradation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hd95", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve() if args.out else paths.home() / "results_demo"
    (out / "per_model").mkdir(parents=True, exist_ok=True)
    df = make(args.models, args.rho, args.spread, seed=args.seed, hd95=args.hd95)
    for mid, g in df.groupby("model_id"):
        g.to_csv(out / "per_model" / f"{mid}.csv", index=False)
    print(f"Wrote synthetic results for {args.models} models to {out}")


if __name__ == "__main__":
    main()
