"""Make the one fixed patient-level split (Section 6.6.1).

    python -m ssp.split --data-root /path/to/mnms

Writes configs/split.json. Commit that file. It is created once and is the
same for every model: held-out vendor B patients are used for scoring only.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np

from . import config, data, paths


def make_split(cases, seed: int, n_test: int, n_val: int, training_vendor: str, real_vendors: list[str]) -> dict:
    labelled = cases[cases["labelled"]]
    pool = sorted(labelled.loc[labelled["vendor"] == training_vendor, "patient"])
    if len(pool) < n_test + n_val + 5:
        raise SystemExit(
            f"Only {len(pool)} labelled vendor {training_vendor} patients; "
            f"need at least {n_test + n_val + 5} for n_test={n_test}, n_val={n_val}."
        )
    order = np.random.default_rng(seed).permutation(len(pool))
    shuffled = [pool[i] for i in order]
    real = {v: sorted(labelled.loc[labelled["vendor"] == v, "patient"]) for v in real_vendors}
    empty = [v for v, p in real.items() if not p]
    if empty:
        raise SystemExit(f"No labelled patients for vendor(s) {empty}. The real-shift arm cannot run.")
    return {
        "created": date.today().isoformat(),
        "seed": seed,
        "training_vendor": training_vendor,
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val:]),
        "real": real,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root")
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (do not do this after training)")
    args = ap.parse_args()

    out = paths.configs() / "split.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists. The split is made once. Use --force only before any training.")
    cfg = config.data()
    root = paths.data_root(args.data_root)
    cases = data.discover_cases(root)
    print("Checking that every label file has annotations on ED and ES ...")
    ok = cases[cases["labelled"]].apply(data.has_both_phases_labelled, axis=1)
    bad = sorted(cases.loc[ok[~ok].index, "patient"])
    cases.loc[ok[~ok].index, "labelled"] = False
    if bad:
        print(f"  {len(bad)} patient(s) have a label file without annotations on ED and ES; not used: {', '.join(bad)}")
    split = make_split(cases, cfg["split"]["seed"], args.n_test or cfg["split"]["n_test"],
                       args.n_val or cfg["split"]["n_val"], cfg["training_vendor"], cfg["real_vendors"])
    split["mock"] = paths.is_mock(root)
    split["excluded_unlabelled"] = bad
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=1)
    print(f"Wrote {out}")
    print(f"  vendor {cfg['training_vendor']}: train {len(split['train'])}, val {len(split['val'])}, held-out test {len(split['test'])}")
    for v, p in split["real"].items():
        print(f"  vendor {v}: {len(p)} labelled patients (real-shift arm)")


if __name__ == "__main__":
    main()
