"""Score one trained model on the evaluation sets of Section 6.6.1.

    python -m ssp.evaluate --model m01 --arm clean     # held-out vendor B, uncorrupted
    python -m ssp.evaluate --model m01 --arm sim       # same patients, K x S corruptions
    python -m ssp.evaluate --model m01 --arm real      # unseen vendors A, C, D
    python -m ssp.evaluate --model m01 --arm all

Rows go to results/per_model/<model>.csv. Re-running an arm replaces that
arm's rows for that model.

The 'sim' and 'real' arms refuse to run on real data until
configs/severity_grid.yaml says `status: frozen`. That is the pre-registration
rule of the proposal, enforced in code.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, corruptions, data, infer, metrics, paths, preprocess, results


def score(pred, lab, spacing, with_hd95: bool, meta: dict) -> list[dict]:
    rows = []
    for name, value in data.STRUCTURES.items():
        p, r = pred == value, lab == value
        rows.append({**meta, "structure": name, "dice": metrics.dice(p, r),
                     "hd95": metrics.hd95(p, r, spacing) if with_hd95 else np.nan,
                     "pred_empty": bool(r.any() and not p.any())})
    return rows


def evaluate_arm(arm, net, model_id, cases, split, grid, target, device, with_hd95) -> pd.DataFrame:
    by_patient = cases.set_index("patient")
    rows = []

    def run(patient, vendor, phase, image, label, spacing, corruption="none", severity=0):
        img, lab, sp = preprocess.prepare(image, label, spacing, target)
        pred = infer.predict_volume(net, img, device)
        meta = dict(model_id=model_id, arm=arm, vendor=vendor, corruption=corruption, severity=severity,
                    patient=patient, phase=phase)
        rows.extend(score(pred, lab, sp, with_hd95, meta))

    if arm in ("clean", "sim"):
        vendor = split["training_vendor"]
        for patient in split["test"]:
            case = by_patient.loc[patient].copy(); case["patient"] = patient
            for phase in data.PHASES:
                image, label, spacing = data.load_volume(case, phase)
                if arm == "clean":
                    run(patient, vendor, phase, image, label, spacing)
                else:
                    for corruption, severity in corruptions.conditions(grid):
                        bad = corruptions.corrupt(image, spacing, corruption, severity, grid, key=f"{patient}|{phase}")
                        run(patient, vendor, phase, bad, label, spacing, corruption, severity)
    else:
        for vendor, patients in split["real"].items():
            for patient in patients:
                case = by_patient.loc[patient].copy(); case["patient"] = patient
                for phase in data.PHASES:
                    image, label, spacing = data.load_volume(case, phase)
                    run(patient, vendor, phase, image, label, spacing)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=["clean", "sim", "real", "all"])
    ap.add_argument("--data-root")
    ap.add_argument("--hd95", action="store_true", help="also compute HD95 (critical metric C1, slower)")
    ap.add_argument("--device")
    args = ap.parse_args()

    root = paths.data_root(args.data_root)
    mock = paths.is_mock(root)
    arms = list(results.ARMS) if args.arm == "all" else [args.arm]
    grid = config.severity_grid()
    if not mock and not config.grid_is_frozen() and any(a in ("sim", "real") for a in arms):
        raise SystemExit(
            "Blocked: configs/severity_grid.yaml is still 'status: draft'.\n"
            "The simulated and real arms may only run after the grid is frozen and committed\n"
            "(see docs/FREEZE_CHECKLIST.md). The clean arm can run at any time."
        )
    split, dcfg = config.split(), config.data()
    if bool(split.get("mock")) != mock:
        raise SystemExit("configs/split.json was made for a different dataset (mock vs real). Re-make the split.")

    device = infer.pick_device(args.device)
    net, _ = infer.load_model(args.model, device)
    cases = data.discover_cases(root)
    rdir = paths.results(root)
    for arm in arms:
        df = evaluate_arm(arm, net, args.model, cases, split, grid, dcfg["target_spacing_mm"], device, args.hd95)
        path = results.write_arm(rdir, args.model, arm, df)
        case_mean = df.groupby("patient")["dice"].mean().mean()
        results.update_run(rdir, args.model, **{
            f"{arm}_evaluated_at": datetime.now().isoformat(timespec="seconds"),
            f"{arm}_commit": config.git_commit(), f"{arm}_mean_dice": round(float(case_mean), 4),
        })
        print(f"[{args.model}] {arm:5s}: {len(df)} rows, mean Dice {case_mean:.4f} -> {path}")


if __name__ == "__main__":
    main()
