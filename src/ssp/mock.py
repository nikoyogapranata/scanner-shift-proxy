"""Make a small fake dataset shaped like M&Ms, for testing the pipeline anywhere.

    python -m ssp.mock --out data/mock

The "hearts" are drawn circles and the four "vendors" differ in resolution,
contrast and noise. It has no scientific meaning. It exists so the whole
pipeline can be run on a laptop in minutes, without the real data.

A marker file (.mock) makes the code send all results to results_mock/ and
lets the pipeline run before the severity grid is frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

VENDORS = {  # pixel size mm, blood brightness, noise level, smooth shading strength
    "A": dict(pix=1.20, blood=0.95, noise=0.06, shade=0.35, name="Siemens"),
    "B": dict(pix=1.40, blood=0.80, noise=0.03, shade=0.05, name="Philips"),
    "C": dict(pix=1.30, blood=0.70, noise=0.05, shade=0.20, name="GE"),
    "D": dict(pix=1.60, blood=0.60, noise=0.08, shade=0.10, name="Canon"),
}


def _case(rng, v: dict, fov_mm: float, slices: int, frames: int):
    n = int(round(fov_mm / v["pix"]))
    yy, xx = np.meshgrid(np.arange(n), np.arange(n))
    cx, cy = n / 2 + rng.normal(0, 4), n / 2 + rng.normal(0, 4)
    r_lv0, wall = rng.uniform(20, 26) / v["pix"], rng.uniform(7, 10) / v["pix"]
    img = np.zeros((n, n, slices, frames), np.float32)
    lab = np.zeros((n, n, slices, frames), np.uint8)
    shade = 1 + v["shade"] * ((xx - n / 2) / n + (yy - n / 2) / n)
    for t in range(frames):
        contraction = 1.0 if t == 0 else (0.65 if t == frames // 2 else 0.85)
        for z in range(slices):
            taper = 1 - 0.35 * abs(z - (slices - 1) / 2) / max((slices - 1) / 2, 1)
            r_lv = r_lv0 * contraction * taper
            d = np.hypot(xx - cx, yy - cy)
            lv, myo = d < r_lv, (d >= r_lv) & (d < r_lv + wall)
            d_rv = np.hypot(xx - (cx - r_lv - wall - r_lv * 0.6), yy - cy)
            rv = (d_rv < r_lv * 0.9) & ~lv & ~myo
            body = np.hypot((xx - n / 2) / (n * 0.45), (yy - n / 2) / (n * 0.38)) < 1
            sl = 0.25 * body + ndimage.gaussian_filter(rng.random((n, n)), 3) * 0.3 * body
            sl[myo], sl[lv], sl[rv] = 0.35, v["blood"], v["blood"] * 0.9
            sl = ndimage.gaussian_filter(sl, 0.7) * shade + rng.normal(0, v["noise"], (n, n))
            img[:, :, z, t] = np.clip(sl, 0, None) * 1000
            lab[:, :, z, t] = 1 * lv + 2 * myo + 3 * rv
    keep = np.zeros(frames, bool)
    keep[[0, frames // 2]] = True          # only ED and ES are annotated, as in M&Ms
    lab[..., ~keep] = 0
    return img, lab, 0, frames // 2


def make(out: Path, counts: dict, fov_mm: float = 150.0, slices: int = 6, frames: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for vendor, n_patients in counts.items():
        v = VENDORS[vendor]
        for i in range(n_patients):
            code = f"{vendor}{i:03d}MK"
            img, lab, ed, es = _case(rng, v, fov_mm, slices, frames)
            folder = out / "OpenDataset" / ("Training" if vendor == "B" else "Testing") / code
            folder.mkdir(parents=True, exist_ok=True)
            affine = np.diag([v["pix"], v["pix"], 10.0, 1.0])
            nib.save(nib.Nifti1Image(img, affine), folder / f"{code}_sa.nii.gz")
            nib.save(nib.Nifti1Image(lab, affine), folder / f"{code}_sa_gt.nii.gz")
            rows.append({"External code": code, "VendorName": v["name"], "Vendor": vendor, "Centre": 1, "ED": ed, "ES": es})
    pd.DataFrame(rows).to_csv(out / "OpenDataset" / "mock_dataset_information.csv", index=False)
    (out / ".mock").write_text("Synthetic data made by ssp.mock. Not M&Ms.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/mock")
    ap.add_argument("--n-b", type=int, default=60, help="vendor B patients (training vendor)")
    ap.add_argument("--n-other", type=int, default=12, help="patients for each of vendors A, C, D")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    make(out, {"B": args.n_b, "A": args.n_other, "C": args.n_other, "D": args.n_other}, seed=args.seed)
    print(f"Mock dataset written to {out}")
    print(f"Use it with:  --data-root {out}   (or set MNMS_ROOT)")


if __name__ == "__main__":
    main()
