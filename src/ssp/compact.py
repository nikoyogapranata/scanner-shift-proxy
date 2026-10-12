"""Make a compact copy of an M&Ms folder: only the ED and ES frames, compressed.

    python -m ssp.compact --data-root <original> --out <compact folder>

The original 4D files hold every time frame of the cardiac cycle (often 25 to
35), but the study uses only the two annotated ones. Keeping just those two,
gzip-compressed and in the original data type, shrinks the data from about
70 GB to a few GB without changing a single value the code reads.

The copy has the same layout the loader expects: `<code>_sa.nii.gz`,
`<code>_sa_gt.nii.gz` and a metadata CSV, with ED = frame 0 and ES = frame 1.
Patients whose label file has no annotations on ED and ES are left out.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import data, paths

META_NAME = "mnms_compact_information.csv"


def _two_frames(path: str, ed: int, es: int, is_label: bool):
    img = nib.load(path)
    frames = np.stack([np.asarray(img.dataobj[..., ed]), np.asarray(img.dataobj[..., es])], axis=-1)
    header = img.header.copy()
    if is_label:
        frames = np.rint(frames).astype(np.uint8)
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(1, 0)
    else:
        header.set_data_dtype(img.get_data_dtype())      # keep the original type; nibabel rescales if needed
    zooms = list(img.header.get_zooms())
    out = nib.Nifti1Image(frames, img.affine, header)
    out.header.set_zooms(tuple(zooms[:3]) + (zooms[3] if len(zooms) > 3 else 1.0,))
    return out


def build(src_root: Path, out_root: Path, verify: int = 5) -> pd.DataFrame:
    cases = data.discover_cases(src_root)
    cases = cases[cases["labelled"]]
    keep = cases[cases.apply(data.has_both_phases_labelled, axis=1)]
    print(f"{len(cases)} patients with label files, {len(keep)} with annotations on ED and ES. Writing to {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (_, case) in enumerate(keep.iterrows(), 1):
        code, ed, es = case["patient"], int(case["ed"]), int(case["es"])
        folder = out_root / code
        folder.mkdir(exist_ok=True)
        nib.save(_two_frames(case["image"], ed, es, False), folder / f"{code}_sa.nii.gz")
        nib.save(_two_frames(case["label"], ed, es, True), folder / f"{code}_sa_gt.nii.gz")
        rows.append({"External code": code, "Vendor": case["vendor"], "Centre": case["centre"],
                     "ED": 0, "ES": 1, "original_ED": ed, "original_ES": es})
        if i % 25 == 0 or i == len(keep):
            print(f"  {i}/{len(keep)}")
    meta = pd.DataFrame(rows)
    meta.to_csv(out_root / META_NAME, index=False)

    if verify:
        check(src_root, out_root, n=verify)
    return meta


def check(src_root: Path, out_root: Path, n: int = 5) -> None:
    """Compare n random patients, both phases, between the original and the compact copy."""
    orig = data.discover_cases(src_root).set_index("patient", drop=False)
    comp = data.discover_cases(out_root).set_index("patient", drop=False)
    worst = 0.0
    for code in random.Random(0).sample(sorted(comp.index), min(n, len(comp))):
        for phase in data.PHASES:
            a_img, a_lab, a_sp = data.load_volume(orig.loc[code], phase)
            b_img, b_lab, b_sp = data.load_volume(comp.loc[code], phase)
            if not np.array_equal(a_lab, b_lab) or a_img.shape != b_img.shape or not np.allclose(a_sp, b_sp):
                raise SystemExit(f"Compact copy differs from the original for {code} {phase}")
            span = float(a_img.max() - a_img.min()) or 1.0
            worst = max(worst, float(np.abs(a_img - b_img).max()) / span)
    if worst > 1e-3:
        raise SystemExit(f"Compact images differ from the originals by up to {worst:.2e} of their range")
    print(f"Verified {min(n, len(comp))} patients: labels identical, images within {worst:.1e} of their range.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="the original dataset folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", type=int, default=5, help="number of patients to compare afterwards (0 to skip)")
    args = ap.parse_args()
    build(paths.data_root(args.data_root), Path(args.out).resolve(), args.verify)


if __name__ == "__main__":
    main()
