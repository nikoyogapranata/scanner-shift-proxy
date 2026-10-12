"""Find the cases in an M&Ms folder and load one volume at a time.

Written for the layout of the official release: a metadata CSV with one row per
patient (a patient code, a vendor letter, the ED and ES frame numbers) and, per
patient, a 4D image `<code>_sa.nii.gz` and a 4D label `<code>_sa_gt.nii.gz`
somewhere below the root. Folder names do not matter.

Run notebooks/01_dataset_audit.ipynb first. If your copy is laid out
differently, `discover_cases` is the only function that needs changing.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PHASES = ("ED", "ES")
STRUCTURES = {"LV": 1, "MYO": 2, "RV": 3}


def _find_metadata(root: Path) -> pd.DataFrame:
    for path in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        vendor = [c for k, c in cols.items() if "vendor" in k]
        code = [c for k, c in cols.items() if "code" in k or k in ("patient", "subject", "id")]
        if vendor and code and "ed" in cols and "es" in cols:
            # prefer the single-letter column (A/B/C/D) over the manufacturer name
            vendor.sort(key=lambda c: df[c].astype(str).str.len().mean())
            out = pd.DataFrame({
                "patient": df[code[0]].astype(str).str.strip(),
                "vendor": df[vendor[0]].astype(str).str.strip().str.upper(),
                "ed": df[cols["ed"]].astype(int),
                "es": df[cols["es"]].astype(int),
            })
            centre = [c for k, c in cols.items() if "centre" in k or "center" in k]
            out["centre"] = df[centre[0]].astype(str) if centre else ""
            return out
    raise SystemExit(
        f"No metadata CSV with patient code, vendor, ED and ES columns found under {root}.\n"
        "Without it the vendor of each scan is unknown and the study cannot run."
    )


def discover_cases(root: Path) -> pd.DataFrame:
    """One row per patient: vendor, ED/ES frames, file paths, whether it is labelled."""
    meta = _find_metadata(root)
    images, labels = {}, {}
    for path in root.rglob("*.nii*"):
        name = path.name
        code = name.split("_")[0]
        if "_gt" in name:
            labels[code] = path
        else:
            images[code] = path
    meta["image"] = meta["patient"].map(lambda c: str(images.get(c, "")))
    meta["label"] = meta["patient"].map(lambda c: str(labels.get(c, "")))
    meta["labelled"] = (meta["image"] != "") & (meta["label"] != "")
    if not meta["labelled"].any():
        raise SystemExit(f"No labelled cases (<code>_sa.nii.gz + <code>_sa_gt.nii.gz) found under {root}.")
    return meta.sort_values("patient").reset_index(drop=True)


def load_volume(case: pd.Series, phase: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Return (image [X,Y,Z] float32, label [X,Y,Z] uint8, voxel size in mm) for ED or ES."""
    frame = int(case["ed"] if phase == "ED" else case["es"])
    img = nib.load(case["image"])
    lab = nib.load(case["label"])
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    image = np.asarray(img.dataobj[..., frame], dtype=np.float32)
    label = np.rint(np.asarray(lab.dataobj[..., frame])).astype(np.uint8)   # labels may be stored as floats
    if image.shape != label.shape:
        raise ValueError(f"{case['patient']} {phase}: image {image.shape} and label {label.shape} differ")
    return image, label, spacing


def has_both_phases_labelled(case: pd.Series) -> bool:
    """True if the label file has annotations on both the ED and the ES frame."""
    lab = nib.load(case["label"])
    if len(lab.shape) < 4:
        return False
    return all(np.asarray(lab.dataobj[..., int(case[c])]).any() for c in ("ed", "es"))
