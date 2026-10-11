"""Dice and HD95 for one structure in one volume (Sections 6.6.2, 6.6.6 C1, 6.6.7).

Rules for empty masks, fixed in advance:
  reference empty      -> both metrics are NaN (the structure is excluded for that volume)
  prediction empty     -> Dice 0, HD95 = diagonal of the volume in mm
"""
from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """pred and ref are boolean masks of the same shape."""
    if not ref.any():
        return float("nan")
    inter = np.logical_and(pred, ref).sum()
    return float(2.0 * inter / (pred.sum() + ref.sum()))


def volume_diagonal_mm(shape, spacing) -> float:
    return float(np.sqrt(sum((n * s) ** 2 for n, s in zip(shape, spacing))))


def hd95(pred: np.ndarray, ref: np.ndarray, spacing) -> float:
    """95th percentile Hausdorff distance in mm, using MONAI's implementation."""
    if not ref.any():
        return float("nan")
    if not pred.any():
        return volume_diagonal_mm(pred.shape, spacing)
    import torch
    from monai.metrics import compute_hausdorff_distance

    p = torch.from_numpy(pred[None, None].astype(np.uint8))
    r = torch.from_numpy(ref[None, None].astype(np.uint8))
    value = compute_hausdorff_distance(p, r, include_background=True, percentile=95, spacing=list(spacing))
    return float(value.item())
