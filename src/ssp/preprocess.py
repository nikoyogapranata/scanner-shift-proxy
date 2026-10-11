"""Bring every scan to the same in-plane resolution and intensity scale.

Scans are resampled by physical pixel size, never squashed to a fixed square,
so anatomy keeps its shape whatever the scanner's field of view was.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def resample_inplane(volume: np.ndarray, spacing, target: float, is_label: bool = False) -> np.ndarray:
    """Resample X and Y to `target` mm per pixel. Slices (Z) are left as they are."""
    zoom = (spacing[0] / target, spacing[1] / target, 1.0)
    if is_label:
        return ndimage.zoom(volume, zoom, order=0, mode="nearest", grid_mode=True).astype(np.uint8)
    return ndimage.zoom(volume.astype(np.float32), zoom, order=1, mode="nearest", grid_mode=True)


def normalise(volume: np.ndarray) -> np.ndarray:
    """Clip to the 0.5 to 99.5 percentile range, then zero mean and unit variance per volume."""
    lo, hi = np.percentile(volume, [0.5, 99.5])
    clipped = np.clip(volume, lo, hi)
    return ((clipped - clipped.mean()) / (clipped.std() + 1e-8)).astype(np.float32)


def prepare(image: np.ndarray, label: np.ndarray | None, spacing, target: float):
    """Resample and normalise one volume. Returns (image, label, new spacing)."""
    img = normalise(resample_inplane(image, spacing, target))
    lab = None if label is None else resample_inplane(label, spacing, target, is_label=True)
    return img, lab, (target, target, float(spacing[2]))


def pad_to_multiple(arr: np.ndarray, k: int = 16):
    """Pad the two in-plane axes of [X,Y,...] up to a multiple of k. Returns (padded, original size)."""
    x, y = arr.shape[:2]
    px, py = (-x) % k, (-y) % k
    pad = [(0, px), (0, py)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad, mode="constant", constant_values=float(arr.min())), (x, y)


def crop_or_pad_2d(image: np.ndarray, label: np.ndarray, size: int, centre: tuple[int, int]):
    """Cut a size x size patch around `centre` from one slice, padding where the slice ends."""
    out_i = np.full((size, size), float(image.min()), dtype=np.float32)
    out_l = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    x0, y0 = centre[0] - half, centre[1] - half
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, image.shape[0]), min(y0 + size, image.shape[1])
    out_i[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = image[sx0:sx1, sy0:sy1]
    out_l[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = label[sx0:sx1, sy0:sy1]
    return out_i, out_l
