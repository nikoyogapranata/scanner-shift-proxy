"""Simulated scanner shift (Section 6.3): four corruption families.

Bias field, blur and gamma use TorchIO. Resolution reduction uses SciPy (see
`_lower_resolution` for why).

Each corruption is applied to the raw volume, before resampling and
normalisation, because a scanner changes the image at acquisition time.

Every call is deterministic. The random seed comes from the case and the
corruption, so every model is scored on identical inputs.
"""
from __future__ import annotations

import hashlib

import numpy as np

CORRUPTIONS = ("bias", "blur", "gamma", "resolution")


def _seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def corrupt(image: np.ndarray, spacing, corruption: str, severity: int, grid: dict, key: str) -> np.ndarray:
    """Return a corrupted copy of `image` [X,Y,Z].

    severity runs from 1 to grid['levels']; `key` identifies the case and phase.
    """
    import torch
    import torchio as tio

    if corruption not in grid["corruptions"]:
        raise ValueError(f"Unknown corruption '{corruption}'")
    value = float(grid["corruptions"][corruption]["values"][severity - 1])
    # The seed depends on the case and the corruption, not on the severity, so one
    # case gets the same bias-field shape and gamma direction at every level and
    # only the strength changes.
    seed = _seed(f"{key}|{corruption}")

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    lo, hi = float(image.min()), float(image.max())
    scale = (hi - lo) or 1.0
    unit = torch.from_numpy(((image - lo) / scale).astype(np.float32))[None]   # [1,X,Y,Z] in [0,1]
    img = tio.ScalarImage(tensor=unit, affine=affine)

    if corruption == "bias":
        torch.manual_seed(seed)
        out = tio.RandomBiasField(coefficients=value, order=3)(img)
    elif corruption == "blur":
        out = tio.Blur(std=(value, value, 0.0))(img)            # in-plane only, in mm
    elif corruption == "gamma":
        sign = 1.0 if seed % 2 == 0 else -1.0                   # brighter or darker, fixed per case
        out = tio.Gamma(gamma=float(np.exp(sign * value)))(img)
    elif corruption == "resolution":
        out = None
        result = _lower_resolution(unit[0].numpy(), value)
    if out is not None:
        result = out.data[0].numpy().astype(np.float32)
    if result.shape != image.shape:
        raise RuntimeError(f"{corruption} changed the volume shape {image.shape} -> {result.shape}")
    return result * scale + lo


def _lower_resolution(unit: np.ndarray, factor: float) -> np.ndarray:
    """Simulate acquiring the slice with pixels `factor` times larger, in-plane.

    Low-pass filter (each larger pixel averages over its area), sample on the
    coarse grid, then interpolate back to the original grid. Done with SciPy:
    TorchIO's resampler was not monotone in the factor on test images.
    """
    from scipy import ndimage

    x, y = unit.shape[:2]
    sigma = 0.5 * float(np.sqrt(max(factor ** 2 - 1.0, 0.0)))          # in pixels of the original grid
    low = ndimage.gaussian_filter(unit, (sigma, sigma, 0.0), mode="nearest")
    nx, ny = max(2, round(x / factor)), max(2, round(y / factor))
    down = ndimage.zoom(low, (nx / x, ny / y, 1.0), order=1, mode="nearest", grid_mode=True)
    up = ndimage.zoom(down, (x / down.shape[0], y / down.shape[1], 1.0), order=1, mode="nearest", grid_mode=True)
    return up[:x, :y].astype(np.float32)


def conditions(grid: dict) -> list[tuple[str, int]]:
    """All (corruption, severity) pairs in the grid: K x S conditions."""
    return [(c, s) for c in grid["corruptions"] for s in range(1, grid["levels"] + 1)]
