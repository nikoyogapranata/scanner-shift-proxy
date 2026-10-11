"""End to end on the mock dataset: data -> split -> train -> evaluate. Needs torch, monai, torchio."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("torchio")

from ssp import config, corruptions, mock  # noqa: E402

pytestmark = pytest.mark.slow


def _run(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)


def test_corruptions_are_deterministic_and_grow_with_severity(home):
    grid = config.severity_grid()
    rng = np.random.default_rng(0)
    from scipy import ndimage
    image = ndimage.gaussian_filter(rng.random((96, 88, 5)), (2, 2, 0)).astype(np.float32) * 1000
    spacing = (1.4, 1.4, 10.0)
    for name in corruptions.CORRUPTIONS:
        change = []
        for s in range(1, grid["levels"] + 1):
            a = corruptions.corrupt(image, spacing, name, s, grid, key="p|ED")
            b = corruptions.corrupt(image, spacing, name, s, grid, key="p|ED")
            assert a.shape == image.shape and np.array_equal(a, b), name
            change.append(float(np.abs(a - image).mean()))
        assert change == sorted(change) and change[0] > 0, (name, change)
    assert len(corruptions.conditions(grid)) == len(grid["corruptions"]) * grid["levels"]


def test_mock_pipeline_end_to_end(home, monkeypatch):
    root = home / "data" / "mock"
    mock.make(root, {"B": 16, "A": 2, "C": 2, "D": 2}, fov_mm=110, slices=3)
    monkeypatch.setenv("MNMS_ROOT", str(root))
    cfg = yaml.safe_load(open(home / "configs" / "models.yaml"))
    cfg["defaults"].update(epochs=1, patch_size=96, batch_size=4, channels=[8, 16, 32])
    yaml.safe_dump(cfg, open(home / "configs" / "models.yaml", "w"))

    assert _run("ssp.split", "--n-test", "4", "--n-val", "2").returncode == 0
    assert _run("ssp.split").returncode != 0                       # the split is made once
    r = _run("ssp.train", "--model", "m01"); assert r.returncode == 0, r.stderr[-800:]
    r = _run("ssp.evaluate", "--model", "m01", "--arm", "all"); assert r.returncode == 0, r.stderr[-800:]

    df = pd.read_csv(home / "results_mock" / "per_model" / "m01.csv")
    grid = config.severity_grid()
    k_s = len(grid["corruptions"]) * grid["levels"]
    assert (df["arm"] == "clean").sum() == 4 * 2 * 3               # patients x phases x structures
    assert (df["arm"] == "sim").sum() == 4 * 2 * 3 * k_s
    assert (df["arm"] == "real").sum() == 6 * 2 * 3
    assert df["dice"].between(0, 1).all()
    assert not (home / "results" / "per_model" / "m01.csv").exists()   # mock results stay out of results/
    assert _run("ssp.status", "--mock").returncode == 0

    # On data without the mock marker, a draft grid must block the simulated and real arms.
    (root / ".mock").unlink()
    r = _run("ssp.evaluate", "--model", "m01", "--arm", "real")
    assert r.returncode != 0 and "Blocked" in (r.stderr + r.stdout)
