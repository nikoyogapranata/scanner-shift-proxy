import numpy as np
import pandas as pd
import pytest

from ssp import metrics, preprocess, results, split


def test_dice_values_and_empty_rules():
    ref = np.zeros((8, 8, 2), bool); ref[2:6, 2:6] = True
    assert metrics.dice(ref, ref) == 1.0
    half = ref.copy(); half[4:6] = False
    assert metrics.dice(half, ref) == pytest.approx(2 * half.sum() / (half.sum() + ref.sum()))
    assert metrics.dice(np.zeros_like(ref), ref) == 0.0           # empty prediction scores zero
    assert np.isnan(metrics.dice(ref, np.zeros_like(ref)))        # absent reference is excluded


def test_hd95_rules_without_and_with_monai():
    ref = np.zeros((20, 20, 2), bool); ref[5:10, 5:10] = True
    spacing = (1.5, 1.5, 10.0)
    assert np.isnan(metrics.hd95(ref, np.zeros_like(ref), spacing))
    assert metrics.hd95(np.zeros_like(ref), ref, spacing) == pytest.approx(metrics.volume_diagonal_mm(ref.shape, spacing))
    pytest.importorskip("monai")
    shifted = np.roll(ref, 2, axis=0)
    assert metrics.hd95(ref, ref, spacing) == 0.0
    assert metrics.hd95(shifted, ref, spacing) == pytest.approx(3.0)   # 2 pixels x 1.5 mm


def test_resampling_keeps_physical_size():
    vol = np.random.rand(100, 80, 4).astype(np.float32)
    out = preprocess.resample_inplane(vol, (2.5, 1.25, 10.0), 1.25)
    assert out.shape == (200, 80, 4)
    lab = preprocess.resample_inplane((vol > 0.5).astype(np.uint8) * 3, (2.5, 1.25, 10.0), 1.25, is_label=True)
    assert set(np.unique(lab)) <= {0, 3}
    padded, size = preprocess.pad_to_multiple(out, 32)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0 and size == (200, 80)


def _cases(n_b=60, n_other=5):
    rows = [{"patient": f"B{i:03d}", "vendor": "B", "labelled": True} for i in range(n_b)]
    rows += [{"patient": f"{v}{i:03d}", "vendor": v, "labelled": True} for v in "ACD" for i in range(n_other)]
    rows += [{"patient": "C999", "vendor": "C", "labelled": False}]
    return pd.DataFrame(rows)


def test_split_is_fixed_disjoint_and_patient_level():
    a = split.make_split(_cases(), 2026, 30, 15, "B", ["A", "C", "D"])
    b = split.make_split(_cases(), 2026, 30, 15, "B", ["A", "C", "D"])
    assert a["test"] == b["test"] and a["train"] == b["train"]
    assert len(a["test"]) == 30 and len(a["val"]) == 15 and len(a["train"]) == 15
    assert not (set(a["test"]) & set(a["train"])) and not (set(a["val"]) & set(a["train"])) and not (set(a["test"]) & set(a["val"]))
    assert "C999" not in a["real"]["C"]                            # unlabelled patients are never used
    with pytest.raises(SystemExit):
        split.make_split(_cases(n_b=20), 2026, 30, 15, "B", ["A", "C", "D"])


def test_writing_an_arm_replaces_only_that_arm(tmp_path):
    (tmp_path / "per_model").mkdir()
    row = dict(model_id="m01", vendor="B", corruption="none", severity=0, patient="p", phase="ED",
               structure="LV", hd95=np.nan, pred_empty=False)
    results.write_arm(tmp_path, "m01", "clean", pd.DataFrame([{**row, "arm": "clean", "dice": 0.9}]))
    results.write_arm(tmp_path, "m01", "real", pd.DataFrame([{**row, "arm": "real", "dice": 0.5}]))
    results.write_arm(tmp_path, "m01", "clean", pd.DataFrame([{**row, "arm": "clean", "dice": 0.8}]))
    df = results.load_all(tmp_path)
    assert len(df) == 2
    assert df.set_index("arm")["dice"].to_dict() == {"clean": 0.8, "real": 0.5}
