"""The analysis must reproduce Subsection 6.6 exactly. These tests pin that down."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ssp import analyze, config, results, synth


def _rows(model, arm, vendor, corruption, severity, patient, dice):
    return [(model, arm, vendor, corruption, severity, patient, ph, s, dice, np.nan, False)
            for ph in ("ED", "ES") for s in ("LV", "MYO", "RV")]


def _tiny():
    r = []
    for p in ("p1", "p2"):
        r += _rows("m1", "clean", "B", "none", 0, p, 0.9)
        r += _rows("m1", "sim", "B", "bias", 1, p, 0.8)
        r += _rows("m1", "sim", "B", "bias", 2, p, 0.6)
    for p in ("a1", "a2", "a3"):
        r += _rows("m1", "real", "A", "none", 0, p, 0.7)
    r += _rows("m1", "real", "C", "none", 0, "c1", 0.5)
    return pd.DataFrame(r, columns=results.COLUMNS)


def _cond(df):
    return analyze.condition_means(analyze.case_scores(df))


def test_drops_follow_equations_4_and_5():
    d = analyze.drops(_cond(_tiny())).loc["m1"]
    assert d["clean"] == pytest.approx(0.9)
    assert d["d_sim"] == pytest.approx(0.9 - 0.7)            # unweighted mean of the two conditions
    assert d["d_real"] == pytest.approx(0.9 - 0.6)           # vendors weigh equally: (0.7 + 0.5) / 2, not pooled 0.65


def test_severity_weighting_and_relative_drop():
    cond = _cond(_tiny())
    w = analyze.drops(cond, alpha=0.5).loc["m1"]
    assert w["d_sim"] == pytest.approx(0.9 - (0.8 * 0.5 + 0.6 * 0.25) / 0.75)
    rel = analyze.drops(cond, relative=True).loc["m1"]
    assert rel["d_sim"] == pytest.approx(0.2 / 0.9)


def test_filters_by_vendor_and_corruption():
    cond = _cond(_tiny())
    assert analyze.drops(cond, vendors=["C"]).loc["m1", "d_real"] == pytest.approx(0.4)
    assert analyze.drops(cond, severities=[2]).loc["m1", "d_sim"] == pytest.approx(0.3)


def test_hd95_degradation_is_increase():
    df = _tiny()
    df["hd95"] = np.where(df["arm"] == "clean", 2.0, 5.0)
    cond = analyze.condition_means(analyze.case_scores(df, value="hd95"), value="hd95")
    d = analyze.drops(cond, value="hd95", higher_is_better=False).loc["m1"]
    assert d["d_sim"] == pytest.approx(3.0) and d["d_real"] == pytest.approx(3.0)


def test_decision_bands(home):
    cfg = config.analysis()
    assert analyze.band(0.70, cfg) == "supported"
    assert analyze.band(0.699, cfg) == "inconclusive"
    assert analyze.band(0.40, cfg) == "inconclusive"
    assert analyze.band(0.399, cfg) == "not supported"
    assert analyze.band(float("nan"), cfg) == "undefined"


def test_spearman_matches_scipy_and_handles_constant_input():
    rng = np.random.default_rng(0)
    x, y = rng.random(12), rng.random(12)
    assert analyze.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    assert np.isnan(analyze.spearman(np.ones(12), y))


def test_permutation_p_value_extremes():
    x = np.arange(10.0)
    assert analyze.permutation_p(x, x, 20000, 0) < 0.001
    assert analyze.permutation_p(x, -x, 20000, 0) > 0.99


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    x = rng.random(12)
    y = x + rng.normal(0, 0.2, 12)
    lo, hi = analyze.bootstrap_ci(x, y, 5000, 0)
    assert lo <= analyze.spearman(x, y) <= hi


def test_strong_relation_is_supported(home):
    out, table = analyze.run(synth.make(rho=0.97, spread=0.04, seed=3), config.analysis())
    assert out["F"]["band"] == "supported"
    assert out["F"]["rho"] >= 0.70 and out["F"]["p_one_sided"] < 0.05
    assert out["C7"]["interpretable"]
    assert len(table) == 12 and set(out["C3"]) == {"A", "C", "D"}
    assert set(out["C4"]["by_corruption"]) == {"bias", "blur", "gamma", "resolution"}


def test_no_relation_is_not_supported(home):
    out, _ = analyze.run(synth.make(rho=0.0, spread=0.04, seed=1), config.analysis())
    assert out["F"]["rho"] < 0.70 and out["F"]["band"] != "supported"


def test_identical_models_are_uninterpretable(home):
    out, _ = analyze.run(synth.make(rho=0.9, spread=0.0, seed=2), config.analysis())
    assert not out["C7"]["interpretable"]
    assert out["F"]["band"].startswith("uninterpretable")


def test_baseline_floor_excludes_failed_models(home):
    df = synth.make(rho=0.9, seed=4)
    df.loc[df["model_id"] == "m05", "dice"] *= 0.5
    cfg = {**config.analysis(), "min_baseline_dice": 0.7}
    out, table = analyze.run(df, cfg)
    assert out["baseline"]["excluded_models"] == ["m05"] and "m05" not in table.index


def test_model_missing_an_arm_is_reported_not_analysed(home):
    df = synth.make(rho=0.9, seed=5)
    df = df[~((df["model_id"] == "m02") & (df["arm"] == "real"))]
    out, table = analyze.run(df, config.analysis())
    assert out["models_missing_an_arm"] == ["m02"] and len(table) == 11


def test_c1_only_when_hd95_present(home):
    cfg = config.analysis()
    assert analyze.run(synth.make(seed=6), cfg)[0]["C1"] is None
    assert analyze.run(synth.make(seed=6, hd95=True), cfg)[0]["C1"]["n_models"] == 12


def test_report_and_outputs_are_written(home):
    out, table = analyze.run(synth.make(seed=7), config.analysis())
    text = analyze.report(out, table)
    assert "Final metric F" in text and "| m01 |" in text
    analyze.scatter(table, out, home / "scatter.png")
    assert (home / "scatter.png").stat().st_size > 1000
