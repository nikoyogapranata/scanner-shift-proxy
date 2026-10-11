"""Compute the final metric F and the critical metrics C1 to C7 (Subsection 6.6).

    python -m ssp.analyze                 # real results in results/
    python -m ssp.analyze --mock          # results from the mock dataset
    python -m ssp.analyze --results-dir results_demo

Reads the per-model result tables and writes, into <results>/analysis/:
    report.md            the full report, readable as is
    summary.json         the same numbers, machine-readable
    per_model.csv        one row per model: clean Dice, both drops, both ranks
    scatter.png          simulated drop against real drop, one point per model

No model is run here. Everything is regrouping of stored scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, paths, results

STRUCTURES = ["LV", "MYO", "RV"]
CASE_KEYS = ["model_id", "arm", "vendor", "corruption", "severity", "patient"]
COND_KEYS = ["model_id", "arm", "vendor", "corruption", "severity"]


# ----------------------------------------------------------------------------
# From rows to model-level drops (Equations 2 to 5)
# ----------------------------------------------------------------------------
def case_scores(df: pd.DataFrame, value: str = "dice", structures: list[str] | None = None) -> pd.DataFrame:
    """Equation 2: one score per model, patient and condition (mean over structures and phases)."""
    d = df if structures is None else df[df["structure"].isin(structures)]
    return d.groupby(CASE_KEYS, as_index=False)[value].mean()


def condition_means(cases: pd.DataFrame, value: str = "dice") -> pd.DataFrame:
    """Equation 3: mean over patients, per model and condition."""
    return cases.groupby(COND_KEYS, as_index=False)[value].mean()


def drops(cond: pd.DataFrame, value: str = "dice", *, vendors=None, corruptions=None, severities=None,
          alpha: float | None = None, relative: bool = False, higher_is_better: bool = True) -> pd.DataFrame:
    """Equations 4 and 5: clean score, simulated drop and real drop for every model.

    alpha       weight severity s by alpha**s (C6); None means unweighted
    relative    divide each drop by the clean score (C6)
    """
    clean = cond[cond["arm"] == "clean"].set_index("model_id")[value]

    sim = cond[cond["arm"] == "sim"]
    if corruptions is not None:
        sim = sim[sim["corruption"].isin(corruptions)]
    if severities is not None:
        sim = sim[sim["severity"].isin(severities)]
    if alpha is None:
        sim_mean = sim.groupby("model_id")[value].mean()
    else:
        w = alpha ** sim["severity"].astype(float)
        sim_mean = (sim[value] * w).groupby(sim["model_id"]).sum() / w.groupby(sim["model_id"]).sum()

    real = cond[cond["arm"] == "real"]
    if vendors is not None:
        real = real[real["vendor"].isin(vendors)]
    real_mean = real.groupby(["model_id", "vendor"])[value].mean().groupby("model_id").mean()  # vendors weigh equally

    out = pd.DataFrame({"clean": clean, "sim": sim_mean, "real": real_mean}).dropna()
    sign = 1.0 if higher_is_better else -1.0
    out["d_sim"] = sign * (out["clean"] - out["sim"])
    out["d_real"] = sign * (out["clean"] - out["real"])
    if relative:
        out["d_sim"] /= out["clean"]
        out["d_real"] /= out["clean"]
    return out


# ----------------------------------------------------------------------------
# Correlation and its uncertainty (Equation 6, C5)
# ----------------------------------------------------------------------------
def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _rowwise_spearman(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    rx = stats.rankdata(bx, axis=1)
    ry = stats.rankdata(by, axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(1) / den, np.nan)


def bootstrap_ci(x, y, n: int, seed: int, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap over models, keeping each model's two drops together."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    rho = _rowwise_spearman(x[idx], y[idx])
    lo, hi = np.nanpercentile(rho, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def permutation_p(x, y, n: int, seed: int) -> float:
    """One-sided p-value for H1: rho > 0, by permuting the pairing of the models."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = spearman(x, y)
    if np.isnan(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n, len(y))), axis=1)
    rho = _rowwise_spearman(np.broadcast_to(x, (n, len(x))), y[perm])
    return float((1 + np.sum(rho >= observed - 1e-12)) / (1 + n))


def band(rho: float, cfg: dict) -> str:
    if np.isnan(rho):
        return "undefined"
    if rho >= cfg["decision"]["supported_at"]:
        return "supported"
    if rho < cfg["decision"]["rejected_below"]:
        return "not supported"
    return "inconclusive"


def leave_one_out(d: pd.DataFrame, cfg: dict) -> dict:
    full_band = band(spearman(d["d_sim"], d["d_real"]), cfg)
    rows = []
    for m in d.index:
        rest = d.drop(index=m)
        rho = spearman(rest["d_sim"], rest["d_real"])
        rows.append({"dropped": m, "rho": rho, "band": band(rho, cfg)})
    rhos = [r["rho"] for r in rows]
    return {
        "min": float(np.nanmin(rhos)), "max": float(np.nanmax(rhos)),
        "band_changes_when_dropping": [r["dropped"] for r in rows if r["band"] != full_band],
    }


def correlate(d: pd.DataFrame, cfg: dict, with_ci: bool = True) -> dict:
    rho = spearman(d["d_sim"], d["d_real"])
    out = {"n_models": int(len(d)), "rho": rho}
    if with_ci and len(d) >= 4 and not np.isnan(rho):
        out["ci95"] = bootstrap_ci(d["d_sim"], d["d_real"], cfg["bootstrap_models"], cfg["seed"])
    return out


# ----------------------------------------------------------------------------
# C7: are the inputs meaningful?
# ----------------------------------------------------------------------------
def patient_noise(cases: pd.DataFrame, value: str, n_boot: int, seed: int, higher_is_better: bool = True) -> pd.DataFrame:
    """Standard error of each model's two drops, from a bootstrap over patients."""
    rng = np.random.default_rng(seed)
    sign = 1.0 if higher_is_better else -1.0
    rows = []
    for model_id, g in cases.groupby("model_id"):
        clean = g[g["arm"] == "clean"].set_index("patient")[value]
        sim = g[g["arm"] == "sim"].groupby("patient")[value].mean()      # mean over the K x S conditions
        paired = pd.concat([clean, sim], axis=1, keys=["clean", "sim"]).dropna()
        c, s = paired["clean"].to_numpy(), paired["sim"].to_numpy()
        n = len(c)
        if n < 2:
            continue
        idx = rng.integers(0, n, size=(n_boot, n))
        clean_b = c[idx].mean(1)
        d_sim_b = sign * (clean_b - s[idx].mean(1))                      # same patients in both terms

        vendor_means = []
        for _, gv in g[g["arm"] == "real"].groupby("vendor"):
            v = gv[value].to_numpy()
            vendor_means.append(v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1))
        if not vendor_means:
            continue
        d_real_b = sign * (clean_b - np.mean(vendor_means, axis=0))
        rows.append({"model_id": model_id, "se_sim": float(d_sim_b.std(ddof=1)), "se_real": float(d_real_b.std(ddof=1))})
    return pd.DataFrame(rows).set_index("model_id")


def validity(d: pd.DataFrame, noise: pd.DataFrame) -> dict:
    sd_sim, sd_real = float(d["d_sim"].std(ddof=1)), float(d["d_real"].std(ddof=1))
    se_sim, se_real = float(noise["se_sim"].mean()), float(noise["se_real"].mean())
    return {
        "between_model_sd": {"sim": sd_sim, "real": sd_real},
        "mean_patient_se": {"sim": se_sim, "real": se_real},
        "spread_exceeds_noise": {"sim": bool(sd_sim > se_sim), "real": bool(sd_real > se_real)},
        "interpretable": bool(sd_sim > se_sim and sd_real > se_real),
        "share_negative_drop": {"sim": float((d["d_sim"] < 0).mean()), "real": float((d["d_real"] < 0).mean())},
    }


# ----------------------------------------------------------------------------
# The whole analysis
# ----------------------------------------------------------------------------
def run(df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    seed = cfg["seed"]
    out: dict = {"decision_rule": cfg["decision"]}

    cases = case_scores(df)
    cond = condition_means(cases)

    # C7 (first part): baseline adequacy and exclusion of failed training runs
    baseline = cond[cond["arm"] == "clean"].set_index("model_id")["dice"]
    floor = cfg.get("min_baseline_dice")
    excluded = sorted(baseline[baseline < floor].index) if floor is not None else []
    if excluded:
        df = df[~df["model_id"].isin(excluded)]
        cases, cond = case_scores(df), condition_means(case_scores(df))
    out["baseline"] = {"min_baseline_dice": floor, "excluded_models": excluded,
                       "clean_dice": {k: float(v) for k, v in baseline.items()}}

    d = drops(cond)
    incomplete = sorted(set(df["model_id"]) - set(d.index))
    out["models_missing_an_arm"] = incomplete
    if len(d) < 4:
        raise SystemExit(f"Only {len(d)} models have clean, sim and real results. Need at least 4 to correlate.")

    # F and C5
    rho = spearman(d["d_sim"], d["d_real"])
    out["F"] = {
        "n_models": int(len(d)), "rho": rho, "band": band(rho, cfg),
        "ci95": bootstrap_ci(d["d_sim"], d["d_real"], cfg["bootstrap_models"], seed),
        "p_one_sided": permutation_p(d["d_sim"], d["d_real"], cfg["permutations"], seed),
        "leave_one_out": leave_one_out(d, cfg),
    }

    # C7 (second part): spread against noise, sign of drops
    out["C7"] = validity(d, patient_noise(cases, "dice", cfg["bootstrap_patients"], seed))
    if not out["C7"]["interpretable"]:
        out["F"]["band"] = "uninterpretable (C7 failed)"

    # C2: per structure
    out["C2"] = {s: correlate(drops(condition_means(case_scores(df, structures=[s]))), cfg) for s in STRUCTURES}
    # C3: per unseen vendor
    out["C3"] = {v: correlate(drops(cond, vendors=[v]), cfg) for v in sorted(cond.loc[cond["arm"] == "real", "vendor"].unique())}
    # C4: per corruption family, then per severity level
    sim = cond[cond["arm"] == "sim"]
    out["C4"] = {
        "by_corruption": {c: correlate(drops(cond, corruptions=[c]), cfg) for c in sorted(sim["corruption"].unique())},
        "by_severity": {int(s): correlate(drops(cond, severities=[s]), cfg) for s in sorted(sim["severity"].unique())},
    }
    # C6: one analytic choice changed at a time
    weighted, relative = drops(cond, alpha=cfg["severity_weight_alpha"]), drops(cond, relative=True)
    out["C6"] = {
        "kendall_tau": float(stats.kendalltau(d["d_sim"], d["d_real"]).statistic),
        "pearson_r": float(stats.pearsonr(d["d_sim"], d["d_real"]).statistic),
        "relative_drop_rho": spearman(relative["d_sim"], relative["d_real"]),
        "severity_weighted_rho": spearman(weighted["d_sim"], weighted["d_real"]),
    }
    # C1: boundary-based, only if HD95 was computed
    if df["hd95"].notna().any():
        h_cases = case_scores(df, value="hd95")
        dh = drops(condition_means(h_cases, value="hd95"), value="hd95", higher_is_better=False)
        out["C1"] = {**correlate(dh, cfg), "band": band(spearman(dh["d_sim"], dh["d_real"]), cfg),
                     "empty_predictions": int(df["pred_empty"].astype(bool).sum())}
    else:
        out["C1"] = None

    table = d.copy()
    table["rank_sim"] = table["d_sim"].rank()
    table["rank_real"] = table["d_real"].rank()
    return out, table


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def _fmt(c: dict | None) -> str:
    if not c or np.isnan(c.get("rho", float("nan"))):
        return "not available"
    ci = f" (95% CI {c['ci95'][0]:.2f} to {c['ci95'][1]:.2f})" if "ci95" in c else ""
    return f"{c['rho']:.2f}{ci}, n = {c['n_models']}"


def report(out: dict, table: pd.DataFrame) -> str:
    F, C7 = out["F"], out["C7"]
    L = ["# Analysis report", "",
         f"Models analysed: {F['n_models']}",
         f"Excluded by baseline floor ({out['baseline']['min_baseline_dice']}): {out['baseline']['excluded_models'] or 'none'}",
         f"Models missing an arm (not analysed): {out['models_missing_an_arm'] or 'none'}", "",
         "## Final metric F", "",
         f"- Spearman correlation of simulated and real Dice drop: **{F['rho']:.3f}**",
         f"- Outcome under the pre-specified rule: **{F['band']}**",
         f"- 95% bootstrap interval: {F['ci95'][0]:.2f} to {F['ci95'][1]:.2f}",
         f"- One-sided permutation p-value: {F['p_one_sided']:.4f}",
         f"- Leave-one-model-out range: {F['leave_one_out']['min']:.2f} to {F['leave_one_out']['max']:.2f}",
         f"- Outcome changes when dropping: {F['leave_one_out']['band_changes_when_dropping'] or 'no single model'}", "",
         "## C7 Input validity", "",
         f"- Between-model SD of the drop: simulated {C7['between_model_sd']['sim']:.4f}, real {C7['between_model_sd']['real']:.4f}",
         f"- Mean patient-level SE of one model's drop: simulated {C7['mean_patient_se']['sim']:.4f}, real {C7['mean_patient_se']['real']:.4f}",
         f"- Spread exceeds noise: simulated {C7['spread_exceeds_noise']['sim']}, real {C7['spread_exceeds_noise']['real']}",
         f"- F interpretable: **{C7['interpretable']}**",
         f"- Share of models with a negative drop: simulated {C7['share_negative_drop']['sim']:.0%}, real {C7['share_negative_drop']['real']:.0%}", "",
         "## C2 Per structure", ""]
    L += [f"- {k}: {_fmt(v)}" for k, v in out["C2"].items()]
    L += ["", "## C3 Per unseen vendor", ""] + [f"- Vendor {k}: {_fmt(v)}" for k, v in out["C3"].items()]
    L += ["", "## C4 Per corruption", ""] + [f"- {k}: {_fmt(v)}" for k, v in out["C4"]["by_corruption"].items()]
    L += ["", "Per severity level:", ""] + [f"- Level {k}: {_fmt(v)}" for k, v in out["C4"]["by_severity"].items()]
    L += ["", "## C6 Sensitivity to analytic choices", "",
          f"- Kendall tau: {out['C6']['kendall_tau']:.2f}",
          f"- Pearson r: {out['C6']['pearson_r']:.2f}",
          f"- Spearman on relative drops: {out['C6']['relative_drop_rho']:.2f}",
          f"- Spearman with severity weighting: {out['C6']['severity_weighted_rho']:.2f}", "",
          "## C1 Boundary-based (HD95)", ""]
    L += [f"- {_fmt(out['C1'])}, outcome: {out['C1']['band']}, empty predictions: {out['C1']['empty_predictions']}"] if out["C1"] \
        else ["- Not computed. Re-run evaluation with --hd95 to fill this in (Tier 3)."]
    L += ["", "## Per-model table", "", "| model | clean Dice | sim drop | real drop | rank sim | rank real |", "|---|---|---|---|---|---|"]
    L += [f"| {m} | {r.clean:.3f} | {r.d_sim:.3f} | {r.d_real:.3f} | {r.rank_sim:.0f} | {r.rank_real:.0f} |" for m, r in table.iterrows()]
    return "\n".join(L) + "\n"


def scatter(table: pd.DataFrame, out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, series = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=surface)
    ax.set_facecolor(surface)
    ax.scatter(table["d_sim"], table["d_real"], s=64, color=series, edgecolor=surface, linewidth=2, zorder=3)
    for m, r in table.iterrows():
        ax.annotate(m, (r.d_sim, r.d_real), xytext=(6, 5), textcoords="offset points", fontsize=8, color=muted)
    F = out["F"]
    ax.set_title(f"Simulated vs real Dice drop, one point per model\n"
                 f"Spearman {F['rho']:.2f} (95% CI {F['ci95'][0]:.2f} to {F['ci95'][1]:.2f}), outcome: {F['band']}",
                 fontsize=10, color=ink, loc="left")
    ax.set_xlabel("Dice drop under simulated shift", color=muted)
    ax.set_ylabel("Dice drop under real cross-vendor shift", color=muted)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.tick_params(colors=muted, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", help="folder holding per_model/ (default: results/)")
    ap.add_argument("--mock", action="store_true", help="analyse results_mock/ instead of results/")
    args = ap.parse_args()

    rdir = Path(args.results_dir).resolve() if args.results_dir else paths.home() / ("results_mock" if args.mock else "results")
    cfg = config.analysis()
    out, table = run(results.load_all(rdir), cfg)

    adir = rdir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "report.md").write_text(report(out, table), encoding="utf-8")
    (adir / "summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    table.to_csv(adir / "per_model.csv")
    scatter(table, out, adir / "scatter.png")
    F = out["F"]
    print(f"F = {F['rho']:.3f}  (95% CI {F['ci95'][0]:.2f} to {F['ci95'][1]:.2f}),  outcome: {F['band']},  models: {F['n_models']}")
    print(f"Report written to {adir / 'report.md'}")


if __name__ == "__main__":
    main()
