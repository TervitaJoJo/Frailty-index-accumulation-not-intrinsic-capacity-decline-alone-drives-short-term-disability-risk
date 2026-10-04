"""Cross-cohort calibration and predicted-risk strata for incident disability.

This stage reuses the frozen Stage 66 endpoint and Stage 68 transport models.
It exports only aggregate calibration and risk-stratum summaries plus figures;
no IDs, individual predictions or row-level records are written.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
FIG = OUT / "supplementary_figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 20260925
BOOT = 400
CONTINUOUS = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
BASE_TERMS = ["age_z", "sex", "education_z", "baseline_fi_z"]
FULL_TERMS = BASE_TERMS + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]


def load_stage68():
    p = OUT / "68_leave_one_cohort_out_disability_validation.py"
    spec = importlib.util.spec_from_file_location("stage68_calibration", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def calibration_stats(y, p):
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q)).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, np.asarray(y, dtype=int))
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def bootstrap_calibration(y, p, key):
    rng = np.random.default_rng(SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100000)
    ints, slopes = [], []
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        a, b = calibration_stats(y[idx], p[idx])
        if np.isfinite(a) and np.isfinite(b):
            ints.append(a); slopes.append(b)
    return (float(np.quantile(ints, .025)), float(np.quantile(ints, .975)),
            float(np.quantile(slopes, .025)), float(np.quantile(slopes, .975))) if ints else (np.nan, np.nan, np.nan, np.nan)


def calibration_row(y, p, sample, cohort, window, standardization, model, role):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    intercept, slope = calibration_stats(y, p)
    il, ih, sl, sh = bootstrap_calibration(y, p, f"{sample}|{cohort}|{window}|{standardization}|{model}")
    mean_p = float(np.mean(p)); obs = float(np.mean(y))
    return {"sample": sample, "role": role, "cohort": cohort, "window": window, "standardization": standardization, "model": model, "n": len(y), "events": int(y.sum()), "observed_risk": obs, "mean_predicted_risk": mean_p, "observed_expected_ratio": obs / mean_p if mean_p > 0 else np.nan, "calibration_intercept": intercept, "calibration_intercept_low": il, "calibration_intercept_high": ih, "calibration_slope": slope, "calibration_slope_low": sl, "calibration_slope_high": sh, "brier": float(brier_score_loss(y, p)), "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan}


def risk_strata(y, p, sample, cohort, window, standardization, model, role, groups=5):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    # Rank before qcut so tied predictions still form deterministic strata.
    g = pd.qcut(pd.Series(p).rank(method="first"), groups, labels=False) + 1
    rows = []
    for k, idx in pd.Series(np.arange(len(y))).groupby(g):
        ii = idx.to_numpy(dtype=int); yy = y[ii]; pp = p[ii]
        lo, hi = proportion_confint(int(yy.sum()), len(yy), alpha=0.05, method="wilson")
        rows.append({"sample": sample, "role": role, "cohort": cohort, "window": window, "standardization": standardization, "model": model, "risk_stratum": int(k), "n": len(ii), "events": int(yy.sum()), "mean_predicted_risk": float(pp.mean()), "observed_risk": float(yy.mean()), "observed_low": float(lo), "observed_high": float(hi), "observed_minus_predicted": float(yy.mean() - pp.mean()), "observed_expected_ratio": float(yy.mean() / pp.mean()) if pp.mean() > 0 else np.nan})
    return rows


def fit_train_predict(train, test):
    tr = train.dropna(subset=["event", *FULL_TERMS]).copy(); te = test.dropna(subset=["event", *FULL_TERMS]).copy()
    mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(tr[BASE_TERMS], tr.event.astype(int))
    mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(tr[FULL_TERMS], tr.event.astype(int))
    return tr, te, mb.predict_proba(te[BASE_TERMS])[:, 1], mf.predict_proba(te[FULL_TERMS])[:, 1], mb, mf


def main():
    stage68 = load_stage68(); stage66 = stage68.load_stage66()
    builders = {"ELSA": stage66.build_elsa, "CHARLS": stage66.build_charls, "HRS": stage66.build_hrs}
    raw = {name: stage68.harmonize_sex(fn()[0]) for name, fn in builders.items()}
    cal_rows = []; strata_rows = []

    # Leave-one-cohort-out: primary within-cohort scaling plus the stricter
    # pooled-training-reference sensitivity.
    for heldout in builders:
        train_names = [x for x in builders if x != heldout]
        for scale in ["within_cohort", "pooled_training_reference"]:
            scaled = {}
            if scale == "within_cohort":
                for name in builders:
                    scaled[name], _ = stage68.standardize(raw[name], CONTINUOUS)
            else:
                ref = pd.concat([raw[x] for x in train_names], ignore_index=True)
                for name in builders:
                    scaled[name], _ = stage68.standardize(raw[name], CONTINUOUS, reference=ref)
            train = pd.concat([scaled[x] for x in train_names], ignore_index=True); test = scaled[heldout]
            tr, te, pb, pf, _, _ = fit_train_predict(train, test)
            for model, p in [("base", pb), ("base_plus_IC", pf)]:
                cal_rows.append(calibration_row(te.event, p, "leave_one_cohort_out", heldout, str(te.window.iloc[0]), scale, model, "heldout_validation"))
                strata_rows.extend(risk_strata(te.event, p, "leave_one_cohort_out", heldout, str(te.window.iloc[0]), scale, model, "heldout_validation"))

    # Pooled development and fixed-coefficient SHARE validation, matching Stage 66.
    dev_parts = []
    for name in builders:
        z, _ = stage68.standardize(raw[name], CONTINUOUS); dev_parts.append(z)
    dev = pd.concat(dev_parts, ignore_index=True).dropna(subset=["event", *FULL_TERMS]).copy()
    mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(dev[BASE_TERMS], dev.event.astype(int))
    mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(dev[FULL_TERMS], dev.event.astype(int))
    for model, p in [("base", mb.predict_proba(dev[BASE_TERMS])[:, 1]), ("base_plus_IC", mf.predict_proba(dev[FULL_TERMS])[:, 1])]:
        cal_rows.append(calibration_row(dev.event, p, "development_pooled", "ELSA+CHARLS+HRS", "anchor_windows", "within_cohort", model, "apparent_development"))
        strata_rows.extend(risk_strata(dev.event, p, "development_pooled", "ELSA+CHARLS+HRS", "anchor_windows", "within_cohort", model, "apparent_development"))
    for target, window in [(7, "6->7"), (8, "6->8")]:
        sh, _ = stage66.build_share(target); sh = stage68.harmonize_sex(sh); sh, _ = stage68.standardize(sh, CONTINUOUS)
        sh = sh.dropna(subset=["event", *FULL_TERMS]).copy()
        for model, p in [("base", mb.predict_proba(sh[BASE_TERMS])[:, 1]), ("base_plus_IC", mf.predict_proba(sh[FULL_TERMS])[:, 1])]:
            cal_rows.append(calibration_row(sh.event, p, "external_validation", "SHARE", window, "within_cohort", model, "fixed_coefficient_external"))
            strata_rows.extend(risk_strata(sh.event, p, "external_validation", "SHARE", window, "within_cohort", model, "fixed_coefficient_external"))

    cal = pd.DataFrame(cal_rows); strata = pd.DataFrame(strata_rows)
    cal.to_csv(TAB / "cross_cohort_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(TAB / "cross_cohort_risk_strata.csv", index=False, encoding="utf-8-sig")

    # Primary figure: calibration by predicted-risk quintile for within-cohort
    # held-out validation and fixed-coefficient SHARE validation.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
        primary = strata[((strata["sample"] == "leave_one_cohort_out") & (strata["standardization"] == "within_cohort") & (strata["model"] == "base_plus_IC")) | ((strata["sample"] == "external_validation") & (strata["model"] == "base_plus_IC"))]
        panels = [("ELSA", "6->7"), ("CHARLS", "3->4"), ("HRS", "10->11"), ("SHARE", "6->7"), ("SHARE", "6->8")]
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharex=True, sharey=True)
        for ax, (cohort, window) in zip(axes, panels):
            q = primary[(primary["cohort"] == cohort) & (primary["window"] == window)]
            if q.empty:
                ax.axis("off"); continue
            ax.errorbar(q.mean_predicted_risk, q.observed_risk, yerr=[q.observed_risk-q.observed_low, q.observed_high-q.observed_risk], fmt="o-", color="#185a7a", capsize=2, lw=1.1, ms=4)
            ax.plot([0, .35], [0, .35], ls="--", color="#777", lw=.8)
            ax.set_title(f"{cohort} {window}", fontsize=9); ax.set_xlim(0, .35); ax.set_ylim(0, .35); ax.set_xlabel("Predicted")
        axes[0].set_ylabel("Observed risk")
        fig.suptitle("External and leave-one-cohort-out calibration by risk quintile", y=1.02, fontsize=11)
        fig.tight_layout(); fig.savefig(FIG / "FigureS8_cross_cohort_calibration_quintiles.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / "FigureS8_cross_cohort_calibration_quintiles.pdf", bbox_inches="tight"); plt.close(fig)

        # Compare base and IC-augmented risk strata in external validation.
        ext = strata[(strata["sample"] == "external_validation") & (strata["window"] == "6->7")]
        fig, ax = plt.subplots(figsize=(5.7, 4.2))
        for model, color, label in [("base", "#777777", "Base"), ("base_plus_IC", "#185a7a", "Base + IC")]:
            q = ext[ext["model"] == model]
            ax.errorbar(q.risk_stratum, q.observed_risk, yerr=[q.observed_risk-q.observed_low, q.observed_high-q.observed_risk], fmt="o-", color=color, capsize=2, lw=1.2, ms=4, label=label)
        ax.set_xticks(range(1, 6)); ax.set_xlabel("Predicted-risk quintile (SHARE 6→7)"); ax.set_ylabel("Observed incident disability risk"); ax.legend(frameon=False); ax.set_ylim(0, .35); ax.set_title("External risk stratification")
        fig.tight_layout(); fig.savefig(FIG / "FigureS9_external_risk_stratification.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / "FigureS9_external_risk_stratification.pdf", bbox_inches="tight"); plt.close(fig)
    except Exception as exc:
        (OUT / "cross_cohort_calibration_figure_error.txt").write_text(str(exc), encoding="utf-8")

    memo = """# Cross-cohort calibration and risk stratification

This stage reuses the frozen incident-disability model and fixed coefficients. Calibration intercepts and slopes were estimated in leave-one-cohort-out validation, pooled development, and fixed-coefficient SHARE validation; 400 bootstrap resamples provide uncertainty intervals. Risk-stratum tables use predicted-risk quintiles and Wilson intervals for observed event risk.

The primary display uses within-cohort standardisation, consistent with the failed scalar-invariance gate. The pooled-training-reference scale remains a sensitivity boundary analysis and is retained in the aggregate table. These displays assess transportability and risk ranking; they do not establish a clinical action threshold.
"""
    (OUT / "cross_cohort_calibration_risk_stratification_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 37, "script": "73_cross_cohort_calibration_risk_stratification.py", "calibration": ["intercept", "slope", "bootstrap percentile intervals", "Brier", "AUC"], "risk_stratification": "predicted-risk quintiles with Wilson observed-risk intervals", "samples": ["leave-one-cohort-out ELSA/CHARLS/HRS", "pooled development", "SHARE 6->7 and 6->8 fixed-coefficient validation"], "output_level": "aggregate only"}
    (OUT / "cross_cohort_calibration_risk_stratification_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(cal.to_string(index=False)); print("\nRisk strata:"); print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
