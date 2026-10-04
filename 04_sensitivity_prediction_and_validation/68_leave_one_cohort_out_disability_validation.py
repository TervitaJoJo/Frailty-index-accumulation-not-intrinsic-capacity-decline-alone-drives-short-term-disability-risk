"""Leave-one-cohort-out transportability audit for incident disability.

The script reuses the frozen endpoint construction from 66_incident_disability_prediction.py.
It fits the base and IC-augmented logistic models in two cohorts and evaluates the
fixed coefficients in the held-out third cohort. The primary analysis uses the
within-cohort standardisation already used in the disability analysis. A sensitivity
analysis applies one pooled mean/SD from the two training cohorts to both training
and held-out cohorts, which makes the scale transport explicit.

Only aggregate metrics are exported; no identifiers, scores or person-level
predictions are written to disk.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
FIG = OUT / "supplementary_figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 20260924
BOOTSTRAP_B = 400
CONTINUOUS = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
BASE_TERMS = ["age_z", "sex", "education_z", "baseline_fi_z"]
FULL_TERMS = BASE_TERMS + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]


def load_stage66():
    path = OUT / "66_incident_disability_prediction.py"
    spec = importlib.util.spec_from_file_location("stage66_incident_disability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def standardize(d: pd.DataFrame, cols: list[str], reference: pd.DataFrame | None = None):
    """Add z-scored predictors and return parameters used."""
    out = d.copy()
    ref = d if reference is None else reference
    pars = {}
    for c in cols:
        x = pd.to_numeric(ref[c], errors="coerce")
        mu = float(x.mean())
        sd = float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out[c + "_z"] = (pd.to_numeric(out[c], errors="coerce") - mu) / sd
        pars[c] = {"mean": mu, "sd": sd}
    return out, pars


def harmonize_sex(d: pd.DataFrame) -> pd.DataFrame:
    """Put sex on the common 0/1 scale before pooled transport checks.

    The legacy stage-66 ELSA builder retains the source 1/2 coding, whereas
    CHARLS and HRS are already recoded to 0/1. This recode is applied only in
    this validation layer and is audited below; it does not alter the frozen
    stage-66 aggregate results.
    """
    out = d.copy()
    x = pd.to_numeric(out["sex"], errors="coerce")
    vals = set(x.dropna().unique().tolist())
    if vals and vals.issubset({1.0, 2.0}):
        out["sex"] = (x - 1.0).where(x.isin([1.0, 2.0]))
    return out


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q)).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def bootstrap_metrics(y: np.ndarray, p_base: np.ndarray, p_full: np.ndarray, key: str):
    rng = np.random.default_rng(SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100000)
    auc_b, auc_f, bri_b, bri_f = [], [], [], []
    if len(y) <= 100 or len(np.unique(y)) < 2:
        return auc_b, auc_f, bri_b, bri_f
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        auc_b.append(roc_auc_score(y[idx], p_base[idx]))
        auc_f.append(roc_auc_score(y[idx], p_full[idx]))
        bri_b.append(brier_score_loss(y[idx], p_base[idx]))
        bri_f.append(brier_score_loss(y[idx], p_full[idx]))
    return auc_b, auc_f, bri_b, bri_f


def paired_delta(y: np.ndarray, p_base: np.ndarray, p_full: np.ndarray, key: str):
    """Paired bootstrap uncertainty for ΔAUC and ΔBrier."""
    rng = np.random.default_rng(SEED + int(hashlib.sha256((key + "|delta").encode()).hexdigest()[:8], 16) % 100000)
    da, db = [], []
    if len(y) <= 100 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        da.append(roc_auc_score(y[idx], p_full[idx]) - roc_auc_score(y[idx], p_base[idx]))
        db.append(brier_score_loss(y[idx], p_full[idx]) - brier_score_loss(y[idx], p_base[idx]))
    return q025(da), q975(da), q025(db), q975(db)


def q025(x):
    return float(np.quantile(x, 0.025)) if len(x) else np.nan


def q975(x):
    return float(np.quantile(x, 0.975)) if len(x) else np.nan


def evaluate(y: np.ndarray, p: np.ndarray, cohort: str, window: str, model: str,
             sample: str, n_boot: int = BOOTSTRAP_B) -> dict:
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    brier = float(brier_score_loss(y, p))
    intercept, slope = calibration_stats(y, p)
    rng = np.random.default_rng(SEED + int(hashlib.sha256(f"{sample}|{cohort}|{model}".encode()).hexdigest()[:8], 16) % 100000)
    auc_b, bri_b = [], []
    if n_boot and len(y) > 100 and len(np.unique(y)) == 2:
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), size=len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            auc_b.append(roc_auc_score(y[idx], p[idx]))
            bri_b.append(brier_score_loss(y[idx], p[idx]))
    return {
        "sample": sample, "train_cohorts": "+".join(cohort[0]),
        "heldout_cohort": cohort[1], "window": window, "standardization": cohort[2],
        "model": model, "n": int(len(y)), "events": int(y.sum()),
        "event_rate": float(y.mean()), "auc_horizon_cindex": auc,
        "auc_low": q025(auc_b), "auc_high": q975(auc_b), "brier": brier,
        "brier_low": q025(bri_b), "brier_high": q975(bri_b),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "mean_predicted_risk": float(np.mean(p)), "observed_risk": float(np.mean(y)),
        "observed_expected_ratio": float(np.mean(y) / np.mean(p)) if np.mean(p) > 0 else np.nan,
    }


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame):
    tr = train.dropna(subset=["event", *FULL_TERMS]).copy()
    te = test.dropna(subset=["event", *FULL_TERMS]).copy()
    if len(tr) < 100 or len(te) < 50 or tr.event.nunique() < 2 or te.event.nunique() < 2:
        return None
    mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(tr[BASE_TERMS], tr.event.astype(int))
    mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(tr[FULL_TERMS], tr.event.astype(int))
    pb = mb.predict_proba(te[BASE_TERMS])[:, 1]
    pf = mf.predict_proba(te[FULL_TERMS])[:, 1]
    return tr, te, pb, pf, mb, mf


def main():
    stage66 = load_stage66()
    builders = {"ELSA": stage66.build_elsa, "CHARLS": stage66.build_charls, "HRS": stage66.build_hrs}
    raw = {}
    infos = {}
    for name, fn in builders.items():
        d, info = fn()
        raw[name] = harmonize_sex(d)
        infos[name] = info

    results = []
    incremental = []
    pars_out = {}
    cohort_names = list(builders)
    for heldout in cohort_names:
        train_names = [x for x in cohort_names if x != heldout]
        target_window = str(raw[heldout]["window"].iloc[0])
        # Two scaling schemes: primary within-cohort z scores and a pooled
        # training-reference scale for a stricter transportability check.
        for scale in ["within_cohort", "pooled_training_reference"]:
            scaled = {}
            pars = {}
            if scale == "within_cohort":
                for name in cohort_names:
                    scaled[name], pars[name] = standardize(raw[name], CONTINUOUS)
            else:
                ref = pd.concat([raw[x] for x in train_names], ignore_index=True)
                pooled_ref = ref.loc[:, CONTINUOUS]
                pooled_pars = {}
                for c in CONTINUOUS:
                    x = pd.to_numeric(pooled_ref[c], errors="coerce")
                    sd = float(x.std(ddof=1))
                    pooled_pars[c] = {"mean": float(x.mean()), "sd": sd if np.isfinite(sd) and sd > 0 else 1.0}
                for name in cohort_names:
                    scaled[name], _ = standardize(raw[name], CONTINUOUS, reference=pooled_ref)
                pars = {name: pooled_pars for name in cohort_names}
            train = pd.concat([scaled[x] for x in train_names], ignore_index=True)
            test = scaled[heldout]
            scored = fit_and_score(train, test)
            if scored is None:
                continue
            tr, te, pb, pf, mb, mf = scored
            descriptor = (tuple(train_names), heldout, scale)
            for model, p in [("base", pb), ("base_plus_IC", pf)]:
                results.append(evaluate(te.event.to_numpy(dtype=int), p, descriptor, target_window, model, "leave_one_cohort_out"))
            y = te.event.to_numpy(dtype=int)
            dauc_l, dauc_h, dbri_l, dbri_h = paired_delta(y, pb, pf, f"{'+'.join(train_names)}|{heldout}|{scale}")
            results[-2]["delta_auc_from_pair"] = np.nan
            results[-1]["delta_auc_from_pair"] = float(roc_auc_score(y, pf) - roc_auc_score(y, pb))
            results[-2]["train_n"] = int(len(tr)); results[-1]["train_n"] = int(len(tr))
            results[-2]["train_events"] = int(tr.event.sum()); results[-1]["train_events"] = int(tr.event.sum())
            incremental.append({
                "sample": "leave_one_cohort_out", "train_cohorts": "+".join(train_names),
                "heldout_cohort": heldout, "window": target_window, "standardization": scale,
                "n": int(len(te)), "events": int(y.sum()),
                "delta_auc": float(roc_auc_score(y, pf) - roc_auc_score(y, pb)),
                "delta_brier": float(brier_score_loss(y, pf) - brier_score_loss(y, pb)),
                "delta_auc_low": dauc_l, "delta_auc_high": dauc_h,
                "delta_brier_low": dbri_l, "delta_brier_high": dbri_h,
            })
            pars_out[f"{'+'.join(train_names)}_to_{heldout}_{scale}"] = pars

    metrics_df = pd.DataFrame(results)
    incremental_df = pd.DataFrame(incremental)
    metrics_df.to_csv(TAB / "leave_one_cohort_out_metrics.csv", index=False, encoding="utf-8-sig")
    incremental_df.to_csv(TAB / "leave_one_cohort_out_incremental.csv", index=False, encoding="utf-8-sig")
    (TAB / "leave_one_cohort_out_standardization_parameters.json").write_text(json.dumps(pars_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact forest-style figure of delta AUC and its bootstrap interval. This
    # is a transportability display, not a clinical decision curve.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        plot = []
        for _, row in incremental_df.iterrows():
            b = metrics_df[(metrics_df.train_cohorts == row.train_cohorts) &
                           (metrics_df.heldout_cohort == row.heldout_cohort) &
                           (metrics_df.standardization == row.standardization)]
            a1 = b.loc[b.model == "base", "auc_horizon_cindex"].iloc[0]
            a2 = b.loc[b.model == "base_plus_IC", "auc_horizon_cindex"].iloc[0]
            plot.append({"label": f"{row.train_cohorts} → {row.heldout_cohort}\n{row.standardization.replace('_', ' ')}",
                         "delta_auc": a2 - a1,
                         "delta_low": float(row.delta_auc_low),
                         "delta_high": float(row.delta_auc_high)})
        pdf = pd.DataFrame(plot)
        yloc = np.arange(len(pdf))[::-1]
        ax.axvline(0, color="#666666", lw=0.9)
        xerr = np.vstack([pdf.delta_auc - pdf.delta_low, pdf.delta_high - pdf.delta_auc])
        ax.errorbar(pdf.delta_auc, yloc, xerr=xerr, fmt="o", ms=6, color="#185a7a",
                    ecolor="#185a7a", elinewidth=1.3, capsize=3, zorder=3)
        ax.set_yticks(yloc); ax.set_yticklabels(pdf.label)
        ax.set_xlabel("Incremental horizon AUC (base + IC minus base)")
        ax.set_title("Leave-one-cohort-out transportability")
        fig.tight_layout()
        fig.savefig(FIG / "FigureS7_leave_one_cohort_out_delta_auc.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG / "FigureS7_leave_one_cohort_out_delta_auc.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        (OUT / "leave_one_cohort_out_figure_error.txt").write_text(str(exc), encoding="utf-8")

    memo = f"""# Leave-one-cohort-out incident disability validation

This analysis addresses transportability beyond apparent performance. For each
target cohort, the base model (age, sex, education and baseline outcome-disjoint
FI) and the IC-augmented model were fitted in the other two primary cohorts and
then evaluated with fixed coefficients in the held-out cohort. The endpoint is
the prespecified fixed-window incident ADL/IADL limitation outcome.

The primary scaling scheme standardises each continuous predictor within its own
cohort, consistent with the failed scalar-invariance gate. A stricter sensitivity
uses means and standard deviations estimated only in the two training cohorts for
both training and held-out data. Metrics are horizon AUC, Brier score,
calibration intercept and slope, and observed/expected risk; 400 nonparametric
bootstrap resamples provide uncertainty intervals for AUC and Brier score.

The analysis is a validation of fixed-window risk ranking and calibration. It does
not imply continuous-time onset prediction or a universal IC level across cohorts.

Rows generated: {len(metrics_df)} metric rows; {len(incremental_df)} incremental comparisons.
"""
    (OUT / "leave_one_cohort_out_validation_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 32, "script": "68_leave_one_cohort_out_disability_validation.py",
           "training_sets": "each pair of ELSA, CHARLS and HRS", "heldout": cohort_names,
           "standardization": ["within_cohort", "pooled_training_reference"],
           "metrics": ["horizon AUC", "Brier", "calibration intercept", "calibration slope", "observed/expected"],
           "bootstrap_replicates": BOOTSTRAP_B, "output_level": "aggregate only"}
    (OUT / "leave_one_cohort_out_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print("\nIncremental:")
    print(incremental_df.to_string(index=False))


if __name__ == "__main__":
    main()
