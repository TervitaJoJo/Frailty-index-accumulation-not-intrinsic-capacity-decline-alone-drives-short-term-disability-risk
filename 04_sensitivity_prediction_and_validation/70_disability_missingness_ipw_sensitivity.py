"""Baseline-observable IPW sensitivity for incident disability prediction.

This stage asks whether complete-case IC prediction results are sensitive to
selection into the complete predictor set. The inclusion model uses only
baseline-observable age, sex, education, baseline FI and domain-observation
flags. Stabilised inverse-probability weights are clipped for numerical
stability. No person-level weights or predictions are exported.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SEED = 20260924
BASE_TERMS = ["age_z", "sex", "education_z", "baseline_fi_z"]
FULL_TERMS = BASE_TERMS + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
CONTINUOUS = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]


def load_stage66():
    path = OUT / "66_incident_disability_prediction.py"
    spec = importlib.util.spec_from_file_location("stage66_for_ipw", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def harmonize_sex(d):
    out = d.copy()
    x = pd.to_numeric(out["sex"], errors="coerce")
    vals = set(x.dropna().unique().tolist())
    if vals and vals.issubset({1.0, 2.0}):
        out["sex"] = (x - 1.0).where(x.isin([1.0, 2.0]))
    return out


def standardize(d):
    out = d.copy()
    for c in CONTINUOUS:
        x = pd.to_numeric(out[c], errors="coerce")
        mu, sd = float(x.mean()), float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out[c + "_z"] = (x - mu) / sd
    return out


def rank_weighted_auc(y, p, w):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); w = np.asarray(w, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float); ranks[order] = np.arange(1, len(p) + 1)
    # Tie correction is negligible for the continuous linear predictor; use
    # mid-ranks to remain correct if predictions tie.
    sr = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    return float((w[pos] * (sr[pos] - 0.5 * w[pos])).sum() / (w[pos].sum() * w[neg].sum())) if False else float(
        ((w[pos, None] * w[neg][None, :]) * (p[pos, None] > p[neg][None, :])).sum() +
        0.5 * ((w[pos, None] * w[neg][None, :]) * (p[pos, None] == p[neg][None, :])).sum()
    ) / float(w[pos].sum() * w[neg].sum())


def weighted_auc(y, p, w):
    # Efficient rank formulation with weighted cumulative negatives.
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); w = np.asarray(w, dtype=float)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    yy, ww = y[order], w[order]
    neg_cum = np.cumsum(ww * (yy == 0)) - ww * (yy == 0)
    pos_concord = (ww * (yy == 1) * neg_cum).sum()
    # Add half of tied pairs.
    tied = 0.0
    for _, idx in pd.Series(p).groupby(p).groups.items():
        z = np.asarray(idx, dtype=int); wp = w[z][y[z] == 1].sum(); wn = w[z][y[z] == 0].sum(); tied += wp * wn
    return float((pos_concord + 0.5 * tied) / (w[y == 1].sum() * w[y == 0].sum()))


def weighted_brier(y, p, w):
    return float(np.average((np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2, weights=np.asarray(w, dtype=float)))


def calibration(y, p, w):
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit, y, sample_weight=w)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def main():
    mod = load_stage66()
    builders = {"ELSA": mod.build_elsa, "CHARLS": mod.build_charls, "HRS": mod.build_hrs}
    results, diagnostics = [], []
    for cohort, fn in builders.items():
        d, info = fn()
        d = harmonize_sex(standardize(d))
        d["complete_ic_model"] = d[FULL_TERMS].notna().all(axis=1).astype(int)
        # The frame already contains participants with baseline independence
        # and an observed follow-up endpoint; only predictor completeness is
        # reweighted here.
        eligible = d.dropna(subset=["event"]).copy()
        covars = ["age", "sex", "education", "baseline_fi"]
        X = eligible[covars].copy()
        for c in covars:
            x = pd.to_numeric(X[c], errors="coerce")
            X[c] = x.fillna(x.median() if x.notna().any() else 0.0)
        for c in ["cognition", "locomotion", "grip_vitality", "psychological"]:
            X[c + "_observed"] = eligible[c].notna().astype(float)
        yinc = eligible["complete_ic_model"].astype(int).to_numpy()
        if len(np.unique(yinc)) < 2:
            continue
        incmod = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(X, yinc)
        pcomp = np.clip(incmod.predict_proba(X)[:, 1], 0.05, 0.99)
        sw = np.clip(yinc.mean() / pcomp, 0.1, 10.0)
        cc = eligible.loc[eligible.complete_ic_model.eq(1)].copy()
        # Map the inclusion weights back to complete cases without writing them.
        cc_w = sw[eligible.complete_ic_model.to_numpy() == 1]
        for model, terms in [("base", BASE_TERMS), ("base_plus_IC", FULL_TERMS)]:
            # Fit the apparent complete-case model and the IPW model separately.
            m_unw = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(cc[terms], cc.event.astype(int))
            p_unw = m_unw.predict_proba(cc[terms])[:, 1]
            cal_i_u, cal_s_u = calibration(cc.event.to_numpy(dtype=int), p_unw, np.ones(len(cc)))
            results.append({"cohort": cohort, "window": info["window"], "model": model, "weighting": "complete_case_unweighted",
                            "n": int(len(cc)), "events": int(cc.event.sum()), "auc": float(roc_auc_score(cc.event, p_unw)),
                            "brier": weighted_brier(cc.event, p_unw, np.ones(len(cc))), "calibration_intercept": cal_i_u,
                            "calibration_slope": cal_s_u, "mean_predicted": float(np.mean(p_unw)), "observed": float(cc.event.mean())})
            m_ipw = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(cc[terms], cc.event.astype(int), sample_weight=cc_w)
            p_ipw = m_ipw.predict_proba(cc[terms])[:, 1]
            cal_i_w, cal_s_w = calibration(cc.event.to_numpy(dtype=int), p_ipw, cc_w)
            results.append({"cohort": cohort, "window": info["window"], "model": model, "weighting": "baseline_observable_IPW",
                            "n": int(len(cc)), "events": int(cc.event.sum()), "auc": weighted_auc(cc.event, p_ipw, cc_w),
                            "brier": weighted_brier(cc.event, p_ipw, cc_w), "calibration_intercept": cal_i_w,
                            "calibration_slope": cal_s_w, "mean_predicted": float(np.average(p_ipw, weights=cc_w)),
                            "observed": float(np.average(cc.event, weights=cc_w))})
        diagnostics.append({"cohort": cohort, "window": info["window"], "eligible_n": int(len(eligible)),
                            "complete_n": int(len(cc)), "complete_rate": float(yinc.mean()), "weight_mean": float(sw.mean()),
                            "weight_p99": float(np.quantile(sw, .99)), "complete_weight_mean": float(cc_w.mean()),
                            "complete_weight_max": float(cc_w.max())})

    metrics = pd.DataFrame(results); diag = pd.DataFrame(diagnostics)
    metrics.to_csv(TAB / "incident_disability_missingness_ipw_metrics.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(TAB / "incident_disability_missingness_ipw_diagnostics.csv", index=False, encoding="utf-8-sig")
    # A compact delta table facilitates manuscript QA.
    rows = []
    for (cohort, model), g in metrics.groupby(["cohort", "model"]):
        u = g.loc[g.weighting == "complete_case_unweighted"].iloc[0]
        w = g.loc[g.weighting == "baseline_observable_IPW"].iloc[0]
        rows.append({"cohort": cohort, "model": model, "auc_unweighted": u.auc, "auc_IPW": w.auc,
                     "delta_auc_IPW_minus_unweighted": w.auc - u.auc, "brier_unweighted": u.brier,
                     "brier_IPW": w.brier, "delta_brier_IPW_minus_unweighted": w.brier - u.brier})
    pd.DataFrame(rows).to_csv(TAB / "incident_disability_missingness_ipw_incremental.csv", index=False, encoding="utf-8-sig")
    memo = f"""# Incident-disability missingness IPW sensitivity

This sensitivity models selection into the complete IC predictor set from
baseline-observable age, sex, education, baseline FI and four domain-observation
flags. Stabilised inverse-probability weights were clipped to [0.1, 10]. The
eligible frame already required baseline ADL/IADL independence and an observed
follow-up endpoint; the analysis therefore targets predictor completeness rather
than imputing an unobserved disability outcome.

The weighted model and weighted AUC/Brier summaries are robustness diagnostics.
They do not correct structural module absence and do not replace the primary
partial-metric latent-score analysis.

Rows: {len(metrics)} metric rows across {len(diagnostics)} cohorts.
"""
    (OUT / "incident_disability_missingness_ipw_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 34, "script": "70_disability_missingness_ipw_sensitivity.py", "cohorts": list(builders),
           "ipw_covariates": ["age", "sex", "education", "baseline_fi", "four domain observed flags"],
           "weight_clip": [0.1, 10.0], "output_level": "aggregate only"}
    (OUT / "incident_disability_missingness_ipw_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False)); print("\nDiagnostics:"); print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
