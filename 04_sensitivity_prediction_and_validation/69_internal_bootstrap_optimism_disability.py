"""Internal bootstrap optimism correction for the pooled disability models.

The primary development model is the pooled ELSA+CHARLS+HRS fixed-window
incident disability model from stage 66. Predictors are standardised within
cohort before pooling. A stratified-by-cohort bootstrap preserves the cohort
composition and estimates optimism in apparent AUC and Brier score. No
individual-level predictions are saved.
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


OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SEED = 20260924
B = 400
BASE_TERMS = ["age_z", "sex", "education_z", "baseline_fi_z"]
FULL_TERMS = BASE_TERMS + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
CONTINUOUS = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]


def load_stage66():
    path = OUT / "66_incident_disability_prediction.py"
    spec = importlib.util.spec_from_file_location("stage66_for_bootstrap", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def standardize(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    for c in CONTINUOUS:
        x = pd.to_numeric(out[c], errors="coerce")
        mu, sd = float(x.mean()), float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out[c + "_z"] = (x - mu) / sd
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, terms: list[str]):
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(train[terms], train.event.astype(int))
    return model.predict_proba(test[terms])[:, 1]


def metric(y, p):
    return float(roc_auc_score(y, p)), float(brier_score_loss(y, p))


def main():
    mod = load_stage66()
    ds = []
    for fn in [mod.build_elsa, mod.build_charls, mod.build_hrs]:
        d, _ = fn()
        d = standardize(d)
        d = d.dropna(subset=["event", *FULL_TERMS]).copy()
        ds.append(d)
    dev = pd.concat(ds, ignore_index=True)

    # Apparent performance of the prespecified model in the full development set.
    app = {}
    for name, terms in [("base", BASE_TERMS), ("base_plus_IC", FULL_TERMS)]:
        p = fit_predict(dev, dev, terms)
        app[name] = metric(dev.event.to_numpy(dtype=int), p)

    rng = np.random.default_rng(SEED)
    rows = []
    cohort_index = {g: np.flatnonzero(dev.cohort.to_numpy() == g) for g in dev.cohort.unique()}
    for b in range(B):
        pieces = []
        for g, idx in cohort_index.items():
            take = rng.integers(0, len(idx), size=len(idx))
            pieces.append(dev.iloc[idx[take]])
        boot = pd.concat(pieces, ignore_index=True)
        for name, terms in [("base", BASE_TERMS), ("base_plus_IC", FULL_TERMS)]:
            try:
                p_boot = fit_predict(boot, boot, terms)
                p_orig = fit_predict(boot, dev, terms)
                auc_app, bri_app = metric(boot.event.to_numpy(dtype=int), p_boot)
                auc_test, bri_test = metric(dev.event.to_numpy(dtype=int), p_orig)
                rows.append({"bootstrap": b + 1, "model": name,
                             "auc_bootstrap_apparent": auc_app, "auc_original_test": auc_test,
                             "auc_optimism": auc_app - auc_test,
                             "brier_bootstrap_apparent": bri_app, "brier_original_test": bri_test,
                             "brier_optimism": bri_app - bri_test})
            except Exception:
                continue

    boot_df = pd.DataFrame(rows)
    out = []
    for name in ["base", "base_plus_IC"]:
        b = boot_df[boot_df.model == name]
        mean_auc_opt = float(b.auc_optimism.mean())
        mean_bri_opt = float(b.brier_optimism.mean())
        # For the bootstrap distribution, use the original-sample test
        # performance as the optimism-corrected sampling distribution proxy.
        out.append({"analysis": "pooled_development_internal_bootstrap",
                    "model": name, "development_n": int(len(dev)),
                    "development_events": int(dev.event.sum()), "bootstrap_replicates": int(len(b)),
                    "apparent_auc": app[name][0], "mean_auc_optimism": mean_auc_opt,
                    "optimism_corrected_auc": float(app[name][0] - mean_auc_opt),
                    "bootstrap_test_auc_low": float(b.auc_original_test.quantile(.025)),
                    "bootstrap_test_auc_high": float(b.auc_original_test.quantile(.975)),
                    "apparent_brier": app[name][1], "mean_brier_optimism": mean_bri_opt,
                    "optimism_corrected_brier": float(app[name][1] - mean_bri_opt),
                    "bootstrap_test_brier_low": float(b.brier_original_test.quantile(.025)),
                    "bootstrap_test_brier_high": float(b.brier_original_test.quantile(.975))})
    summary = pd.DataFrame(out)
    boot_df.to_csv(TAB / "internal_bootstrap_optimism_replicates.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TAB / "internal_bootstrap_optimism_summary.csv", index=False, encoding="utf-8-sig")

    # Incremental optimism correction is also useful as a compact reviewer-facing result.
    b0 = summary.loc[summary.model == "base"].iloc[0]
    b1 = summary.loc[summary.model == "base_plus_IC"].iloc[0]
    delta = pd.DataFrame([{
        "analysis": "pooled_development_internal_bootstrap",
        "apparent_delta_auc": float(b1.apparent_auc - b0.apparent_auc),
        "optimism_corrected_delta_auc": float(b1.optimism_corrected_auc - b0.optimism_corrected_auc),
        "apparent_delta_brier": float(b1.apparent_brier - b0.apparent_brier),
        "optimism_corrected_delta_brier": float(b1.optimism_corrected_brier - b0.optimism_corrected_brier),
    }])
    delta.to_csv(TAB / "internal_bootstrap_optimism_incremental.csv", index=False, encoding="utf-8-sig")

    memo = f"""# Internal bootstrap optimism correction

The ELSA, CHARLS and HRS development samples were pooled after within-cohort
standardisation. We refitted each model in 400 stratified-by-cohort bootstrap
samples. Optimism was defined as bootstrap apparent performance minus performance
of that bootstrap model in the original development sample. The corrected value
is the full-sample apparent estimate minus mean optimism.

The results quantify internal overfitting of the pooled fixed-window disability
models. They do not replace the leave-one-cohort-out transportability analysis,
which evaluates performance in genuinely held-out cohorts.

Valid bootstrap model fits: {len(boot_df)} model-by-replicate rows.
"""
    (OUT / "internal_bootstrap_optimism_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 33, "script": "69_internal_bootstrap_optimism_disability.py",
           "development": "ELSA+CHARLS+HRS", "stratification": "cohort",
           "replicates": B, "metrics": ["AUC", "Brier"], "output_level": "aggregate and bootstrap summaries"}
    (OUT / "internal_bootstrap_optimism_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nIncremental optimism-corrected result:")
    print(delta.to_string(index=False))


if __name__ == "__main__":
    main()
