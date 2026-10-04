"""Paired apparent bootstrap intervals for fixed-coefficient proxy predictions.

The script reproduces the frozen Stage-73 proxy transport setup for SHARE and
adds paired bootstrap intervals for delta AUC and delta Brier. It writes only
aggregate rows and does not modify manuscript files.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(r"PATH_TO_IC_FRAILTY")
TABLES = ROOT / "tables"
OUT = ROOT / "robustness_gate"
OUT.mkdir(exist_ok=True)
BOOT = 400
CONTINUOUS = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
BASE_TERMS = ["age_z", "sex", "education_z", "baseline_fi_z"]
FULL_TERMS = BASE_TERMS + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def paired(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, key: str):
    seed = 20260925 + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100000
    rng = np.random.default_rng(seed)
    da, db = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        da.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
        db.append(brier_score_loss(y[idx], p1[idx]) - brier_score_loss(y[idx], p0[idx]))
    return {
        "delta_auc": float(roc_auc_score(y, p1) - roc_auc_score(y, p0)),
        "delta_auc_low": float(np.quantile(da, 0.025)),
        "delta_auc_high": float(np.quantile(da, 0.975)),
        "delta_brier": float(brier_score_loss(y, p1) - brier_score_loss(y, p0)),
        "delta_brier_low": float(np.quantile(db, 0.025)),
        "delta_brier_high": float(np.quantile(db, 0.975)),
        "bootstrap_replicates": len(da),
    }


def main():
    stage66 = load(ROOT / "66_incident_disability_prediction.py", "stage66_for_bootstrap")
    stage68 = load(ROOT / "68_leave_one_cohort_out_disability_validation.py", "stage68_for_bootstrap")
    builders = {"ELSA": stage66.build_elsa, "CHARLS": stage66.build_charls, "HRS": stage66.build_hrs}
    raw = {name: stage68.harmonize_sex(fn()[0]) for name, fn in builders.items()}
    scaled = {name: stage68.standardize(d, CONTINUOUS)[0] for name, d in raw.items()}
    dev = pd.concat(list(scaled.values()), ignore_index=True).dropna(subset=["event", *FULL_TERMS]).copy()
    base = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(dev[BASE_TERMS], dev.event.astype(int))
    full = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(dev[FULL_TERMS], dev.event.astype(int))
    rows = []
    for target, window in [(7, "6->7"), (8, "6->8")]:
        sh, _ = stage66.build_share(target)
        sh = stage68.harmonize_sex(sh)
        sh = stage68.standardize(sh, CONTINUOUS)[0]
        sh = sh.dropna(subset=["event", *FULL_TERMS]).copy()
        y = sh.event.to_numpy(dtype=int)
        p0 = base.predict_proba(sh[BASE_TERMS])[:, 1]
        p1 = full.predict_proba(sh[FULL_TERMS])[:, 1]
        r = paired(y, p0, p1, f"SHARE|{window}")
        r.update({"track": "proxy", "sample": "SHARE", "window": window, "n": len(y), "events": int(y.sum()), "prediction_type": "fixed_coefficient_external", "interval_type": "paired_apparent_bootstrap"})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "proxy_external_delta_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

