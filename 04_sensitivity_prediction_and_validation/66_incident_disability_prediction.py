"""Criterion-validity analysis for incident disability.

This script builds an anonymised, fixed-window incident ADL/IADL endpoint from
the four ageing cohorts and evaluates incremental prediction by baseline
four-domain IC proxies beyond age, sex and the outcome-disjoint FI.

The endpoint is deliberately a fixed follow-up window. Exact incident dates
are not available in the harmonised files, so the discrimination metric is
reported as horizon AUC (equivalent to a C-index for this fixed-window binary
endpoint), rather than as a misleading continuous-time survival estimate.
No individual identifiers or person-level scores are written to disk.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import platform
import sys
import hashlib

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

SEED = 20260924
BOOTSTRAP_B = 400


def num(x: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce").astype(float)
    z = z.mask(z < -90)
    if lo is not None:
        z = z.mask(z < lo)
    if hi is not None:
        z = z.mask(z > hi)
    return z


def valid_binary(x: pd.Series) -> pd.Series:
    z = num(x)
    return z.where(z.isin([0.0, 1.0]))


def read_dta(path: Path, cols: list[str]) -> pd.DataFrame:
    # Metadata-only read prevents a typo in a cohort-specific manifest from
    # silently changing the sample.
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in dict.fromkeys(cols) if c in meta.column_names]
    missing = [c for c in dict.fromkeys(cols) if c not in meta.column_names]
    if missing:
        print(f"{path.name}: omitted unavailable columns: {missing}", file=sys.stderr)
    d, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    return d


def mean_min(d: pd.DataFrame, cols: list[str], transforms: dict[str, object] | None = None,
             minimum: int = 1) -> pd.Series:
    vals = []
    for c in cols:
        if c not in d:
            continue
        x = num(d[c])
        if transforms and c in transforms:
            x = transforms[c](x)
        vals.append(x.rename(c))
    if not vals:
        return pd.Series(np.nan, index=d.index)
    z = pd.concat(vals, axis=1)
    return z.mean(axis=1).where(z.notna().sum(axis=1).ge(minimum))


def max_min(d: pd.DataFrame, cols: list[str], transforms: dict[str, object] | None = None) -> pd.Series:
    vals = []
    for c in cols:
        if c not in d:
            continue
        x = num(d[c])
        if transforms and c in transforms:
            x = transforms[c](x)
        vals.append(x.rename(c))
    if not vals:
        return pd.Series(np.nan, index=d.index)
    z = pd.concat(vals, axis=1)
    return z.max(axis=1, skipna=True).where(z.notna().any(axis=1))


def fi_score(d: pd.DataFrame, disease_cols: list[str], shlt_col: str, bmi_col: str) -> pd.Series:
    """Outcome-disjoint 8-component FI, >=6 observed components.

    All four files use the conventional 1=excellent and 5=poor self-rated
    health coding. The deficit is therefore (SHLT-1)/4. A BMI below 18.5 or
    at least 30 is a deficit, as frozen in the analysis manifest.
    """
    z = pd.DataFrame(index=d.index)
    for c in disease_cols:
        z[c] = valid_binary(d[c]) if c in d else np.nan
    shlt = num(d[shlt_col], 1, 5) if shlt_col in d else pd.Series(np.nan, index=d.index)
    z["self_rated_health"] = (shlt - 1.0) / 4.0
    bmi = num(d[bmi_col], 0, 100) if bmi_col in d else pd.Series(np.nan, index=d.index)
    z["bmi"] = (bmi.lt(18.5) | bmi.ge(30.0)).where(bmi.notna())
    z = z.astype(float)
    n_obs = z.notna().sum(axis=1)
    total = z.sum(axis=1, skipna=True)
    return total.div(n_obs).where(n_obs.ge(6))


def safe_id(x: pd.Series) -> pd.Series:
    # IDs are used only in memory for joins. They are never written.
    return x.astype("string")


def endpoint_counts(adl: pd.Series, iadl: pd.Series) -> tuple[pd.Series, pd.Series]:
    a = num(adl, 0, 30)
    i = num(iadl, 0, 30)
    observed = a.notna() & i.notna()
    none = observed & a.add(i).eq(0)
    any_limit = observed & a.add(i).gt(0)
    return none, any_limit


def make_long_pair(d: pd.DataFrame, id_col: str, wave_col: str, baseline: int,
                   target: int, active_baseline: pd.Series, active_target: pd.Series,
                   base_fields: dict[str, pd.Series], target_none: pd.Series,
                   target_event: pd.Series, cohort: str, window: str) -> pd.DataFrame:
    """Merge baseline predictors with a fixed-window disability endpoint."""
    d = d.copy()
    d[".__id"] = safe_id(d[id_col])
    d[".__wave"] = pd.to_numeric(d[wave_col], errors="coerce")
    base = d.loc[active_baseline & d[".__wave"].eq(baseline)].copy()
    fut = d.loc[active_target & d[".__wave"].eq(target)].copy()
    base = base.drop_duplicates(".__id", keep="first")
    fut = fut.drop_duplicates(".__id", keep="first")
    b = pd.DataFrame({".__id": base[".__id"].values}, index=base.index)
    for k, v in base_fields.items():
        b[k] = pd.to_numeric(v.loc[base.index], errors="coerce").to_numpy()
    b["baseline_none"] = target_none.loc[base.index].to_numpy()
    # outcome indicators are passed in as wave-specific Series indexed like d
    f = pd.DataFrame({".__id": fut[".__id"].values}, index=fut.index)
    f["event"] = target_event.loc[fut.index].to_numpy()
    f["target_observed"] = (target_none.loc[fut.index] | target_event.loc[fut.index]).to_numpy()
    out = b.reset_index(drop=True).merge(f.reset_index(drop=True), on=".__id", how="inner")
    out = out.loc[out["baseline_none"].eq(True) & out["target_observed"].eq(True)].copy()
    out["event"] = out["event"].astype(int)
    out["cohort"] = cohort
    out["window"] = window
    return out.drop(columns=[".__id"])


def make_wide_pair(d: pd.DataFrame, id_col: str, active_b: pd.Series, active_f: pd.Series,
                   base_fields: dict[str, pd.Series], base_none: pd.Series, fut_event: pd.Series,
                   fut_observed: pd.Series, cohort: str, window: str) -> pd.DataFrame:
    d = d.copy()
    d[".__id"] = safe_id(d[id_col])
    bmask = active_b & d[".__id"].notna()
    fmask = active_f & d[".__id"].notna()
    b = pd.DataFrame({".__id": d.loc[bmask, ".__id"].values}, index=d.index[bmask])
    for k, v in base_fields.items():
        b[k] = pd.to_numeric(v.loc[bmask], errors="coerce").to_numpy()
    b["baseline_none"] = base_none.loc[bmask].to_numpy()
    f = pd.DataFrame({".__id": d.loc[fmask, ".__id"].values}, index=d.index[fmask])
    f["event"] = fut_event.loc[fmask].to_numpy()
    f["target_observed"] = fut_observed.loc[fmask].to_numpy()
    b = b.reset_index(drop=True).drop_duplicates(".__id", keep="first")
    f = f.reset_index(drop=True).drop_duplicates(".__id", keep="first")
    out = b.merge(f, on=".__id", how="inner")
    out = out.loc[out["baseline_none"].eq(True) & out["target_observed"].eq(True)].copy()
    out["event"] = out["event"].astype(int)
    out["cohort"] = cohort
    out["window"] = window
    return out.drop(columns=[".__id"])


def build_elsa() -> tuple[pd.DataFrame, dict]:
    path = ROOT / "ELSA/Working_data/elsa.dta"
    cols = ["idauniqc", "wave", "tcog_z_z", "walkra", "walk100a", "gripsum", "cesd",
            "agey", "ragender", "raeducl", "shlt", "mbmi", "hibpe", "diabe", "hearte", "stroke",
            "cancre", "arthre", "adltot6", "iadltot2_e"]
    d = read_dta(path, cols)
    wave = pd.to_numeric(d["wave"], errors="coerce")
    active = wave.isin([6, 7])
    d = d.loc[active].copy()
    wave = pd.to_numeric(d["wave"], errors="coerce")
    # indicators point in the capacity direction (higher = better)
    cog = num(d["tcog_z_z"])
    loc = mean_min(d, ["walkra", "walk100a"], {"walkra": lambda x: 1 - x, "walk100a": lambda x: 1 - x})
    grip = num(d["gripsum"], 0, 100)
    psych = 1 - num(d["cesd"], 0, 8) / 8.0
    fi = fi_score(d, ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"], "shlt", "mbmi")
    adl = num(d["adltot6"], 0, 30)
    iadl = num(d["iadltot2_e"], 0, 30)
    none, event = endpoint_counts(adl, iadl)
    active_b = wave.eq(6)
    active_f = wave.eq(7)
    bfields = {"age": num(d["agey"]), "sex": num(d["ragender"]), "education": num(d["raeducl"]), "baseline_fi": fi,
               "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych}
    pairs = make_long_pair(d, "idauniqc", "wave", 6, 7, active_b, active_f, bfields,
                           none, event, "ELSA", "6->7")
    info = {"cohort": "ELSA", "window": "6->7", "years": 2, "endpoint": "adltot6+iadltot2_e"}
    return pairs, info


def build_charls() -> tuple[pd.DataFrame, dict]:
    path = ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"
    cols = ["ID", "inw3", "inw4", "r3orient", "r3tr20", "r3walk100a", "r3lgrip", "r3rgrip",
            "r3cesd10", "r3agey", "ragender", "raeduc_c", "r3shlt", "r3mbmi", "r3hibpe", "r3diabe",
            "r3hearte", "r3stroke", "r3cancre", "r3arthre", "r3adlwa", "r3iadla",
            "r4adlwa", "r4iadla"]
    d = read_dta(path, cols)
    active_b = num(d["inw3"]).eq(1)
    active_f = num(d["inw4"]).eq(1)
    cog = mean_min(d, ["r3orient", "r3tr20"], {"r3orient": lambda x: x / 4.0, "r3tr20": lambda x: x / 20.0}, minimum=2)
    loc = 1 - num(d["r3walk100a"], 0, 1)
    grip = max_min(d, ["r3lgrip", "r3rgrip"], {"r3lgrip": lambda x: x / 100.0, "r3rgrip": lambda x: x / 100.0})
    psych = 1 - num(d["r3cesd10"], 0, 30) / 30.0
    fi = fi_score(d, ["r3hibpe", "r3diabe", "r3hearte", "r3stroke", "r3cancre", "r3arthre"], "r3shlt", "r3mbmi")
    b_none, _ = endpoint_counts(d["r3adlwa"], d["r3iadla"])
    _, f_event = endpoint_counts(d["r4adlwa"], d["r4iadla"])
    f_obs = num(d["r4adlwa"], 0, 30).notna() & num(d["r4iadla"], 0, 30).notna()
    bfields = {"age": num(d["r3agey"]), "sex": (num(d["ragender"]) - 1).where(num(d["ragender"]).isin([1, 2])), "education": num(d["raeduc_c"]),
               "baseline_fi": fi,
               "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych}
    pairs = make_wide_pair(d, "ID", active_b, active_f, bfields, b_none, f_event, f_obs, "CHARLS", "3->4")
    return pairs, {"cohort": "CHARLS", "window": "3->4", "years": 3, "endpoint": "r4adlwa+r4iadla"}


def build_hrs() -> tuple[pd.DataFrame, dict]:
    path = ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
    cols = ["hhidpn", "inw10", "inw11", "r10proxy", "r11proxy", "r10imrc", "r10dlrc", "r10ser7",
            "r10walkra", "r10walk1a", "r10walksa", "r10grpl", "r10grpr", "r10cesd", "r10agey_m",
            "ragender", "raeduc", "r10shlt", "r10bmi", "r10hibpe", "r10diabe", "r10hearte", "r10stroke",
            "r10cancre", "r10arthre", "r10adl5a", "r10iadl5a", "r11adl5a", "r11iadl5a"]
    d = read_dta(path, cols)
    active_b = num(d["inw10"]).eq(1) & num(d["r10proxy"]).eq(0)
    active_f = num(d["inw11"]).eq(1) & num(d["r11proxy"]).eq(0)
    cog = mean_min(d, ["r10imrc", "r10dlrc", "r10ser7"], {"r10imrc": lambda x: x / 10.0, "r10dlrc": lambda x: x / 10.0, "r10ser7": lambda x: x / 5.0}, minimum=2)
    loc = mean_min(d, ["r10walkra", "r10walk1a", "r10walksa"], {c: lambda x: 1 - (x > 0).astype(float) for c in ["r10walkra", "r10walk1a", "r10walksa"]})
    grip = max_min(d, ["r10grpl", "r10grpr"], {"r10grpl": lambda x: x / 100.0, "r10grpr": lambda x: x / 100.0})
    psych = 1 - num(d["r10cesd"], 0, 8) / 8.0
    fi = fi_score(d, ["r10hibpe", "r10diabe", "r10hearte", "r10stroke", "r10cancre", "r10arthre"], "r10shlt", "r10bmi")
    b_none, _ = endpoint_counts(d["r10adl5a"], d["r10iadl5a"])
    _, f_event = endpoint_counts(d["r11adl5a"], d["r11iadl5a"])
    f_obs = num(d["r11adl5a"], 0, 30).notna() & num(d["r11iadl5a"], 0, 30).notna()
    bfields = {"age": num(d["r10agey_m"]), "sex": (num(d["ragender"]) - 1).where(num(d["ragender"]).isin([1, 2])), "education": num(d["raeduc"]),
               "baseline_fi": fi,
               "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych}
    pairs = make_wide_pair(d, "hhidpn", active_b, active_f, bfields, b_none, f_event, f_obs, "HRS", "10->11")
    return pairs, {"cohort": "HRS", "window": "10->11", "years": 2, "endpoint": "r11adl5a+r11iadl5a", "proxy_excluded": True}


SHARE_ITEMS = ["walkra", "dressa", "batha", "eata", "beda", "toilta", "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "leavhsa", "laundrya"]


def share_endpoint(d: pd.DataFrame, wave: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Housework is omitted because the household module is structurally absent
    # for a large fraction of SHARE respondents. At least 10 of 14 items must
    # be observed in each person-wave.
    z = pd.concat([num(d[c], 0, 1) for c in SHARE_ITEMS], axis=1)
    nobs = z.notna().sum(axis=1)
    observed = nobs.ge(10)
    none = observed & z.max(axis=1, skipna=True).eq(0)
    event = observed & z.max(axis=1, skipna=True).gt(0)
    return none, event, observed


def build_share(target_wave: int = 7) -> tuple[pd.DataFrame, dict]:
    path = next((ROOT / "SHARE").rglob("Working_data/share.dta"))
    cols = ["mergeid", "wave", "agey", "ragender", "raeducl", "imrc", "dlrc", "orient", "walkra", "walk100a", "lgrip", "rgrip",
            "eurod", "shlt", "bmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", *SHARE_ITEMS]
    d = read_dta(path, cols)
    d[".__id"] = safe_id(d["mergeid"])
    d[".__wave"] = pd.to_numeric(d["wave"], errors="coerce")
    d = d.loc[d[".__wave"].isin([6, target_wave]) & d[".__id"].notna()].copy()
    d = d.drop_duplicates([".__id", ".__wave"], keep="first")
    base = d[".__wave"].eq(6)
    fut = d[".__wave"].eq(target_wave)
    cog = mean_min(d, ["imrc", "dlrc", "orient"], {"imrc": lambda x: x / 10.0, "dlrc": lambda x: x / 10.0, "orient": lambda x: x / 4.0}, minimum=2)
    loc = mean_min(d, ["walkra", "walk100a"], {"walkra": lambda x: 1 - x, "walk100a": lambda x: 1 - x})
    grip = max_min(d, ["lgrip", "rgrip"], {"lgrip": lambda x: x / 100.0, "rgrip": lambda x: x / 100.0})
    psych = 1 - num(d["eurod"], 0, 12) / 12.0
    fi = fi_score(d, ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"], "shlt", "bmi")
    b_none, _, b_obs = share_endpoint(d, 6)
    _, f_event, f_obs = share_endpoint(d, target_wave)
    bfields = {"age": num(d.loc[base, "agey"]) if "agey" in d else pd.Series(np.nan, index=d.index),
               "sex": num(d.loc[base, "ragender"]) if "ragender" in d else pd.Series(np.nan, index=d.index),
               "baseline_fi": fi, "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych}
    # build_wide_pair expects full-index series, so re-expand baseline fields
    base_fields = {k: (v if isinstance(v, pd.Series) and len(v) == len(d) else pd.Series(np.nan, index=d.index)) for k, v in bfields.items()}
    base_fields["age"] = num(d["agey"]) if "agey" in d else pd.Series(np.nan, index=d.index)
    base_fields["sex"] = num(d["ragender"]) if "ragender" in d else pd.Series(np.nan, index=d.index)
    base_fields["education"] = num(d["raeducl"]) if "raeducl" in d else pd.Series(np.nan, index=d.index)
    pairs = make_wide_pair(d, "mergeid", base, fut, base_fields, b_none, f_event, f_obs, "SHARE", f"6->{target_wave}")
    return pairs, {"cohort": "SHARE", "window": f"6->{target_wave}", "years": 2 if target_wave == 7 else 4,
                   "endpoint": "14-item ADL/IADL any limitation", "min_items": 10, "target_wave": target_wave}


def cohort_standardize(d: pd.DataFrame, cols: list[str], reference: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Standardize continuous predictors within cohort/window.

    The no-scalar-invariance finding makes within-cohort scaling preferable to
    pretending that raw proxy units are directly comparable. SHARE is scaled
    on its own baseline distribution during external validation, while the
    fitted coefficients remain fixed.
    """
    out = d.copy()
    ref = reference if reference is not None else d
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


def design_matrix(d: pd.DataFrame, full: bool) -> tuple[pd.DataFrame, list[str]]:
    base = ["age_z", "sex", "education_z", "baseline_fi_z"]
    ic = ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
    terms = base + (ic if full else [])
    return d[terms].astype(float), terms


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q)).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def metric_row(y: np.ndarray, p: np.ndarray, cohort: str, window: str, model: str, sample: str,
               n_boot: int = BOOTSTRAP_B) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    brier = float(brier_score_loss(y, p))
    ci, cs = calibration_stats(y, p)
    stable_key = f"{cohort}|{window}|{model}|{sample}".encode("utf-8")
    stable_offset = int(hashlib.sha256(stable_key).hexdigest()[:8], 16) % 100000
    rng = np.random.default_rng(SEED + stable_offset)
    auc_b, bri_b = [], []
    if n_boot and len(y) > 100 and len(np.unique(y)) == 2:
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), size=len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            auc_b.append(roc_auc_score(y[idx], p[idx]))
            bri_b.append(brier_score_loss(y[idx], p[idx]))
    return {"sample": sample, "cohort": cohort, "window": window, "model": model,
            "n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean()),
            "auc_horizon_cindex": auc, "auc_low": float(np.quantile(auc_b, .025)) if auc_b else np.nan,
            "auc_high": float(np.quantile(auc_b, .975)) if auc_b else np.nan,
            "brier": brier, "brier_low": float(np.quantile(bri_b, .025)) if bri_b else np.nan,
            "brier_high": float(np.quantile(bri_b, .975)) if bri_b else np.nan,
            "calibration_intercept": ci, "calibration_slope": cs,
            "mean_predicted_risk": float(np.mean(p)), "observed_risk": float(np.mean(y)),
            "observed_expected_ratio": float(np.mean(y) / np.mean(p)) if np.mean(p) > 0 else np.nan}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, full: bool) -> tuple[np.ndarray, np.ndarray, LogisticRegression, list[str]]:
    terms = ["age_z", "sex", "education_z", "baseline_fi_z"] + (["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"] if full else [])
    tr = train.dropna(subset=["event", *terms]).copy()
    te = test.dropna(subset=["event", *terms]).copy()
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, class_weight=None)
    m.fit(tr[terms], tr["event"].astype(int))
    return m.predict_proba(tr[terms])[:, 1], m.predict_proba(te[terms])[:, 1], m, terms


def dca_rows(y: np.ndarray, p_base: np.ndarray, p_full: np.ndarray, cohort: str, sample: str, window: str = "") -> list[dict]:
    y = np.asarray(y, dtype=int)
    rows = []
    # The 5-30% range spans the observed event prevalence and avoids making a
    # clinical recommendation in the absence of a validated threshold.
    for threshold in np.arange(0.05, 0.301, 0.01):
        def nb(p):
            pred = p >= threshold
            return float(pred[y == 1].sum() / len(y) - pred[y == 0].sum() / len(y) * threshold / (1 - threshold))
        treat_all = float(y.mean() - (1 - y.mean()) * threshold / (1 - threshold))
        rows += [{"sample": sample, "cohort": cohort, "window": window, "threshold": round(float(threshold), 2), "model": "base", "net_benefit": nb(p_base)},
                 {"sample": sample, "cohort": cohort, "window": window, "threshold": round(float(threshold), 2), "model": "base_plus_IC", "net_benefit": nb(p_full)},
                 {"sample": sample, "cohort": cohort, "window": window, "threshold": round(float(threshold), 2), "model": "treat_all", "net_benefit": treat_all},
                 {"sample": sample, "cohort": cohort, "window": window, "threshold": round(float(threshold), 2), "model": "treat_none", "net_benefit": 0.0}]
    return rows


def coefficient_rows(model: LogisticRegression, terms: list[str], cohort: str, window: str,
                     sample: str, model_name: str) -> list[dict]:
    """Return odds ratios per one within-cohort SD (or sex unit)."""
    rows = []
    for term, beta in zip(terms, model.coef_[0]):
        rows.append({"sample": sample, "cohort": cohort, "window": window, "model": model_name,
                     "term": term, "log_odds": float(beta), "odds_ratio_per_unit": float(np.exp(beta))})
    rows.append({"sample": sample, "cohort": cohort, "window": window, "model": model_name,
                 "term": "intercept", "log_odds": float(model.intercept_[0]),
                 "odds_ratio_per_unit": np.nan})
    return rows


def main() -> None:
    pairs = []
    infos = []
    builders = [build_elsa, build_charls, build_hrs]
    for fn in builders:
        d, info = fn()
        pairs.append(d)
        infos.append(info)
    share7, info7 = build_share(7)
    share8, info8 = build_share(8)
    pairs.extend([share7, share8])
    infos.extend([info7, info8])

    # Apply within-cohort standardisation separately for each fixed window.
    std_pairs = []
    params = []
    continuous = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
    for d in pairs:
        if d.empty:
            continue
        z, pars = cohort_standardize(d, continuous)
        std_pairs.append(z)
        params.append({"cohort": str(d["cohort"].iloc[0]), "window": str(d["window"].iloc[0]), "parameters": pars})
    by_key = {(str(d["cohort"].iloc[0]), str(d["window"].iloc[0])): d for d in std_pairs}

    # Sample audit before any model fitting.
    sample_rows = []
    for d in std_pairs:
        cohort, window = str(d["cohort"].iloc[0]), str(d["window"].iloc[0])
        model_terms = ["age_z", "sex", "education_z", "baseline_fi_z", "cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
        complete = d.dropna(subset=["event", *model_terms])
        sample_rows.append({"cohort": cohort, "window": window, "eligible_baseline_no_disability": int(len(d)),
                            "incident_events": int(d["event"].sum()), "incidence_pct": float(d["event"].mean() * 100),
                            "complete_model_n": int(len(complete)), "complete_model_events": int(complete["event"].sum()),
                            "complete_model_incidence_pct": float(complete["event"].mean() * 100) if len(complete) else np.nan})
    pd.DataFrame(sample_rows).to_csv(TAB / "incident_disability_cohort_summary.csv", index=False, encoding="utf-8-sig")

    development = pd.concat([by_key[("ELSA", "6->7")], by_key[("CHARLS", "3->4")], by_key[("HRS", "10->11")]], ignore_index=True)
    metrics = []
    coefficients = []
    incremental = []
    calibration = []
    dca = []
    # Internal cohort-specific apparent metrics and pooled development metrics.
    for key, d in list(by_key.items()):
        if key[0] == "SHARE":
            continue
        dd = d.dropna(subset=["event", "age_z", "sex", "education_z", "baseline_fi_z", "cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]).copy()
        for full, name in [(False, "base"), (True, "base_plus_IC")]:
            terms = ["age_z", "sex", "education_z", "baseline_fi_z"] + (["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"] if full else [])
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(dd[terms], dd.event)
            p = m.predict_proba(dd[terms])[:, 1]
            metrics.append(metric_row(dd.event.values, p, key[0], key[1], name, "internal_cohort"))
            coefficients.extend(coefficient_rows(m, terms, key[0], key[1], "internal_cohort", name))
        mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(dd[["age_z", "sex", "education_z", "baseline_fi_z"]], dd.event)
        mf2 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(dd[["age_z", "sex", "education_z", "baseline_fi_z", "cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]], dd.event)
        pb = mb.predict_proba(dd[["age_z", "sex", "education_z", "baseline_fi_z"]])[:, 1]
        pf = mf2.predict_proba(dd[["age_z", "sex", "education_z", "baseline_fi_z", "cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]])[:, 1]
        incremental.append({"sample": "internal_cohort", "cohort": key[0], "window": key[1], "n": int(len(dd)), "events": int(dd.event.sum()),
                            "delta_auc": float(roc_auc_score(dd.event, pf) - roc_auc_score(dd.event, pb)),
                            "delta_brier": float(brier_score_loss(dd.event, pf) - brier_score_loss(dd.event, pb))})
        # Calibration deciles are generated for the full model only.
        mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(dd[["age_z", "sex", "education_z", "baseline_fi_z", "cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]], dd.event)
        p = mf.predict_proba(dd[["age_z", "sex", "education_z", "baseline_fi_z", "cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]])[:, 1]
        tmp = pd.DataFrame({"y": dd.event.values, "p": p})
        tmp["decile"] = pd.qcut(tmp.p.rank(method="first"), 10, labels=False) + 1
        for dec, q in tmp.groupby("decile", sort=True):
            calibration.append({"sample": "internal_cohort", "cohort": key[0], "window": key[1], "model": "base_plus_IC", "decile": int(dec), "n": int(len(q)), "mean_predicted": float(q.p.mean()), "observed": float(q.y.mean())})

    # Pooled development model and fixed-coefficient external validation.
    terms_base = ["age_z", "sex", "education_z", "baseline_fi_z"]
    terms_full = terms_base + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
    dev = development.dropna(subset=["event", *terms_full]).copy()
    m_base = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(dev[terms_base], dev.event)
    m_full = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(dev[terms_full], dev.event)
    pbase_dev = m_base.predict_proba(dev[terms_base])[:, 1]
    pfull_dev = m_full.predict_proba(dev[terms_full])[:, 1]
    metrics.extend([metric_row(dev.event.values, pbase_dev, "ELSA+CHARLS+HRS", "anchor_windows", "base", "development_pooled"),
                    metric_row(dev.event.values, pfull_dev, "ELSA+CHARLS+HRS", "anchor_windows", "base_plus_IC", "development_pooled")])
    coefficients.extend(coefficient_rows(m_base, terms_base, "ELSA+CHARLS+HRS", "anchor_windows", "development_pooled", "base"))
    coefficients.extend(coefficient_rows(m_full, terms_full, "ELSA+CHARLS+HRS", "anchor_windows", "development_pooled", "base_plus_IC"))
    incremental.append({"sample": "development_pooled", "cohort": "ELSA+CHARLS+HRS", "window": "anchor_windows",
                        "n": int(len(dev)), "events": int(dev.event.sum()),
                        "delta_auc": float(roc_auc_score(dev.event, pfull_dev) - roc_auc_score(dev.event, pbase_dev)),
                        "delta_brier": float(brier_score_loss(dev.event, pfull_dev) - brier_score_loss(dev.event, pbase_dev))})
    dca.extend(dca_rows(dev.event.values, pbase_dev, pfull_dev, "ELSA+CHARLS+HRS", "development_pooled", "anchor_windows"))
    # External SHARE uses its own baseline z scaling, but the coefficients and
    # model intercept are held fixed from the three-cohort development set.
    for share_key in [("SHARE", "6->7"), ("SHARE", "6->8")]:
        sh = by_key.get(share_key)
        if sh is None or sh.empty:
            continue
        sh = sh.dropna(subset=["event", *terms_full]).copy()
        if len(sh) == 0 or sh.event.nunique() < 2:
            continue
        pbase = m_base.predict_proba(sh[terms_base])[:, 1]
        pfull = m_full.predict_proba(sh[terms_full])[:, 1]
        metrics.extend([metric_row(sh.event.values, pbase, "SHARE", share_key[1], "base", "external_validation"),
                        metric_row(sh.event.values, pfull, "SHARE", share_key[1], "base_plus_IC", "external_validation")])
        coefficients.extend(coefficient_rows(m_base, terms_base, "SHARE", share_key[1], "external_validation", "base"))
        coefficients.extend(coefficient_rows(m_full, terms_full, "SHARE", share_key[1], "external_validation", "base_plus_IC"))
        incremental.append({"sample": "external_validation", "cohort": "SHARE", "window": share_key[1],
                            "n": int(len(sh)), "events": int(sh.event.sum()),
                            "delta_auc": float(roc_auc_score(sh.event, pfull) - roc_auc_score(sh.event, pbase)),
                            "delta_brier": float(brier_score_loss(sh.event, pfull) - brier_score_loss(sh.event, pbase))})
        dca.extend(dca_rows(sh.event.values, pbase, pfull, "SHARE", "external_validation", share_key[1]))
        tmp = pd.DataFrame({"y": sh.event.values, "p": pfull})
        tmp["decile"] = pd.qcut(tmp.p.rank(method="first"), 10, labels=False) + 1
        for dec, q in tmp.groupby("decile", sort=True):
            calibration.append({"sample": "external_validation", "cohort": "SHARE", "window": share_key[1], "model": "base_plus_IC", "decile": int(dec), "n": int(len(q)), "mean_predicted": float(q.p.mean()), "observed": float(q.y.mean())})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(TAB / "incident_disability_prediction_metrics.csv", index=False, encoding="utf-8-sig")
    coef_df = pd.DataFrame(coefficients)
    coef_df.to_csv(TAB / "incident_disability_prediction_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(incremental).to_csv(TAB / "incident_disability_incremental_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(calibration).to_csv(TAB / "incident_disability_calibration.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dca).to_csv(TAB / "incident_disability_dca.csv", index=False, encoding="utf-8-sig")
    (TAB / "incident_disability_standardization_parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    memo = """# Incident disability criterion-validity analysis

## Frozen endpoint

The primary endpoint is new ADL/IADL limitation during one prespecified follow-up window among participants with no baseline ADL/IADL limitation. ELSA uses `adltot6` plus `iadltot2_e` (wave 6 to 7), CHARLS uses `r4adlwa` plus `r4iadla` among wave-3 participants with both summaries equal to zero, and HRS uses `r11adl5a` plus `r11iadl5a` among non-proxy wave-10 participants with both summaries equal to zero. SHARE uses 14 consistently observed ADL/IADL items, excluding the structurally sparse housework item; at least 10 observed items are required per person-wave. Wave 6 to 7 is the primary external window and wave 6 to 8 is a longer-horizon sensitivity.

## Prediction models

The base model contains age, sex and the baseline outcome-disjoint eight-component FI. The full model adds the four baseline IC proxy domains. Continuous predictors are standardized within each cohort-window because the scalar-invariance gate was not met; SHARE is therefore an external test of relative domain information with the primary-cohort coefficients and intercept fixed, not a test of a universal IC level. Metrics include horizon AUC (equivalent to a C-index for the fixed-window binary endpoint), Brier score, calibration intercept and slope, and observed/expected risk. Confidence intervals use 400 nonparametric bootstrap resamples.

Decision-curve analysis is reported as an exploratory operating-characteristic diagnostic over 5–30% risk thresholds. No clinical threshold is asserted because this analysis does not validate a treatment decision rule.

The endpoint is measured at the follow-up interview rather than an exact onset date. A continuous-time Cox C-index or time-dependent AUC would therefore imply temporal precision that is not present in the source files; the fixed-window horizon AUC is the estimand used here. No individual IDs or person-level predictions are written.
"""
    (OUT / "incident_disability_prediction_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 31, "date": "2026-09-24", "script": "66_incident_disability_prediction.py", "endpoint": "incident any ADL/IADL limitation after baseline no limitation", "primary_cohorts": ["ELSA 6->7", "CHARLS 3->4", "HRS 10->11"], "external_validation": ["SHARE 6->7", "SHARE 6->8 sensitivity"], "base_model": ["age", "sex", "education", "baseline outcome-disjoint FI"], "full_model": "base + four IC proxies", "metrics": ["fixed-window horizon AUC (C-index equivalent)", "Brier", "calibration intercept/slope", "observed/expected", "exploratory DCA"], "bootstrap_replicates": BOOTSTRAP_B, "output_level": "aggregate only; no IDs or person-level scores"}
    (OUT / "incident_disability_prediction_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(sample_rows).to_string(index=False))
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
