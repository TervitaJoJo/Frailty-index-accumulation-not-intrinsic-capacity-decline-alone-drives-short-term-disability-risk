"""Exploratory MCID feasibility and uncertainty-aware DCA.

This stage does not claim a validated MCID.  The source cohorts contain no
harmonized patient-global change anchor or clinician-rated transition anchor in
the frozen analysis manifest.  We therefore audit anchor availability and
estimate distribution-based/prognostic change thresholds only as exploratory
signals.

Decision-curve analysis is repeated from the frozen incident-disability model
and augmented with bootstrap confidence intervals for net benefit.  Thresholds
are operating-characteristic values, not validated treatment or referral
thresholds.  No person-level data or predictions are written.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import re
import hashlib

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SEED = 20260925
BOOT = 400
THRESHOLDS = [0.20, 0.50, 1.00]
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.301, 0.01), 2)


def load66():
    p = OUT / "66_incident_disability_prediction.py"
    spec = importlib.util.spec_from_file_location("stage66_mcid", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_dta(mod, path, cols):
    return mod.read_dta(path, cols)


def safe_id(x):
    return x.astype("string")


def standardize(x):
    x = pd.to_numeric(x, errors="coerce")
    s = x.std(ddof=1)
    return (x - x.mean()) / s if np.isfinite(s) and s > 0 else pd.Series(np.nan, index=x.index)


def anchor_audit():
    """Metadata-level audit; matched words are not treated as validated anchors."""
    specs = {
        "ELSA": ROOT / "ELSA/Working_data/elsa.dta",
        "CHARLS": ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
        "HRS": ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
        "SHARE": next((ROOT / "SHARE").rglob("Working_data/share.dta")),
    }
    patt = re.compile(r"global|transition|change|improv|worsen|better|worse|anchor", re.I)
    rows = []
    for cohort, path in specs.items():
        _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
        labels = getattr(meta, "column_names_to_labels", {}) or {}
        hits = []
        for c in meta.column_names:
            label = str(labels.get(c, ""))
            if patt.search(c) or patt.search(label):
                hits.append(f"{c}: {label}" if label else c)
        rows.append({
            "cohort": cohort,
            "metadata_columns": len(meta.column_names),
            "matched_change_anchor_terms": len(hits),
            "matched_terms_preview": " | ".join(hits[:20]),
            "validated_clinical_anchor_available": False,
            "interpretation": "No harmonized patient-global or clinician-rated change anchor was prespecified; matched names/labels require construct validation and are not an MCID anchor.",
        })
    return pd.DataFrame(rows)


def merge_wave_pairs(base, fut, id_col, outcome_base, outcome_future):
    b = base.copy(); f = fut.copy()
    b["id"] = safe_id(b[id_col]); f["id"] = safe_id(f[id_col])
    b = b.drop_duplicates("id", keep="first"); f = f.drop_duplicates("id", keep="first")
    out = b.merge(f, on="id", suffixes=("_base", "_future"), how="inner")
    out["baseline_none"] = out[out.columns[out.columns.str.endswith("_base")].isin([])] if False else out["none_base"]
    return out


def build_elsa(mod):
    path = ROOT / "ELSA/Working_data/elsa.dta"
    cols = ["idauniqc", "wave", "tcog_z_z", "walkra", "walk100a", "gripsum", "cesd", "agey", "ragender", "raeducl", "shlt", "mbmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "adltot6", "iadltot2_e"]
    d = read_dta(mod, path, cols)
    d = d.loc[mod.num(d["wave"]).isin([6, 7])].copy()
    cog = mod.num(d["tcog_z_z"])
    loc = mod.mean_min(d, ["walkra", "walk100a"], {"walkra": lambda x: 1 - x, "walk100a": lambda x: 1 - x})
    grip = mod.num(d["gripsum"], 0, 100)
    psych = 1 - mod.num(d["cesd"], 0, 8) / 8.0
    fi = mod.fi_score(d, ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"], "shlt", "mbmi")
    none, event = mod.endpoint_counts(mod.num(d["adltot6"], 0, 30), mod.num(d["iadltot2_e"], 0, 30))
    row = pd.DataFrame({"id": safe_id(d["idauniqc"]), "wave": mod.num(d["wave"]), "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych, "age": mod.num(d["agey"]), "sex": mod.num(d["ragender"]), "education": mod.num(d["raeducl"]), "baseline_fi": fi, "none": none, "event": event, "observed": none | event})
    b = row.loc[row.wave.eq(6)].copy(); f = row.loc[row.wave.eq(7)].copy()
    out = b.merge(f, on="id", suffixes=("_base", "_future"), how="inner")
    if "event_future" not in out.columns:
        out["event_future"] = out["event"]
    if "observed_future" not in out.columns:
        out["observed_future"] = out["observed"]
    out = out.loc[out.none_base & out.observed_future].copy()
    out["cohort"] = "ELSA"; out["window"] = "6->7"
    return out


def build_charls(mod):
    path = ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"
    cols = ["ID", "inw3", "inw4", "r3orient", "r4orient", "r3tr20", "r4tr20", "r3walk100a", "r4walk100a", "r3lgrip1", "r3lgrip2", "r3rgrip1", "r3rgrip2", "r4lgrip1", "r4lgrip2", "r4rgrip1", "r4rgrip2", "r3cesd10", "r4cesd10", "r3agey", "ragender", "raeduc_c", "r3shlt", "r3mbmi", "r3hibpe", "r4hibpe", "r3diabe", "r4diabe", "r3hearte", "r4hearte", "r3stroke", "r4stroke", "r3cancre", "r4cancre", "r3arthre", "r4arthre", "r3adlwa", "r3iadla", "r4adlwa", "r4iadla"]
    d = read_dta(mod, path, cols)
    def wf(w):
        p = f"r{w}"
        cog = mod.mean_min(d, [p + "orient", p + "tr20"], {p + "orient": lambda x: x / 4.0, p + "tr20": lambda x: x / 20.0}, minimum=2)
        loc = 1 - mod.num(d[p + "walk100a"], 0, 1)
        grip = mod.max_min(d, [p + "lgrip1", p + "lgrip2", p + "rgrip1", p + "rgrip2"], {c: lambda x: x / 100.0 for c in [p + "lgrip1", p + "lgrip2", p + "rgrip1", p + "rgrip2"]})
        psych = 1 - mod.num(d[p + "cesd10"], 0, 30) / 30.0
        fi = mod.fi_score(d, [p + "hibpe", p + "diabe", p + "hearte", p + "stroke", p + "cancre", p + "arthre"], p + "shlt", p + "mbmi")
        return pd.DataFrame({"id": safe_id(d["ID"]), "active": mod.num(d[f"inw{w}"]).eq(1), "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych, "age": mod.num(d["r3agey"]), "sex": (mod.num(d["ragender"]) - 1).where(mod.num(d["ragender"]).isin([1, 2])), "education": mod.num(d["raeduc_c"]), "baseline_fi": fi})
    b = wf(3); f = wf(4)
    b["none"] = mod.endpoint_counts(d["r3adlwa"], d["r3iadla"])[0]
    f["event"], f["none"] = mod.endpoint_counts(d["r4adlwa"], d["r4iadla"])[1], mod.endpoint_counts(d["r4adlwa"], d["r4iadla"])[0]
    f["observed"] = mod.num(d["r4adlwa"], 0, 30).notna() & mod.num(d["r4iadla"], 0, 30).notna()
    b = b.loc[b.active].copy(); f = f.loc[f.active].copy()
    out = b.merge(f, on="id", suffixes=("_base", "_future"), how="inner")
    if "event_future" not in out.columns:
        out["event_future"] = out["event"]
    if "observed_future" not in out.columns:
        out["observed_future"] = out["observed"]
    out = out.loc[out.none_base & out.observed_future].copy()
    out["cohort"] = "CHARLS"; out["window"] = "3->4"
    return out


def build_hrs(mod):
    path = ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
    cols = ["hhidpn", "inw10", "inw11", "r10proxy", "r11proxy", "r10imrc", "r11imrc", "r10dlrc", "r11dlrc", "r10ser7", "r11ser7", "r10walkra", "r11walkra", "r10walk1a", "r11walk1a", "r10walksa", "r11walksa", "r10grpl", "r11grpl", "r10grpr", "r11grpr", "r10cesd", "r11cesd", "r10agey_m", "ragender", "raeduc", "r10shlt", "r11shlt", "r10bmi", "r11bmi", "r10hibpe", "r11hibpe", "r10diabe", "r11diabe", "r10hearte", "r11hearte", "r10stroke", "r11stroke", "r10cancre", "r11cancre", "r10arthre", "r11arthre", "r10adl5a", "r10iadl5a", "r11adl5a", "r11iadl5a"]
    d = read_dta(mod, path, cols)
    def wf(w):
        p = f"r{w}"
        cog = mod.mean_min(d, [p + "imrc", p + "dlrc", p + "ser7"], {p + "imrc": lambda x: x / 10.0, p + "dlrc": lambda x: x / 10.0, p + "ser7": lambda x: x / 5.0}, minimum=2)
        loc = mod.mean_min(d, [p + "walkra", p + "walk1a", p + "walksa"], {c: lambda x: 1 - (x > 0).astype(float) for c in [p + "walkra", p + "walk1a", p + "walksa"]})
        grip = mod.max_min(d, [p + "grpl", p + "grpr"], {p + "grpl": lambda x: x / 100.0, p + "grpr": lambda x: x / 100.0})
        psych = 1 - mod.num(d[p + "cesd"], 0, 8) / 8.0
        fi = mod.fi_score(d, [p + "hibpe", p + "diabe", p + "hearte", p + "stroke", p + "cancre", p + "arthre"], p + "shlt", p + "bmi")
        return pd.DataFrame({"id": safe_id(d["hhidpn"]), "active": mod.num(d[f"inw{w}"]).eq(1) & mod.num(d[f"r{w}proxy"]).eq(0), "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych, "age": mod.num(d["r10agey_m"]), "sex": (mod.num(d["ragender"]) - 1).where(mod.num(d["ragender"]).isin([1, 2])), "education": mod.num(d["raeduc"]), "baseline_fi": fi})
    b = wf(10); f = wf(11)
    b["none"] = mod.endpoint_counts(d["r10adl5a"], d["r10iadl5a"])[0]
    f["event"], f["none"] = mod.endpoint_counts(d["r11adl5a"], d["r11iadl5a"])[1], mod.endpoint_counts(d["r11adl5a"], d["r11iadl5a"])[0]
    f["observed"] = mod.num(d["r11adl5a"], 0, 30).notna() & mod.num(d["r11iadl5a"], 0, 30).notna()
    b = b.loc[b.active].copy(); f = f.loc[f.active].copy()
    out = b.merge(f, on="id", suffixes=("_base", "_future"), how="inner")
    if "event_future" not in out.columns:
        out["event_future"] = out["event"]
    if "observed_future" not in out.columns:
        out["observed_future"] = out["observed"]
    out = out.loc[out.none_base & out.observed_future].copy()
    out["cohort"] = "HRS"; out["window"] = "10->11"
    return out


def build_share(mod):
    path = next((ROOT / "SHARE").rglob("Working_data/share.dta"))
    items = ["walkra", "dressa", "batha", "eata", "beda", "toilta", "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "leavhsa", "laundrya"]
    cols = ["mergeid", "wave", "agey", "ragender", "raeducl", "imrc", "dlrc", "orient", "walkra", "walk100a", "lgrip", "rgrip", "eurod", "shlt", "bmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", *items]
    d = read_dta(mod, path, cols)
    d = d.loc[mod.num(d["wave"]).isin([6, 8]) & d["mergeid"].notna()].copy()
    d["id"] = safe_id(d["mergeid"]); d["wave_n"] = mod.num(d["wave"])
    def wf(w):
        z = d.loc[d.wave_n.eq(w)].copy()
        cog = mod.mean_min(z, ["imrc", "dlrc", "orient"], {"imrc": lambda x: x / 10.0, "dlrc": lambda x: x / 10.0, "orient": lambda x: x / 4.0}, minimum=2)
        loc = mod.mean_min(z, ["walkra", "walk100a"], {"walkra": lambda x: 1 - x, "walk100a": lambda x: 1 - x})
        grip = mod.max_min(z, ["lgrip", "rgrip"], {"lgrip": lambda x: x / 100.0, "rgrip": lambda x: x / 100.0})
        psych = 1 - mod.num(z["eurod"], 0, 12) / 12.0
        fi = mod.fi_score(z, ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"], "shlt", "bmi")
        zz = pd.concat([mod.num(z[c], 0, 1) for c in items], axis=1)
        obs = zz.notna().sum(axis=1).ge(10)
        none = obs & zz.max(axis=1, skipna=True).eq(0)
        event = obs & zz.max(axis=1, skipna=True).gt(0)
        return pd.DataFrame({"id": safe_id(z["mergeid"]), "active": z["id"].notna(), "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych, "age": mod.num(z["agey"]), "sex": mod.num(z["ragender"]), "education": mod.num(z["raeducl"]), "baseline_fi": fi, "none": none, "event": event, "observed": obs})
    b = wf(6); f = wf(8)
    out = b.merge(f, on="id", suffixes=("_base", "_future"), how="inner")
    if "event_future" not in out.columns:
        out["event_future"] = out["event"]
    if "observed_future" not in out.columns:
        out["observed_future"] = out["observed"]
    out = out.loc[out.none_base & out.observed_future].copy()
    out["cohort"] = "SHARE"; out["window"] = "6->8"
    return out


def add_change_metrics(d):
    domains = ["cognition", "locomotion", "grip_vitality", "psychological"]
    z = d.copy()
    for dom in domains:
        base = pd.to_numeric(z[f"{dom}_base"], errors="coerce")
        fut = pd.to_numeric(z[f"{dom}_future"], errors="coerce")
        mu = base.mean(); sd = base.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0: sd = 1.0
        z[f"{dom}_base_z"] = (base - mu) / sd
        z[f"{dom}_future_z"] = (fut - mu) / sd
        z[f"{dom}_loss_z"] = z[f"{dom}_base_z"] - z[f"{dom}_future_z"]
    bcomp = z[[f"{x}_base_z" for x in domains]].mean(axis=1, skipna=False)
    fcomp3 = z[[f"{x}_future_z" for x in domains]].mean(axis=1, skipna=True).where(z[[f"{x}_future_z" for x in domains]].notna().sum(axis=1).ge(3))
    fcomp4 = z[[f"{x}_future_z" for x in domains]].mean(axis=1, skipna=False)
    z["ic_base_comp_z"] = bcomp
    z["ic_loss_comp3_z"] = bcomp - fcomp3
    z["ic_loss_comp4_z"] = bcomp - fcomp4
    return z


def glm_threshold(d, indicator, label, threshold, cohort, window):
    terms = ["age_base", "sex_base", "education_base", "baseline_fi_base", "ic_base_comp_z", indicator]
    x = d[["event_future", *terms]].dropna().copy()
    if len(x) < 100 or x.event_future.nunique() < 2 or x[indicator].nunique() < 2:
        return {"cohort": cohort, "window": window, "measure": label, "threshold_sd": threshold, "n": len(x), "events": int(x.event_future.sum()) if len(x) else 0, "decline_prevalence": float(x[indicator].mean()) if len(x) else np.nan, "event_risk_decline": float(x.loc[x[indicator].eq(1), "event_future"].mean()) if (x[indicator] == 1).any() else np.nan, "event_risk_no_decline": float(x.loc[x[indicator].eq(0), "event_future"].mean()) if (x[indicator] == 0).any() else np.nan, "adjusted_or": np.nan, "ci_low": np.nan, "ci_high": np.nan, "status": "insufficient_variation"}
    X = sm.add_constant(x[terms].astype(float), has_constant="add")
    try:
        fit = sm.GLM(x.event_future.astype(float), X, family=sm.families.Binomial()).fit(cov_type="HC3")
        beta = float(fit.params[indicator]); ci = fit.conf_int().loc[indicator]
        status = "ok"
        vals = (float(np.exp(beta)), float(np.exp(ci[0])), float(np.exp(ci[1])))
    except Exception as exc:
        vals = (np.nan, np.nan, np.nan); status = f"fit_failed:{type(exc).__name__}"
    return {"cohort": cohort, "window": window, "measure": label, "threshold_sd": threshold, "n": len(x), "events": int(x.event_future.sum()), "decline_prevalence": float(x[indicator].mean()), "event_risk_decline": float(x.loc[x[indicator].eq(1), "event_future"].mean()), "event_risk_no_decline": float(x.loc[x[indicator].eq(0), "event_future"].mean()), "adjusted_or": vals[0], "ci_low": vals[1], "ci_high": vals[2], "status": status}


def run_mcid(mod):
    frames = [build_elsa(mod), build_charls(mod), build_hrs(mod), build_share(mod)]
    all_results = []
    availability = []
    for d in frames:
        d = add_change_metrics(d)
        cohort, window = str(d.cohort.iloc[0]), str(d.window.iloc[0])
        domains = ["cognition", "locomotion", "grip_vitality", "psychological"]
        for dom in domains:
            for thr in THRESHOLDS:
                ind = f"{dom}_loss_z_ge_{str(thr).replace('.', '')}"
                d[ind] = (d[f"{dom}_loss_z"] >= thr).astype(float).where(d[f"{dom}_loss_z"].notna())
                all_results.append(glm_threshold(d, ind, f"{dom}_decline", thr, cohort, window))
        for comp in ["ic_loss_comp3_z", "ic_loss_comp4_z"]:
            for thr in THRESHOLDS:
                ind = f"{comp}_ge_{str(thr).replace('.', '')}"
                d[ind] = (d[comp] >= thr).astype(float).where(d[comp].notna())
                all_results.append(glm_threshold(d, ind, "IC_composite_decline_" + (">=3_domains" if "comp3" in comp else "all4"), thr, cohort, window))
        availability.append({"cohort": cohort, "window": window, "n_pairs_no_baseline_disability": len(d), "n_all4_baseline": int(d[[f"{x}_base" for x in domains]].notna().all(axis=1).sum()), "n_all4_future": int(d[[f"{x}_future" for x in domains]].notna().all(axis=1).sum()), "n_composite_ge3_change": int(d["ic_loss_comp3_z"].notna().sum()), "n_composite_all4_change": int(d["ic_loss_comp4_z"].notna().sum()), "incident_events": int(d.event_future.sum())})
    return pd.DataFrame(all_results), pd.DataFrame(availability)


def fixed_effect_meta(mcid):
    """Fixed-effect synthesis of candidate prognostic ORs, for stability only."""
    rows = []
    d = mcid.loc[(mcid.status == "ok") & mcid.adjusted_or.notna()].copy()
    for (measure, thr), g in d.groupby(["measure", "threshold_sd"], sort=True):
        if len(g) < 2:
            continue
        log_or = np.log(g.adjusted_or.astype(float).to_numpy())
        se = (np.log(g.ci_high.astype(float).to_numpy()) - np.log(g.ci_low.astype(float).to_numpy())) / (2 * 1.96)
        w = 1 / (se ** 2); pooled = float(np.sum(w * log_or) / np.sum(w)); pooled_se = float(np.sqrt(1 / np.sum(w)))
        q = float(np.sum(w * (log_or - pooled) ** 2)); df = len(g) - 1
        i2 = max(0.0, (q - df) / q * 100) if q > 0 else 0.0
        rows.append({"measure": measure, "threshold_sd": thr, "cohorts": len(g), "pooled_or_fixed": float(np.exp(pooled)), "ci_low": float(np.exp(pooled - 1.96 * pooled_se)), "ci_high": float(np.exp(pooled + 1.96 * pooled_se)), "Q": q, "I2_pct": i2})
    return pd.DataFrame(rows)


def net_benefit(y, p, t):
    y = np.asarray(y, int); p = np.asarray(p, float)
    pred = p >= t
    return float((pred & (y == 1)).sum() / len(y) - (pred & (y == 0)).sum() / len(y) * t / (1 - t))


def dca_bootstrap(y, pb, pf, sample, cohort, window):
    y = np.asarray(y, int); pb = np.asarray(pb, float); pf = np.asarray(pf, float)
    seed_key = f"{sample}|{cohort}|{window}".encode("utf-8")
    seed_offset = int(hashlib.sha256(seed_key).hexdigest()[:8], 16) % 100000
    rng = np.random.default_rng(SEED + seed_offset)
    rows = []
    prevalence = float(y.mean())
    for t in DCA_THRESHOLDS:
        b = net_benefit(y, pb, t); f = net_benefit(y, pf, t); diff = f - b
        ta = prevalence - (1 - prevalence) * t / (1 - t)
        bs = []; fs = []; ds = []
        for _ in range(BOOT):
            idx = rng.integers(0, len(y), len(y))
            bs.append(net_benefit(y[idx], pb[idx], t)); fs.append(net_benefit(y[idx], pf[idx], t)); ds.append(fs[-1] - bs[-1])
        rows.append({"sample": sample, "cohort": cohort, "window": window, "threshold": float(t), "n": len(y), "event_rate": prevalence, "base_net_benefit": b, "base_low": np.quantile(bs, .025), "base_high": np.quantile(bs, .975), "ic_net_benefit": f, "ic_low": np.quantile(fs, .025), "ic_high": np.quantile(fs, .975), "delta_net_benefit": diff, "delta_low": np.quantile(ds, .025), "delta_high": np.quantile(ds, .975), "treat_all_net_benefit": ta, "treat_none_net_benefit": 0.0})
    return rows


def run_dca(mod):
    pairs = []
    for fn in [mod.build_elsa, mod.build_charls, mod.build_hrs]:
        d, _ = fn(); pairs.append(d)
    sh7, _ = mod.build_share(7); sh8, _ = mod.build_share(8)
    def prep(d):
        return mod.cohort_standardize(d, ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"])[0]
    dev = pd.concat([prep(x) for x in pairs], ignore_index=True)
    base_terms = ["age_z", "sex", "education_z", "baseline_fi_z"]
    full_terms = base_terms + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
    dev = dev.dropna(subset=["event", *full_terms]).copy()
    mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(dev[base_terms], dev.event)
    mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(dev[full_terms], dev.event)
    rows = []
    rows += dca_bootstrap(dev.event, mb.predict_proba(dev[base_terms])[:, 1], mf.predict_proba(dev[full_terms])[:, 1], "development_apparent", "ELSA+CHARLS+HRS", "anchor_windows")
    for sh, window in [(sh7, "6->7"), (sh8, "6->8")]:
        sh = prep(sh).dropna(subset=["event", *full_terms]).copy()
        rows += dca_bootstrap(sh.event, mb.predict_proba(sh[base_terms])[:, 1], mf.predict_proba(sh[full_terms])[:, 1], "external_validation", "SHARE", window)
    return pd.DataFrame(rows)


def main():
    mod = load66()
    anchor = anchor_audit()
    anchor.to_csv(TAB / "mcid_anchor_audit.csv", index=False, encoding="utf-8-sig")
    mcid, avail = run_mcid(mod)
    mcid.to_csv(TAB / "mcid_candidate_thresholds.csv", index=False, encoding="utf-8-sig")
    avail.to_csv(TAB / "mcid_change_availability.csv", index=False, encoding="utf-8-sig")
    fixed_effect_meta(mcid).to_csv(TAB / "mcid_candidate_meta_fixed_effect.csv", index=False, encoding="utf-8-sig")
    dca = run_dca(mod)
    dca.to_csv(TAB / "incident_disability_dca_bootstrap.csv", index=False, encoding="utf-8-sig")
    key = dca[dca.threshold.isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])].copy()
    key.to_csv(TAB / "incident_disability_dca_key_thresholds.csv", index=False, encoding="utf-8-sig")
    memo = """# MCID and DCA exploratory package

## MCID feasibility

The four cohorts do not contain a harmonized patient-global change rating or clinician-rated transition anchor prespecified in the analysis manifest. Therefore a validated minimal clinically important difference (MCID) cannot be identified from these data. The candidate-threshold table is explicitly distribution-based/prognostic: 0.2, 0.5 and 1.0 within-cohort baseline SD of domain or composite capacity decline. Adjusted odds ratios describe association with incident disability and are not MCID estimates.

## Decision-curve analysis

DCA was recomputed for the frozen incident-disability models over 5–30% risk thresholds and augmented with 400 bootstrap percentile intervals. The thresholds are operating-characteristic values spanning the observed event rates; no treatment, referral or rehabilitation action threshold has been validated. Development results are apparent and external SHARE results are the more informative utility check.
"""
    (OUT / "mcid_dca_exploration_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 36, "script": "72_mcid_dca_exploration.py", "mcid": {"validated_anchor_available": False, "candidate_thresholds_sd": THRESHOLDS, "interpretation": "exploratory prognostic/distribution-based thresholds; not MCID"}, "dca": {"threshold_range": "0.05-0.30", "bootstrap_replicates": BOOT, "development": "apparent", "external": "SHARE 6->7 and 6->8", "clinical_threshold_validated": False}, "output_level": "aggregate only"}
    (OUT / "mcid_dca_exploration_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MCID availability:"); print(avail.to_string(index=False)); print("\nMCID candidates:"); print(mcid.to_string(index=False)); print("\nDCA key thresholds:"); print(key.to_string(index=False))


if __name__ == "__main__":
    main()
