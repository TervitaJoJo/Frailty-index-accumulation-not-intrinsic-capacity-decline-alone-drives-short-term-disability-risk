"""Fixed-window competing-event sensitivity for incident disability.

The primary disability analysis requires a completed follow-up interview. This
stage asks whether deaths that prevent follow-up materially change the risk
ranking. Because ELSA and CHARLS do not provide an interval-aligned death
status in the audited working extracts, the analysis is restricted to HRS and
SHARE and is explicitly a fixed-window competing-event sensitivity. Outcomes
are mutually exclusive: 0 = alive without observed disability, 1 = incident
disability, 2 = death before the follow-up window closes.

No individual identifiers, predictions or weights are written.
"""
from __future__ import annotations

from pathlib import Path
import json
import importlib.util
import hashlib

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SEED = 20260925
BOOT = 400
BASE = ["age_z", "sex", "education_z", "baseline_fi_z"]
FULL = BASE + ["cognition_z", "locomotion_z", "grip_vitality_z", "psychological_z"]
CONT = ["age", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
SHARE_ITEMS = ["walkra", "dressa", "batha", "eata", "beda", "toilta", "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "leavhsa", "laundrya"]


def load66():
    p = OUT / "66_incident_disability_prediction.py"
    spec = importlib.util.spec_from_file_location("stage66_competing", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def harmonize_sex(d):
    out = d.copy(); x = pd.to_numeric(out["sex"], errors="coerce")
    if set(x.dropna().unique()).issubset({1.0, 2.0}): out["sex"] = (x - 1).where(x.isin([1, 2]))
    return out


def std(d):
    out = d.copy()
    for c in CONT:
        x = pd.to_numeric(out[c], errors="coerce"); mu = float(x.mean()); sd = float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0: sd = 1.0
        out[c + "_z"] = (x - mu) / sd
    return out


def hrs_frame(mod):
    path = ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
    cols = ["hhidpn", "inw10", "inw11", "r10proxy", "r11proxy", "r10imrc", "r10dlrc", "r10ser7",
            "r10walkra", "r10walk1a", "r10walksa", "r10grpl", "r10grpr", "r10cesd", "r10agey_m",
            "ragender", "raeduc", "r10shlt", "r10bmi", "r10hibpe", "r10diabe", "r10hearte", "r10stroke",
            "r10cancre", "r10arthre", "r10adl5a", "r10iadl5a", "r11adl5a", "r11iadl5a", "radyear", "radmonth"]
    d = mod.read_dta(path, cols)
    b = mod.num(d["inw10"]).eq(1) & mod.num(d["r10proxy"]).eq(0)
    f = mod.num(d["inw11"]).eq(1) & mod.num(d["r11proxy"]).eq(0)
    cog = mod.mean_min(d, ["r10imrc", "r10dlrc", "r10ser7"], {"r10imrc": lambda x: x / 10, "r10dlrc": lambda x: x / 10, "r10ser7": lambda x: x / 5}, minimum=2)
    loc = mod.mean_min(d, ["r10walkra", "r10walk1a", "r10walksa"], {c: lambda x: 1 - (x > 0).astype(float) for c in ["r10walkra", "r10walk1a", "r10walksa"]})
    grip = mod.max_min(d, ["r10grpl", "r10grpr"], {"r10grpl": lambda x: x / 100, "r10grpr": lambda x: x / 100})
    psych = 1 - mod.num(d["r10cesd"], 0, 8) / 8
    fi = mod.fi_score(d, ["r10hibpe", "r10diabe", "r10hearte", "r10stroke", "r10cancre", "r10arthre"], "r10shlt", "r10bmi")
    b_none, _ = mod.endpoint_counts(d["r10adl5a"], d["r10iadl5a"])
    _, f_event = mod.endpoint_counts(d["r11adl5a"], d["r11iadl5a"])
    f_obs = mod.num(d["r11adl5a"], 0, 30).notna() & mod.num(d["r11iadl5a"], 0, 30).notna()
    out = pd.DataFrame({"id": mod.safe_id(d["hhidpn"]), "active_b": b, "active_f": f, "age": mod.num(d["r10agey_m"]),
                        "sex": (mod.num(d["ragender"]) - 1).where(mod.num(d["ragender"]).isin([1, 2])), "education": mod.num(d["raeduc"]),
                        "baseline_fi": fi, "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych,
                        "baseline_none": b_none, "target_event": f_event, "target_observed": f_obs,
                        "death_year": mod.num(d["radyear"], 1900, 2030), "death_month": mod.num(d["radmonth"], 1, 12)})
    # HRS wave 10 is 2010 and wave 11 is 2012. The strict definition treats
    # deaths in 2010–2011 as deaths before the next interview; 2012 is a broad
    # sensitivity because target interview month is not available in RAND.
    out["death_strict"] = out.death_year.isin([2010, 2011])
    out["death_broad"] = out.death_year.isin([2010, 2011, 2012])
    out["cohort"] = "HRS"; out["window"] = "10->11"
    return out


def share_frame(mod, target_wave=7):
    path = next((ROOT / "SHARE").rglob("Working_data/share.dta"))
    cols = ["mergeid", "wave", "agey", "ragender", "raeducl", "imrc", "dlrc", "orient", "walkra", "walk100a", "lgrip", "rgrip", "eurod", "shlt", "bmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "retyr", "retmon", "radyear", "radmonth", *SHARE_ITEMS]
    d = mod.read_dta(path, cols); d = d.loc[mod.num(d["wave"]).isin([6, target_wave]) & d["mergeid"].notna()].copy()
    d["id"] = mod.safe_id(d["mergeid"]); d["wave_n"] = mod.num(d["wave"]); d = d.drop_duplicates(["id", "wave_n"], keep="first")
    base = d.wave_n.eq(6); fut = d.wave_n.eq(target_wave)
    cog = mod.mean_min(d, ["imrc", "dlrc", "orient"], {"imrc": lambda x: x / 10, "dlrc": lambda x: x / 10, "orient": lambda x: x / 4}, minimum=2)
    loc = mod.mean_min(d, ["walkra", "walk100a"], {"walkra": lambda x: 1 - x, "walk100a": lambda x: 1 - x})
    grip = mod.max_min(d, ["lgrip", "rgrip"], {"lgrip": lambda x: x / 100, "rgrip": lambda x: x / 100})
    psych = 1 - mod.num(d["eurod"], 0, 12) / 12
    fi = mod.fi_score(d, ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"], "shlt", "bmi")
    z = pd.concat([mod.num(d[c], 0, 1) for c in SHARE_ITEMS], axis=1); obs = z.notna().sum(axis=1).ge(10)
    b_none = obs & z.max(axis=1, skipna=True).eq(0); f_event = obs & z.max(axis=1, skipna=True).gt(0); f_obs = obs
    base_year = mod.num(d["retyr"], 2000, 2030)
    out = pd.DataFrame({"id": d.id, "active_b": base, "active_f": fut, "age": mod.num(d["agey"]), "sex": mod.num(d["ragender"]), "education": mod.num(d["raeducl"]), "baseline_fi": fi, "cognition": cog, "locomotion": loc, "grip_vitality": grip, "psychological": psych, "baseline_none": b_none, "target_event": f_event, "target_observed": f_obs, "death_year": mod.num(d["radyear"], 1900, 2030), "death_month": mod.num(d["radmonth"], 1, 12), "interview_year": base_year})
    # Convert the person-wave rows into a baseline-wide frame. A death date is
    # repeated across waves, so it must be retained from the baseline row and
    # the follow-up endpoint must come from the target-wave row.
    b = out.loc[out.active_b].drop_duplicates("id", keep="first").copy()
    f = out.loc[out.active_f, ["id", "target_event", "target_observed"]].drop_duplicates("id", keep="first").copy()
    f["has_target"] = True
    out = b.drop(columns=["active_f", "target_event", "target_observed"]).merge(f, on="id", how="left")
    out["active_f"] = out["has_target"].fillna(False)
    out["target_event"] = out["target_event"].fillna(False)
    out["target_observed"] = out["target_observed"].fillna(False)
    out = out.drop(columns=["has_target"])
    # Wave 7 was fielded after wave 6. For a strict, reproducible sensitivity,
    # classify death in 2016–2017 as before the next interview and retain a
    # broader 2016–2018 definition because individual target months are sparse.
    out["death_strict"] = out.death_year.isin([2016, 2017])
    out["death_broad"] = out.death_year.isin([2016, 2017, 2018])
    out["cohort"] = "SHARE"; out["window"] = "6->7"
    return out


def classify(d, death_col):
    x = d.loc[d.active_b & d.baseline_none].copy()
    # Death takes priority only when the follow-up interview is absent. This
    # avoids labelling a person as dead before a completed target interview.
    x["status"] = np.nan
    x.loc[x.active_f & x.target_observed & ~x[death_col], "status"] = x.loc[x.active_f & x.target_observed & ~x[death_col], "target_event"].astype(int)
    x.loc[~(x.active_f & x.target_observed) & x[death_col], "status"] = 2
    return x.dropna(subset=["status"]).copy().astype({"status": int})


def multiclass_metrics(y, p, cohort, window, model, definition):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    one = np.eye(3)[y]
    rows = [{"cohort": cohort, "window": window, "death_definition": definition, "model": model, "n": len(y), "events_disability": int((y == 1).sum()), "events_death": int((y == 2).sum()), "multiclass_brier": float(np.mean(np.sum((p - one) ** 2, axis=1))), "multiclass_logloss": float(log_loss(y, p, labels=[0, 1, 2]))}]
    for k, label in [(1, "disability"), (2, "death")]:
        rows[0][f"auc_{label}_vs_other"] = float(roc_auc_score((y == k).astype(int), p[:, k])) if len(np.unique(y == k)) == 2 else np.nan
        rows[0][f"observed_{label}_risk"] = float(np.mean(y == k))
    return rows[0]


def bootstrap_deltas(y, pb, pf, seed_key, b=BOOT):
    """Paired apparent bootstrap intervals for model increments."""
    y = np.asarray(y, dtype=int); pb = np.asarray(pb, dtype=float); pf = np.asarray(pf, dtype=float)
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16) % 1000000
    rng = np.random.default_rng(SEED + seed)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y)); yy = y[idx]
        if len(np.unique(yy)) < 3:
            continue
        try:
            vals.append([
                roc_auc_score(yy == 1, pf[idx, 1]) - roc_auc_score(yy == 1, pb[idx, 1]),
                roc_auc_score(yy == 2, pf[idx, 2]) - roc_auc_score(yy == 2, pb[idx, 2]),
                np.mean(np.sum((pf[idx] - np.eye(3)[yy]) ** 2, axis=1)) - np.mean(np.sum((pb[idx] - np.eye(3)[yy]) ** 2, axis=1)),
                log_loss(yy, pf[idx], labels=[0, 1, 2]) - log_loss(yy, pb[idx], labels=[0, 1, 2]),
            ])
        except ValueError:
            continue
    if not vals:
        return {}
    a = np.asarray(vals); names = ["delta_disability_auc", "delta_death_auc", "delta_multiclass_brier", "delta_logloss"]
    return {f"{name}_{bound}": float(np.quantile(a[:, j], q)) for j, name in enumerate(names) for bound, q in [("low", .025), ("high", .975)]}


def main():
    mod = load66(); frames = [(hrs_frame(mod), "strict"), (hrs_frame(mod), "broad"), (share_frame(mod), "strict"), (share_frame(mod), "broad")]
    rows = []; increments = []; sample_rows = []
    for d, definition in frames:
        dc = classify(d, "death_" + definition)
        z = std(harmonize_sex(dc)); z = z.dropna(subset=["status", *FULL]).copy()
        if len(z) < 100 or z.status.nunique() < 3: continue
        for model_name, terms in [("base", BASE), ("base_plus_IC", FULL)]:
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(z[terms], z.status)
            p = m.predict_proba(z[terms])
            rows.append(multiclass_metrics(z.status, p, str(z.cohort.iloc[0]), str(z.window.iloc[0]), model_name, definition))
        mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(z[BASE], z.status)
        mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(z[FULL], z.status)
        pb, pf = mb.predict_proba(z[BASE]), mf.predict_proba(z[FULL])
        inc_row = {"cohort": str(z.cohort.iloc[0]), "window": str(z.window.iloc[0]), "death_definition": definition, "n": len(z), "events_disability": int((z.status == 1).sum()), "events_death": int((z.status == 2).sum()), "delta_disability_auc": roc_auc_score(z.status.eq(1), pf[:, 1]) - roc_auc_score(z.status.eq(1), pb[:, 1]), "delta_death_auc": roc_auc_score(z.status.eq(2), pf[:, 2]) - roc_auc_score(z.status.eq(2), pb[:, 2]), "delta_multiclass_brier": np.mean(np.sum((pf - np.eye(3)[z.status]) ** 2, axis=1)) - np.mean(np.sum((pb - np.eye(3)[z.status]) ** 2, axis=1)), "delta_logloss": log_loss(z.status, pf, labels=[0, 1, 2]) - log_loss(z.status, pb, labels=[0, 1, 2])}
        inc_row.update(bootstrap_deltas(z.status, pb, pf, f"{z.cohort.iloc[0]}|{z.window.iloc[0]}|{definition}"))
        increments.append(inc_row)
        sample_rows.append({"cohort": str(z.cohort.iloc[0]), "window": str(z.window.iloc[0]), "death_definition": definition, "n": len(z), "status_0_alive_no_disability": int((z.status == 0).sum()), "status_1_disability": int((z.status == 1).sum()), "status_2_death": int((z.status == 2).sum())})
    metrics = pd.DataFrame(rows); inc = pd.DataFrame(increments); samp = pd.DataFrame(sample_rows)
    metrics.to_csv(TAB / "mortality_competing_event_metrics.csv", index=False, encoding="utf-8-sig")
    inc.to_csv(TAB / "mortality_competing_event_incremental.csv", index=False, encoding="utf-8-sig")
    samp.to_csv(TAB / "mortality_competing_event_samples.csv", index=False, encoding="utf-8-sig")
    memo = """# Fixed-window mortality competing-event sensitivity\n\nThis exploratory sensitivity was restricted to HRS and SHARE because the audited ELSA and CHARLS extracts do not support an interval-aligned death endpoint. The mutually exclusive outcome is alive without observed disability, incident disability, or death before the follow-up interview/window closes. HRS uses a strict 2010–2011 death definition and a broad 2010–2012 sensitivity; SHARE uses strict 2016–2017 and broad 2016–2018 definitions. Because exact target interview dates are unavailable or sparse, this is a fixed-window multinomial competing-event analysis, not a continuous-time Fine–Gray model.\n\nThe base and IC-augmented multinomial models report multiclass Brier/log-loss and one-vs-rest AUC for disability and death. Results are exploratory and do not replace the primary disability model.\n"""
    (OUT / "mortality_competing_event_memo.md").write_text(memo, encoding="utf-8")
    run = {"stage": 35, "script": "71_mortality_competing_event_sensitivity.py", "cohorts": ["HRS", "SHARE"], "outcomes": ["alive_no_disability", "incident_disability", "death"], "definitions": ["strict", "broad"], "model": "multinomial fixed-window competing-event sensitivity", "continuous_time": False, "output_level": "aggregate only"}
    (OUT / "mortality_competing_event_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(samp.to_string(index=False)); print("\nMetrics:"); print(metrics.to_string(index=False)); print("\nIncremental:"); print(inc.to_string(index=False))


if __name__ == "__main__": main()
