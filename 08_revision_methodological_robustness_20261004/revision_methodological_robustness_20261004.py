"""Methodological robustness analyses for the IC--frailty trajectory manuscript.

This script is aggregate-only. It evaluates incremental information beyond FI
accumulation, continuous change and interaction, survey-weight sensitivity, and
FI content sensitivity under the frozen three-cohort trajectory state design.
Participant identifiers and individual predictions remain in memory only.
"""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

PKG = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("IC_FRAILTY_STATISTICAL_MODELING", "PATH_TO_STATISTICAL_MODELING"))
OUT = PKG
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG))
import trajectory_states_analysis_20260930 as ts  # noqa: E402
from revision_data import META, predictors  # noqa: E402

SEED = 20261004
BOOTSTRAP_B = 400
Z = 1.959963984540054
STATES = ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"]
DISEASES = ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"]


def q025(values):
    return float(np.quantile(values, 0.025)) if len(values) else np.nan


def q975(values):
    return float(np.quantile(values, 0.975)) if len(values) else np.nan


def load_development():
    out = {}
    for cohort in ["ELSA", "CHARLS", "HRS"]:
        raw, _ = ts.load_cohort(cohort)
        z, _ = ts.prepare_analysis(raw)
        q = ts.assign_states(z, "primary_sign")
        q["event"] = q.outcome_category.eq("disability").astype(int)
        q["ic_decline"] = q.state.isin(["ic_decline_only", "coupled"]).astype(int)
        q["fi_accumulation"] = q.state.isin(["fi_accumulation_only", "coupled"]).astype(int)
        q["interaction"] = q.ic_decline * q.fi_accumulation
        out[cohort] = q
    return out


def robust_contrast(fit, terms):
    beta = fit.params.loc[terms].to_numpy(dtype=float)
    cov = fit.cov_params().loc[terms, terms].to_numpy(dtype=float)
    est = float(beta.sum())
    se = float(np.sqrt(np.ones(len(beta)) @ cov @ np.ones(len(beta))))
    return {
        "log_odds": est,
        "OR": float(np.exp(est)),
        "OR_low": float(np.exp(est - Z * se)),
        "OR_high": float(np.exp(est + Z * se)),
        "p": float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan,
    }


def fit_nested_incremental(dev):
    q = pd.concat(dev.values(), ignore_index=True)
    q = q.loc[q.outcome_category.isin(["disability", "disability_free", "death"])].copy()
    cov = ["age_baseline", "male", "education", "baseline_fi"]
    keep = ["event", "fi_accumulation", "ic_decline", "interaction", "cohort", *cov]
    q = q.dropna(subset=keep).copy()
    rhs_base = "event ~ fi_accumulation + age_baseline + male + education + baseline_fi + C(cohort)"
    rhs_full = rhs_base + " + ic_decline + interaction"
    base_plain = smf.glm(rhs_base, data=q, family=sm.families.Binomial()).fit()
    full_plain = smf.glm(rhs_full, data=q, family=sm.families.Binomial()).fit()
    base = smf.glm(rhs_base, data=q, family=sm.families.Binomial()).fit(cov_type="HC3")
    full = smf.glm(rhs_full, data=q, family=sm.families.Binomial()).fit(cov_type="HC3")
    lr = 2 * (full_plain.llf - base_plain.llf)
    lr_p = float(chi2.sf(lr, full_plain.df_model - base_plain.df_model))
    contrast = robust_contrast(full, ["ic_decline", "interaction"])
    rows = [{
        "analysis": "nested_incremental_state_model",
        "n": int(len(q)),
        "events": int(q.event.sum()),
        "base_model": rhs_base,
        "full_model": rhs_full,
        "likelihood_ratio": float(lr),
        "likelihood_ratio_df": int(full_plain.df_model - base_plain.df_model),
        "likelihood_ratio_p": lr_p,
        "coupled_vs_fi_only_OR": contrast["OR"],
        "coupled_vs_fi_only_OR_low": contrast["OR_low"],
        "coupled_vs_fi_only_OR_high": contrast["OR_high"],
        "coupled_vs_fi_only_p": contrast["p"],
    }]
    pd.DataFrame(rows).to_csv(OUT / "nested_incremental_state_model.csv", index=False)
    return q, rhs_base, rhs_full


def fit_continuous_change(dev):
    parts = []
    for cohort, q in dev.items():
        z = q.copy()
        for col in ["delta_ic", "delta_fi"]:
            mu = z[col].mean()
            sd = z[col].std(ddof=1)
            z[col + "_z"] = (z[col] - mu) / (sd if np.isfinite(sd) and sd > 0 else 1.0)
        z["delta_interaction"] = z.delta_ic_z * z.delta_fi_z
        parts.append(z)
    q = pd.concat(parts, ignore_index=True)
    q = q.loc[q.outcome_category.isin(["disability", "disability_free", "death"])].copy()
    rhs = "event ~ delta_ic_z + delta_fi_z + delta_interaction + age_baseline + male + education + baseline_fi + C(cohort)"
    keep = ["event", "delta_ic_z", "delta_fi_z", "delta_interaction", "age_baseline", "male", "education", "baseline_fi", "cohort"]
    q = q.dropna(subset=keep)
    fit = smf.glm(rhs, data=q, family=sm.families.Binomial()).fit(cov_type="HC3")
    rows = []
    for term in ["delta_ic_z", "delta_fi_z", "delta_interaction"]:
        b = float(fit.params[term]); se = float(fit.bse[term])
        rows.append({
            "analysis": "continuous_change_interaction",
            "term": term,
            "n": int(len(q)),
            "events": int(q.event.sum()),
            "OR_per_SD": float(np.exp(b)),
            "OR_low": float(np.exp(b - Z * se)),
            "OR_high": float(np.exp(b + Z * se)),
            "p": float(fit.pvalues[term]),
            "formula": rhs,
        })
    pd.DataFrame(rows).to_csv(OUT / "continuous_change_interaction.csv", index=False)


def design_matrix(q, full):
    d = q[["age_baseline", "male", "education", "baseline_fi", "fi_accumulation", "ic_decline", "interaction", "cohort"]].copy()
    for c in ["age_baseline", "education", "baseline_fi"]:
        d[c] = d[c].astype(float)
        d[c] = (d[c] - d[c].mean()) / (d[c].std(ddof=1) or 1.0)
    x = pd.get_dummies(d, columns=["cohort"], drop_first=False, dtype=float)
    cols = ["age_baseline", "male", "education", "baseline_fi", "fi_accumulation"]
    if full:
        cols += ["ic_decline", "interaction"]
    cols += [c for c in x.columns if c.startswith("cohort_")]
    return x[cols].to_numpy(float)


def bootstrap_incremental_metrics(q):
    q = q.copy().reset_index(drop=True)
    x_base = design_matrix(q, False)
    x_full = design_matrix(q, True)
    y = q.event.to_numpy(dtype=int)
    base_fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(x_base, y)
    full_fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(x_full, y)
    p_base = base_fit.predict_proba(x_base)[:, 1]
    p_full = full_fit.predict_proba(x_full)[:, 1]
    apparent = {
        "auc_base": float(roc_auc_score(y, p_base)),
        "auc_full": float(roc_auc_score(y, p_full)),
        "delta_auc": float(roc_auc_score(y, p_full) - roc_auc_score(y, p_base)),
        "brier_base": float(brier_score_loss(y, p_base)),
        "brier_full": float(brier_score_loss(y, p_full)),
        "delta_brier": float(brier_score_loss(y, p_full) - brier_score_loss(y, p_base)),
    }
    rng = np.random.default_rng(SEED)
    strata = {c: np.flatnonzero(q.cohort.to_numpy() == c) for c in q.cohort.unique()}
    rows = []
    for b in range(BOOTSTRAP_B):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata.values()])
        try:
            mb = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(x_base[idx], y[idx])
            mf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(x_full[idx], y[idx])
            pb_boot = mb.predict_proba(x_base[idx])[:, 1]
            pf_boot = mf.predict_proba(x_full[idx])[:, 1]
            pb_test = mb.predict_proba(x_base)[:, 1]
            pf_test = mf.predict_proba(x_full)[:, 1]
            auc_app_b = roc_auc_score(y[idx], pf_boot) - roc_auc_score(y[idx], pb_boot)
            auc_test_b = roc_auc_score(y, pf_test) - roc_auc_score(y, pb_test)
            bri_app_b = brier_score_loss(y[idx], pf_boot) - brier_score_loss(y[idx], pb_boot)
            bri_test_b = brier_score_loss(y, pf_test) - brier_score_loss(y, pb_test)
            rows.append({"bootstrap": b + 1, "delta_auc_apparent": auc_app_b, "delta_auc_test": auc_test_b,
                         "delta_brier_apparent": bri_app_b, "delta_brier_test": bri_test_b})
        except Exception:
            continue
    boot = pd.DataFrame(rows)
    corrected = {
        "optimism_corrected_delta_auc": apparent["delta_auc"] - float((boot.delta_auc_apparent - boot.delta_auc_test).mean()),
        "optimism_corrected_delta_brier": apparent["delta_brier"] - float((boot.delta_brier_apparent - boot.delta_brier_test).mean()),
        "bootstrap_delta_auc_low": q025(boot.delta_auc_test.tolist()),
        "bootstrap_delta_auc_high": q975(boot.delta_auc_test.tolist()),
        "bootstrap_delta_brier_low": q025(boot.delta_brier_test.tolist()),
        "bootstrap_delta_brier_high": q975(boot.delta_brier_test.tolist()),
    }
    row = {"analysis": "nested_incremental_prediction_metrics", "n": int(len(q)), "events": int(y.sum()), **apparent, **corrected, "bootstrap_replicates": int(len(boot))}
    pd.DataFrame([row]).to_csv(OUT / "nested_incremental_prediction_metrics.csv", index=False)
    boot.to_csv(OUT / "nested_incremental_prediction_bootstrap.csv", index=False)


def read_weight_table(cohort):
    if cohort == "ELSA":
        p = ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_data_v2.dta"
        d, _ = pyreadstat.read_dta(str(p), usecols=["idauniq", "w6xwgt"])
        d["id"] = d.idauniq.astype(str)
        return d[["id", "w6xwgt"]].rename(columns={"w6xwgt": "weight"})
    if cohort == "CHARLS":
        p = META[cohort]["path"]
        d, _ = pyreadstat.read_dta(str(p), usecols=["ID", "r3wtrespb"])
        d["id"] = d.ID.astype(str)
        return d[["id", "r3wtrespb"]].rename(columns={"r3wtrespb": "weight"})
    p = META[cohort]["path"]
    d, _ = pyreadstat.read_dta(str(p), usecols=["hhidpn", "r10wtresp"])
    d["id"] = d.hhidpn.astype(str)
    return d[["id", "r10wtresp"]].rename(columns={"r10wtresp": "weight"})


def survey_weight_sensitivity(dev):
    risk_rows = []
    model_parts = []
    for cohort, q in dev.items():
        w = read_weight_table(cohort)
        q = q.copy()
        q["id"] = q.id.astype(str)
        q = q.merge(w, on="id", how="left", validate="one_to_one")
        q["weight"] = pd.to_numeric(q.weight, errors="coerce")
        q = q.loc[q.weight.gt(0)].copy()
        q["weight_norm"] = q.weight / q.weight.mean()
        q["event"] = q.outcome_category.eq("disability").astype(int)
        for state, h in q.groupby("state", observed=True):
            den = h.weight_norm.sum()
            risk_rows.append({"cohort": cohort, "state": state, "n": int(len(h)), "weighted_denominator": float(den),
                              "unweighted_disability_risk": float(h.event.mean()),
                              "survey_weighted_disability_risk": float(np.average(h.event, weights=h.weight_norm))})
        model_parts.append(q)
    pooled = pd.concat(model_parts, ignore_index=True)
    rhs = "event ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi + C(cohort)"
    keep = ["event", "state", "age_baseline", "male", "education", "baseline_fi", "cohort", "weight_norm"]
    g = pooled.dropna(subset=keep).copy()
    fit = smf.glm(rhs, data=g, family=sm.families.Binomial(), freq_weights=g.weight_norm).fit(cov_type="HC3")
    term = "C(state, Treatment(reference='preserved_low'))[T.coupled]"
    b = float(fit.params[term]); se = float(fit.bse[term])
    model_row = {"analysis": "survey_weighted_state_sensitivity", "n": int(len(g)), "events": int(np.sum(g.event * g.weight_norm)),
                 "weighted_coupled_vs_preserved_OR": float(np.exp(b)), "OR_low": float(np.exp(b - Z * se)),
                 "OR_high": float(np.exp(b + Z * se)), "p": float(fit.pvalues[term]), "formula": rhs}
    pd.DataFrame(risk_rows).to_csv(OUT / "survey_weighted_state_risks.csv", index=False)
    pd.DataFrame([model_row]).to_csv(OUT / "survey_weighted_state_model.csv", index=False)


def component_fi(d, minimum):
    z = d[DISEASES].apply(pd.to_numeric, errors="coerce")
    z = z.where(z.isin([0, 1]))
    return z.mean(axis=1).where(z.notna().sum(axis=1).ge(minimum))


def source_component_frames(cohort):
    spec = ts.COHORT_SPEC[cohort]
    raw = pyreadstat.read_dta(
        META[cohort]["path"], usecols=ts.required_columns(cohort)
    )[0]
    if cohort == "CHARLS":
        for wave in spec["waves"]:
            if f"r{wave}shlt" not in raw.columns and f"r{wave}shlta" in raw.columns:
                raw[f"r{wave}shlt"] = raw[f"r{wave}shlta"]
            if f"r{wave}mbmi" not in raw.columns:
                raw[f"r{wave}mbmi"] = np.nan
    if cohort == "ELSA":
        raw = raw.loc[pd.to_numeric(raw["wave"], errors="coerce").isin(spec["waves"])].copy()
        frames = {}
        for wave in [spec["base"], spec["intermediate"]]:
            sub = raw.loc[pd.to_numeric(raw["wave"], errors="coerce").eq(wave)].copy()
            sub = sub.drop_duplicates(spec["id"], keep="last")
            p = predictors(sub, cohort, "")
            p["id"] = sub[spec["id"]].astype(str).to_numpy()
            frames[wave] = p
        return frames
    frames = {}
    for wave in [spec["base"], spec["intermediate"]]:
        pfx = f"r{wave}"
        p = predictors(raw, cohort, pfx)
        p["id"] = raw[spec["id"]].astype(str).to_numpy()
        frames[wave] = p
    return frames


def fi_content_sensitivity(dev):
    rows = []
    for cohort, q in dev.items():
        spec = ts.COHORT_SPEC[cohort]
        frames = source_component_frames(cohort)
        b = frames[spec["base"]][["id", *DISEASES]].copy()
        i = frames[spec["intermediate"]][["id", *DISEASES]].copy()
        b["fi_disease_min5"] = component_fi(b, 5)
        i["fi_disease_min5"] = component_fi(i, 5)
        b["fi_disease_complete6"] = component_fi(b, 6)
        i["fi_disease_complete6"] = component_fi(i, 6)
        alt = b[["id", "fi_disease_min5", "fi_disease_complete6"]].merge(
            i[["id", "fi_disease_min5", "fi_disease_complete6"]], on="id", suffixes=("_base", "_intermediate"), validate="one_to_one")
        q2 = q.copy(); q2["id"] = q2.id.astype(str)
        q2 = q2.merge(alt, on="id", how="left", validate="one_to_one")
        for definition in ["fi_disease_min5", "fi_disease_complete6"]:
            q2["delta_alt"] = q2[f"{definition}_intermediate"] - q2[f"{definition}_base"]
            q2["fi_alt_accumulation"] = q2.delta_alt.gt(0)
            q2["alt_state"] = np.select(
                [q2.ic_decline.eq(False) & q2.fi_alt_accumulation.eq(False),
                 q2.ic_decline.eq(True) & q2.fi_alt_accumulation.eq(False),
                 q2.ic_decline.eq(False) & q2.fi_alt_accumulation.eq(True),
                 q2.ic_decline.eq(True) & q2.fi_alt_accumulation.eq(True)], STATES, default=None)
            g = q2.loc[q2.outcome_category.isin(["disability", "disability_free", "death"])].copy()
            g["event"] = g.outcome_category.eq("disability").astype(int)
            keep = ["event", "alt_state", "age_baseline", "male", "education", f"{definition}_base", "cohort"]
            g = g.dropna(subset=keep)
            if len(g) == 0 or g.alt_state.nunique() < 4:
                continue
            rhs = f"event ~ C(alt_state, Treatment(reference='fi_accumulation_only')) + age_baseline + male + education + {definition}_base + C(cohort)"
            fit = smf.glm(rhs, data=g, family=sm.families.Binomial()).fit(cov_type="HC3")
            term = "C(alt_state, Treatment(reference='fi_accumulation_only'))[T.coupled]"
            bcoef = float(fit.params[term]); se = float(fit.bse[term])
            rows.append({"cohort": cohort, "definition": definition, "n": int(len(g)), "events": int(g.event.sum()),
                         "coupled_vs_alt_FI_only_OR": float(np.exp(bcoef)), "OR_low": float(np.exp(bcoef - Z * se)),
                         "OR_high": float(np.exp(bcoef + Z * se)), "p": float(fit.pvalues[term]), "formula": rhs})
    pd.DataFrame(rows).to_csv(OUT / "fi_content_sensitivity.csv", index=False)


def main():
    dev = load_development()
    q, _, _ = fit_nested_incremental(dev)
    fit_continuous_change(dev)
    bootstrap_incremental_metrics(q)
    survey_weight_sensitivity(dev)
    fi_content_sensitivity(dev)
    run = {
        "script": Path(__file__).name,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_B,
        "cohorts": ["ELSA", "CHARLS", "HRS"],
        "analyses": ["nested_incremental_state_model", "continuous_change_interaction",
                     "optimism_corrected_incremental_auc_brier", "survey_weighted_state_sensitivity",
                     "FI excluding self-rated health and BMI, minimum 5/6 or complete 6/6 disease components"],
        "aggregate_only": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "run_info.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
