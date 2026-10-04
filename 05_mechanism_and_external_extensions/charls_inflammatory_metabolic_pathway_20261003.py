"""CHARLS inflammatory--metabolic pathway feasibility audit.

The script reuses the audited CHARLS directional-state construction and merges
the 2011 blood panel through ID_w1. It is an independent exploratory analysis;
it does not alter the primary manuscript or claim causal mediation.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf


PKG = Path(r"PATH_TO_IC_FRAILTY_V2")
ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
HARM = ROOT / "CHARLS" / "Harmonized_CHARLS" / "H_CHARLS_D_Data.dta"
BLOOD = ROOT / "CHARLS" / "2011" / "Blood_20140429.dta"
OUT = PKG / "analysis_audit" / "charls_inflammatory_metabolic_20261003"
OUT.mkdir(parents=True, exist_ok=True)


def load_main_module():
    path = PKG / "scripts" / "trajectory_states_analysis_20260930.py"
    spec = importlib.util.spec_from_file_location("charls_main_trajectory", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def clean(x):
    z = pd.to_numeric(x, errors="coerce")
    return z.mask(z <= 0)


def standardize(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if np.isfinite(sd) and sd > 0 else x * np.nan


def safe_glm(formula, data):
    try:
        fit = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(cov_type="HC3")
        rows = []
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            se = fit.bse[term]
            rows.append({"term": term, "OR": float(np.exp(coef)), "CI_low": float(np.exp(coef - 1.959964 * se)), "CI_high": float(np.exp(coef + 1.959964 * se)), "p": float(fit.pvalues[term])})
        return rows
    except Exception as exc:
        return [{"term": "MODEL_FAILED", "error": str(exc)}]


def safe_ols(formula, data):
    try:
        fit = smf.ols(formula, data=data).fit(cov_type="HC3")
        rows = []
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            se = fit.bse[term]
            rows.append({"term": term, "beta": float(coef), "CI_low": float(coef - 1.959964 * se), "CI_high": float(coef + 1.959964 * se), "p": float(fit.pvalues[term])})
        return rows
    except Exception as exc:
        return [{"term": "MODEL_FAILED", "error": str(exc)}]


def safe_weighted_glm(formula, data):
    try:
        fit = smf.glm(formula, data=data, family=sm.families.Binomial(), freq_weights=data["analysis_weight"]).fit(cov_type="HC3")
        rows = []
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            se = fit.bse[term]
            rows.append({"term": term, "OR": float(np.exp(coef)), "CI_low": float(np.exp(coef - 1.959964 * se)), "CI_high": float(np.exp(coef + 1.959964 * se)), "p": float(fit.pvalues[term])})
        return rows
    except Exception as exc:
        return [{"term": "MODEL_FAILED", "error": str(exc)}]


def safe_weighted_ols(formula, data):
    try:
        fit = smf.wls(formula, data=data, weights=data["analysis_weight"]).fit(cov_type="HC3")
        rows = []
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            se = fit.bse[term]
            rows.append({"term": term, "beta": float(coef), "CI_low": float(coef - 1.959964 * se), "CI_high": float(coef + 1.959964 * se), "p": float(fit.pvalues[term])})
        return rows
    except Exception as exc:
        return [{"term": "MODEL_FAILED", "error": str(exc)}]


def weighted_mean(x, w):
    ok = x.notna() & w.notna() & (w > 0)
    return float(np.average(x.loc[ok], weights=w.loc[ok])) if ok.any() else np.nan


def main():
    main = load_main_module()
    raw, _ = main.load_cohort("CHARLS")
    z, trajectory_mask = main.prepare_analysis(raw)
    z["id"] = z["id"].astype(str)

    # Harmonized CHARLS full ID maps to the 11-character biomarker ID through
    # ID_w1; direct matching on the 12-character full ID would silently fail.
    mapping = pyreadstat.read_dta(str(HARM), usecols=["ID", "ID_w1"])[0]
    mapping["id"] = mapping["ID"].astype(str)
    mapping["ID_w1"] = mapping["ID_w1"].astype(str)
    z = z.merge(mapping[["id", "ID_w1"]], on="id", how="left", validate="one_to_one")
    blood_cols = ["ID", "bloodweight", "newglu", "newcho", "newtg", "newhdl", "newldl", "newcrp", "newhba1c"]
    blood = pyreadstat.read_dta(str(BLOOD), usecols=blood_cols)[0]
    blood["ID_w1"] = blood["ID"].astype(str)
    for c in ["newglu", "newcho", "newtg", "newhdl", "newldl", "newcrp", "newhba1c", "bloodweight"]:
        blood[c] = pd.to_numeric(blood[c], errors="coerce")
    blood = blood.drop_duplicates("ID_w1", keep="first")
    q = z.merge(blood.drop(columns=["ID"]), on="ID_w1", how="left", validate="one_to_one")

    # Biomarkers are baseline measures collected before the W1--W3 trajectory
    # outcome window. CRP and triglycerides are log-transformed.
    for c in ["newglu", "newcho", "newtg", "newhdl", "newldl", "newcrp", "newhba1c"]:
        q[c] = clean(q[c])
    q["crp_log"] = np.log(q["newcrp"])
    q["trig_log"] = np.log(q["newtg"])
    q["tyg"] = np.log(q["newglu"] * q["newtg"] / 2.0)
    q["disability_event"] = q.outcome_category.eq("disability").astype(int)

    biomarkers = {
        "crp_log": "baseline_log_CRP",
        "newglu": "baseline_glucose",
        "newhba1c": "baseline_HbA1c",
        "newcho": "baseline_total_cholesterol",
        "newhdl": "baseline_HDL",
        "newldl": "baseline_LDL",
        "newtg": "baseline_triglycerides",
        "trig_log": "baseline_log_triglycerides",
        "tyg": "baseline_TyG",
    }
    thresholds = [
        ("primary_sign", 0.0, 0.0),
        ("cohort_median", q.delta_ic.median(), q.delta_fi.median()),
        ("meaningful_0.2SD", -0.2 * q.delta_ic.std(ddof=1), 0.2 * q.delta_fi.std(ddof=1)),
        ("meaningful_0.5SD", -0.5 * q.delta_ic.std(ddof=1), 0.5 * q.delta_fi.std(ddof=1)),
    ]
    counts, outcomes, summary, assoc, models, atten, coverage = [], [], [], [], [], [], []
    weighted_assoc, weighted_models = [], []
    for definition, ic_cut, fi_cut in thresholds:
        qq = q.copy()
        qq["ic_decline"] = qq.delta_ic < ic_cut
        qq["fi_accumulation"] = qq.delta_fi > fi_cut
        qq["state"] = np.select([~qq.ic_decline & ~qq.fi_accumulation, qq.ic_decline & ~qq.fi_accumulation, ~qq.ic_decline & qq.fi_accumulation, qq.ic_decline & qq.fi_accumulation], ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"], default="unknown")
        qq["state"] = pd.Categorical(qq["state"], categories=["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"])
        known = qq.outcome_category.ne("unknown")
        for state, h in qq.groupby("state", observed=False):
            if len(h) == 0:
                continue
            hk = h.loc[known.loc[h.index]]
            counts.append({"definition": definition, "state": state, "n": len(h), "percent": len(h) / len(qq)})
            outcomes.append({"definition": definition, "state": state, "n": len(h), "known_outcome_n": len(hk), "disability_events": int(hk.disability_event.sum()), "disability_risk": float(hk.disability_event.mean()) if len(hk) else np.nan, "weighted_biomarker_mean_glucose": weighted_mean(h.newglu, h.bloodweight)})
            for b, label in biomarkers.items():
                vals = pd.to_numeric(h[b], errors="coerce")
                summary.append({"definition": definition, "state": state, "biomarker": label, "n": int(vals.notna().sum()), "mean": vals.mean(), "sd": vals.std(ddof=1), "median": vals.median(), "weighted_mean": weighted_mean(vals, h.bloodweight)})
        for b, label in biomarkers.items():
            dat = qq[["state", "age_baseline", "male", "baseline_fi", b, "bloodweight"]].dropna().copy()
            coverage.append({"definition": definition, "biomarker": label, "available": len(dat), "trajectory_n": len(qq), "coverage": len(dat) / len(qq)})
            if len(dat) < 200 or dat.state.nunique() < 2:
                continue
            dat["biomarker_z"] = standardize(dat[b])
            for row in safe_ols("biomarker_z ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi", dat):
                row.update({"definition": definition, "biomarker": label, "analysis": "biomarker_by_state", "n": len(dat)})
                assoc.append(row)
            dat["analysis_weight"] = dat["bloodweight"] / dat["bloodweight"].mean()
            for row in safe_weighted_ols("biomarker_z ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi", dat):
                row.update({"definition": definition, "biomarker": label, "analysis": "weighted_biomarker_by_state", "n": len(dat)})
                weighted_assoc.append(row)
            d3 = qq[["disability_event", "state", "age_baseline", "male", "baseline_fi", b]].dropna().copy()
            d3 = d3.join(qq[["bloodweight"]], how="left")
            d3["biomarker_z"] = standardize(d3[b])
            state_formula = "disability_event ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi"
            full_formula = state_formula + " + biomarker_z"
            state_rows = safe_glm(state_formula, d3)
            full_rows = safe_glm(full_formula, d3)
            for row in full_rows:
                row.update({"definition": definition, "biomarker": label, "analysis": "disability_state_plus_biomarker", "n": len(d3), "events": int(d3.disability_event.sum())})
                models.append(row)
            d3["analysis_weight"] = d3["bloodweight"] / d3["bloodweight"].mean()
            for row in safe_weighted_glm(full_formula, d3):
                row.update({"definition": definition, "biomarker": label, "analysis": "weighted_disability_state_plus_biomarker", "n": len(d3), "events": int(d3.disability_event.sum())})
                weighted_models.append(row)
            coupled_state = next((r for r in state_rows if str(r.get("term", "")).endswith("T.coupled]")), {})
            coupled_full = next((r for r in full_rows if str(r.get("term", "")).endswith("T.coupled]")), {})
            atten.append({"definition": definition, "biomarker": label, "n": len(d3), "events": int(d3.disability_event.sum()), "coupled_OR_state_only": coupled_state.get("OR"), "coupled_OR_state_plus_biomarker": coupled_full.get("OR")})

    pd.DataFrame(counts).to_csv(OUT / "charls_state_counts_by_threshold.csv", index=False)
    pd.DataFrame(outcomes).to_csv(OUT / "charls_state_disability_outcomes_by_threshold.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "charls_state_biomarker_summary.csv", index=False)
    pd.DataFrame(assoc).to_csv(OUT / "charls_biomarker_state_associations.csv", index=False)
    pd.DataFrame(models).to_csv(OUT / "charls_disability_models_with_biomarkers.csv", index=False)
    pd.DataFrame(atten).to_csv(OUT / "charls_disability_state_attenuation.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "charls_biomarker_coverage.csv", index=False)
    pd.DataFrame(weighted_assoc).to_csv(OUT / "charls_weighted_biomarker_state_associations.csv", index=False)
    pd.DataFrame(weighted_models).to_csv(OUT / "charls_weighted_disability_models_with_biomarkers.csv", index=False)
    flow = [
        {"stage": "CHARLS trajectory gate", "n": len(q)},
        {"stage": "known W1-W4 outcome", "n": int(q.outcome_category.ne("unknown").sum())},
        {"stage": "linked 2011 blood record", "n": int(q.bloodweight.notna().sum())},
        {"stage": "linked CRP record", "n": int(q.crp_log.notna().sum())},
    ]
    pd.DataFrame(flow).to_csv(OUT / "charls_mechanism_flow.csv", index=False)
    log = {"run_utc": datetime.now(timezone.utc).isoformat(), "trajectory_n": len(q), "known_outcome_n": int(q.outcome_category.ne("unknown").sum()), "blood_link_n": int(q.bloodweight.notna().sum()), "crp_n": int(q.crp_log.notna().sum()), "blood_file": str(BLOOD), "id_bridge": "H_CHARLS_D_Data.ID -> ID_w1 -> Blood_20140429.ID"}
    (OUT / "charls_mechanism_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    memo = f"""# CHARLS inflammatory--metabolic pathway feasibility audit

Run (UTC): {log['run_utc']}

The audited CHARLS state frame contained {log['trajectory_n']:,} participants with {log['known_outcome_n']:,} known W1–W4 outcomes. The 2011 blood panel linked to {log['blood_link_n']:,} state-frame participants; baseline CRP was available for {log['crp_n']:,}. Linkage required the documented ID bridge from the 12-character harmonized full ID to `ID_w1`, then to the 11-character 2011 blood ID. Direct full-ID matching would produce zero overlap.

The blood measures precede the W1–W3 IC–deficit trajectory and W4 outcome window. CRP and triglycerides were log-transformed; TyG used the standard mg/dL formula because the CHARLS blood file reports glucose and triglycerides in mg/dL. Biomarker-by-state models are unweighted HC3 exploratory regressions. The blood response weight is retained in the descriptive output; design-based weighted confirmation is required before manuscript use.

State-specific tables and models are threshold-stratified. A stable metabolic pattern across the primary sign, median and 0.2/0.5-SD rules would support replication of a pathway-consistent correlate. These analyses are associations and do not estimate mediation or causality.
"""
    (OUT / "charls_mechanism_method_memo.md").write_text(memo, encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
