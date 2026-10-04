"""ELSA inflammatory--metabolic correlate audit.

This module reuses the audited ELSA 6-8-9 directional-state frame and links
the wave-6 nurse/blood panel. It is a secondary association analysis. It does
not change the primary manuscript estimand and does not estimate mediation.
"""
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf


PKG = Path(r"PATH_TO_IC_FRAILTY_V2")
ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = PKG / "analysis_audit" / "elsa_inflammatory_metabolic_20261003"
OUT.mkdir(parents=True, exist_ok=True)
BLOOD = ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_nurse_data_v2.dta"


def load_main_module():
    path = PKG / "scripts" / "trajectory_states_analysis_20260930.py"
    spec = importlib.util.spec_from_file_location("elsa_main_trajectory", path)
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


def safe_ols(formula, data, weighted=False):
    try:
        if weighted:
            fit = smf.wls(formula, data=data, weights=data["analysis_weight"]).fit(cov_type="HC3")
        else:
            fit = smf.ols(formula, data=data).fit(cov_type="HC3")
        rows = []
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            se = fit.bse[term]
            rows.append({
                "term": term,
                "beta": float(coef),
                "CI_low": float(coef - 1.959964 * se),
                "CI_high": float(coef + 1.959964 * se),
                "p": float(fit.pvalues[term]),
            })
        return rows
    except Exception as exc:
        return [{"term": "MODEL_FAILED", "error": str(exc)}]


def safe_glm(formula, data, weighted=False):
    try:
        if weighted:
            fit = smf.glm(
                formula,
                data=data,
                family=sm.families.Binomial(),
                freq_weights=data["analysis_weight"],
            ).fit(cov_type="HC3")
        else:
            fit = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(cov_type="HC3")
        rows = []
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            se = fit.bse[term]
            rows.append({
                "term": term,
                "OR": float(np.exp(coef)),
                "CI_low": float(np.exp(coef - 1.959964 * se)),
                "CI_high": float(np.exp(coef + 1.959964 * se)),
                "p": float(fit.pvalues[term]),
            })
        return rows
    except Exception as exc:
        return [{"term": "MODEL_FAILED", "error": str(exc)}]


def main():
    main = load_main_module()
    raw, _ = main.load_cohort("ELSA")
    z, _ = main.prepare_analysis(raw)
    z["id"] = z["id"].astype(str)

    blood_cols = ["idauniq", "chol", "trig", "hscrp", "hba1c", "w6bldwt"]
    blood = pyreadstat.read_dta(str(BLOOD), usecols=blood_cols)[0]
    blood["id"] = blood["idauniq"].astype(str)
    for c in ["chol", "trig", "hscrp", "hba1c", "w6bldwt"]:
        blood[c] = clean(blood[c])
    blood = blood.drop_duplicates("id", keep="first")
    q = z.merge(blood.drop(columns=["idauniq"]), on="id", how="left", validate="one_to_one")
    q["disability_event"] = q.outcome_category.eq("disability").astype(int)
    q["crp_log"] = np.log(q["hscrp"])
    q["trig_log"] = np.log(q["trig"])

    biomarkers = {
        "crp_log": "baseline_log_CRP",
        "hba1c": "baseline_HbA1c",
        "trig_log": "baseline_log_triglycerides",
        "chol": "baseline_total_cholesterol",
        "trig": "baseline_triglycerides",
    }
    thresholds = [
        ("primary_sign", 0.0, 0.0),
        ("cohort_median", q.delta_ic.median(), q.delta_fi.median()),
        ("meaningful_0.2SD", -0.2 * q.delta_ic.std(ddof=1), 0.2 * q.delta_fi.std(ddof=1)),
        ("meaningful_0.5SD", -0.5 * q.delta_ic.std(ddof=1), 0.5 * q.delta_fi.std(ddof=1)),
    ]
    counts, outcomes, coverage, assoc, weighted_assoc, models, weighted_models, attenuation = [], [], [], [], [], [], [], []

    for definition, ic_cut, fi_cut in thresholds:
        qq = q.copy()
        qq["ic_decline"] = qq.delta_ic < ic_cut
        qq["fi_accumulation"] = qq.delta_fi > fi_cut
        qq["state"] = np.select(
            [~qq.ic_decline & ~qq.fi_accumulation,
             qq.ic_decline & ~qq.fi_accumulation,
             ~qq.ic_decline & qq.fi_accumulation,
             qq.ic_decline & qq.fi_accumulation],
            ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"],
            default="unknown",
        )
        qq["state"] = pd.Categorical(
            qq["state"],
            categories=["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"],
        )
        known = qq.outcome_category.ne("unknown")
        for state, h in qq.groupby("state", observed=False):
            if len(h) == 0:
                continue
            hk = h.loc[known.loc[h.index]]
            outcomes.append({
                "definition": definition,
                "state": state,
                "n": len(h),
                "known_outcome_n": len(hk),
                "disability_events": int(hk.disability_event.sum()),
                "disability_risk": float(hk.disability_event.mean()) if len(hk) else np.nan,
            })
            counts.append({"definition": definition, "state": state, "n": len(h), "percent": len(h) / len(qq)})

        for b, label in biomarkers.items():
            dat = qq[["state", "age_baseline", "male", "baseline_fi", b, "w6bldwt"]].dropna().copy()
            coverage.append({
                "definition": definition,
                "biomarker": label,
                "available": len(dat),
                "trajectory_n": len(qq),
                "coverage": len(dat) / len(qq),
            })
            if len(dat) < 200 or dat.state.nunique() < 2:
                continue
            dat["biomarker_z"] = standardize(dat[b])
            for row in safe_ols("biomarker_z ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi", dat):
                row.update({"definition": definition, "biomarker": label, "analysis": "biomarker_by_state", "n": len(dat)})
                assoc.append(row)
            dat["analysis_weight"] = dat["w6bldwt"] / dat["w6bldwt"].mean()
            for row in safe_ols("biomarker_z ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi", dat, weighted=True):
                row.update({"definition": definition, "biomarker": label, "analysis": "weighted_biomarker_by_state", "n": len(dat)})
                weighted_assoc.append(row)

            d3 = qq[["disability_event", "state", "age_baseline", "male", "baseline_fi", b, "w6bldwt"]].dropna().copy()
            d3["biomarker_z"] = standardize(d3[b])
            state_formula = "disability_event ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi"
            full_formula = state_formula + " + biomarker_z"
            state_rows = safe_glm(state_formula, d3)
            full_rows = safe_glm(full_formula, d3)
            for row in full_rows:
                row.update({"definition": definition, "biomarker": label, "analysis": "disability_state_plus_biomarker", "n": len(d3), "events": int(d3.disability_event.sum())})
                models.append(row)
            d3["analysis_weight"] = d3["w6bldwt"] / d3["w6bldwt"].mean()
            for row in safe_glm(full_formula, d3, weighted=True):
                row.update({"definition": definition, "biomarker": label, "analysis": "weighted_disability_state_plus_biomarker", "n": len(d3), "events": int(d3.disability_event.sum())})
                weighted_models.append(row)
            coupled_state = next((r for r in state_rows if str(r.get("term", "")).endswith("T.coupled]")), {})
            coupled_full = next((r for r in full_rows if str(r.get("term", "")).endswith("T.coupled]")), {})
            attenuation.append({
                "definition": definition,
                "biomarker": label,
                "n": len(d3),
                "events": int(d3.disability_event.sum()),
                "coupled_OR_state_only": coupled_state.get("OR"),
                "coupled_OR_state_plus_biomarker": coupled_full.get("OR"),
            })

    for name, rows in {
        "elsa_state_counts_by_threshold.csv": counts,
        "elsa_state_disability_outcomes_by_threshold.csv": outcomes,
        "elsa_biomarker_coverage.csv": coverage,
        "elsa_biomarker_state_associations.csv": assoc,
        "elsa_weighted_biomarker_state_associations.csv": weighted_assoc,
        "elsa_disability_models_with_biomarkers.csv": models,
        "elsa_weighted_disability_models_with_biomarkers.csv": weighted_models,
        "elsa_disability_state_attenuation.csv": attenuation,
    }.items():
        pd.DataFrame(rows).to_csv(OUT / name, index=False)

    flow = [
        {"stage": "ELSA trajectory gate", "n": len(q)},
        {"stage": "known outcome", "n": int(q.outcome_category.ne("unknown").sum())},
        {"stage": "linked wave-6 blood record", "n": int(q.w6bldwt.notna().sum())},
        {"stage": "linked wave-6 CRP", "n": int(q.crp_log.notna().sum())},
    ]
    pd.DataFrame(flow).to_csv(OUT / "elsa_mechanism_flow.csv", index=False)
    log = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_n": len(q),
        "known_outcome_n": int(q.outcome_category.ne("unknown").sum()),
        "blood_link_n": int(q.w6bldwt.notna().sum()),
        "crp_n": int(q.crp_log.notna().sum()),
        "blood_file": str(BLOOD),
        "id_bridge": "ELSA Working_data.idauniqc -> wave-6 nurse idauniq",
        "available_measures": list(biomarkers.values()),
        "tyg_available": False,
    }
    (OUT / "elsa_mechanism_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    memo = f"""# ELSA inflammatory--metabolic pathway feasibility audit

Run (UTC): {log['run_utc']}

The audited ELSA 6-8-9 trajectory frame contained {log['trajectory_n']:,} participants with {log['known_outcome_n']:,} known outcomes. The wave-6 nurse/blood panel linked to {log['blood_link_n']:,} participants; CRP was available for {log['crp_n']:,}. Available measures were baseline CRP, HbA1c, triglycerides and cholesterol. The wave-6 file did not contain a glucose measure suitable for a TyG calculation, so TyG was not constructed in ELSA.

Biomarker contrasts use age-, sex- and baseline-FI-adjusted models; the sensitivity analysis uses the wave-6 blood sampling weight with robust standard errors. State models follow the primary-sign, cohort-median and 0.2/0.5-SD directional rules used in the main analysis. Results are associative and do not estimate mediation.
"""
    (OUT / "elsa_mechanism_method_memo.md").write_text(memo, encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
