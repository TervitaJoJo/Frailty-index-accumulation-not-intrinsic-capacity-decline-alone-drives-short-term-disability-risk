"""Robustness audit for the CHARLS and ELSA biomarker extension.

The audit addresses biomarker-subset selection, multiple testing and overlap
with the baseline FI. It writes aggregate estimates only.
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
from statsmodels.stats.multitest import multipletests


PKG = Path(r"PATH_TO_IC_FRAILTY_V2")
ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = PKG / "analysis_audit" / "mechanism_robustness_20261003"
OUT.mkdir(parents=True, exist_ok=True)


def main_module():
    p = PKG / "scripts" / "trajectory_states_analysis_20260930.py"
    spec = importlib.util.spec_from_file_location("trajectory_main", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def clean(x):
    z = pd.to_numeric(x, errors="coerce")
    return z.mask(z <= 0)


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if np.isfinite(sd) and sd > 0 else x * np.nan


def coupled_row(rows):
    for r in rows:
        if str(r.get("term", "")).endswith("T.coupled]"):
            return r
    return {}


def weighted_ols(formula, data):
    fit = smf.wls(formula, data=data, weights=data["analysis_weight"]).fit(cov_type="HC3")
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


def selection_weights(q, available, base_weight):
    # Use baseline variables and exposure state; outcome is never used.
    d = q[["age_baseline", "male", "education", "baseline_fi", "state"]].copy()
    d["available"] = available.astype(int)
    d = d.dropna().copy()
    fit = smf.glm(
        "available ~ age_baseline + male + education + baseline_fi + C(state)",
        data=d,
        family=sm.families.Binomial(),
    ).fit()
    p = pd.Series(fit.predict(d), index=d.index).clip(0.02, 0.98)
    pi = d.available.mean()
    ipcw = pi / p
    lo, hi = ipcw.quantile([0.01, 0.99])
    ipcw_trim = ipcw.clip(lo, hi)
    d["ipcw"] = ipcw
    d["ipcw_trim"] = ipcw_trim
    d["base_weight"] = pd.to_numeric(base_weight.loc[d.index], errors="coerce").fillna(1.0)
    d["analysis_weight"] = d["base_weight"] * d["ipcw_trim"]
    return d, {
        "selection_model_n": len(d),
        "available_n": int(d.available.sum()),
        "availability": float(d.available.mean()),
        "ipcw_mean": float(ipcw.mean()),
        "ipcw_min": float(ipcw.min()),
        "ipcw_max": float(ipcw.max()),
        "ipcw_trim_low": float(lo),
        "ipcw_trim_high": float(hi),
        "analysis_weight_mean": float(d.analysis_weight.mean()),
        "effective_sample_size": float(d.analysis_weight.sum() ** 2 / (d.analysis_weight ** 2).sum()),
    }


def add_states(q, definition):
    if definition == "primary_sign":
        ic_cut, fi_cut = 0.0, 0.0
    elif definition == "cohort_median":
        ic_cut, fi_cut = q.delta_ic.median(), q.delta_fi.median()
    elif definition == "meaningful_0.2SD":
        ic_cut, fi_cut = -0.2 * q.delta_ic.std(ddof=1), 0.2 * q.delta_fi.std(ddof=1)
    else:
        ic_cut, fi_cut = -0.5 * q.delta_ic.std(ddof=1), 0.5 * q.delta_fi.std(ddof=1)
    out = q.copy()
    ic_decline = out.delta_ic < ic_cut
    fi_acc = out.delta_fi > fi_cut
    out["state"] = np.select(
        [~ic_decline & ~fi_acc, ic_decline & ~fi_acc, ~ic_decline & fi_acc, ic_decline & fi_acc],
        ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"],
        default="unknown",
    )
    out["state"] = pd.Categorical(out.state, categories=["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"])
    # Exclude diabetes and BMI from the baseline FI for content-overlap sensitivity.
    fi_components = ["hibpe", "hearte", "stroke", "cancre", "arthre", "srh"]
    comp = out[fi_components]
    out["baseline_fi_excl_diabetes_bmi"] = comp.mean(axis=1).where(comp.notna().sum(axis=1) >= 5)
    return out


def prepare_cohort(cohort):
    mod = main_module()
    raw, _ = mod.load_cohort(cohort)
    z, _ = mod.prepare_analysis(raw)
    z["id"] = z.id.astype(str)
    if cohort == "CHARLS":
        f = ROOT / "CHARLS" / "Harmonized_CHARLS" / "H_CHARLS_D_Data.dta"
        map_df = pyreadstat.read_dta(str(f), usecols=["ID", "ID_w1"])[0]
        map_df["id"] = map_df.ID.astype(str)
        map_df["ID_w1"] = map_df.ID_w1.astype(str)
        z = z.merge(map_df[["id", "ID_w1"]], on="id", how="left", validate="one_to_one")
        blood = pyreadstat.read_dta(
            str(ROOT / "CHARLS" / "2011" / "Blood_20140429.dta"),
            usecols=["ID", "bloodweight", "newcrp", "newhba1c"],
        )[0]
        blood["ID_w1"] = blood.ID.astype(str)
        for c in ["bloodweight", "newcrp", "newhba1c"]:
            blood[c] = clean(blood[c])
        blood = blood.drop_duplicates("ID_w1")
        q = z.merge(blood.drop(columns=["ID"]), on="ID_w1", how="left", validate="one_to_one")
        q["crp_log"] = np.log(q.newcrp)
        q["hba1c"] = q.newhba1c
        base_weight = q.bloodweight
        measures = {"crp_log": "log_CRP", "hba1c": "HbA1c"}
    else:
        blood = pyreadstat.read_dta(
            str(ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_nurse_data_v2.dta"),
            usecols=["idauniq", "w6bldwt", "hscrp", "hba1c"],
        )[0]
        blood["id"] = blood.idauniq.astype(str)
        for c in ["w6bldwt", "hscrp", "hba1c"]:
            blood[c] = clean(blood[c])
        blood = blood.drop_duplicates("id")
        q = z.merge(blood.drop(columns=["idauniq"]), on="id", how="left", validate="one_to_one")
        q["crp_log"] = np.log(q.hscrp)
        base_weight = q.w6bldwt
        q["hba1c"] = q.hba1c
        measures = {"crp_log": "log_CRP", "hba1c": "HbA1c"}
    return q, base_weight, measures


def main():
    selection_rows, contrast_rows, overlap_rows, fdr_rows = [], [], [], []
    for cohort in ["CHARLS", "ELSA"]:
        q, base_weight, measures = prepare_cohort(cohort)
        for definition in ["primary_sign", "cohort_median", "meaningful_0.2SD", "meaningful_0.5SD"]:
            qq = add_states(q, definition)
            for b, label in measures.items():
                available = qq[b].notna() & qq["age_baseline"].notna() & qq["male"].notna() & qq["education"].notna() & qq["baseline_fi"].notna()
                try:
                    sel, diag = selection_weights(qq, available, base_weight)
                    diag.update({"cohort": cohort, "definition": definition, "biomarker": label})
                    selection_rows.append(diag)
                    use = qq.loc[sel.index].copy()
                    use = use.loc[available.loc[sel.index]].copy()
                    use["biomarker_z"] = zscore(use[b])
                    use["analysis_weight"] = sel.loc[use.index, "analysis_weight"]
                    # Standard baseline-FI model.
                    rows = weighted_ols("biomarker_z ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi", use)
                    r = coupled_row(rows)
                    contrast_rows.append({"cohort": cohort, "definition": definition, "biomarker": label, "model": "selection_plus_blood_weight", **r, "n": len(use)})
                    # Content-overlap sensitivity.
                    use2 = use.dropna(subset=["baseline_fi_excl_diabetes_bmi"]).copy()
                    rows2 = weighted_ols("biomarker_z ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + baseline_fi_excl_diabetes_bmi", use2)
                    r2 = coupled_row(rows2)
                    overlap_rows.append({"cohort": cohort, "definition": definition, "biomarker": label, "model": "selection_plus_blood_weight_FI_excl_diabetes_BMI", **r2, "n": len(use2)})
                except Exception as exc:
                    selection_rows.append({"cohort": cohort, "definition": definition, "biomarker": label, "error": str(exc)})

    contrasts = pd.DataFrame(contrast_rows)
    if len(contrasts):
        for cohort, idx in contrasts.groupby("cohort").groups.items():
            contrasts.loc[idx, "q_bh_within_cohort"] = multipletests(contrasts.loc[idx, "p"].astype(float), method="fdr_bh")[1]
    contrasts.to_csv(OUT / "biomarker_contrasts_selection_weighted.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "biomarker_contrasts_selection_weighted_FI_overlap.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "biomarker_selection_weight_diagnostics.csv", index=False)
    log = {"run_utc": datetime.now(timezone.utc).isoformat(), "cohorts": ["CHARLS", "ELSA"], "biomarkers": ["log_CRP", "HbA1c"], "estimand": "biomarker contrast by directional state in the trajectory frame"}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
