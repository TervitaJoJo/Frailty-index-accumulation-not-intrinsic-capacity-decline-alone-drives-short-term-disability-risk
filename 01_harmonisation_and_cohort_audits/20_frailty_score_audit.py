"""Score the provisional primary outcome-disjoint FI at anchor waves.

This is an aggregate audit only. It writes no person-level rows or IDs. The
primary BMI rule is fixed at BMI <18.5 or >=30; a cohort-percentile variant is
reported as sensitivity.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"


def valid(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)


SPECS = {
    "ELSA": {"path":ROOT / "ELSA/Working_data/elsa.dta", "layout":"long", "wave":"wave", "value":6, "shlt":"shlt", "bmi":"mbmi", "shlt_direction":"poor_is_low", "disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"}},
    "SHARE": {"path":next(ROOT.glob("SHARE/**/Working_data/share.dta")), "layout":"long", "wave":"wave", "value":6, "shlt":"shlt", "bmi":"bmi", "shlt_direction":"poor_is_low", "disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"}},
    "CHARLS": {"path":ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta", "layout":"wide", "anchor_wave":3, "active":"inw3", "shlt":"r3shlt", "bmi":"r3mbmi", "shlt_direction":"poor_is_high", "disease":{"hypertension":"r3hibpe","diabetes":"r3diabe","heart_disease":"r3hearte","stroke":"r3stroke","cancer":"r3cancre","arthritis":"r3arthre"}},
    "HRS": {"path":ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta", "layout":"wide", "anchor_wave":10, "active":"inw10", "shlt":"r10shlt", "bmi":"r10bmi", "shlt_direction":"poor_is_high", "disease":{"hypertension":"r10hibpe","diabetes":"r10diabe","heart_disease":"r10hearte","stroke":"r10stroke","cancer":"r10cancre","arthritis":"r10arthre"}},
    "LASI": {"path":next(ROOT.glob("LASI/**/Working_data/lasi.dta")), "layout":"wide", "anchor_wave":1, "active_year":"r1iwy", "shlt":"r1shlt", "bmi":"r1mbmi", "shlt_direction":"poor_is_low", "disease":{"hypertension":"r1hibpe","diabetes":"r1diabe","heart_disease":"r1hearte","stroke":"r1stroke","cancer":"r1cancre","arthritis":"r1arthre"}},
}

summary = []
quantiles = []
component_rows = []
for dataset, spec in SPECS.items():
    _, meta = pyreadstat.read_dta(str(spec["path"]), metadataonly=True)
    needed = [spec["shlt"], spec["bmi"], *spec["disease"].values()]
    if spec["layout"] == "long":
        needed.append(spec["wave"])
    else:
        needed.append(spec.get("active", spec.get("active_year")))
    use = [c for c in needed if c in meta.column_names]
    df, _ = pyreadstat.read_dta(str(spec["path"]), usecols=use, apply_value_formats=False)
    if spec["layout"] == "long":
        active = df[spec["wave"]].eq(spec["value"])
    elif spec.get("active"):
        active = valid(df[spec["active"]]).eq(1)
    else:
        active = df[spec["active_year"]].notna()
    sub = df.loc[active]
    scores = pd.DataFrame(index=sub.index)
    for component, col in spec["disease"].items():
        x = valid(sub[col])
        scores[component] = x.eq(1).where(x.notna())
    shlt = valid(sub[spec["shlt"]])
    scores["self_rated_health"] = ((5 - shlt) / 4 if spec["shlt_direction"] == "poor_is_low" else (shlt - 1) / 4).where(shlt.notna())
    bmi = valid(sub[spec["bmi"]])
    scores["bmi"] = (bmi.lt(18.5) | bmi.ge(30)).where(bmi.notna())
    scores = scores.astype(float)
    observed = scores.notna().sum(axis=1)
    primary = scores.sum(axis=1, min_count=1).div(observed).where(observed.ge(6))
    complete = scores.sum(axis=1, min_count=8).div(8).where(observed.eq(8))
    # Sensitivity BMI percentile thresholds are computed within this anchor
    # cohort and are reported only as an aggregate robustness diagnostic.
    bv = bmi.dropna()
    if len(bv):
        q10, q90 = bv.quantile(.10), bv.quantile(.90)
        scores_pct = scores.copy()
        scores_pct["bmi"] = (bmi.le(q10) | bmi.ge(q90)).where(bmi.notna())
        scores_pct = scores_pct.astype(float)
        pct_score = scores_pct.sum(axis=1, min_count=1).div(scores_pct.notna().sum(axis=1)).where(scores_pct.notna().sum(axis=1).ge(6))
    else:
        pct_score = pd.Series(np.nan, index=sub.index)
    valid_primary = primary.dropna(); valid_complete = complete.dropna(); valid_pct = pct_score.dropna()
    row = {"dataset":dataset,"anchor_wave":spec.get("value", spec.get("anchor_wave",1)),"n_active":len(sub),"n_scored_min6":len(valid_primary),"pct_scored_min6":round(float(len(valid_primary)/len(sub)*100),2),"n_complete8":len(valid_complete),"pct_complete8":round(float(len(valid_complete)/len(sub)*100),2),"primary_fi_mean":round(float(valid_primary.mean()),4),"primary_fi_sd":round(float(valid_primary.std(ddof=1)),4),"primary_fi_median":round(float(valid_primary.median()),4),"primary_fi_p25":round(float(valid_primary.quantile(.25)),4),"primary_fi_p75":round(float(valid_primary.quantile(.75)),4),"primary_fi_p90":round(float(valid_primary.quantile(.90)),4),"percentile_bmi_fi_mean":round(float(valid_pct.mean()),4) if len(valid_pct) else None,"mean_abs_difference_percentile_minus_clinical":round(float((pct_score-primary).dropna().mean()),4) if (pct_score-primary).notna().any() else None}
    summary.append(row)
    for q in [0.1,0.25,0.5,0.75,0.9]:
        quantiles.append({"dataset":dataset,"score_version":"clinical_bmi","quantile":q,"value":round(float(valid_primary.quantile(q)),4)})
        if len(valid_pct): quantiles.append({"dataset":dataset,"score_version":"percentile_bmi_sensitivity","quantile":q,"value":round(float(valid_pct.quantile(q)),4)})
    for component in scores:
        x = scores[component]
        component_rows.append({"dataset":dataset,"component":component,"n_observed":int(x.notna().sum()),"pct_observed":round(float(x.notna().mean()*100),2),"mean_deficit":round(float(x.mean()),4) if x.notna().any() else None})

pd.DataFrame(summary).to_csv(TAB / "frailty_score_audit_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(quantiles).to_csv(TAB / "frailty_score_audit_quantiles.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(component_rows).to_csv(TAB / "frailty_score_audit_components.csv", index=False, encoding="utf-8-sig")

memo = """# Primary frailty score audit

The selected primary score uses six chronic disease indicators, label-aware self-rated health, and a clinical BMI deficit (`BMI <18.5 or BMI >=30`). A score is formed when at least 6 of 8 components are observed, with the denominator equal to the number observed. The 8/8 complete score and a within-anchor 10th/90th percentile BMI sensitivity are also reported.

This is a scoring feasibility audit. It does not apply survey weights, imputation, causal adjustment, or inferential cutoffs for frailty categories.
"""
(OUT / "frailty_score_audit.md").write_text(memo, encoding="utf-8")
print("wrote frailty score audit")
