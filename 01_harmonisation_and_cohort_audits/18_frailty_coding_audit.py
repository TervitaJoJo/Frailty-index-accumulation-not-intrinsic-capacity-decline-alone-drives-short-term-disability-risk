"""Audit outcome-disjoint FI coding options at the selected anchor waves."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"


def valid(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)


FILES = {
    "ELSA": ROOT / "ELSA/Working_data/elsa.dta",
    "SHARE": next(ROOT.glob("SHARE/**/Working_data/share.dta")),
    "CHARLS": ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
    "HRS": ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
    "LASI": next(ROOT.glob("LASI/**/Working_data/lasi.dta")),
}

ANCHOR = {
    "ELSA": {"layout":"long", "wave":"wave", "value":6, "shlt":"shlt", "bmi":"mbmi", "disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"}, "shlt_direction":"poor_is_low"},
    "SHARE": {"layout":"long", "wave":"wave", "value":6, "shlt":"shlt", "bmi":"bmi", "disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"}, "shlt_direction":"poor_is_low"},
    "CHARLS": {"layout":"wide", "active":"inw3", "shlt":"r3shlt", "bmi":"r3mbmi", "disease":{"hypertension":"r3hibpe","diabetes":"r3diabe","heart_disease":"r3hearte","stroke":"r3stroke","cancer":"r3cancre","arthritis":"r3arthre"}, "shlt_direction":"poor_is_high"},
    "HRS": {"layout":"wide", "active":"inw10", "shlt":"r10shlt", "bmi":"r10bmi", "disease":{"hypertension":"r10hibpe","diabetes":"r10diabe","heart_disease":"r10hearte","stroke":"r10stroke","cancer":"r10cancre","arthritis":"r10arthre"}, "shlt_direction":"poor_is_high"},
    "LASI": {"layout":"wide", "active_year":"r1iwy", "shlt":"r1shlt", "bmi":"r1mbmi", "disease":{"hypertension":"r1hibpe","diabetes":"r1diabe","heart_disease":"r1hearte","stroke":"r1stroke","cancer":"r1cancre","arthritis":"r1arthre"}, "shlt_direction":"poor_is_low"},
}

prevalence = []
bmi_rows = []
label_rows = []
for dataset, spec in ANCHOR.items():
    path = FILES[dataset]
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    needed = [spec["shlt"], spec["bmi"], *spec["disease"].values()]
    if spec["layout"] == "long":
        needed.append(spec["wave"])
    else:
        needed.append(spec.get("active", spec.get("active_year")))
    use = [c for c in needed if c in meta.column_names]
    df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    if spec["layout"] == "long":
        active = df[spec["wave"]].eq(spec["value"])
    elif spec.get("active"):
        active = valid(df[spec["active"]]).eq(1)
    else:
        active = df[spec["active_year"]].notna()
    sub = df.loc[active].copy()
    for component, col in spec["disease"].items():
        x = valid(sub[col]) if col in sub else pd.Series(np.nan, index=sub.index)
        observed = x.notna()
        yes = x.eq(1)
        prevalence.append({"dataset":dataset,"component":component,"n_active":len(sub),"n_valid":int(observed.sum()),"pct_valid":round(float(observed.mean()*100),2),"deficit_yes_n":int(yes.sum()),"deficit_yes_pct_valid":round(float(yes.sum()/observed.sum()*100),2) if observed.sum() else None,"value_label_name":meta.variable_to_label.get(col,""),"value_labels":str(meta.value_labels.get(meta.variable_to_label.get(col,""),{}))})
    shlt = valid(sub[spec["shlt"]]) if spec["shlt"] in sub else pd.Series(np.nan,index=sub.index)
    if spec["shlt_direction"] == "poor_is_low":
        shlt_deficit = (5 - shlt) / 4
    else:
        shlt_deficit = (shlt - 1) / 4
    prevalence.append({"dataset":dataset,"component":"self_rated_health","n_active":len(sub),"n_valid":int(shlt.notna().sum()),"pct_valid":round(float(shlt.notna().mean()*100),2),"deficit_mean":round(float(shlt_deficit.mean()),4) if shlt.notna().any() else None,"deficit_ge_0_5_pct_valid":round(float((shlt_deficit.ge(.5).sum()/shlt.notna().sum())*100),2) if shlt.notna().sum() else None,"value_label_name":meta.variable_to_label.get(spec["shlt"],""),"value_labels":str(meta.value_labels.get(meta.variable_to_label.get(spec["shlt"],""),{}))})
    bmi = valid(sub[spec["bmi"]]) if spec["bmi"] in sub else pd.Series(np.nan,index=sub.index)
    y = bmi.dropna()
    if len(y):
        q10, q90 = y.quantile(.10), y.quantile(.90)
        q20, q80 = y.quantile(.20), y.quantile(.80)
        clinical = y.lt(18.5) | y.ge(30)
        outside20 = y.le(q20) | y.ge(q80)
        outside10 = y.le(q10) | y.ge(q90)
        bmi_rows.append({"dataset":dataset,"n_active":len(sub),"n_valid":len(y),"pct_valid":round(float(len(y)/len(sub)*100),2),"mean":round(float(y.mean()),3),"median":round(float(y.median()),3),"p01":round(float(y.quantile(.01)),3),"p99":round(float(y.quantile(.99)),3),"clinical_bmi_lt18_5_or_ge30_pct":round(float(clinical.mean()*100),2),"q10_q90_extreme_pct":round(float(outside10.mean()*100),2),"q20_q80_extreme_pct":round(float(outside20.mean()*100),2),"q10":round(float(q10),3),"q90":round(float(q90),3),"q20":round(float(q20),3),"q80":round(float(q80),3),"value_label_name":meta.variable_to_label.get(spec["bmi"],"")})

pd.DataFrame(prevalence).to_csv(TAB / "frailty_coding_prevalence.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(bmi_rows).to_csv(TAB / "frailty_bmi_sensitivity.csv", index=False, encoding="utf-8-sig")

memo = """# Frailty coding audit

At the selected anchor waves, binary disease candidates are treated as deficits when the harmonized numeric value equals 1; negative special codes remain missing. Self-rated health is mapped according to labels: ELSA/SHARE use poor-is-low and are reversed, whereas CHARLS/HRS/LASI use poor-is-high and are mapped directly to a 0–1 deficit scale.

BMI is intentionally not frozen by this audit. It compares a clinical threshold (`BMI <18.5 or >=30`) with cohort-specific outer-percentile alternatives. The clinical option has the clearest interpretation but may have different prevalence across countries because the source BMI is measured in some cohorts and self-reported in others. The percentile options improve distributional comparability but weaken clinical interpretation.

Review `frailty_coding_prevalence.csv` and `frailty_bmi_sensitivity.csv` before finalizing the FI scoring rule.
"""
(OUT / "frailty_coding_audit.md").write_text(memo, encoding="utf-8")
print("wrote frailty coding audit")
