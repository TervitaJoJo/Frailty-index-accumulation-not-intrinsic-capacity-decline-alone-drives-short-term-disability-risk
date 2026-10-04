"""Compare available cognition indicators in anchor waves."""
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"


def valid(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)


rows = []
correlations = []
specs = {
    "ELSA": (ROOT / "ELSA/Working_data/elsa.dta", ["wave"], {"wave": 6}, ["memory_z", "orient_z", "tcog_z_z", "imrc", "dlrc", "orient", "ser7"]),
    "CHARLS": (ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta", ["inw3"], {"inw3": 1}, ["r3orient", "r3tr20"]),
    "HRS": (ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta", ["inw10"], {"inw10": 1}, ["r10imrc", "r10dlrc", "r10ser7", "r10cog27", "r10cogtot"]),
}
for dataset, (path, active_cols, active_values, variables) in specs.items():
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in [*active_cols, *variables] if c in meta.column_names]
    df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    active = pd.Series(True, index=df.index)
    for c in active_cols:
        active &= valid(df[c]).eq(active_values[c])
    sub = pd.DataFrame({v: valid(df[v]) for v in variables if v in df}).loc[active]
    for v in sub:
        y = sub[v].dropna()
        rows.append({"dataset":dataset,"anchor_wave":6 if dataset != "HRS" else 10,"variable":v,"variable_label":meta.column_names_to_labels.get(v,""),"n_active":len(sub),"n_valid":int(y.size),"pct_valid":round(float(y.size/len(sub)*100),2) if len(sub) else None,"mean":round(float(y.mean()),4) if len(y) else None,"sd":round(float(y.std(ddof=1)),4) if len(y)>1 else None})
    corr = sub.corr(method="spearman", min_periods=2)
    for a in corr.index:
        for b in corr.columns:
            if a < b:
                correlations.append({"dataset":dataset,"anchor_wave":6 if dataset != "HRS" else 10,"variable_a":a,"variable_b":b,"n_pairwise":int(sub[[a,b]].dropna().shape[0]),"spearman":round(float(corr.loc[a,b]),4) if pd.notna(corr.loc[a,b]) else None})

# SHARE raw cognition, wave 6, flag-aware and restricted to implicat=1 for
# this exploratory indicator audit.
raw_path = next((ROOT / "SHARE").rglob("sharew6_rel9-0-0_gv_imputations.dta"))
_, meta = pyreadstat.read_dta(str(raw_path), metadataonly=True)
use = [c for c in ["implicat","orienti","memory","math10","orienti_f","memory_f"] if c in meta.column_names]
raw, _ = pyreadstat.read_dta(str(raw_path), usecols=use, apply_value_formats=False)
raw = raw.loc[raw["implicat"].eq(1)].copy()
for v in ["orienti", "memory", "math10"]:
    if v not in raw:
        continue
    x = valid(raw[v])
    if v in ["orienti", "memory"] and f"{v}_f" in raw:
        flag = pd.to_numeric(raw[f"{v}_f"], errors="coerce")
        x = x.where(flag.between(3,13))
    y = x.dropna()
    rows.append({"dataset":"SHARE","anchor_wave":6,"variable":v,"variable_label":meta.column_names_to_labels.get(v,""),"n_active":len(raw),"n_valid":int(y.size),"pct_valid":round(float(y.size/len(raw)*100),2),"mean":round(float(y.mean()),4) if len(y) else None,"sd":round(float(y.std(ddof=1)),4) if len(y)>1 else None})

pd.DataFrame(rows).to_csv(TAB / "anchor_cognition_indicator_coverage.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(correlations).to_csv(TAB / "anchor_cognition_indicator_spearman.csv", index=False, encoding="utf-8-sig")

memo = """# Anchor-wave cognition indicator audit

The anchor-wave cognition audit shows whether a domain can be represented by multiple comparable indicators rather than a single precomputed total. HRS has immediate recall, delayed recall, serial-7 and a 27-point cognition summary with materially higher coverage than `cogtot`; ELSA has memory/orientation summaries and raw recall indicators; CHARLS has orientation and total recall. SHARE wave 6 has flag-usable orientation and memory in the raw imputation file.

This supports a two-level strategy: build a cohort-specific cognition domain score from the available indicators, then compare the four-domain IC latent structure across cohorts. Indicator-level CFA/IRT remains a sensitivity analysis because item sets and scales differ.
"""
(OUT / "anchor_cognition_indicator_audit.md").write_text(memo, encoding="utf-8")
print("wrote cognition indicator audit")
