"""Coverage audit for a fixed outcome-disjoint frailty core.

The audit evaluates whether the same eight candidate components can be
observed in the principal cohorts. It reports aggregate person-wave counts;
it does not assign deficit values, impute, weight, or export IDs.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT_ROOT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT_ROOT / "tables"

FILES = {
    "ELSA": PROJECT_ROOT / "ELSA/Working_data/elsa.dta",
    "SHARE": next(PROJECT_ROOT.glob("SHARE/**/Working_data/share.dta")),
    "CHARLS": PROJECT_ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
    "HRS": PROJECT_ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
    "LASI": next(PROJECT_ROOT.glob("LASI/**/Working_data/lasi.dta")),
}

# These are the eight primary disjoint candidates. The script evaluates
# observation coverage only; the final deficit cut-points remain provisional.
CORE = ["hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis", "self_rated_health", "bmi"]
SPECS = {
    "ELSA": {"layout": "long", "id": "idauniqc", "wave": "wave", "active": None, "vars": {"hypertension":"hibpe", "diabetes":"diabe", "heart_disease":"hearte", "stroke":"stroke", "cancer":"cancre", "arthritis":"arthre", "self_rated_health":"shlt", "bmi":"mbmi"}},
    "SHARE": {"layout": "long", "id": "mergeid", "wave": "wave", "active": None, "vars": {"hypertension":"hibpe", "diabetes":"diabe", "heart_disease":"hearte", "stroke":"stroke", "cancer":"cancre", "arthritis":"arthre", "self_rated_health":"shlt", "bmi":"bmi"}},
    "CHARLS": {"layout": "wide", "id": "ID", "waves": [1,2,3,4], "active":"inw{wave}", "vars": {"hypertension":"r{wave}hibpe", "diabetes":"r{wave}diabe", "heart_disease":"r{wave}hearte", "stroke":"r{wave}stroke", "cancer":"r{wave}cancre", "arthritis":"r{wave}arthre", "self_rated_health":"r{wave}shlt", "bmi":"r{wave}mbmi"}},
    "HRS": {"layout": "wide", "id": "hhidpn", "waves": [8,9,10,11,12,13], "active":"inw{wave}", "vars": {"hypertension":"r{wave}hibpe", "diabetes":"r{wave}diabe", "heart_disease":"r{wave}hearte", "stroke":"r{wave}stroke", "cancer":"r{wave}cancre", "arthritis":"r{wave}arthre", "self_rated_health":"r{wave}shlt", "bmi":"r{wave}bmi"}},
    "LASI": {"layout": "wide", "id": "hhid", "waves": [1], "active_year":"r{wave}iwy", "vars": {"hypertension":"r{wave}hibpe", "diabetes":"r{wave}diabe", "heart_disease":"r{wave}hearte", "stroke":"r{wave}stroke", "cancer":"r{wave}cancre", "arthritis":"r{wave}arthre", "self_rated_health":"r{wave}shlt", "bmi":"r{wave}mbmi"}},
}


def valid(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)


rows: list[dict[str, object]] = []
for dataset, path in FILES.items():
    spec = SPECS[dataset]
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    needed = {spec["id"]}
    if spec["layout"] == "long":
        needed.add(spec["wave"])
        needed.update(spec["vars"].values())
    else:
        for w in spec["waves"]:
            needed.add(spec["active"].format(wave=w) if spec.get("active") else spec["active_year"].format(wave=w))
            needed.update(v.format(wave=w) for v in spec["vars"].values())
    use = [c for c in needed if c in meta.column_names]
    df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    waves = sorted(df[spec["wave"]].dropna().unique().tolist()) if spec["layout"] == "long" and spec["wave"] in df else ["all_current_waves"]
    if spec["layout"] == "wide":
        waves = spec["waves"]
    for w in waves:
        if spec["layout"] == "long":
            # Long files contain one row per person-wave; restrict each
            # summary to that wave rather than repeating the full file.
            active = df[spec["wave"]].eq(w)
            actual = {k: v for k, v in spec["vars"].items() if v in df}
        else:
            active_col = spec["active"].format(wave=w) if spec.get("active") else spec["active_year"].format(wave=w)
            active = valid(df[active_col]).eq(1) if spec.get("active") else df[active_col].notna()
            actual = {k: v.format(wave=w) for k, v in spec["vars"].items() if v.format(wave=w) in df}
        observed = pd.DataFrame(index=df.index)
        for component in CORE:
            col = actual.get(component)
            observed[component] = valid(df[col]).notna() if col else False
        sub = observed.loc[active]
        n_obs = sub.sum(axis=1)
        row: dict[str, object] = {"dataset": dataset, "wave": w, "n_active": int(active.sum()), "n_ids": int(df.loc[active, spec["id"]].nunique()) if spec["id"] in df else None}
        row.update({f"{k}_pct_observed": round(float(sub[k].mean() * 100), 2) if len(sub) else None for k in CORE})
        row["n_complete_8"] = int((n_obs == 8).sum())
        row["pct_complete_8"] = round(float((n_obs == 8).mean() * 100), 2) if len(sub) else None
        row["n_at_least_6_of_8"] = int((n_obs >= 6).sum())
        row["pct_at_least_6_of_8"] = round(float((n_obs >= 6).mean() * 100), 2) if len(sub) else None
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(TAB / "frailty_core_coverage_by_wave.csv", index=False, encoding="utf-8-sig")
overall = (out.groupby("dataset", as_index=False)
           .agg(n_active=("n_active", "sum"), n_complete_8=("n_complete_8", "sum"), n_at_least_6_of_8=("n_at_least_6_of_8", "sum")))
overall["pct_complete_8"] = (overall["n_complete_8"] / overall["n_active"] * 100).round(2)
overall["pct_at_least_6_of_8"] = (overall["n_at_least_6_of_8"] / overall["n_active"] * 100).round(2)
overall.to_csv(TAB / "frailty_core_coverage_overall.csv", index=False, encoding="utf-8-sig")

memo = """# Outcome-disjoint frailty core coverage

This audit uses the provisional eight-component core (hypertension, diabetes, heart disease, stroke, cancer, arthritis, self-rated health and BMI) and reports observation coverage only. A value is counted as observed when it is numeric and not a negative special code. The result is not a scored frailty index.

The eight-item complete case rate is expected to be constrained by measured BMI in ELSA and CHARLS. The `at_least_6_of_8` summary is a feasibility screen for a denominator rule that may be used later, after cut-points and missing-data assumptions are frozen. It should not be interpreted as a final sample definition.
"""
(OUT_ROOT / "frailty_core_coverage.md").write_text(memo, encoding="utf-8")
print("wrote", TAB / "frailty_core_coverage_by_wave.csv")
