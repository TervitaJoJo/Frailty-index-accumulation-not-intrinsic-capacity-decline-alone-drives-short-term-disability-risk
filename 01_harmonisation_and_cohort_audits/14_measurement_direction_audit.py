"""Audit direction, special codes, and scale distributions for primary items."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

FILES = {
    "ELSA": ROOT / "ELSA/Working_data/elsa.dta",
    "SHARE": next(ROOT.glob("SHARE/**/Working_data/share.dta")),
    "CHARLS": ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
    "HRS": ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
}
TARGETS = {
    "ELSA": {"layout":"long", "wave":"wave", "active":None, "vars":["tcog_z_z","walkra","gripsum","cesd","shlt","mbmi"]},
    "SHARE": {"layout":"long", "wave":"wave", "active":None, "vars":["orient","walkra","lgrip","rgrip","eurod","shlt","bmi"]},
    "CHARLS": {"layout":"wide", "waves":[1,2,3,4], "active":"inw{wave}", "vars":["r{wave}orient","r{wave}tr20","r{wave}walk100a","r{wave}lgrip","r{wave}rgrip","r{wave}cesd10","r{wave}shlt","r{wave}mbmi"]},
    "HRS": {"layout":"wide", "waves":[8,9,10,11,12,13], "active":"inw{wave}", "vars":["r{wave}cogtot","r{wave}walkra","r{wave}grpl","r{wave}grpr","r{wave}cesd","r{wave}shlt","r{wave}bmi"]},
}


def valid(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)


summary_rows = []
level_rows = []
for dataset, spec in TARGETS.items():
    path = FILES[dataset]
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    if spec["layout"] == "long":
        use = [spec["wave"], *[v for v in spec["vars"] if v in meta.column_names]]
        df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
        waves = sorted(df[spec["wave"]].dropna().unique().tolist())
    else:
        expanded = []
        for wave_value in spec["waves"]:
            for template in spec["vars"]:
                expanded.append(template.format(wave=wave_value))
        use = [v for v in [*expanded, *[spec["active"].format(wave=w) for w in spec["waves"]]] if v in meta.column_names]
        df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
        waves = spec["waves"]
    for wave in waves:
        if spec["layout"] == "long":
            active = df[spec["wave"]].eq(wave)
            vars_actual = spec["vars"]
        else:
            active_col = spec["active"].format(wave=wave)
            active = valid(df[active_col]).eq(1)
            vars_actual = [v.format(wave=wave) for v in spec["vars"]]
        for var in vars_actual:
            if var not in df:
                continue
            x = valid(df.loc[active, var])
            y = x.dropna()
            label_name = meta.variable_to_label.get(var, "")
            value_map = meta.value_labels.get(label_name, {}) if label_name else {}
            summary_rows.append({
                "dataset": dataset, "wave": wave, "variable": var,
                "variable_label": meta.column_names_to_labels.get(var, ""),
                "value_label_name": label_name,
                "n_active": int(active.sum()), "n_valid": int(y.size),
                "pct_valid_active": round(float(y.size / active.sum() * 100), 2) if active.sum() else None,
                "mean": round(float(y.mean()), 4) if len(y) else None,
                "sd": round(float(y.std(ddof=1)), 4) if len(y) > 1 else None,
                "p01": round(float(y.quantile(.01)), 4) if len(y) else None,
                "p25": round(float(y.quantile(.25)), 4) if len(y) else None,
                "median": round(float(y.median()), 4) if len(y) else None,
                "p75": round(float(y.quantile(.75)), 4) if len(y) else None,
                "p99": round(float(y.quantile(.99)), 4) if len(y) else None,
                "min": round(float(y.min()), 4) if len(y) else None,
                "max": round(float(y.max()), 4) if len(y) else None,
            })
            counts = y.value_counts(dropna=False).sort_index()
            for value, n in counts.items():
                level_rows.append({
                    "dataset": dataset, "wave": wave, "variable": var,
                    "value": value, "n": int(n),
                    "pct_valid": round(float(n / len(y) * 100), 2) if len(y) else None,
                    "value_label": value_map.get(value, ""),
                })

pd.DataFrame(summary_rows).to_csv(TAB / "measurement_direction_audit_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(level_rows).to_csv(TAB / "measurement_direction_audit_levels.csv", index=False, encoding="utf-8-sig")

memo = """# Measurement direction audit

This audit summarizes active-wave coverage, numeric distributions, and value-code frequencies for the provisional IC and frailty components in ELSA, SHARE, CHARLS and HRS. Negative special codes are treated as missing. The output is a direction-and-scale review ledger, not a finalized recoding algorithm.

Expected direction rules to verify against the tables:

- cognition and grip/vitality: higher values should represent better capacity;
- locomotion difficulty and depressive symptoms: higher values usually represent worse status and will require reverse orientation for an IC ability score;
- self-rated health: poorer categories should represent larger frailty deficits;
- chronic disease indicators: disease present should represent larger frailty deficits;
- BMI: a prespecified deficit function is required; do not treat raw BMI as a linear deficit without a sensitivity analysis.

The tables should be reviewed before constructing item-level scores or fitting CFA/IRT models.
"""
(OUT / "measurement_direction_audit.md").write_text(memo, encoding="utf-8")
print("wrote measurement direction audit")
