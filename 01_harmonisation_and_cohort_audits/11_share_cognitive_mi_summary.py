"""Create an aggregate, MI-aware summary of SHARE raw cognitive files.

The SHARE gv_imputations files contain five records per mergeid in many
waves.  This script keeps the imputation index for summaries and never writes
mergeid values or an individual-level remerged file.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT_ROOT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT_ROOT / "tables"
SHARE_ROOT = PROJECT_ROOT / "SHARE"
WAVES = [1, 2, 4, 5, 6, 7, 8]
COGNITIVE = ["orienti", "memory", "math10"]
FLAG_FOR = {"orienti": "orienti_f", "memory": "memory_f"}


def valid_numeric(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    return y.mask(y < -90)


rows: list[dict[str, object]] = []
mi_rows: list[dict[str, object]] = []
for wave in WAVES:
    matches = list(SHARE_ROOT.rglob(f"sharew{wave}_rel9-0-0_gv_imputations.dta"))
    if not matches:
        rows.append({"wave": wave, "file_found": False})
        continue
    path = matches[0]
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    available = [v for v in ["mergeid", "implicat", *COGNITIVE, "orienti_f", "memory_f"] if v in meta.column_names]
    df, _ = pyreadstat.read_dta(str(path), usecols=available, apply_value_formats=False)
    if "mergeid" not in df or "implicat" not in df:
        rows.append({"wave": wave, "file_found": True, "has_mergeid_implicat": False})
        continue
    row: dict[str, object] = {
        "wave": wave,
        "file_found": True,
        "n_rows": int(len(df)),
        "n_unique_mergeid": int(df["mergeid"].nunique()),
        "n_implicat_values": int(df["implicat"].nunique()),
        "implicat_values": ";".join(sorted(df["implicat"].dropna().astype(str).unique().tolist())),
    }
    for var in COGNITIVE:
        if var not in df:
            row[f"{var}_available"] = False
            continue
        x = valid_numeric(df[var])
        row[f"{var}_available"] = True
        row[f"{var}_n_valid_records"] = int(x.notna().sum())
        row[f"{var}_pct_valid_records"] = round(float(x.notna().mean() * 100), 2)
        row[f"{var}_mean_all_MI"] = round(float(x.mean()), 4) if x.notna().any() else None
        row[f"{var}_sd_all_MI"] = round(float(x.std(ddof=1)), 4) if x.notna().sum() > 1 else None
        flag_var = FLAG_FOR.get(var)
        if flag_var and flag_var in df:
            flags = pd.to_numeric(df[flag_var], errors="coerce")
            usable = flags.between(3, 13, inclusive="both")
            structural = flags.isin([-99, 1, 2])
            imputed_missing = flags.eq(14)
            row[f"{var}_n_flag_usable_records"] = int(usable.sum())
            row[f"{var}_pct_flag_usable_records"] = round(float(usable.mean() * 100), 2)
            row[f"{var}_pct_structural_or_not_designed_records"] = round(float(structural.mean() * 100), 2)
            row[f"{var}_pct_imputed_missing_flag14_records"] = round(float(imputed_missing.mean() * 100), 2)
            valid_ids = df.loc[usable, "mergeid"].nunique()
            row[f"{var}_n_ids_flag_usable"] = int(valid_ids)
            row[f"{var}_pct_ids_flag_usable"] = round(float(valid_ids / df["mergeid"].nunique() * 100), 2) if df["mergeid"].nunique() else None
        for imp, sub in df.groupby("implicat", dropna=False):
            xi = valid_numeric(sub[var])
            mi_rows.append({
                "wave": wave,
                "implicat": imp,
                "variable": var,
                "n_rows": int(len(sub)),
                "n_unique_mergeid": int(sub["mergeid"].nunique()),
                "n_valid": int(xi.notna().sum()),
                "pct_valid": round(float(xi.notna().mean() * 100), 2),
                "mean": round(float(xi.mean()), 4) if xi.notna().any() else None,
                "sd": round(float(xi.std(ddof=1)), 4) if xi.notna().sum() > 1 else None,
            })
    # Within-person imputation variation: calculated as an aggregate proportion.
    for var in COGNITIVE:
        if var not in df:
            continue
        valid = df.assign(_x=valid_numeric(df[var])).groupby("mergeid")
        valid_groups = 0
        variable_groups = 0
        for _, sub in valid:
            if FLAG_FOR.get(var) and FLAG_FOR[var] in sub:
                flags = pd.to_numeric(sub[FLAG_FOR[var]], errors="coerce")
                x = sub.loc[flags.between(3, 13, inclusive="both"), "_x"].dropna()
            else:
                x = sub["_x"].dropna()
            if len(x):
                valid_groups += 1
                variable_groups += int(x.nunique() > 1)
        row[f"{var}_pct_ids_with_MI_variation"] = round(float(variable_groups / valid_groups * 100), 2) if valid_groups else None
    rows.append(row)

pd.DataFrame(rows).to_csv(TAB / "share_raw_cognitive_mi_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(mi_rows).to_csv(TAB / "share_raw_cognitive_mi_by_implicat.csv", index=False, encoding="utf-8-sig")

memo = """# SHARE cognition: MI-aware remerge plan

The raw `gv_imputations` files are summarized with `mergeid` retained only in memory and `implicat` retained as an analysis dimension. The outputs contain aggregate counts and distributions; no IDs or person-level data are exported. SHARE flags are used to distinguish regular/imputed observations (flags 3–13) from structural or not-designed records (−99, 1, 2) and the special imputed-missing flag 14.

For the next modeling stage:

1. Re-merge `orienti`, `memory` and, where available, `math10` onto the selected working-file person-wave roster by `mergeid` and wave.
2. Preserve the five `implicat` records as multiply imputed measurements; never stack them as independent person-waves.
3. Use a proper MI-compatible estimator, or pre-specify `implicat=1` only for a clearly labeled exploratory sensitivity run.
4. Treat wave 7 as a coverage warning: its flag-usable cognitive coverage is much lower than waves 4–6 and 8, and a large proportion is structurally absent or not designed.

See `tables/share_raw_cognitive_mi_summary.csv`, `tables/share_raw_cognitive_mi_by_implicat.csv` and the earlier merge coverage audit.
"""
(OUT_ROOT / "share_raw_cognitive_mi_summary.md").write_text(memo, encoding="utf-8")
print("wrote SHARE MI summaries")
