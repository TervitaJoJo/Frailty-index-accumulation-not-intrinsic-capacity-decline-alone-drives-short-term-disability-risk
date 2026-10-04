"""Audit SHARE raw cognitive files that are absent from the current merged working file."""
from pathlib import Path
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
share_root = PROJECT_ROOT / "SHARE"
rows = []
for wave in [1,2,4,5,6,7,8]:
    matches = list(share_root.rglob(f"sharew{wave}_rel9-0-0_gv_imputations.dta"))
    if not matches:
        rows.append({"wave":wave,"file_found":False})
        continue
    path = matches[0]
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    cols = set(meta.column_names)
    wanted = [c for c in ["mergeid","orienti","memory","math10","orienti_f","memory_f"] if c in cols]
    df, _ = pyreadstat.read_dta(str(path), usecols=wanted, apply_value_formats=False)
    row = {"wave":wave,"file_found":True,"source_file":str(path.relative_to(PROJECT_ROOT)),"n_rows":len(df),"n_unique_mergeid":int(df["mergeid"].nunique()) if "mergeid" in df else 0}
    for var in ["orienti","memory","math10","orienti_f","memory_f"]:
        row[f"{var}_available"] = var in df
        if var in df:
            x = pd.to_numeric(df[var], errors="coerce").mask(lambda z: z < -90)
            row[f"{var}_n_observed"] = int(x.notna().sum())
            row[f"{var}_pct_observed"] = round(float(x.notna().mean()*100),2)
        else:
            row[f"{var}_n_observed"] = 0
            row[f"{var}_pct_observed"] = None
    # SHARE flags distinguish structural absence (`-99`, flag 1/2) from
    # regular observations (3) and imputed values. Numeric values alone can
    # therefore overstate coverage, especially in wave 7.
    for value_var, flag_var in [("orienti", "orienti_f"), ("memory", "memory_f"), ("math10", None)]:
        if flag_var and flag_var in df:
            flags = pd.to_numeric(df[flag_var], errors="coerce")
            usable = flags.between(3, 13, inclusive="both")
            structural = flags.isin([-99, 1, 2])
            imputed_missing = flags.eq(14)
            row[f"{value_var}_n_flag_usable_records"] = int(usable.sum())
            row[f"{value_var}_pct_flag_usable_records"] = round(float(usable.mean() * 100), 2)
            row[f"{value_var}_pct_structural_or_not_designed_records"] = round(float(structural.mean() * 100), 2)
            row[f"{value_var}_pct_imputed_missing_flag14_records"] = round(float(imputed_missing.mean() * 100), 2)
            valid_ids = df.loc[usable, "mergeid"].nunique()
            row[f"{value_var}_n_ids_flag_usable"] = int(valid_ids)
            row[f"{value_var}_pct_ids_flag_usable"] = round(float(valid_ids / df["mergeid"].nunique() * 100), 2) if df["mergeid"].nunique() else None
        elif value_var in df:
            x = pd.to_numeric(df[value_var], errors="coerce").mask(lambda z: z < -90)
            row[f"{value_var}_n_flag_usable_records"] = int(x.notna().sum())
            row[f"{value_var}_pct_flag_usable_records"] = round(float(x.notna().mean() * 100), 2)
    if "mergeid" in df:
        mi_rows = []
        for _, s in df.groupby("mergeid"):
            if "memory" in s:
                x = pd.to_numeric(s["memory"], errors="coerce").mask(lambda z: z < -90).dropna()
                mi_rows.append((x.nunique() > 1, len(x) > 0))
        row["memory_pct_ids_with_valid_value"] = round(float(sum(v for _, v in mi_rows) / len(mi_rows) * 100),2) if mi_rows else None
        row["memory_pct_ids_with_MI_variation"] = round(float(sum(v for v, ok in mi_rows if ok) / sum(ok for _, ok in mi_rows) * 100),2) if any(ok for _, ok in mi_rows) else None
    rows.append(row)

aud = pd.DataFrame(rows)
aud.to_csv(TAB / "share_raw_cognitive_audit.csv", index=False, encoding="utf-8-sig")
# Compare merge-id coverage with the current working file without exporting IDs.
merged_path = next(share_root.rglob("Working_data/share.dta"))
merged, _ = pyreadstat.read_dta(str(merged_path), usecols=["mergeid","wave"], apply_value_formats=False)
comparisons=[]
for wave, sub in merged.groupby("wave"):
    matches = list(share_root.rglob(f"sharew{int(wave)}_rel9-0-0_gv_imputations.dta"))
    if not matches: continue
    raw, _ = pyreadstat.read_dta(str(matches[0]), usecols=["mergeid"], apply_value_formats=False)
    a=set(sub.mergeid.dropna().astype(str)); b=set(raw.mergeid.dropna().astype(str))
    comparisons.append({"wave":int(wave),"working_n_unique":len(a),"raw_cognitive_n_unique":len(b),"n_id_intersection":len(a&b),"pct_working_ids_in_raw":round(len(a&b)/len(a)*100,2) if a else None})
pd.DataFrame(comparisons).to_csv(TAB / "share_raw_cognitive_merge_check.csv", index=False, encoding="utf-8-sig")

memo = """# SHARE raw cognitive remerge audit

The current merged SHARE working file retains `orient` and `raccmathperf`, but the raw `gv_imputations` files contain `orienti` and, from wave 4 onward, `memory`. This audit confirms the available raw cognitive variables, their imputation flags, and merge-ID overlap with the current working file without exporting IDs. Numeric values are not treated as observed when the SHARE flag marks the item as missing by design or not designed for the respondent.

Use these raw scores to build a wave-specific cognitive composite before final CFA/IRT. Wave 7 has a large structural/non-designed component in both cognitive tests and should be excluded from the primary longitudinal cognition comparison or modeled as a prespecified sensitivity wave. Do not silently treat the current merged-file cognition coverage as the true SHARE coverage.
"""
(OUT / "share_raw_cognitive_audit.md").write_text(memo, encoding="utf-8")
print("wrote", TAB / "share_raw_cognitive_audit.csv")
