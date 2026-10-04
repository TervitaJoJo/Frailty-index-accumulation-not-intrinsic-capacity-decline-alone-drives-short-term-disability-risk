"""Aggregate audit of SHARE wave-6 self-reported mobility candidates."""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
share_parent = next(p for p in (ROOT / "SHARE").iterdir() if p.is_dir())
path = share_parent / "Working_data" / "share.dta"
df, meta = pyreadstat.read_dta(str(path), usecols=["wave", "walkra", "walk100a"], apply_value_formats=False)
df = df.loc[df["wave"].eq(6)].copy()
for c in ["walkra", "walk100a"]:
    x = pd.to_numeric(df[c], errors="coerce")
    df[c] = x.where(x.isin([0, 1]))
df["mobility_sum"] = df[["walkra", "walk100a"]].sum(axis=1, min_count=2)
df["mobility_hierarchical"] = pd.NA
df.loc[df["walk100a"].eq(0), "mobility_hierarchical"] = 0
df.loc[df["walk100a"].eq(1) & df["walkra"].eq(0), "mobility_hierarchical"] = 1
df.loc[df["walkra"].eq(1), "mobility_hierarchical"] = 2
rows = []
for c in ["walkra", "walk100a", "mobility_sum", "mobility_hierarchical"]:
    x = df[c].dropna()
    rows.append({"indicator": c, "n_active": len(df), "n_valid": len(x),
                 "pct_valid": 100 * len(x) / len(df), "mean": x.mean(), "sd": x.std(),
                 "n_unique": x.nunique(), "category_counts": ";".join(f"{k}:{int(v)}" for k, v in x.value_counts().sort_index().items())})
out = pd.DataFrame(rows)
TAB.mkdir(parents=True, exist_ok=True)
out.to_csv(TAB / "share_mobility_composite_audit.csv", index=False, encoding="utf-8-sig")
memo = """# SHARE mobility composite audit

Wave-6 self-reported mobility items are nested distance thresholds. `walkra` (difficulty walking across a room) is rare, while `walk100a` (difficulty walking 100 metres) is more prevalent. Their sum and a hierarchical severity score are retained as aggregate candidates; no individual-level derived file is written.

The two-item CFA should remain a diagnostic because the rare nested item produced a boundary/negative residual variance. A preregistered composite or a domain-score treatment can be compared before the cross-cohort model is frozen.

See `tables/share_mobility_composite_audit.csv`.
"""
(OUT / "stage7_share_mobility_composite_audit.md").write_text(memo, encoding="utf-8")
(OUT / "stage7_share_mobility_composite_run_info.json").write_text(json.dumps({"wave": 6, "source": "SHARE Working_data/share.dta", "scope": "aggregate-only"}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote SHARE mobility composite audit")
