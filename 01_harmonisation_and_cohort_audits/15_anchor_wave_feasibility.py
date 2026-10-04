"""Assess an anchor-wave design for cross-national IC--frailty comparison.

The anchor-wave screen is exploratory. It uses one wave per cohort to avoid
confounding the first cross-national measurement comparison with different
module schedules. SHARE cognitive coverage is re-read from raw gv_imputations
using implicat=1 only for this aggregate feasibility screen; no individual IDs
are exported and this is not the final MI estimator.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

ANCHOR = {"ELSA": 6, "SHARE": 6, "CHARLS": 3, "HRS": 10, "LASI": 1}
FILES = {
    "ELSA": ROOT / "ELSA/Working_data/elsa.dta",
    "SHARE": next(ROOT.glob("SHARE/**/Working_data/share.dta")),
    "CHARLS": ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
    "HRS": ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
    "LASI": next(ROOT.glob("LASI/**/Working_data/lasi.dta")),
}


def valid(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)


def max_valid(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return pd.concat([valid(df[c]) for c in present], axis=1).max(axis=1, skipna=True)


def indicators(df: pd.DataFrame, names: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for key, cols in names.items():
        if len(cols) == 1:
            out[key] = valid(df[cols[0]]) if cols[0] in df else np.nan
        else:
            out[key] = max_valid(df, cols)
    return out


rows: list[dict[str, object]] = []
detail: list[dict[str, object]] = []

# ELSA, CHARLS, HRS, LASI use the current harmonized working files.
specs = {
    "ELSA": {
        "layout": "long", "wave_col": "wave", "wave": 6,
        "ic": {"cognition": ["tcog_z_z"], "locomotion": ["walkra"], "grip_vitality": ["gripsum"], "psychological": ["cesd"]},
        "frailty": {"hypertension":["hibpe"], "diabetes":["diabe"], "heart_disease":["hearte"], "stroke":["stroke"], "cancer":["cancre"], "arthritis":["arthre"], "self_rated_health":["shlt"], "bmi":["mbmi"]},
    },
    "CHARLS": {
        "layout": "wide", "wave": 3, "active": "inw3",
        "ic": {"cognition": ["r3orient", "r3tr20"], "locomotion": ["r3walk100a"], "grip_vitality": ["r3lgrip", "r3rgrip"], "psychological": ["r3cesd10"]},
        "frailty": {"hypertension":["r3hibpe"], "diabetes":["r3diabe"], "heart_disease":["r3hearte"], "stroke":["r3stroke"], "cancer":["r3cancre"], "arthritis":["r3arthre"], "self_rated_health":["r3shlt"], "bmi":["r3mbmi"]},
    },
    "HRS": {
        "layout": "wide", "wave": 10, "active": "inw10",
        "ic": {"cognition": ["r10cog27"], "locomotion": ["r10walkra"], "grip_vitality": ["r10grpl", "r10grpr"], "psychological": ["r10cesd"]},
        "frailty": {"hypertension":["r10hibpe"], "diabetes":["r10diabe"], "heart_disease":["r10hearte"], "stroke":["r10stroke"], "cancer":["r10cancre"], "arthritis":["r10arthre"], "self_rated_health":["r10shlt"], "bmi":["r10bmi"]},
    },
    "LASI": {
        "layout": "wide", "wave": 1, "active_year": "r1iwy",
        "ic": {"cognition": ["r1cog_total"], "locomotion": ["r1walk100a"], "grip_vitality": ["r1lgrip", "r1rgrip"], "psychological": ["r1cesd10_l"]},
        "frailty": {"hypertension":["r1hibpe"], "diabetes":["r1diabe"], "heart_disease":["r1hearte"], "stroke":["r1stroke"], "cancer":["r1cancre"], "arthritis":["r1arthre"], "self_rated_health":["r1shlt"], "bmi":["r1mbmi"]},
    },
}

for dataset, spec in specs.items():
    _, meta = pyreadstat.read_dta(str(FILES[dataset]), metadataonly=True)
    all_cols = {c for group in [spec["ic"], spec["frailty"]] for cols in group.values() for c in cols}
    all_cols = [c for c in all_cols if c in meta.column_names]
    if spec["layout"] == "long":
        all_cols = [spec["wave_col"], *all_cols]
    else:
        active_col = spec.get("active", spec.get("active_year"))
        all_cols = [active_col, *all_cols]
    df, _ = pyreadstat.read_dta(str(FILES[dataset]), usecols=list(dict.fromkeys([c for c in all_cols if c])), apply_value_formats=False)
    if spec["layout"] == "long":
        active = df[spec["wave_col"]].eq(spec["wave"])
    elif spec.get("active"):
        active = valid(df[spec["active"]]).eq(1)
    else:
        active = df[spec["active_year"]].notna()
    ic = indicators(df.loc[active], spec["ic"])
    frail = indicators(df.loc[active], spec["frailty"])
    ic_complete = ic.notna().all(axis=1)
    n_frail = frail.notna().sum(axis=1)
    row = {"dataset": dataset, "anchor_wave": spec["wave"], "n_active": int(active.sum()), "ic_complete_4_domains_n": int(ic_complete.sum()), "ic_complete_4_domains_pct": round(float(ic_complete.mean()*100), 2) if len(ic) else None, "frailty_complete_8_n": int((n_frail == 8).sum()), "frailty_complete_8_pct": round(float((n_frail == 8).mean()*100), 2) if len(frail) else None, "frailty_at_least_6_n": int((n_frail >= 6).sum()), "frailty_at_least_6_pct": round(float((n_frail >= 6).mean()*100), 2) if len(frail) else None, "joint_ic_frailty_min6_n": int((ic_complete & (n_frail >= 6)).sum()), "joint_ic_frailty_min6_pct": round(float((ic_complete & (n_frail >= 6)).mean()*100), 2) if len(ic) else None}
    for col in ic.columns:
        row[f"ic_{col}_pct"] = round(float(ic[col].notna().mean()*100), 2) if len(ic) else None
        detail.append({"dataset":dataset,"anchor_wave":spec["wave"],"section":"IC","component":col,"n_active":len(ic),"n_observed":int(ic[col].notna().sum()),"pct_observed":round(float(ic[col].notna().mean()*100),2) if len(ic) else None})
    for col in frail.columns:
        row[f"frailty_{col}_pct"] = round(float(frail[col].notna().mean()*100), 2) if len(frail) else None
        detail.append({"dataset":dataset,"anchor_wave":spec["wave"],"section":"frailty","component":col,"n_active":len(frail),"n_observed":int(frail[col].notna().sum()),"pct_observed":round(float(frail[col].notna().mean()*100),2) if len(frail) else None})
    rows.append(row)

# SHARE requires the raw cognition module. Use implicat=1 for this aggregate
# screen, retaining the flag-aware distinction between usable and structural
# records. Other domains come from the current working file.
share_path = FILES["SHARE"]
_, share_meta = pyreadstat.read_dta(str(share_path), metadataonly=True)
share_cols = ["mergeid", "wave", "walkra", "lgrip", "rgrip", "eurod", "shlt", "bmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"]
share_cols = [c for c in share_cols if c in share_meta.column_names]
share, _ = pyreadstat.read_dta(str(share_path), usecols=share_cols, apply_value_formats=False)
share = share.loc[share["wave"].eq(6)].copy()
raw_path = next((ROOT / "SHARE").rglob("sharew6_rel9-0-0_gv_imputations.dta"))
raw, _ = pyreadstat.read_dta(str(raw_path), usecols=["mergeid", "implicat", "orienti", "memory", "orienti_f", "memory_f"], apply_value_formats=False)
raw = raw.loc[raw["implicat"].eq(1)].copy()
share = share.merge(raw, on="mergeid", how="left", validate="one_to_one")
ic = pd.DataFrame({"cognition": valid(share["orienti"]).where(pd.to_numeric(share["orienti_f"], errors="coerce").between(3,13) & pd.to_numeric(share["memory_f"], errors="coerce").between(3,13)), "locomotion": valid(share["walkra"]), "grip_vitality": max_valid(share,["lgrip","rgrip"]), "psychological": valid(share["eurod"])})
frail = indicators(share, {"hypertension":["hibpe"], "diabetes":["diabe"], "heart_disease":["hearte"], "stroke":["stroke"], "cancer":["cancre"], "arthritis":["arthre"], "self_rated_health":["shlt"], "bmi":["bmi"]})
ic_complete = ic.notna().all(axis=1); n_frail = frail.notna().sum(axis=1)
row = {"dataset":"SHARE", "anchor_wave":6, "n_active":len(share), "ic_complete_4_domains_n":int(ic_complete.sum()), "ic_complete_4_domains_pct":round(float(ic_complete.mean()*100),2), "frailty_complete_8_n":int((n_frail==8).sum()), "frailty_complete_8_pct":round(float((n_frail==8).mean()*100),2), "frailty_at_least_6_n":int((n_frail>=6).sum()), "frailty_at_least_6_pct":round(float((n_frail>=6).mean()*100),2), "joint_ic_frailty_min6_n":int((ic_complete & (n_frail>=6)).sum()), "joint_ic_frailty_min6_pct":round(float((ic_complete & (n_frail>=6)).mean()*100),2)}
for col in ic.columns:
    row[f"ic_{col}_pct"] = round(float(ic[col].notna().mean()*100),2)
    detail.append({"dataset":"SHARE","anchor_wave":6,"section":"IC","component":col,"n_active":len(ic),"n_observed":int(ic[col].notna().sum()),"pct_observed":round(float(ic[col].notna().mean()*100),2)})
for col in frail.columns:
    row[f"frailty_{col}_pct"] = round(float(frail[col].notna().mean()*100),2)
    detail.append({"dataset":"SHARE","anchor_wave":6,"section":"frailty","component":col,"n_active":len(frail),"n_observed":int(frail[col].notna().sum()),"pct_observed":round(float(frail[col].notna().mean()*100),2)})
rows.append(row)

pd.DataFrame(rows).sort_values("dataset").to_csv(TAB / "anchor_wave_feasibility_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(detail).to_csv(TAB / "anchor_wave_feasibility_components.csv", index=False, encoding="utf-8-sig")

memo = """# Anchor-wave feasibility screen

To make the first cross-national construct comparison interpretable, this screen uses one anchor wave per cohort: ELSA 6, SHARE 6, CHARLS 3, HRS 10 and LASI 1. These waves were chosen because the current audits show concurrent availability of the four IC domains and the eight outcome-disjoint frailty candidates.

SHARE cognition is read from raw `gv_imputations` at `implicat=1` only for coverage screening. The final analysis must preserve all five imputations and use a proper MI-compatible estimator.

The summary reports domain-specific coverage, four-domain IC completeness, frailty completeness under the selected minimum-6-of-8 rule, and the joint usable proportion. It is a feasibility result, not a final sample definition.
"""
(OUT / "anchor_wave_feasibility.md").write_text(memo, encoding="utf-8")
print("wrote anchor wave feasibility")
