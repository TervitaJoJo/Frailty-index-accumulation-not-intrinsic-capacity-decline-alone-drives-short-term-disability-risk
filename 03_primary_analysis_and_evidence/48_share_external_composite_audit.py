"""Stage 20: SHARE wave-6 external validation using a prespecified mobility composite.

All person-level data remain in memory. Only aggregate coverage, correlations,
and country-level counts are written.
"""
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(r"PATH_TO_IC_FRAILTY")
SHARE_ROOT = OUT / "_share_ascii_real"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

def unique_path(pattern):
    matches = list(SHARE_ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one SHARE source for {pattern}; found {len(matches)}")
    return matches[0]

def read(path, columns):
    return pyreadstat.read_dta(str(path), usecols=list(dict.fromkeys(columns)), apply_value_formats=False)[0]

def bounded(x, lo, hi):
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x.between(lo, hi)).astype(float)

def binary(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x.isin([0, 1])).astype(float)

def exclusive_list_total(df, columns):
    values = df[columns].apply(lambda x: bounded(x, 0, 10))
    n = values.notna().sum(axis=1)
    return values.sum(axis=1, min_count=1).where(n.eq(1)), n

def score_fi(df):
    disease_names = ["hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis"]
    parts = pd.DataFrame({n: binary(df[n]) for n in disease_names}, index=df.index)
    shlt = bounded(df["shlt"], 1, 5)
    parts["self_rated_health"] = (5 - shlt) / 4
    bmi = pd.to_numeric(df["bmi"], errors="coerce").where(lambda z: z.gt(0))
    parts["bmi"] = (bmi.lt(18.5) | bmi.ge(30)).astype(float).where(bmi.notna())
    n = parts.notna().sum(axis=1)
    primary = parts.sum(axis=1, min_count=1).div(n.where(n.gt(0))).where(n.ge(6))
    complete8 = parts.sum(axis=1, min_count=8).div(8).where(n.eq(8))
    wo_srh = parts.drop(columns="self_rated_health")
    n7 = wo_srh.notna().sum(axis=1)
    without_srh = wo_srh.sum(axis=1, min_count=1).div(n7.where(n7.gt(0))).where(n7.ge(6))
    return primary, complete8, without_srh, n, parts

def spearman_rows(frame, sample):
    cols = ["cognition_full4", "locomotion_hierarchical", "grip_vitality", "psychological_full4", "fi_primary"]
    rows = []
    for a, b in combinations(cols, 2):
        p = frame[[a, b]].dropna()
        rho = p[a].corr(p[b], method="spearman") if len(p) > 2 and p[a].nunique() > 1 and p[b].nunique() > 1 else np.nan
        rows.append({"sample": sample, "construct_a": a, "construct_b": b, "n_pairwise": len(p), "spearman": rho})
    return rows

def main():
    base_path = SHARE_ROOT / "Working_data" / "share.dta"
    base_cols = ["mergeid", "wave", "walkra", "walk100a", "lgrip", "rgrip", "shlt", "bmi",
                 "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"]
    base = read(base_path, base_cols)
    base = base.loc[base["wave"].eq(6)].copy()
    if base["mergeid"].duplicated().any():
        raise ValueError("SHARE wave-6 working file has duplicate mergeid")

    cf_path = unique_path("Raw_data/**/*w6*cf.dta")
    cf_cols = ["mergeid", "country", "cf003_", "cf004_", "cf005_", "cf006_", "cf010_"] + [f"cf{i}tot" for i in range(104, 108)] + [f"cf{i}tot" for i in range(113, 117)]
    cf = read(cf_path, cf_cols)
    gh_path = unique_path("Raw_data/**/*w6*gv_health.dta")
    gh = read(gh_path, ["mergeid", "euro1", "euro2", "euro5", "euro11"])
    for x in [cf, gh]:
        if x["mergeid"].duplicated().any():
            raise ValueError("SHARE module has duplicate mergeid")
    d = base.merge(cf, on="mergeid", how="left", validate="one_to_one").merge(gh, on="mergeid", how="left", validate="one_to_one")

    orient = d[["cf003_", "cf004_", "cf005_", "cf006_"]].apply(lambda x: bounded(x, 1, 2).map({1: 1.0, 2: 0.0}))
    orientation = orient.sum(axis=1, min_count=4) / 4
    immediate, n_im = exclusive_list_total(d, [f"cf{i}tot" for i in range(104, 108)])
    delayed, n_del = exclusive_list_total(d, [f"cf{i}tot" for i in range(113, 117)])
    verbal = bounded(d["cf010_"], 0, 100) / 100
    d["cognition_full4"] = pd.concat([immediate / 10, delayed / 10, orientation, verbal], axis=1).mean(axis=1, skipna=False)
    d["cognition_at_least3"] = pd.concat([immediate / 10, delayed / 10, orientation, verbal], axis=1).mean(axis=1, skipna=True).where(pd.concat([immediate / 10, delayed / 10, orientation, verbal], axis=1).notna().sum(axis=1).ge(3))

    walkra = binary(d["walkra"])
    walk100 = binary(d["walk100a"])
    severity = pd.Series(np.nan, index=d.index, dtype=float)
    severity.loc[walk100.eq(0)] = 0
    severity.loc[walk100.eq(1) & walkra.eq(0)] = 1
    severity.loc[walkra.eq(1)] = 2
    d["locomotion_hierarchical"] = 1 - severity / 2
    d["locomotion_sum"] = 1 - (walkra + walk100) / 2

    grip = pd.concat([pd.to_numeric(d["lgrip"], errors="coerce"), pd.to_numeric(d["rgrip"], errors="coerce")], axis=1).where(lambda x: x.ge(0)).max(axis=1)
    d["grip_vitality"] = grip / 100
    psych = pd.DataFrame({c: binary(d[c]) for c in ["euro1", "euro2", "euro5", "euro11"]})
    d["psychological_full4"] = 1 - psych.mean(axis=1, skipna=False)
    d["psychological_at_least3"] = 1 - psych.mean(axis=1, skipna=True).where(psych.notna().sum(axis=1).ge(3))

    disease_map = {"hypertension": "hibpe", "diabetes": "diabe", "heart_disease": "hearte", "stroke": "stroke", "cancer": "cancre", "arthritis": "arthre"}
    for n, c in disease_map.items():
        d[n] = d[c]
    d["shlt"] = d["shlt"]
    d["bmi"] = d["bmi"]
    d["fi_primary"], d["fi_complete8"], d["fi_without_srh"], d["fi_observed_n"], components = score_fi(d)

    # Aggregate descriptive coverage; source mergeid is removed before output.
    domains = ["cognition_full4", "cognition_at_least3", "locomotion_hierarchical", "locomotion_sum", "grip_vitality", "psychological_full4", "psychological_at_least3", "fi_primary", "fi_complete8", "fi_without_srh"]
    desc = []
    for c in domains:
        x = d[c].dropna()
        desc.append({"construct": c, "n_active": len(d), "n_valid": len(x), "pct_valid": 100 * len(x) / len(d), "mean": x.mean(), "sd": x.std(), "n_unique": x.nunique()})
    domain_complete = d[["cognition_full4", "locomotion_hierarchical", "grip_vitality", "psychological_full4", "fi_primary"]].notna().all(axis=1)
    desc.append({"construct": "joint_four_domain_fi_complete", "n_active": len(d), "n_valid": int(domain_complete.sum()), "pct_valid": 100 * domain_complete.mean()})

    corr = spearman_rows(d, "primary_full_component_composite")
    # Pairwise sensitivity: mobility sum, cognition/psychology at-least-3,
    # and FI definitions are evaluated on matched records.
    sens = d.rename(columns={"locomotion_hierarchical": "locomotion_hierarchical"}).copy()
    sens["locomotion_hierarchical"] = d["locomotion_sum"]
    sens["cognition_full4"] = d["cognition_at_least3"]
    sens["psychological_full4"] = d["psychological_at_least3"]
    for fi_name in ["fi_primary", "fi_complete8", "fi_without_srh"]:
        temp = sens.copy()
        if fi_name != "fi_primary":
            temp["fi_primary"] = d[fi_name]
        corr.extend(spearman_rows(temp, f"sensitivity_{fi_name}"))

    country = []
    for c, g in d.groupby("country", dropna=False):
        joint = g[["cognition_full4", "locomotion_hierarchical", "grip_vitality", "psychological_full4", "fi_primary"]].notna().all(axis=1)
        country.append({"country": c, "n_active": len(g), "cognition_full4_pct": 100 * g["cognition_full4"].notna().mean(), "mobility_pct": 100 * g["locomotion_hierarchical"].notna().mean(), "joint_four_domain_fi_pct": 100 * joint.mean()})

    pd.DataFrame(desc).round(6).to_csv(TAB / "stage20_share_external_descriptives.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(corr).round(6).to_csv(TAB / "stage20_share_external_spearman.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(country).round(6).to_csv(TAB / "stage20_share_external_country_coverage.csv", index=False, encoding="utf-8-sig")
    coverage = {"n_active": len(d), "n_immediate_valid": int(immediate.notna().sum()), "n_delayed_valid": int(delayed.notna().sum()), "n_orientation_valid": int(orientation.notna().sum()), "n_verbal_valid": int(verbal.notna().sum()), "n_fi_primary": int(d["fi_primary"].notna().sum()), "n_joint_four_domain_fi": int(domain_complete.sum()), "n_mobility_hierarchical": int(d["locomotion_hierarchical"].notna().sum()), "n_mobility_sum": int(d["locomotion_sum"].notna().sum()), "n_ambiguous_immediate_wordlists": int(n_im.gt(1).sum()), "n_ambiguous_delayed_wordlists": int(n_del.gt(1).sum())}
    (OUT / "stage20_share_external_run_info.json").write_text(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "design": "SHARE wave 6 external validation; hierarchical mobility composite primary; aggregate-only", "fi_rule": "six diseases + self-rated health + BMI; minimum 6/8; observed denominator; BMI<18.5 or >=30", "coverage": coverage, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, ensure_ascii=False, indent=2), encoding="utf-8")
    memo = [
        "# Stage 20 SHARE wave-6 external validation using mobility composite", "",
        f"SHARE wave 6 active sample: **{len(d):,}**. The primary mobility indicator is a prespecified hierarchical severity composite: no 100-metre difficulty=0, 100-metre difficulty without room difficulty=1, room difficulty=2; it is reversed to higher capacity. The simple sum is sensitivity only.", "",
        "Cognition uses objective orientation, immediate recall, delayed recall and verbal fluency; psychological capacity uses four EURO-D symptom indicators in the higher-capacity direction; grip/vitality uses the maximum observed handgrip divided by 100. FI uses the frozen eight-component rule and is not treated as a gold-standard frailty diagnosis.", "",
        f"Primary joint availability (all four domains + FI): **{int(domain_complete.sum()):,} ({100*domain_complete.mean():.1f}%)**. The exact aggregate coverages are in `tables/stage20_share_external_descriptives.csv`.", "",
        "The primary external check is the sign and magnitude of pairwise Spearman associations between the four domain scores and FI. Sensitivity rows replace the hierarchical mobility composite with the simple sum, permit at least 3/4 cognition or psychological indicators, and use complete-8 or omit-self-rated-health FI definitions. These are external structure checks, not latent mean comparisons or causal analyses.", "",
        "The original two-item mobility CFA remains a diagnostic because the rare room-walking item produced a boundary/negative residual variance. A composite that reproduces the expected locomotion–FI direction supports external structural transportability; a discrepant result defines a transportability boundary and should be reported.", "",
        "Outputs: `tables/stage20_share_external_descriptives.csv`, `tables/stage20_share_external_spearman.csv`, `tables/stage20_share_external_country_coverage.csv`, and `stage20_share_external_run_info.json`. No person-level derived file is written."
    ]
    (OUT / "stage20_share_external_composite_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print("Wrote Stage 20 SHARE external composite audit outputs.")

if __name__ == "__main__":
    main()
