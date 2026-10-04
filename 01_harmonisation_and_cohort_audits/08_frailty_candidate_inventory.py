"""Inventory non-IC candidate deficits for an outcome-disjoint frailty score."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT_ROOT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT_ROOT / "tables"
FILES = {
    "ELSA": PROJECT_ROOT / "ELSA/Working_data/elsa.dta",
    "SHARE": next(PROJECT_ROOT.glob("SHARE/**/Working_data/share.dta")),
    "KLoSA": next(PROJECT_ROOT.glob("KLoSA/**/Working_data/klosa.dta")),
    "MHAS": next(PROJECT_ROOT.glob("MHAS/**/Working_data/mhas.dta")),
    "CHARLS": PROJECT_ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
    "HRS": PROJECT_ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
    "LASI": next(PROJECT_ROOT.glob("LASI/**/Working_data/lasi.dta")),
}

# These are candidates for a frailty construct independent of the primary IC
# domains. They are not yet scored: labels, cut-points and special codes must
# be audited before construction.
ROWS = [
    ("all","chronic_disease","hypertension","hibpe","ever/current hypertension diagnosis","binary disease deficit","common candidate"),
    ("all","chronic_disease","diabetes","diabe","ever/current diabetes diagnosis","binary disease deficit","common candidate"),
    ("all","chronic_disease","heart_disease","hearte","heart disease/condition","binary disease deficit","common candidate"),
    ("all","chronic_disease","stroke","stroke","stroke history","binary disease deficit","common candidate"),
    ("all","chronic_disease","cancer","cancre","cancer history","binary disease deficit","common candidate"),
    ("all","chronic_disease","arthritis","arthre","arthritis history","binary disease deficit","common candidate"),
    ("all","chronic_disease","lung_disease","lunge","lung disease history","binary disease deficit","common candidate"),
    ("all","chronic_disease","kidney_disease","kidneye","kidney disease history","binary disease deficit","optional; unavailable in some files"),
    ("all","global_health","self_rated_health","shlt","self-rated health","ordinal health deficit","common candidate; direction differs by file"),
    ("all","nutrition_body","bmi","bmi","body-mass index","continuous or category deficit","cut-points and units require audit"),
    ("all","symptom","pain","painlv","pain severity/frequency","ordinal symptom deficit","not present in CHARLS/HRS current files"),
    ("all","event","fall","fall","fall history/count","binary/count deficit","variable meaning differs by file"),
    ("all","event","hospitalization","hosp1y","hospitalization in previous year","binary event deficit","not present in all current files"),
    ("all","health_behavior","smoking","smokev","ever/current smoking","behavioral deficit candidate","not a core FI component; sensitivity only"),
    ("all","health_behavior","alcohol","drink","alcohol use","behavioral deficit candidate","coding and cultural comparability risk; sensitivity only"),
]

# Dataset-specific templates for the common candidates.
SPECS = {
    "ELSA": {"layout":"long", "waves":[None], "id":"idauniqc", "wave_col":"wave", "active":None, "vars": {"hibpe":"hibpe","diabe":"diabe","hearte":"hearte","stroke":"stroke","cancre":"cancre","arthre":"arthre","lunge":"lunge","kidneye":"","shlt":"shlt","bmi":"mbmi","painlv":"painlv","fall":"fall1y","hosp1y":"","smokev":"smokev","drink":"drink"}},
    "SHARE": {"layout":"long", "waves":[None], "id":"mergeid", "wave_col":"wave", "active":None, "vars": {"hibpe":"hibpe","diabe":"diabe","hearte":"hearte","stroke":"stroke","cancre":"cancre","arthre":"arthre","lunge":"lunge","kidneye":"kidneye","shlt":"shlt","bmi":"bmi","painlv":"painlv","fall":"fall_s","hosp1y":"hosp1y","smokev":"smokev","drink":"drinkx"}},
    "KLoSA": {"layout":"long", "waves":[None], "id":"pid", "wave_col":"wave", "active":None, "vars": {"hibpe":"hibpe","diabe":"diabe","hearte":"hearte","stroke":"stroke","cancre":"cancre","arthre":"arthre","lunge":"lunge","kidneye":"","shlt":"shlt","bmi":"bmi","painlv":"painlv_1","fall":"fall","hosp1y":"","smokev":"smokev","drink":"drinkev"}},
    "MHAS": {"layout":"long", "waves":[None], "id":"rahhidnp", "wave_col":"wave", "active":None, "vars": {"hibpe":"hibpe","diabe":"diabe","hearte":"hearte","stroke":"stroke","cancre":"cancre","arthre":"arthre","lunge":"lunge","kidneye":"","shlt":"shlt","bmi":"bmi","painlv":"painlv","fall":"fall","hosp1y":"hosp1y","smokev":"smokev","drink":"drink"}},
    "CHARLS": {"layout":"wide", "waves":[1,2,3,4], "id":"ID", "active":"inw{wave}", "vars": {"hibpe":"r{wave}hibpe","diabe":"r{wave}diabe","hearte":"r{wave}hearte","stroke":"r{wave}stroke","cancre":"r{wave}cancre","arthre":"r{wave}arthre","lunge":"r{wave}lunge","kidneye":"r{wave}kidneye","shlt":"r{wave}shlt","bmi":"r{wave}mbmi","painlv":"","fall":"","hosp1y":"r{wave}hosp1y","smokev":"r{wave}smokev","drink":"r{wave}drinkev"}},
    "HRS": {"layout":"wide", "waves":[8,9,10,11,12,13], "id":"hhidpn", "active":"inw{wave}", "vars": {"hibpe":"r{wave}hibpe","diabe":"r{wave}diabe","hearte":"r{wave}hearte","stroke":"r{wave}stroke","cancre":"r{wave}cancre","arthre":"r{wave}arthre","lunge":"r{wave}lunge","kidneye":"r{wave}kidneye","shlt":"r{wave}shlt","bmi":"r{wave}bmi","painlv":"","fall":"","hosp1y":"r{wave}hosp","smokev":"r{wave}smokev","drink":"r{wave}drink"}},
    "LASI": {"layout":"wide", "waves":[1], "id":"hhid", "active_year":"r{wave}iwy", "vars": {"hibpe":"r{wave}hibpe","diabe":"r{wave}diabe","hearte":"r{wave}hearte","stroke":"r{wave}stroke","cancre":"r{wave}cancre","arthre":"r{wave}arthre","lunge":"r{wave}lunge","kidneye":"","shlt":"r{wave}shlt","bmi":"r{wave}mbmi","painlv":"r{wave}painlv","fall":"r{wave}fall","hosp1y":"r{wave}hosp1y","smokev":"r{wave}smokev","drink":"r{wave}drinkev"}},
}

def num(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)

coverage = []
ledger = []
for dataset, spec in SPECS.items():
    _, meta = pyreadstat.read_dta(str(FILES[dataset]), metadataonly=True)
    avail = set(meta.column_names)
    raw = {spec["id"]}
    for key, template in spec["vars"].items():
        if not template:
            continue
        for w in spec["waves"]:
            raw.add(template.format(wave=w) if spec["layout"] == "wide" else template)
    for w in spec["waves"]:
        if spec.get("active"):
            raw.add(spec["active"].format(wave=w))
        elif spec.get("active_year"):
            raw.add(spec["active_year"].format(wave=w))
    use = [c for c in raw if c in avail]
    df, _ = pyreadstat.read_dta(str(FILES[dataset]), usecols=use, apply_value_formats=False)
    for _, dom, component, basevar, meaning, scale, role in ROWS:
        template = spec["vars"].get(basevar, "")
        ledger.append({"dataset":dataset,"domain":dom,"component":component,"candidate_expression":template or "not_available","meaning":meaning,"deficit_scale":scale,"role":role})
        for w in spec["waves"]:
            if spec["layout"] == "long":
                active = pd.Series(True, index=df.index)
                actual = [template] if template else []
                wave_label = "all_current_waves"
            else:
                if spec.get("active"):
                    active_col = spec["active"].format(wave=w)
                    active = num(df[active_col]).eq(1) if active_col in df else pd.Series(False, index=df.index)
                else:
                    active_col = spec.get("active_year","").format(wave=w)
                    active = df[active_col].notna() if active_col in df else pd.Series(False, index=df.index)
                actual = [template.format(wave=w)] if template else []
                wave_label = w
            actual = [c for c in actual if c in df]
            obs = num(df[actual[0]]).notna() if actual else pd.Series(False, index=df.index)
            obs_active = obs.loc[active]
            coverage.append({"dataset":dataset,"domain":dom,"component":component,"wave":wave_label,"n_active":int(active.sum()),"n_observed":int(obs_active.sum()),"pct_observed_active":round(float(obs_active.mean()*100),2) if len(obs_active) else np.nan,"available":bool(actual)})

ledger_df = pd.DataFrame(ledger)
coverage_df = pd.DataFrame(coverage)
ledger_df.to_csv(TAB / "outcome_disjoint_frailty_candidate_matrix.csv", index=False, encoding="utf-8-sig")
coverage_df.to_csv(TAB / "outcome_disjoint_frailty_candidate_coverage.csv", index=False, encoding="utf-8-sig")
summary = (coverage_df.groupby(["dataset","domain","component"],as_index=False).agg(mean_pct_observed_active=("pct_observed_active","mean"),min_pct_observed_active=("pct_observed_active","min"),n_waves=("wave","nunique"),available_any=("available","max")))
summary.to_csv(TAB / "outcome_disjoint_frailty_candidate_summary.csv", index=False, encoding="utf-8-sig")

memo = """# Outcome-disjoint frailty: candidate design options

## Candidate components

The inventory prioritizes the common candidate set of hypertension, diabetes, heart disease, stroke, cancer, arthritis, self-rated health and BMI. Lung disease is an optional additional chronic item because it is absent from the current MHAS working file. Falls, pain and hospitalization are candidate extensions but are not available with the same meaning in every current working file. Smoking and alcohol are retained only as sensitivity candidates because their role in a frailty index and their cultural comparability are less defensible.

## Option A — primary discriminant FI (recommended for construct comparison)

Use the common candidate set across the target queues: hypertension, diabetes, heart disease, stroke, cancer, arthritis, self-rated health and BMI. Lung disease is an optional eighth chronic item because it is absent from the current MHAS working file. Exclude cognition, locomotion/function, grip/vitality, psychological symptoms, sensory items, ADL/IADL and death. This produces a deliberately non-IC frailty/health-deficit construct; the trade-off is a shorter index and less resemblance to standard 30-item FIs.

## Option B — expanded disjoint FI

Add falls, pain and recent hospitalization where definitions and coding can be harmonized. Use this in cohorts with adequate coverage and treat the cross-cohort comparison as a sensitivity analysis because current availability is uneven.

## Option C — standard overlapping FI benchmark

Reconstruct a conventional frailty index including IC-related deficits, but report it only as a convergent-validity benchmark. It should not be the primary independent frailty measure because ELSA/SHARE existing composites already include the same IC domains.

The recommended sequence is A as the primary discriminant construct, B as a sensitivity analysis, and C as a benchmark. The final component set should be frozen after reviewing the coverage tables and value labels.
"""
(OUT_ROOT / "frailty_design_options.md").write_text(memo, encoding="utf-8")
print("wrote frailty inventory", len(ledger_df), len(coverage_df))
