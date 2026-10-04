"""Build an explicit four-domain IC and outcome-disjoint frailty blueprint."""
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

# Candidate rows intentionally remain a review ledger rather than a frozen
# harmonization.  `primary_four_domain` is the proposed main analysis;
# `full_IC_sensitivity` is the sensory extension.
CANDIDATES = [
    # dataset, domain, variable expression, raw variables, mode, direction,
    # scale, status, risk
    ("ELSA","cognition","tcog_z_z",["tcog_z_z"],"standardized continuous score","higher is better","survey-specific z score","primary_four_domain","wave/module dependent"),
    ("ELSA","locomotion","walkra",["walkra"],"self-reported difficulty walking across room","higher is worse","binary difficulty","primary_four_domain","self-report; coding direction audit"),
    ("ELSA","locomotion","wspeed",["wspeed"],"objective walking speed/time","depends on unit","nurse-module test","full_IC_sensitivity","subsample; speed versus time requires audit"),
    ("ELSA","grip_vitality","gripsum",["gripsum"],"grip strength summary","higher is better","survey-specific grip summary","primary_four_domain","nurse-module subsample"),
    ("ELSA","psychological","cesd",["cesd"],"CES-D-8","higher is worse","0-8 scale","primary_four_domain","scale harmonization"),
    ("ELSA","sensory","sight;hearing",["sight","hearing"],"self-reported vision and hearing","higher is better in raw labels","ordinal","full_IC_sensitivity","two indicators; label audit"),
    ("SHARE","cognition","orient",["orient"],"orientation score","higher is better","0-4 score in current merged file","primary_four_domain","current merged file lacks full memory score"),
    ("SHARE","locomotion","walkra",["walkra"],"self-reported difficulty walking across room","higher is worse","binary difficulty","primary_four_domain","self-report; coding direction audit"),
    ("SHARE","locomotion","wspeed",["wspeed"],"objective walking test","depends on unit","physical test","full_IC_sensitivity","very sparse in current merged file"),
    ("SHARE","grip_vitality","max(lgrip,rgrip)",["lgrip","rgrip"],"maximum hand grip","higher is better","kg or survey unit","primary_four_domain","measurement protocol audit"),
    ("SHARE","psychological","eurod",["eurod"],"EURO-D","higher is worse","0-12 scale","primary_four_domain","not identical to CES-D"),
    ("SHARE","sensory","dsight;hearing",["dsight","hearing"],"self-reported vision and hearing","higher is better in raw labels","ordinal","full_IC_sensitivity","label audit"),
    ("KLoSA","cognition","cog_total",["cog_total"],"total cognition score","higher is better","survey-specific score","primary_four_domain","score construction audit"),
    ("KLoSA","locomotion","not_available",[],"no clear candidate in current working file","not_available","not_available","not_available","return to raw modules"),
    ("KLoSA","grip_vitality","max(lgrip,rgrip)",["lgrip","rgrip"],"maximum hand grip","higher is better","kg or survey unit","primary_four_domain","measurement protocol audit"),
    ("KLoSA","psychological","cesd10b",["cesd10b"],"CES-D-10","higher is worse","0-10 scale","primary_four_domain","scale harmonization"),
    ("KLoSA","sensory","sighta;hearinga",["sighta","hearinga"],"self-reported vision and hearing","higher is better in raw labels","ordinal","full_IC_sensitivity","label audit"),
    ("MHAS","cognition","orient_m",["orient_m"],"orientation score","higher is better","simplified orientation score","primary_four_domain","raw cognitive module contains richer measures"),
    ("MHAS","locomotion","walkra",["walkra"],"self-reported difficulty walking across room","higher is worse","binary difficulty","primary_four_domain","self-report; coding direction audit"),
    ("MHAS","locomotion","wspeed",["wspeed"],"objective walking test","depends on unit","physical test","full_IC_sensitivity","very sparse"),
    ("MHAS","grip_vitality","max(lgrip,rgrip)",["lgrip","rgrip"],"maximum hand grip","higher is better","kg or survey unit","primary_four_domain","very sparse in current working file"),
    ("MHAS","psychological","cesd_m",["cesd_m"],"CES-D-derived score","higher is worse","survey-specific scale","primary_four_domain","scale harmonization"),
    ("MHAS","sensory","sight;hearing",["sight","hearing"],"self-reported vision and hearing","higher is better in raw labels","ordinal","full_IC_sensitivity","label audit"),
    ("CHARLS","cognition","r{wave}tr20",["r{wave}tr20"],"word recall summary","higher is better","0-10 recall score","primary_four_domain","nurse/interview module; use harmonized cognitive composite if available"),
    ("CHARLS","locomotion","r{wave}walk100a",["r{wave}walk100a"],"difficulty walking 100 m","higher is worse","binary difficulty","primary_four_domain","self-report; label audit"),
    ("CHARLS","locomotion","r{wave}wspeed",["r{wave}wspeed"],"objective walking test","depends on unit","physical test","full_IC_sensitivity","subsample"),
    ("CHARLS","grip_vitality","max(r{wave}lgrip,r{wave}rgrip)",["r{wave}lgrip","r{wave}rgrip"],"maximum hand grip","higher is better","kg or survey unit","primary_four_domain","nurse-module subsample"),
    ("CHARLS","psychological","r{wave}cesd10",["r{wave}cesd10"],"CES-D-10","higher is worse","0-30 scale","primary_four_domain","scale harmonization"),
    ("CHARLS","sensory","not_available",[],"no clear vision/hearing candidate in current harmonized file","not_available","not_available","not_available","do not claim full-IC comparability"),
    ("HRS","cognition","r{wave}cogtot",["r{wave}cogtot"],"total cognition summary","higher is better","survey-specific score","primary_four_domain","wave/module dependent"),
    ("HRS","locomotion","r{wave}walkra",["r{wave}walkra"],"difficulty walking across room","higher is worse","binary/ordinal difficulty","primary_four_domain","self-report; label audit"),
    ("HRS","locomotion","r{wave}timwlk",["r{wave}timwlk"],"timed walk test","higher is worse if seconds","seconds","full_IC_sensitivity","subsample"),
    ("HRS","grip_vitality","max(r{wave}grpl,r{wave}grpr)",["r{wave}grpl","r{wave}grpr"],"maximum hand grip","higher is better","kg or survey unit","primary_four_domain","subsample"),
    ("HRS","psychological","r{wave}cesd",["r{wave}cesd"],"CES-D-derived score","higher is worse","survey-specific scale","primary_four_domain","scale harmonization"),
    ("HRS","sensory","r{wave}eyert;r{wave}earrt",["r{wave}eyert","r{wave}earrt"],"self-reported vision and hearing","depends on label","ordinal","full_IC_sensitivity","label audit"),
    ("LASI","cognition","r1cog_total",["r1cog_total"],"total cognition score","higher is better","survey-specific score","primary_four_domain","single baseline wave"),
    ("LASI","locomotion","r1walk100a",["r1walk100a"],"difficulty walking 100 m","higher is worse","binary difficulty","primary_four_domain","self-report; label audit"),
    ("LASI","locomotion","r1wspeed",["r1wspeed"],"objective walking test","depends on unit","physical test","full_IC_sensitivity","single baseline wave"),
    ("LASI","grip_vitality","max(r1lgrip,r1rgrip)",["r1lgrip","r1rgrip"],"maximum hand grip","higher is better","kg or survey unit","primary_four_domain","single baseline wave"),
    ("LASI","psychological","r1cesd10_l",["r1cesd10_l"],"CES-D-10","higher is worse","survey-specific scale","primary_four_domain","single baseline wave"),
    ("LASI","sensory","r1dsighta;r1hearcnde",["r1dsighta","r1hearcnde"],"self-reported vision and hearing","depends on label","ordinal","full_IC_sensitivity","label audit"),
]
cols = ["dataset","domain","candidate_variable","raw_variables","measurement_mode","direction","scale_or_unit","analysis_role","key_risk"]
ledger = pd.DataFrame(CANDIDATES, columns=cols)
ledger["raw_variables"] = ledger.raw_variables.map(lambda x: ";".join(x))
ledger.to_csv(TAB / "primary_domain_candidate_matrix.csv", index=False, encoding="utf-8-sig")

# Resolve a candidate expression into current-file columns and compute active-wave coverage.
def safe_numeric(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)

def resolve(path):
    return path

def active_long(df):
    return pd.Series(True, index=df.index)

def active_wide(df, active_col, year_col=None):
    if active_col and active_col in df:
        return safe_numeric(df[active_col]).eq(1)
    if year_col and year_col in df:
        return df[year_col].notna()
    return pd.Series(True, index=df.index)

specs = {
    "ELSA": {"layout":"long", "waves":[None], "id":"idauniqc", "wave_col":"wave", "active":None},
    "SHARE": {"layout":"long", "waves":[None], "id":"mergeid", "wave_col":"wave", "active":None},
    "KLoSA": {"layout":"long", "waves":[None], "id":"pid", "wave_col":"wave", "active":None},
    "MHAS": {"layout":"long", "waves":[None], "id":"rahhidnp", "wave_col":"wave", "active":None},
    "CHARLS": {"layout":"wide", "waves":[1,2,3,4], "id":"ID", "active":"inw{wave}"},
    "HRS": {"layout":"wide", "waves":[8,9,10,11,12,13], "id":"hhidpn", "active":"inw{wave}"},
    "LASI": {"layout":"wide", "waves":[1], "id":"hhid", "active_year":"r{wave}iwy"},
}

coverage_rows = []
for dataset, group in ledger.groupby("dataset", sort=False):
    path = FILES[dataset]
    spec = specs[dataset]
    # Expand all candidate raw variables for this dataset.
    raw_needed = set()
    for _, row in group.iterrows():
        for rv in row.raw_variables.split(";"):
            if not rv or rv == "not_available":
                continue
            if spec["layout"] == "wide":
                raw_needed.update(rv.format(wave=w) for w in spec["waves"])
            else:
                raw_needed.add(rv)
    if spec["layout"] == "wide":
        for w in spec["waves"]:
            raw_needed.add(spec["id"])
            raw_needed.add(spec["active"].format(wave=w) if spec.get("active") else spec.get("active_year","").format(wave=w))
    else:
        raw_needed.update([spec["id"], spec["wave_col"]])
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    available = set(meta.column_names)
    use = [c for c in raw_needed if c in available]
    df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    for _, row in group.iterrows():
        expr_vars = row.raw_variables.split(";") if row.raw_variables else []
        if row.candidate_variable == "not_available":
            coverage_rows.append({"dataset":dataset,"domain":row.domain,"candidate_variable":row.candidate_variable,"wave":"all","n_rows":len(df),"n_active":0,"n_observed":0,"pct_observed_active":0.0,"available_in_current_file":False})
            continue
        waves = [None] if spec["layout"] == "long" else spec["waves"]
        for w in waves:
            if spec["layout"] == "long":
                active = active_long(df)
                actual = expr_vars
                wave_label = "all_current_waves"
            else:
                active_col = spec["active"].format(wave=w) if spec.get("active") else ""
                year_col = spec.get("active_year","").format(wave=w) if spec.get("active_year") else ""
                active = active_wide(df, active_col, year_col)
                actual = [v.format(wave=w) for v in expr_vars]
                wave_label = w
            actual = [v for v in actual if v in df]
            if not actual:
                observed = pd.Series(False, index=df.index)
                available_flag = False
            elif len(actual) == 1:
                observed = safe_numeric(df[actual[0]]).notna()
                available_flag = True
            else:
                observed = pd.concat([safe_numeric(df[v]) for v in actual], axis=1).max(axis=1, skipna=True).notna()
                available_flag = True
            act_obs = observed.loc[active]
            coverage_rows.append({"dataset":dataset,"domain":row.domain,"candidate_variable":row.candidate_variable,"wave":wave_label,"n_rows":len(df),"n_active":int(active.sum()),"n_observed":int(act_obs.sum()),"pct_observed_active":round(float(act_obs.mean()*100),2) if len(act_obs) else np.nan,"available_in_current_file":available_flag})

cov = pd.DataFrame(coverage_rows)
cov.to_csv(TAB / "primary_domain_candidate_coverage_by_wave.csv", index=False, encoding="utf-8-sig")
overall = (cov.groupby(["dataset","domain","candidate_variable"], as_index=False)
    .agg(n_active=("n_active","sum"), n_observed=("n_observed","sum"), n_wave_records=("wave","nunique"), available_in_current_file=("available_in_current_file","max")))
overall["pct_observed_active"] = (overall["n_observed"] / overall["n_active"] * 100).round(2)
overall.to_csv(TAB / "primary_domain_candidate_coverage_overall.csv", index=False, encoding="utf-8-sig")

memo = """# Harmonization blueprint: primary four-domain IC

## Proposed primary object

The primary construct is a four-domain IC score using cognition, self-reported locomotion difficulty, grip/vitality, and psychological symptoms. The sensory domain is retained as a prespecified sensitivity extension where the current file has both vision and hearing candidates. Objective walking tests are not the primary locomotion indicator because they are module-subsample measures in most cohorts.

## What is still provisional

The candidate rows are not yet harmonized scores. Units, response direction, age/sex adjustment, proxy interviews, and wave-specific module selection must be frozen after reviewing `tables/candidate_value_label_audit.csv`. SHARE cognition is currently limited by the merged file; the original wave files should be remerged before any final multi-group model.

## Frailty design principle

A primary frailty indicator should be outcome-disjoint from the IC measurement model: exclude cognition, locomotion/function, grip/vitality, psychological symptoms, sensory items, ADL/IADL, and death. Candidate non-IC deficits include chronic diseases, BMI/weight-related deficits, falls, pain and self-rated health. A standard overlapping FI can be retained as a secondary convergent-validity benchmark.

## Files

- `tables/primary_domain_candidate_matrix.csv`: review ledger for all proposed indicators.
- `tables/primary_domain_candidate_coverage_by_wave.csv`: wave-level active-record coverage.
- `tables/primary_domain_candidate_coverage_overall.csv`: overall active-record coverage.
- `tables/existing_frailty_component_audit.csv`: overlap audit for existing ELSA/SHARE composites.
"""
(OUT_ROOT / "harmonization_blueprint.md").write_text(memo, encoding="utf-8")
print("wrote blueprint tables", len(ledger), len(cov))

