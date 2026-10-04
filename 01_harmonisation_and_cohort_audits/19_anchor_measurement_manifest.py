"""Export the reviewable anchor-wave measurement manifest."""
from pathlib import Path
import pandas as pd

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

rows = [
    # ELSA wave 6
    ("ELSA",6,"cognition","imrc + dlrc + orient","continuous composite","higher_better","primary","source-audited observed tests; tcog_z_z sensitivity only"),
    ("ELSA",6,"locomotion","walkra","binary","higher_worse","primary","self-reported difficulty; reverse for IC"),
    ("ELSA",6,"grip_vitality","gripsum","continuous","higher_better","primary","nurse module; partial coverage"),
    ("ELSA",6,"psychological","cesd","0-8","higher_worse","primary","reverse for IC"),
    # SHARE wave 6
    ("SHARE",6,"cognition","cf objective word recall + date orientation","continuous composite","higher_better","primary","gv_imputations.memory excluded: matches self-rated cf103"),
    ("SHARE",6,"locomotion","walkra","binary","higher_worse","primary","reverse for IC"),
    ("SHARE",6,"grip_vitality","max(lgrip,rgrip)","continuous","higher_better","primary","partial nurse module"),
    ("SHARE",6,"psychological","eurod","0-12","higher_worse","primary","reverse for IC"),
    # CHARLS wave 3
    ("CHARLS",3,"cognition","r3orient + r3tr20","mixed","higher_better","primary","dual indicator domain"),
    ("CHARLS",3,"locomotion","r3walk100a","binary","higher_worse","primary","100-metre difficulty; sensitivity to walking wording"),
    ("CHARLS",3,"grip_vitality","max(r3lgrip,r3rgrip)","continuous","higher_better","primary","health examination module"),
    ("CHARLS",3,"psychological","r3cesd10","0-10","higher_worse","primary","reverse for IC"),
    # HRS wave 10
    ("HRS",10,"cognition","r10imrc + r10dlrc + r10ser7","continuous composite","higher_better","primary","serial-7 five items scaled by 5; cog27 sensitivity"),
    ("HRS",10,"locomotion","r10walkra","binary","higher_worse","primary","reverse for IC"),
    ("HRS",10,"grip_vitality","max(r10grpl,r10grpr)","continuous","higher_better","primary","partial nurse module; main missingness risk"),
    ("HRS",10,"psychological","r10cesd","scale","higher_worse","primary","reverse for IC"),
    # LASI wave 1 external validation
    ("LASI",1,"cognition","r1cog_total","continuous","higher_better","external validation","single wave"),
    ("LASI",1,"locomotion","r1walk100a","binary","higher_worse","external validation","100-metre difficulty"),
    ("LASI",1,"grip_vitality","max(r1lgrip,r1rgrip)","continuous","higher_better","external validation","high coverage"),
    ("LASI",1,"psychological","r1cesd10_l","scale","higher_worse","external validation","single wave"),
]
df = pd.DataFrame(rows, columns=["dataset","anchor_wave","domain","indicator_expression","scale","raw_direction","role","audit_note"])
df.to_csv(TAB / "anchor_measurement_manifest.csv", index=False, encoding="utf-8-sig")

frailty = []
for dataset, wave in [("ELSA",6),("SHARE",6),("CHARLS",3),("HRS",10),("LASI",1)]:
    for component in ["hypertension","diabetes","heart_disease","stroke","cancer","arthritis"]:
        frailty.append((dataset,wave,component,"binary disease present=1","primary; deficit if value=1"))
    frailty.append((dataset,wave,"self_rated_health","0-1 label-aware deficit","primary; direction mapped from value labels"))
    frailty.append((dataset,wave,"bmi","BMI-based deficit","pending; clinical threshold vs percentile vs excluded"))
pd.DataFrame(frailty, columns=["dataset","anchor_wave","component","planned_score","status"]).to_csv(TAB / "anchor_frailty_manifest.csv", index=False, encoding="utf-8-sig")

memo = """# Anchor measurement manifest

This file freezes the selected anchor waves and the candidate indicators for the primary four-domain IC comparison. It distinguishes primary candidates from external-validation and sensitivity candidates. The corresponding frailty manifest keeps BMI explicitly pending because the coding audit found substantial cross-cohort differences in measurement source and clinical-threshold prevalence.
"""
(OUT / "anchor_measurement_manifest.md").write_text(memo, encoding="utf-8")
print("wrote anchor measurement manifests")
