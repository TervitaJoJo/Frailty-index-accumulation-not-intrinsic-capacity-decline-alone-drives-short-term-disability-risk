"""Write the provisional, reviewable analysis manifest.

This manifest freezes only choices already supported by the exploratory audit
and the user's selected denominator rule. It intentionally leaves detailed
cut-points, proxy handling, and the SHARE wave-7 status marked for review.
"""
from pathlib import Path
import json
import pandas as pd

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

manifest = {
    "analysis_title": "Cross-national latent construct comparison of intrinsic capacity and outcome-disjoint frailty",
    "status": "exploratory_manifest_v1",
    "primary_cohorts": ["ELSA", "SHARE", "CHARLS", "HRS"],
    "external_validation_cohort": ["LASI"],
    "supplementary_cohorts": ["KLoSA", "MHAS", "HAALSI"],
    "anchor_waves": {"ELSA": 6, "SHARE": 6, "CHARLS": 3, "HRS": 10, "LASI": 1},
    "primary_comparison_design": "anchor_wave_multigroup_latent_comparison",
    "primary_wave_scope_user_selected": True,
    "longitudinal_extension": "all_available_waves_as_sensitivity_or_secondary_analysis_after_module_missingness_audit",
    "ic_primary_domains": [
        {"domain": "cognition", "role": "primary", "measurement": "cohort-specific latent domain; at least two indicators where available"},
        {"domain": "locomotion", "role": "primary", "measurement": "self-reported locomotion difficulty"},
        {"domain": "grip_vitality", "role": "primary", "measurement": "grip strength or vitality indicator; objective module retained for sensitivity"},
        {"domain": "psychological", "role": "primary", "measurement": "cohort-specific depressive symptom scale"},
        {"domain": "sensory", "role": "sensitivity", "measurement": "vision/hearing where both domains are auditable"},
    ],
    "frailty_primary": {
        "construct": "outcome_disjoint_health_deficit_index",
        "components": ["hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis", "self_rated_health", "bmi"],
        "denominator_rule": "minimum 6 of 8 observed; FI denominator equals the number of observed components",
        "complete_8_sensitivity": True,
        "exclude_from_primary": ["cognition", "locomotion", "grip_vitality", "psychological", "sensory", "ADL_IADL", "death"],
        "cutpoints_status": "primary BMI deficit fixed at BMI<18.5 or BMI>=30; cohort-percentile BMI sensitivity (user selected)",
        "self_rated_health_rule": "label-aware 0-1 deficit; poor-is-low cohorts reversed, poor-is-high cohorts mapped directly",
    },
    "share_cognition": {
        "source": "raw gv_imputations",
        "retain_implicat": True,
        "never_stack_as_independent_rows": True,
        "wave_7_status": "excluded_from_primary_longitudinal_cognition; sensitivity_only",
    },
    "not_yet_frozen": [
        "SHARE wave-7 sensitivity-only treatment (user selected)",
        "proxy interview and module-subsample rules",
        "survey weights and MI-compatible estimator",
        "measurement invariance/alignment model specification",
    ],
}
(OUT / "analysis_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

rows = []
for d in manifest["ic_primary_domains"]:
    rows.append({"section": "IC", "name": d["domain"], "role": d["role"], "definition": d["measurement"], "status": "provisional"})
for c in manifest["frailty_primary"]["components"]:
    rows.append({"section": "frailty", "name": c, "role": "primary", "definition": "outcome-disjoint component", "status": "component frozen; cut-point pending"})
rows.extend([
    {"section": "frailty", "name": "denominator", "role": "primary", "definition": "minimum 6 of 8; observed-item denominator", "status": "user selected"},
    {"section": "SHARE", "name": "wave_7", "role": "cognition", "definition": "flag-aware structural missingness", "status": "pending user decision"},
])
pd.DataFrame(rows).to_csv(TAB / "analysis_manifest_v1.csv", index=False, encoding="utf-8-sig")
print("wrote", OUT / "analysis_manifest_v1.json")
