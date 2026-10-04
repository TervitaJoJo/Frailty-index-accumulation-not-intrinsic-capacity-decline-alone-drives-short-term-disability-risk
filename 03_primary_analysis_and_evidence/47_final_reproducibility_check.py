"""Final aggregate-output consistency checks for Stages 15-20."""
from pathlib import Path
import csv
import json
from datetime import datetime, timezone

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

def read(name):
    with (TAB / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def as_bool(x):
    return str(x).strip().lower() == "true"

checks = {}
fit15 = read("stage15_raw_invariance_fit.csv")
fit15b = read("stage15b_partial_scalar_fit.csv")
fit16 = read("stage16_cross_cohort_partial_metric_sem_fit.csv")
coef16 = read("stage16_cross_cohort_partial_metric_sem_coefficients.csv")
fit17 = read("stage17_cross_cohort_fi_sensitivity_fit.csv")
coef17 = read("stage17_cross_cohort_fi_sensitivity_coefficients.csv")
share_corr = read("stage20_share_external_spearman.csv")
share_desc = read("stage20_share_external_descriptives.csv")

models15 = {r["model"]: r for r in fit15}
checks["stage15_required_models"] = all(m in models15 for m in ["configural", "metric", "metric_partial_domain", "scalar_full", "scalar_partial_domain"])
checks["stage15_partial_metric_postcheck"] = as_bool(models15["metric_partial_domain"]["post_check"])
checks["stage15_scalar_partial_postcheck_flagged"] = not as_bool(models15["scalar_partial_domain"]["post_check"])
checks["stage15b_matches_main_scalar_fit"] = abs(float(models15["scalar_partial_domain"]["cfi"]) - float(next(r for r in fit15b if r["model"] == "scalar_partial_domain")["cfi"])) < 1e-8

checks["stage16_all_models_converged"] = all(as_bool(r["converged"]) for r in fit16)
checks["stage16_all_models_postcheck"] = all(as_bool(r["post_check"]) for r in fit16)
free16 = [r for r in coef16 if r["model"] == "partial_metric_free" and r["rhs"] == "locomotion"]
checks["stage16_locomotion_negative_all_cohorts"] = len(free16) == 3 and all(float(r["est"]) < 0 for r in free16)

checks["stage17_all_models_converged"] = all(as_bool(r["converged"]) for r in fit17)
checks["stage17_all_models_postcheck"] = all(as_bool(r["post_check"]) for r in fit17)
loc17 = [r for r in coef17 if r["rhs"] == "locomotion"]
checks["stage17_locomotion_negative_all_definitions_and_cohorts"] = len(loc17) == 9 and all(float(r["est"]) < 0 for r in loc17)
share_loc = next((r for r in share_corr if r["sample"] == "primary_full_component_composite" and r["construct_a"] == "locomotion_hierarchical" and r["construct_b"] == "fi_primary"), None)
share_joint = next((r for r in share_desc if r["construct"] == "joint_four_domain_fi_complete"), None)
checks["stage20_share_locomotion_fi_negative"] = share_loc is not None and float(share_loc["spearman"]) < 0
checks["stage20_share_joint_coverage_reported"] = share_joint is not None and float(share_joint["pct_valid"]) > 80

checked_files = [
    "stage15_raw_invariance_fit.csv", "stage15_raw_invariance_parameters.csv",
    "stage15_raw_invariance_lrt.csv", "stage15_raw_invariance_constraint_audit.csv",
    "stage15b_partial_scalar_fit.csv", "stage15b_partial_scalar_parameters.csv",
    "stage16_cross_cohort_partial_metric_sem_fit.csv", "stage16_cross_cohort_partial_metric_sem_coefficients.csv",
    "stage16_cross_cohort_partial_metric_sem_lrt.csv", "stage17_cross_cohort_fi_sensitivity_fit.csv",
    "stage17_cross_cohort_fi_sensitivity_coefficients.csv",
    "stage20_share_external_descriptives.csv", "stage20_share_external_spearman.csv",
    "stage20_share_external_country_coverage.csv"
]
id_like = []
for name in checked_files:
    rows = read(name)
    if rows:
        bad = [c for c in rows[0] if any(token in c.lower() for token in ["mergeid", "idauniq", "hhid", "personid", "respondent_id"])]
        if bad:
            id_like.append({"file": name, "columns": bad})
checks["no_person_identifier_columns_in_stage15_20_outputs"] = not id_like

result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "identifier_columns_found": id_like, "all_pass": all(checks.values())}
(OUT / "stage19_reproducibility_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
memo = [
    "# Stage 19 reproducibility and output-consistency check", "",
    f"Run time (UTC): {result['timestamp_utc']}", "",
    "The check verifies required model rows, post-check status, Stage 15/15b scalar agreement, stable negative locomotion signs in Stages 16, 17 and SHARE Stage 20, and absence of person-identifier columns in the aggregate outputs.", "",
    f"All checks passed: **{result['all_pass']}**.", "",
    "This is an output audit only. It does not replace final survey-weight, multiple-imputation, missing-data, or external validation analyses."
]
(OUT / "stage19_reproducibility_check.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
