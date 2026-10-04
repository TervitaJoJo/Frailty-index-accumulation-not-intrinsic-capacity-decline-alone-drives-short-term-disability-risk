"""Build an aggregate manuscript-facing Stage 14 result table."""
from pathlib import Path
import pandas as pd

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

rel = pd.read_csv(TAB / "stage8_domain_reliability_discriminant.csv")
rel = rel[["dataset", "domain", "omega", "AVE", "factor_score_determinacy_approx", "pct_all_indicator_complete"]]

fit = pd.read_csv(TAB / "stage11_latent_domain_fi_sem_fit.csv")
fit = fit[fit.outcome.eq("fi_primary")][["dataset", "r2", "cfi", "rmsea", "srmr", "n_outcome_valid"]]

coef = pd.read_csv(TAB / "stage11_latent_domain_fi_sem_coefficients.csv")
coef = coef[coef.outcome.eq("fi_primary")][["dataset", "rhs", "capacity_oriented_std_all", "capacity_oriented_est"]]
coef = coef.rename(columns={"rhs": "domain", "capacity_oriented_std_all": "primary_capacity_std_beta", "capacity_oriented_est": "primary_capacity_slope"})

sens = pd.read_csv(TAB / "stage13_locomotion_sensitivity_span.csv")
sens = sens.rename(columns={"min_capacity_std": "locomotion_sensitivity_min_std_beta", "max_capacity_std": "locomotion_sensitivity_max_std_beta"})

main = rel.merge(fit, on="dataset", how="left").merge(coef, on=["dataset", "domain"], how="left")
main = main.merge(sens[["dataset", "locomotion_sensitivity_min_std_beta", "locomotion_sensitivity_max_std_beta"]], on="dataset", how="left")
main.to_csv(TAB / "stage14_exploratory_main_results.csv", index=False, encoding="utf-8-sig")

summary = fit.merge(sens, on="dataset", how="left")
summary.to_csv(TAB / "stage14_exploratory_cohort_summary.csv", index=False, encoding="utf-8-sig")

memo = [
    "# Stage 14 exploratory main-results integration",
    "",
    "This table integrates the Stage 8 domain reliability/discriminant-validity audit, Stage 11 latent-domain FI SEM, and Stage 13 locomotion sensitivity span. It is a manuscript-facing aggregate table; no new person-level computation is performed.",
    "",
    "The recommended primary narrative is: (1) a correlated four-domain IC structure is supported within ELSA, CHARLS, and HRS; (2) a single general IC factor is rejected by the Heywood diagnostics; (3) locomotion has the most stable independent association with outcome-disjoint FI; and (4) other domain coefficients retain cohort-specific uncertainty and should be presented as secondary structural evidence.",
    "",
    "The table is exploratory and unweighted. It should not be used to report cross-country latent means or causal effects before a final measurement-error and survey-design analysis."
]
(OUT / "stage14_exploratory_main_results_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
print("Wrote Stage 14 integrated exploratory main-results tables.")
