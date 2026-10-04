"""Exploratory pairwise comparison of latent-domain FI coefficients.

Uses Stage 11 primary SEM coefficients. Since FI is on the same 0-1 scale and
first-order latent variances were fixed to one, fitted unstandardized slopes
are compared in FI-per-latent-SD units. Pairwise CIs assume independent cohort
estimates and are descriptive only.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
src = TAB / "stage11_latent_domain_fi_sem_coefficients.csv"
df = pd.read_csv(src)
df = df[(df["outcome"] == "fi_primary") & df["rhs"].isin(["cognition", "locomotion", "grip_vitality", "psychological"])].copy()
sign = {"cognition": 1.0, "locomotion": -1.0, "grip_vitality": 1.0, "psychological": -1.0}
df["capacity_est"] = df["est"] * df["rhs"].map(sign)
df["capacity_se"] = df["se"]
rows = []
for domain, g in df.groupby("rhs", sort=False):
    g = g.set_index("dataset")
    for a, b in combinations(g.index.tolist(), 2):
        diff = float(g.loc[a, "capacity_est"] - g.loc[b, "capacity_est"])
        se = float(np.sqrt(g.loc[a, "capacity_se"] ** 2 + g.loc[b, "capacity_se"] ** 2))
        rows.append({"domain": domain, "cohort_a": a, "cohort_b": b,
                     "capacity_est_a": float(g.loc[a, "capacity_est"]),
                     "capacity_est_b": float(g.loc[b, "capacity_est"]),
                     "difference_a_minus_b": diff, "approx_se": se,
                     "ci95_low": diff - 1.96 * se, "ci95_high": diff + 1.96 * se,
                     "comparison_note": "descriptive independent-estimate approximation"})
pairwise = pd.DataFrame(rows)
pairwise.to_csv(TAB / "stage13_cross_cohort_coefficient_comparisons.csv", index=False, encoding="utf-8-sig")

# Sensitivity span for locomotion from Stage 12.
sens = pd.read_csv(TAB / "stage12_structural_missingness_coefficients.csv")
sens = sens[sens["rhs"].eq("locomotion")].copy()
span = sens.groupby("dataset")["capacity_oriented_std_all"].agg(["min", "max", "mean", "count"]).reset_index()
span.columns = ["dataset", "min_capacity_std", "max_capacity_std", "mean_capacity_std", "n_variants"]
span.to_csv(TAB / "stage13_locomotion_sensitivity_span.csv", index=False, encoding="utf-8-sig")

memo = [
    "# Stage 13 cross-cohort coefficient comparison",
    "",
    "The comparison uses Stage 11 primary latent-domain SEM coefficients. FI shares the same 0–1 scale across cohorts and first-order latent variances were fixed to one, so unstandardized slopes are compared as FI units per latent-domain SD. Pairwise intervals assume independent cohort estimates and are descriptive; they do not account for shared harmonization decisions, sampling weights, or cross-cohort measurement uncertainty.",
    "",
    "Locomotion sensitivity spans are reported separately from the Stage 12 primary, complete-FI, complete-grip, and omit-grip variants. A narrow span supports structural robustness; a broad span flags a domain for measurement-error and missingness discussion.",
    "",
    "Outputs: `tables/stage13_cross_cohort_coefficient_comparisons.csv` and `tables/stage13_locomotion_sensitivity_span.csv`."
]
(OUT / "stage13_cross_cohort_comparison_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
print("Wrote Stage 13 cross-cohort coefficient comparison outputs.")
