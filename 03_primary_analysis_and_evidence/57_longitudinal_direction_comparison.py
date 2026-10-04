from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"PATH_TO_IC_FRAILTY")
TAB = ROOT / "tables"

latent = pd.read_csv(TAB / "stage26_latent_longitudinal_coefficients.csv")
latent = latent[latent["rhs"].isin(["cognition", "locomotion", "grip_vitality", "psychological"])].copy()
latent = latent.rename(columns={"rhs": "domain", "capacity_oriented_std_all": "latent_std"})
latent = latent[["dataset", "window", "domain", "latent_std", "pvalue"]]

proxy = pd.read_csv(TAB / "stage25_longitudinal_predictive_main_window.csv")
proxy = proxy[(proxy["model"] == "future_fi_adjusted_baseline") & proxy["term"].str.endswith("_base")].copy()
proxy["domain"] = proxy["term"].str.replace("_base", "", regex=False)
proxy = proxy[proxy["dataset"].isin(["ELSA", "CHARLS", "HRS"])].copy()
proxy = proxy.rename(columns={"std_beta": "proxy_std", "p_hc3": "proxy_p"})
proxy = proxy[["dataset", "window", "domain", "proxy_std", "proxy_p"]]

out = latent.merge(proxy, on=["dataset", "window", "domain"], how="left")
out["same_direction"] = np.where(out["latent_std"].notna() & out["proxy_std"].notna(), np.sign(out["latent_std"]) == np.sign(out["proxy_std"]), np.nan)
out["absolute_difference"] = (out["latent_std"] - out["proxy_std"]).abs()
out.to_csv(TAB / "stage26_vs_stage25_direction_comparison.csv", index=False, encoding="utf-8-sig")

summary = (out.groupby("domain", as_index=False)
           .agg(n_cohort_comparisons=("same_direction", "count"),
                n_same_direction=("same_direction", lambda x: int(np.nansum(x.astype(float)))),
                mean_abs_difference=("absolute_difference", "mean")))
summary["proportion_same_direction"] = summary["n_same_direction"] / summary["n_cohort_comparisons"]
summary.to_csv(TAB / "stage26_vs_stage25_direction_summary.csv", index=False, encoding="utf-8-sig")

memo = """# Stage 26 versus Stage 25 longitudinal direction comparison

This aggregate table compares capacity-oriented standardized coefficients from the baseline latent SEM (Stage 26) with the complete-case observed-proxy HC3 regression (Stage 25) for ELSA 6->7, CHARLS 3->4, and HRS 10->11. It is a concordance diagnostic, not a pooled estimate. A direction mismatch is evidence of model- or module-dependent structural heterogeneity and should not be resolved by selecting the more favorable model.

Stage 26 uses pairwise WLSMV and four common psychological indicators; Stage 25 uses prespecified observed composites and complete cases. CHARLS latent standard errors are not stable under the WLSMV information matrix, so this comparison emphasizes standardized direction and magnitude rather than latent-model p-values.
"""
(ROOT / "stage26_vs_stage25_direction_comparison_memo.md").write_text(memo, encoding="utf-8")
print(out.to_string(index=False))
print(summary.to_string(index=False))
