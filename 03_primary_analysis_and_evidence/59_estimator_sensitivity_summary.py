from pathlib import Path
import pandas as pd

ROOT = Path(r"PATH_TO_IC_FRAILTY")
TAB = ROOT / "tables"

w = pd.read_csv(TAB / "stage26_latent_longitudinal_fit.csv")
m = pd.read_csv(TAB / "stage28_latent_longitudinal_mlr_fit.csv")
keep = ["dataset", "window", "cfi", "rmsea", "srmr", "r2_full", "r2_base", "delta_r2"]
w = w[keep].rename(columns={c: f"wlsmv_{c}" for c in keep if c not in ["dataset", "window"]})
m = m[keep].rename(columns={c: f"mlr_{c}" for c in keep if c not in ["dataset", "window"]})
out = w.merge(m, on=["dataset", "window"], how="outer")
out["delta_r2_difference_mlr_minus_wlsmv"] = out["mlr_delta_r2"] - out["wlsmv_delta_r2"]
out.to_csv(TAB / "stage28_estimator_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

memo = """# Stage 28 estimator sensitivity summary

This table pairs the primary WLSMV/ordinal-indicator results with the robust MLR/FIML sensitivity model. The MLR model treats ordinal indicators as approximately continuous. It has poorer fit and much smaller incremental R-squared in all cohorts, supporting WLSMV as the primary estimator and defining the longitudinal result as estimator-sensitive rather than universally transportable.
"""
(ROOT / "stage28_estimator_sensitivity_summary.md").write_text(memo, encoding="utf-8")
print(out.to_string(index=False))
