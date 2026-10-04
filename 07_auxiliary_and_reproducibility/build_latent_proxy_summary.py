from pathlib import Path
import pandas as pd

root=Path(r"PATH_TO_IC_FRAILTY")
lat=pd.read_csv(root/"tables/incident_disability_latent_factor_metrics.csv")
pro=pd.read_csv(root/"tables/incident_disability_prediction_metrics.csv")
rows=[]
for _,r in lat.iterrows():
    rows.append({"analysis":"primary_partial_metric_CFA","sample":r["sample"],"cohort":r["cohort"],"window":r["window"],"model":r["model"],"n":r["n"],"events":r["events"],"auc":r["auc_horizon_cindex"],"brier":r["brier"],"cal_intercept":r["calibration_intercept"],"cal_slope":r["calibration_slope"]})
for _,r in pro.iterrows():
    rows.append({"analysis":"proxy_sensitivity_external_transport","sample":r["sample"],"cohort":r["cohort"],"window":r["window"],"model":r["model"],"n":r["n"],"events":r["events"],"auc":r["auc_horizon_cindex"],"brier":r["brier"],"cal_intercept":r["calibration_intercept"],"cal_slope":r["calibration_slope"]})
pd.DataFrame(rows).to_csv(root/"tables/incident_disability_latent_vs_proxy_summary.csv",index=False,encoding="utf-8-sig")
text="""# Incident-disability criterion-validity summary

The primary estimand uses partial-metric CFA factor scores in ELSA, CHARLS and HRS. The observed-proxy model is a measurement sensitivity and supplies the fixed-coefficient SHARE transport because SHARE was kept outside the latent-mean comparison.

| Analysis | Cohort/window | Base AUC | IC AUC | Base Brier | IC Brier |
|---|---|---:|---:|---:|---:|
| CFA primary | ELSA 6→7 | 0.694 | 0.725 | 0.0923 | 0.0889 |
| CFA primary | CHARLS 3→4 | 0.726 | 0.726 | 0.1238 | 0.1237 |
| CFA primary | HRS 10→11 | 0.676 | 0.731 | 0.0853 | 0.0817 |
| CFA primary | pooled anchors | 0.695 | 0.711 | 0.1017 | 0.1006 |
| Proxy external | SHARE 6→7 | 0.722 | 0.763 | 0.0940 | 0.0901 |
| Proxy external | SHARE 6→8 | 0.717 | 0.753 | 0.1081 | 0.1034 |

Calibration for the SHARE 6→7 proxy IC model was acceptable (intercept 0.037, slope 1.064, observed/expected ratio 0.945). The CFA-score models were fitted in the development cohorts, so their near-zero calibration intercepts and slopes near one are development-cohort properties rather than external validation.
"""
(root/"incident_disability_latent_vs_proxy_summary.md").write_text(text,encoding="utf-8")
print(root/"incident_disability_latent_vs_proxy_summary.md")
