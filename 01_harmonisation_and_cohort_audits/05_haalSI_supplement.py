"""Limited HAALSI supplement: wave status and candidate-domain coverage."""
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

SRC = Path(r"PATH_TO_STATISTICAL_MODELING\HAALSI\Wave3\HAALSI W1-3 Longitudinal Dataset.dta")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

spec = {
    1: {"active": None, "cognition": ["w1cn014", "w1cn015", "w1cn016", "w1cn017"], "locomotion": ["w1c_pf_diff_walk"], "psychological": ["w1c_cd_cesd"], "proxy": ["w1rproxy"]},
    2: {"active": "w2status", "cognition": ["w2cn214", "w2c_cn_delrec", "w2im211"], "locomotion": ["w2c_pf_diff_walk"], "psychological": ["w2c_cd_cesd"], "proxy": ["w2rproxy"]},
    3: {"active": "w3status", "cognition": ["w3cn214", "w3c_cn_delrec", "w3im211"], "locomotion": ["w3c_pf_diff_walk"], "psychological": ["w3c_cd_cesd"], "proxy": ["w3rproxy"]},
}
cols = sorted({c for s in spec.values() for v in s.values() if isinstance(v, list) for c in v} | {s["active"] for s in spec.values() if s["active"]})
df, meta = pyreadstat.read_dta(str(SRC), usecols=cols, apply_value_formats=False)
rows = []
for wave, s in spec.items():
    active = pd.Series(True, index=df.index) if s["active"] is None else pd.to_numeric(df[s["active"]], errors="coerce").eq(1)
    active_n = int(active.sum())
    for domain, variables in s.items():
        if domain == "active":
            continue
        for var in variables:
            x = df[var].loc[active]
            if x.dtype == object:
                observed = x.notna() & x.astype(str).ne("")
            else:
                observed = pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90).notna()
            rows.append({"dataset":"HAALSI", "wave":wave, "domain":domain, "candidate_variable":var, "n_active":active_n, "n_observed":int(observed.sum()), "pct_observed_active":round(float(observed.mean()*100),2) if len(observed) else np.nan})
out = pd.DataFrame(rows)
out.to_csv(TAB / "haalSI_candidate_coverage.csv", index=False, encoding="utf-8-sig")
profile = pd.DataFrame([{"dataset":"HAALSI", "wave":w, "n_rows":len(df), "n_active":int((pd.Series(True,index=df.index) if s["active"] is None else pd.to_numeric(df[s["active"]],errors="coerce").eq(1)).sum())} for w,s in spec.items()])
profile.to_csv(TAB / "haalSI_wave_profile.csv", index=False, encoding="utf-8-sig")
text = """# HAALSI supplement\n\nHAALSI has three waves in the local file. Waves 2 and 3 have explicit completion/status flags; wave 1 is treated as the baseline file record. The supplement reports candidate cognitive, self-reported locomotion, CES-D and proxy-interview coverage only. A harmonized cognitive score was not constructed here because wave 1 and waves 2–3 use different raw representations. There is no selected grip candidate in this limited pass.\n\nSee `tables/haalSI_wave_profile.csv` and `tables/haalSI_candidate_coverage.csv`.\n"""
(OUT / "haalSI_supplement.md").write_text(text, encoding="utf-8")
print("wrote", TAB / "haalSI_candidate_coverage.csv")
