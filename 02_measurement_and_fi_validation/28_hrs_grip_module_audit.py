"""Aggregate audit of HRS wave-10 grip-module selection."""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
path = ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
cols = ["inw10", "r10agey_m", "ragender", "r10proxy", "r10grpdom", "r10grpl", "r10grpr", "r10imrc", "r10dlrc", "r10ser7", "r10bwc20", "r10walkra", "r10walk1a", "r10walksa", "r10depres", "r10effort", "r10sleepr", "r10whappy", "r10flone", "r10fsad", "r10going", "r10enlife"]
df, meta = pyreadstat.read_dta(str(path), usecols=cols, apply_value_formats=False)
df = df.loc[df["inw10"].eq(1)].copy()
for c in [c for c in cols if c.startswith("r10") and c not in ["r10agey_m", "r10proxy", "r10grpdom"]]:
    x = pd.to_numeric(df[c], errors="coerce")
    df[c] = x.where(x.between(0, 100))
df["grip_n"] = df[["r10grpl", "r10grpr"]].notna().sum(axis=1)
df["grip_status"] = pd.cut(df["grip_n"], bins=[-1, 0, 1, 2], labels=["none", "one", "both"])
df["age_group"] = pd.cut(pd.to_numeric(df["r10agey_m"], errors="coerce"), bins=[-1, 64, 74, 84, 200], labels=["<65", "65-74", "75-84", "85+"])
df["sex_group"] = df["ragender"].map({1: "male", 2: "female"})
df["proxy_group"] = df["r10proxy"].map({0: "non_proxy", 1: "proxy"})

def summarize(data, group_vars, label):
    if not group_vars:
        grouped = [((), data)]
    else:
        grouped = data.groupby(group_vars, dropna=False, observed=False)
    rows = []
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {"stratum": label, "n": len(g), "pct_both": 100 * g["grip_n"].eq(2).mean(),
               "pct_any": 100 * g["grip_n"].ge(1).mean(), "pct_none": 100 * g["grip_n"].eq(0).mean(),
               "mean_age": pd.to_numeric(g["r10agey_m"], errors="coerce").mean()}
        for name, value in zip(group_vars, key):
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)

overall = summarize(df, [], "overall")
by_age = summarize(df, ["age_group"], "age")
by_sex = summarize(df, ["sex_group"], "sex")
by_proxy = summarize(df, ["proxy_group"], "proxy")
by_age_sex = summarize(df, ["age_group", "sex_group"], "age_by_sex")
summary = pd.concat([overall, by_age, by_sex, by_proxy, by_age_sex], ignore_index=True)
TAB.mkdir(parents=True, exist_ok=True)
summary.to_csv(TAB / "hrs_grip_module_selection_summary.csv", index=False, encoding="utf-8-sig")

status = df["grip_status"].value_counts(dropna=False).rename_axis("grip_status").reset_index(name="n")
status["pct"] = 100 * status["n"] / len(df)
status.to_csv(TAB / "hrs_grip_module_status.csv", index=False, encoding="utf-8-sig")
# Domain-module availability by proxy status. Proxy respondents have no valid
# cognitive or psychological battery in this wave and cannot contribute to a
# four-domain IC measurement model.
domains = {
    "cognition": ["r10imrc", "r10dlrc", "r10ser7", "r10bwc20"],
    "locomotion": ["r10walkra", "r10walk1a", "r10walksa"],
    "grip_vitality": ["r10grpl", "r10grpr"],
    "psychological": ["r10depres", "r10effort", "r10sleepr", "r10whappy", "r10flone", "r10fsad", "r10going", "r10enlife"],
}
proxy_rows = []
for proxy, g in df.groupby("r10proxy", dropna=False, observed=False):
    row = {"proxy": proxy, "n": len(g)}
    for domain, vars_ in domains.items():
        row[domain + "_at_least_2_pct"] = 100 * (g[vars_].notna().sum(axis=1) >= 2).mean()
    all_ok = pd.Series(True, index=g.index)
    for vars_ in domains.values():
        all_ok &= g[vars_].notna().sum(axis=1).ge(2)
    row["all_four_domains_pct"] = 100 * all_ok.mean()
    proxy_rows.append(row)
proxy_summary = pd.DataFrame(proxy_rows)
proxy_summary.to_csv(TAB / "hrs_domain_availability_by_proxy.csv", index=False, encoding="utf-8-sig")


memo = """# HRS grip-module selection audit

This aggregate audit uses active wave-10 HRS respondents (`inw10==1`) and does not write identifiers or person-level derived files. Grip availability is defined from valid left/right grip values; survey-specific negative/special codes are treated as missing.

The audit reports availability overall and by age group, sex, proxy interview status, and age-by-sex strata. Because grip testing is a selected physical-measurement module, the availability pattern must be carried into the missing-data model. The primary HRS CFA may use the available grip records, but latent-score comparisons should report a structural-module sensitivity using the maximum grip summary or a model that conditions on module selection.

Proxy respondents have 0% availability for the cognition and psychological batteries in this wave, so the primary HRS IC measurement sample should be restricted to non-proxy respondents; proxies can remain in frailty descriptives or a separate sensitivity.

See `tables/hrs_grip_module_selection_summary.csv`, `tables/hrs_grip_module_status.csv`, and `tables/hrs_domain_availability_by_proxy.csv`.
"""
(OUT / "stage7_hrs_grip_module_audit.md").write_text(memo, encoding="utf-8")
(OUT / "stage7_hrs_grip_module_run_info.json").write_text(json.dumps({"cohort": "HRS", "wave": 10, "active_rule": "inw10==1", "scope": "aggregate-only", "special_codes": "non-numeric/special values treated as missing"}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote HRS grip-module audit")
