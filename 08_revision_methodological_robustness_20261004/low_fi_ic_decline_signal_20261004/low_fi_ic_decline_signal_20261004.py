"""Low-baseline-FI test of the IC-decline-only early-risk signal.

Aggregate-only analysis for the frozen primary directional-state design.
The main estimand is disability risk among participants with baseline FI <0.125,
comparing IC decline only with preserved/low accumulation. Secondary contrasts
cover FI accumulation only, coupled change and coupled versus FI-only. A pooled
state-by-baseline-FI-band interaction tests whether state associations differ
across the three descriptive FI bands.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

PKG = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG))
import trajectory_states_analysis_20260930 as ts  # noqa: E402

SEED = 20261004
Z = norm.ppf(0.975)
STATES = ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"]
BANDS = ["<0.125", "0.125-<0.25", ">=0.25"]


def wilson(events, total):
    if total <= 0:
        return np.nan, np.nan
    p = events / total
    den = 1 + Z * Z / total
    centre = (p + Z * Z / (2 * total)) / den
    half = Z * np.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total)) / den
    return float(max(0, centre - half)), float(min(1, centre + half))


def contrast(fit, terms):
    beta = fit.params.loc[terms].to_numpy(dtype=float)
    cov = fit.cov_params().loc[terms, terms].to_numpy(dtype=float)
    one = np.ones(len(terms))
    est = float(one @ beta)
    se = float(np.sqrt(one @ cov @ one))
    return {
        "OR": float(np.exp(est)),
        "OR_low": float(np.exp(est - Z * se)),
        "OR_high": float(np.exp(est + Z * se)),
        "p": float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan,
    }


def term(state):
    return f"C(state, Treatment(reference='preserved_low'))[T.{state}]"


def load_primary():
    frames = []
    for cohort in ["ELSA", "CHARLS", "HRS"]:
        raw, _ = ts.load_cohort(cohort)
        z, _ = ts.prepare_analysis(raw)
        z = ts.assign_states(z, "primary_sign")
        z["event"] = z.outcome_category.eq("disability").astype(int)
        frames.append(z)
    q = pd.concat(frames, ignore_index=True)
    q["baseline_fi_band"] = pd.cut(
        q["baseline_fi"], bins=[-0.001, 0.125, 0.25, 1.001],
        labels=BANDS, right=False,
    )
    return q


def risk_table(q):
    rows = []
    for population, g in [("Pooled", q), *list(q.groupby("cohort", sort=True))]:
        for band, b in g.groupby("baseline_fi_band", observed=True, sort=False):
            for state, h in b.groupby("state", sort=False):
                known = h.outcome_category.ne("unknown")
                den = int(known.sum())
                events = int((h.event.eq(1) & known).sum())
                lo, hi = wilson(events, den)
                rows.append({
                    "population": population,
                    "baseline_fi_band": str(band),
                    "state": state,
                    "trajectory_n": int(len(h)),
                    "known_outcome_n": den,
                    "disability_events": events,
                    "disability_risk": events / den if den else np.nan,
                    "risk_low": lo,
                    "risk_high": hi,
                })
    pd.DataFrame(rows).to_csv(OUT / "low_fi_state_risks_all_bands.csv", index=False)


def fit_low_fi(q):
    g = q.loc[q.baseline_fi_band.eq("<0.125") & q.outcome_category.ne("unknown")].copy()
    keep = ["event", "state", "age_baseline", "male", "education", "baseline_fi", "cohort"]
    g = g.dropna(subset=keep)
    rhs = "event ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi + C(cohort)"
    fit = smf.glm(rhs, data=g, family=sm.families.Binomial()).fit(cov_type="HC3")
    rows = []
    for state in STATES[1:]:
        c = contrast(fit, [term(state)])
        rows.append({"analysis": "low_fi_pooled_state_model", "baseline_fi_band": "<0.125",
                     "contrast": f"{state} vs preserved_low", "n": int(len(g)),
                     "events": int(g.event.sum()), **c, "formula": rhs})
    c = contrast(fit, [term("coupled"), term("fi_accumulation_only")])
    rows.append({"analysis": "low_fi_pooled_state_model", "baseline_fi_band": "<0.125",
                 "contrast": "coupled vs fi_accumulation_only", "n": int(len(g)),
                 "events": int(g.event.sum()), **c, "formula": rhs})
    pd.DataFrame(rows).to_csv(OUT / "low_fi_pooled_state_models.csv", index=False)
    cohort_rows = []
    for cohort, cg in g.groupby("cohort", sort=True):
        if cg.state.nunique() < 2 or cg.event.nunique() < 2:
            continue
        rhs_c = "event ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi"
        fc = smf.glm(rhs_c, data=cg, family=sm.families.Binomial()).fit(cov_type="HC3")
        for state in STATES[1:]:
            t = term(state)
            if t in fc.params.index:
                c = contrast(fc, [t])
                cohort_rows.append({"cohort": cohort, "baseline_fi_band": "<0.125",
                                    "contrast": f"{state} vs preserved_low", "n": int(len(cg)),
                                    "events": int(cg.event.sum()), **c, "formula": rhs_c})
    pd.DataFrame(cohort_rows).to_csv(OUT / "low_fi_cohort_state_models.csv", index=False)
    return g


def fit_band_interaction(q):
    g = q.loc[q.outcome_category.ne("unknown")].copy()
    keep = ["event", "state", "baseline_fi_band", "age_baseline", "male", "education", "baseline_fi", "cohort"]
    g = g.dropna(subset=keep)
    rhs_main = "event ~ C(state, Treatment(reference='preserved_low')) + C(baseline_fi_band) + age_baseline + male + education + baseline_fi + C(cohort)"
    rhs_full = "event ~ C(state, Treatment(reference='preserved_low')) * C(baseline_fi_band) + age_baseline + male + education + baseline_fi + C(cohort)"
    plain_main = smf.glm(rhs_main, data=g, family=sm.families.Binomial()).fit()
    plain_full = smf.glm(rhs_full, data=g, family=sm.families.Binomial()).fit()
    lr = 2 * (plain_full.llf - plain_main.llf)
    df = int(plain_full.df_model - plain_main.df_model)
    out = [{"analysis": "state_by_baseline_fi_band_interaction", "n": int(len(g)),
            "events": int(g.event.sum()), "likelihood_ratio": float(lr),
            "df": df, "p": float(chi2.sf(lr, df)), "main_formula": rhs_main,
            "full_formula": rhs_full}]
    pd.DataFrame(out).to_csv(OUT / "state_by_fi_band_interaction.csv", index=False)
    robust = smf.glm(rhs_full, data=g, family=sm.families.Binomial()).fit(cov_type="HC3")
    rows = []
    for state in STATES[1:]:
        base = term(state)
        for band in BANDS[1:]:
            inter = f"{base}:C(baseline_fi_band)[T.{band}]"
            if base in robust.params.index and inter in robust.params.index:
                c = contrast(robust, [base, inter])
                rows.append({"baseline_fi_band": band, "contrast": f"{state} vs preserved_low",
                             "n": int(len(g)), "events": int(g.event.sum()), **c})
    pd.DataFrame(rows).to_csv(OUT / "state_by_fi_band_stratum_contrasts.csv", index=False)


def main():
    q = load_primary()
    risk_table(q)
    low = fit_low_fi(q)
    fit_band_interaction(q)
    run = {
        "script": Path(__file__).name,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "primary_low_fi_threshold": 0.125,
        "cohorts": ["ELSA", "CHARLS", "HRS"],
        "n_low_fi_known_outcome": int(len(low)),
        "events_low_fi": int(low.event.sum()),
        "aggregate_only": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "run_info.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
