"""Aggregate-only IC--frailty longitudinal state analysis.

This script is an extension audit and deliberately keeps participant-level data
in memory. It writes only aggregate counts, risks, model coefficients and a
method log. The primary analysis uses ELSA 6-8-9, CHARLS 1-3-4 and HRS 8-10-12.
HAALSI is handled separately because its mortality linkage has not passed the
same endpoint audit gate.
"""
from pathlib import Path
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import beta
import statsmodels.api as sm
import statsmodels.formula.api as smf

PKG = Path(os.environ.get("IC_FRAILTY_V2_ROOT", "PATH_TO_IC_FRAILTY_V2"))
AUDIT = PKG / "analysis_audit"
OUT = AUDIT / "trajectory_states_20260930"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_data import META, DISEASES, COMMON, predictors, items_outcome, numeric  # noqa: E402

SEED = 20260930
DOMAINS = ["cognition", "locomotion", "grip_vitality", "psychological"]
# A harmonised eight-item endpoint is used in all three cohorts. ELSA wave 9
# does not release the money-management item, so retaining it would turn the
# outcome into structural missingness at the final wave.
ENDPOINT_ITEMS = [x for x in COMMON if x != "moneya"]
COHORT_SPEC = {
    "ELSA": {"id": "idauniqc", "cluster": "hhidc", "base": 6, "intermediate": 8, "outcome": 9,
             "waves": [6, 8, 9], "alive": [0], "death": [1], "proxy": None},
    "CHARLS": {"id": "ID", "cluster": "communityID", "base": 1, "intermediate": 3, "outcome": 4,
               "waves": [1, 3, 4], "alive": [1, 4], "death": [5, 6], "proxy": None},
    "HRS": {"id": "hhidpn", "cluster": "hhid", "base": 8, "intermediate": 10, "outcome": 12,
            "waves": [8, 10, 12], "alive": [1, 4], "death": [2, 3, 5, 6], "proxy": "proxy"},
}

def avg(vals, minimum=1):
    z = pd.concat(vals, axis=1)
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= minimum)

def required_columns(cohort):
    s = COHORT_SPEC[cohort]
    cols = [s["id"], s["cluster"], "ragender"]
    if cohort == "ELSA":
        cols += ["wave", "inw", "iwstat", "raeducl"]
    elif cohort == "CHARLS":
        cols += ["raeduc_c"]
    else:
        cols += ["raeduc"]
    core = [
        "agey_m" if cohort == "HRS" else "agey", "shlt", "shlta",
        "mbmi" if cohort in ["ELSA", "CHARLS"] else "bmi",
        "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge",
        "imrc", "dlrc", "ser7", "orient", "tr20", "walkra", "walk1a", "walksa",
        "walk100a", "tcog_z_z", "gripsum", "grpl", "grpr", "lgrip", "rgrip",
        "cesd", "cesd10", *COMMON,
    ]
    for w in s["waves"]:
        p = "" if cohort == "ELSA" else f"r{w}"
        for v in core:
            q = p + v
            if q in META[cohort]["columns"]:
                cols.append(q)
        if cohort != "ELSA":
            cols.append(f"r{w}iwstat")
            if cohort == "HRS":
                cols.append(f"r{w}proxy")
    return list(dict.fromkeys([c for c in cols if c in META[cohort]["columns"]]))

def make_wave(sub, cohort, wave):
    s = COHORT_SPEC[cohort]
    p = "" if cohort == "ELSA" else f"r{wave}"
    d = sub.reset_index(drop=True)
    x = predictors(d, cohort, p)
    x["disability"] = items_outcome(d, p, cohort, ENDPOINT_ITEMS, len(ENDPOINT_ITEMS))
    x["status"] = numeric(d["iwstat" if cohort == "ELSA" else f"r{wave}iwstat"])
    x["present"] = True
    x["proxy"] = numeric(d[f"r{wave}proxy"]).eq(0) if cohort == "HRS" else True
    x["id"] = d[s["id"]].astype(str).values
    x["cluster_raw"] = d[s["cluster"]].astype(str).values
    x["wave"] = wave
    return x

def load_cohort(cohort):
    s = COHORT_SPEC[cohort]
    d = pyreadstat.read_dta(META[cohort]["path"], usecols=required_columns(cohort))[0]
    if cohort == "CHARLS":
        # CHARLS wave 4 stores self-rated health under shlta in the harmonised
        # file; predictors() uses the canonical shlt name.
        for w in COHORT_SPEC[cohort]["waves"]:
            if f"r{w}shlt" not in d.columns and f"r{w}shlta" in d.columns:
                d[f"r{w}shlt"] = d[f"r{w}shlta"]
            if f"r{w}mbmi" not in d.columns:
                # BMI is not released for CHARLS wave 4 in this harmonised
                # file; retain it as an unobserved FI component at that wave.
                d[f"r{w}mbmi"] = np.nan
            for v in ["lgrip", "rgrip"]:
                if f"r{w}{v}" not in d.columns:
                    d[f"r{w}{v}"] = np.nan
    if cohort == "ELSA":
        d = d.loc[numeric(d["wave"]).isin(s["waves"]) & numeric(d["inw"]).eq(1)].copy()
        d = d.sort_values([s["id"], "wave"]).drop_duplicates([s["id"], "wave"], keep="last")
    frames = {}
    for w in s["waves"]:
        sub = d.loc[numeric(d["wave"]).eq(w)].copy() if cohort == "ELSA" else d.copy()
        frames[w] = make_wave(sub, cohort, w)
    out = frames[s["base"]].copy()
    for w in [s["intermediate"], s["outcome"]]:
        z = frames[w].copy()
        keep = ["id", "status", "present", "proxy", "disability", "age", "baseline_fi", *DOMAINS]
        z = z[keep].rename(columns={c: f"{c}_{w}" for c in keep if c != "id"})
        out = out.merge(z, on="id", how="left", validate="one_to_one")
    b = s["base"]
    out = out.rename(columns={"status": f"status_{b}", "proxy": f"proxy_{b}",
                              "disability": f"disability_{b}", "age": f"age_{b}"})
    out["cohort"] = cohort
    base_ok = out[f"age_{b}"].ge(50) & out[f"proxy_{b}"]
    pars = []
    for dom in DOMAINS:
        mu = out.loc[base_ok, dom].mean()
        sd = out.loc[base_ok, dom].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out[f"{dom}_z_{b}"] = (out[dom] - mu) / sd
        for w in [s["intermediate"], s["outcome"]]:
            out[f"{dom}_z_{w}"] = (out[f"{dom}_{w}"] - mu) / sd
        pars.append({"cohort": cohort, "variable": dom, "baseline_wave": b,
                     "mean": float(mu), "sd": float(sd), "reference_n": int(base_ok.sum())})
    i, o = s["intermediate"], s["outcome"]
    out["delta_ic"] = out[[f"{d}_z_{i}" for d in DOMAINS]].mean(axis=1) - out[[f"{d}_z_{b}" for d in DOMAINS]].mean(axis=1)
    out["delta_fi"] = out[f"baseline_fi_{i}"] - out["baseline_fi"]
    out["alive_base"] = out[f"status_{b}"].isin(s["alive"])
    out["alive_intermediate"] = out[f"status_{i}"].isin(s["alive"])
    out["death_intermediate"] = out[f"status_{i}"].isin(s["death"])
    out["alive_outcome"] = out[f"status_{o}"].isin(s["alive"])
    out["death_outcome"] = out[f"status_{o}"].isin(s["death"])
    return out, pd.DataFrame(pars)

def wilson(k, n):
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0, ctr-half), min(1, ctr+half)

def prepare_analysis(d):
    cohort = d.cohort.iloc[0]
    s = COHORT_SPEC[cohort]
    b, i, o = s["base"], s["intermediate"], s["outcome"]
    m = (
        d[f"age_{b}"].ge(50) & d["alive_base"] & d[f"proxy_{b}"] &
        d[f"disability_{b}"].eq(0) & d[f"disability_{b}"].notna() &
        d["alive_intermediate"] & d[f"proxy_{i}"] &
        d[f"disability_{i}"].eq(0) & d[f"disability_{i}"].notna() &
        d["delta_ic"].notna() & d["delta_fi"].notna() &
        d["baseline_fi"].notna() & d[f"baseline_fi_{i}"].notna()
    )
    z = d.loc[m].copy()
    z["age_baseline"] = z[f"age_{b}"]
    z["outcome_category"] = np.select(
        [z["death_outcome"],
         z["alive_outcome"] & z[f"proxy_{o}"] & z[f"disability_{o}"].eq(1),
         z["alive_outcome"] & z[f"proxy_{o}"] & z[f"disability_{o}"].eq(0)],
        ["death", "disability", "disability_free"], default="unknown")
    return z, m

def assign_states(z, definition):
    o = z.copy()
    if definition == "primary_sign":
        ic_cut, fi_cut = 0.0, 0.0
        rule = "delta_IC<0 and delta_FI>0"
    elif definition == "cohort_median":
        ic_cut, fi_cut = o["delta_ic"].median(), o["delta_fi"].median()
        rule = "within-cohort medians"
    else:
        k = 0.2 if definition.endswith("0.2SD") else 0.5
        ic_cut, fi_cut = -k * o["delta_ic"].std(ddof=1), k * o["delta_fi"].std(ddof=1)
        rule = f"delta_IC<-{k} SD and delta_FI>{k} SD"
    o["ic_decline"] = o["delta_ic"] < ic_cut
    o["fi_accumulation"] = o["delta_fi"] > fi_cut
    o["state"] = np.select(
        [~o.ic_decline & ~o.fi_accumulation, o.ic_decline & ~o.fi_accumulation,
         ~o.ic_decline & o.fi_accumulation, o.ic_decline & o.fi_accumulation],
        ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"], default="unknown")
    o["ic_cut"], o["fi_cut"], o["state_rule"] = ic_cut, fi_cut, rule
    return o

def baseline_flow(raw, mask, cohort):
    s = COHORT_SPEC[cohort]
    b, i, o = s["base"], s["intermediate"], s["outcome"]
    stages = [
        ("raw_baseline", pd.Series(True, index=raw.index)),
        ("baseline_age50_alive", raw[f"age_{b}"].ge(50) & raw.alive_base),
        ("baseline_self_report", raw[f"age_{b}"].ge(50) & raw.alive_base & raw[f"proxy_{b}"]),
        ("baseline_disability_free", raw[f"age_{b}"].ge(50) & raw.alive_base & raw[f"proxy_{b}"] & raw[f"disability_{b}"].eq(0)),
        ("trajectory_complete", mask),
    ]
    rows = []
    for label, m in stages:
        rows.append({"cohort": cohort, "stage": label, "n": int(m.sum())})
    return rows

def outcome_rows(z, definition):
    rows = []
    for cohort, g in z.groupby("cohort", sort=True):
        for state, h in g.groupby("state", sort=False):
            den = int(h.outcome_category.ne("unknown").sum())
            for outcome in ["disability", "death", "disability_free"]:
                n = int(h.outcome_category.eq(outcome).sum())
                lo, hi = wilson(n, den)
                rows.append({"definition": definition, "cohort": cohort, "state": state,
                             "outcome": outcome, "n": n, "denominator_known_outcome": den,
                             "risk": n/den if den else np.nan, "risk_low": lo, "risk_high": hi})
    # The pooled loop above groups by cohort; create pooled state rows explicitly.
    rows = [r for r in rows if r["cohort"] != "Pooled"]
    for state, h in z.groupby("state", sort=False):
        den = int(h.outcome_category.ne("unknown").sum())
        for outcome in ["disability", "death", "disability_free"]:
            n = int(h.outcome_category.eq(outcome).sum()); lo, hi = wilson(n, den)
            rows.append({"definition": definition, "cohort": "Pooled", "state": state,
                         "outcome": outcome, "n": n, "denominator_known_outcome": den,
                         "risk": n/den if den else np.nan, "risk_low": lo, "risk_high": hi})
    return rows

def model_rows(z, definition):
    rows = []
    q = z.loc[z.outcome_category.ne("unknown")].copy()
    if len(q) == 0:
        return rows
    q["disability_event"] = q.outcome_category.eq("disability").astype(int)
    q["death_event"] = q.outcome_category.eq("death").astype(int)
    # Cause-specific logistic summaries treat the other endpoint as zero; risk
    # tables retain death as a separate competing endpoint.
    for outcome in ["disability_event", "death_event"]:
        try:
            fit = smf.glm(f"{outcome} ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi + C(cohort)",
                          data=q, family=sm.families.Binomial()).fit(cov_type="HC3")
            for term, coef in fit.params.items():
                if term == "Intercept" or term.startswith("C(cohort)"):
                    continue
                se = fit.bse[term]; lo = coef - 1.959964*se; hi = coef + 1.959964*se
                rows.append({"definition": definition, "model": "pooled_cause_specific", "outcome": outcome,
                             "term": term, "n": int(fit.nobs), "events": int(q[outcome].sum()),
                             "log_odds": coef, "OR": np.exp(coef), "OR_low": np.exp(lo), "OR_high": np.exp(hi),
                             "p": fit.pvalues[term]})
        except Exception as e:
            rows.append({"definition": definition, "model": "pooled_cause_specific", "outcome": outcome,
                         "term": "MODEL_FAILED", "error": str(e), "n": len(q)})
    # Cohort-specific estimates expose transport heterogeneity that a pooled
    # coefficient could conceal.
    for cohort, cg in q.groupby("cohort", sort=True):
        for outcome in ["disability_event", "death_event"]:
            if cg[outcome].nunique() < 2:
                rows.append({"definition": definition, "model": "cohort_specific_cause_specific", "cohort": cohort,
                             "outcome": outcome, "term": "MODEL_NOT_ESTIMABLE", "n": len(cg),
                             "events": int(cg[outcome].sum())})
                continue
            try:
                fit = smf.glm(f"{outcome} ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi",
                              data=cg, family=sm.families.Binomial()).fit(cov_type="HC3")
                for term, coef in fit.params.items():
                    if term == "Intercept":
                        continue
                    se = fit.bse[term]; lo = coef - 1.959964*se; hi = coef + 1.959964*se
                    rows.append({"definition": definition, "model": "cohort_specific_cause_specific", "cohort": cohort,
                                 "outcome": outcome, "term": term, "n": int(fit.nobs), "events": int(cg[outcome].sum()),
                                 "log_odds": coef, "OR": np.exp(coef), "OR_low": np.exp(lo), "OR_high": np.exp(hi),
                                 "p": fit.pvalues[term]})
            except Exception as e:
                rows.append({"definition": definition, "model": "cohort_specific_cause_specific", "cohort": cohort,
                             "outcome": outcome, "term": "MODEL_FAILED", "error": str(e), "n": len(cg)})
    # A multinomial model was screened during development but was not retained:
    # cohort-specific zero cells caused separation for mortality in ELSA. The
    # competing endpoint is therefore reported by state-specific risks, while
    # the adjusted summaries use the stable cause-specific GLMs above.
    return rows

def bootstrap_pooled_risks(az, n_boot=500):
    """Participant bootstrap for pooled combined disability/death risks.

    This is an uncertainty audit for aggregate risk contrasts; it does not
    create or export participant-level predictions. The primary cohort-specific
    model remains the adjusted GLM reported separately.
    """
    q = az.loc[az.outcome_category.ne("unknown")].reset_index(drop=True)
    if len(q) == 0:
        return {}
    rng = np.random.default_rng(SEED + len(q))
    states = [s for s in ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"] if s in set(q.state)]
    vals = {s: [] for s in states}; diffs = {s: [] for s in states if s != "preserved_low"}
    for _ in range(n_boot):
        b = q.iloc[rng.integers(0, len(q), len(q))]
        rr = {s: b.loc[b.state.eq(s), "outcome_category"].isin(["disability", "death"]).mean() for s in states}
        for s in states: vals[s].append(rr[s])
        for s in diffs: diffs[s].append(rr[s] - rr["preserved_low"])
    out = {}
    for s in states:
        out[s] = {"risk_low": float(np.nanquantile(vals[s], .025)), "risk_high": float(np.nanquantile(vals[s], .975))}
    for s in diffs:
        out[s]["difference_low"] = float(np.nanquantile(diffs[s], .025)); out[s]["difference_high"] = float(np.nanquantile(diffs[s], .975))
    out["preserved_low"]["difference_low"] = 0.0; out["preserved_low"]["difference_high"] = 0.0
    return out

def main():
    all_raw = {}; all_z = []; flow = []; pars = []
    for cohort in ["ELSA", "CHARLS", "HRS"]:
        print(f"Reading {cohort}", flush=True)
        raw, p = load_cohort(cohort)
        z, mask = prepare_analysis(raw)
        all_raw[cohort] = raw
        all_z.append(z)
        pars.append(p)
        flow.extend(baseline_flow(raw, mask, cohort))
        print(cohort, "raw", len(raw), "trajectory", len(z), "known", int(z.outcome_category.ne("unknown").sum()), flush=True)
    pooled = pd.concat(all_z, ignore_index=True)
    defs = ["primary_sign", "cohort_median", "meaningful_0.2SD", "meaningful_0.5SD"]
    count_rows, outcome, model, sens, contrasts = [], [], [], [], []
    for definition in defs:
        az = pd.concat([assign_states(g, definition) for _, g in pooled.groupby("cohort")], ignore_index=True)
        for (cohort, state), h in az.groupby(["cohort", "state"], sort=True):
            count_rows.append({"definition": definition, "cohort": cohort, "state": state,
                               "n": len(h), "percent": len(h)/len(az[az.cohort.eq(cohort)])})
        for state, h in az.groupby("state", sort=False):
            count_rows.append({"definition": definition, "cohort": "Pooled", "state": state,
                               "n": len(h), "percent": len(h)/len(az)})
        outcome.extend(outcome_rows(az, definition))
        model.extend(model_rows(az, definition))
        pooled_outcome = az.loc[az.outcome_category.ne("unknown")].copy()
        boot = bootstrap_pooled_risks(az)
        ref = pooled_outcome.loc[pooled_outcome.state.eq("preserved_low")]
        ref_combined = (ref.outcome_category.isin(["disability", "death"]).mean()) if len(ref) else np.nan
        for state, h in pooled_outcome.groupby("state", sort=False):
            combined = h.outcome_category.isin(["disability", "death"]).mean() if len(h) else np.nan
            contrasts.append({"definition": definition, "cohort": "Pooled", "state": state,
                              "known_outcome_n": len(h), "combined_disability_or_death_risk": combined,
                              "risk_difference_vs_preserved": combined - ref_combined if np.isfinite(ref_combined) else np.nan,
                              "risk_ratio_vs_preserved": combined / ref_combined if np.isfinite(ref_combined) and ref_combined > 0 else np.nan,
                              "combined_risk_low": boot.get(state, {}).get("risk_low", np.nan),
                              "combined_risk_high": boot.get(state, {}).get("risk_high", np.nan),
                              "risk_difference_low": boot.get(state, {}).get("difference_low", np.nan),
                              "risk_difference_high": boot.get(state, {}).get("difference_high", np.nan)})
        for cohort, cg in az.groupby("cohort", sort=False):
            cg = cg.loc[cg.outcome_category.ne("unknown")]
            cref = cg.loc[cg.state.eq("preserved_low")]
            cr = cref.outcome_category.isin(["disability", "death"]).mean() if len(cref) else np.nan
            for state, h in cg.groupby("state", sort=False):
                rr = h.outcome_category.isin(["disability", "death"]).mean() if len(h) else np.nan
                contrasts.append({"definition": definition, "cohort": cohort, "state": state,
                                  "known_outcome_n": len(h), "combined_disability_or_death_risk": rr,
                                  "risk_difference_vs_preserved": rr-cr if np.isfinite(cr) else np.nan,
                                  "risk_ratio_vs_preserved": rr/cr if np.isfinite(cr) and cr > 0 else np.nan})
        for (cohort, state), h in az.groupby(["cohort", "state"], sort=True):
            den = h.outcome_category.ne("unknown").sum()
            sens.append({"definition": definition, "cohort": cohort, "state": state, "n": len(h),
                         "delta_ic_mean": h.delta_ic.mean(), "delta_ic_sd": h.delta_ic.std(ddof=1),
                         "delta_fi_mean": h.delta_fi.mean(), "delta_fi_sd": h.delta_fi.std(ddof=1),
                         "disability_risk": h.outcome_category.eq("disability").sum()/den if den else np.nan,
                         "death_risk": h.outcome_category.eq("death").sum()/den if den else np.nan})
        az.groupby(["cohort", "state"], as_index=False).agg(n=("state", "size"), delta_ic_mean=("delta_ic", "mean"), delta_fi_mean=("delta_fi", "mean")).to_csv(OUT / f"state_summary_{definition}.csv", index=False)
    pd.DataFrame(flow).to_csv(OUT / "trajectory_state_flow.csv", index=False)
    pd.concat(pars, ignore_index=True).to_csv(OUT / "trajectory_state_standardisation.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUT / "trajectory_state_counts.csv", index=False)
    pd.DataFrame(outcome).to_csv(OUT / "trajectory_state_outcomes.csv", index=False)
    pd.DataFrame(model).to_csv(OUT / "trajectory_state_models.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT / "trajectory_state_sensitivity.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "trajectory_state_contrasts.csv", index=False)
    primary = pd.concat([assign_states(g, "primary_sign") for _, g in pooled.groupby("cohort")], ignore_index=True)
    rows = []
    for (cohort, state), h in primary.groupby(["cohort", "state"], sort=True):
        row = {"cohort": cohort, "state": state, "n": len(h)}
        for c in ["age_baseline", "male", "education", "baseline_fi"]:
            if c in h: row[c + "_mean"] = h[c].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "trajectory_state_baseline_profile.csv", index=False)
    # Clinically interpretable cross-classification: the state pattern is
    # evaluated within coarse short-FI bands. The bands are descriptive because
    # this is an eight-component short deficit burden, not a validated 30-item FI.
    primary["baseline_fi_band"] = pd.cut(primary["baseline_fi"], bins=[-0.001, 0.125, 0.25, 1.001],
                                           labels=["<0.125", "0.125-<0.25", ">=0.25"], right=False)
    strat=[]
    strat_groups = [(cohort, g) for cohort, g in primary.groupby("cohort", sort=True)] + [("Pooled", primary)]
    for cohort, cg in strat_groups:
      for (band, state), h in cg.groupby(["baseline_fi_band", "state"], observed=True):
        known=h.outcome_category.ne("unknown"); den=int(known.sum()); dis=int((h.outcome_category.eq("disability") & known).sum()); death=int((h.outcome_category.eq("death") & known).sum())
        strat.append({"cohort":cohort,"baseline_fi_band":str(band),"state":state,"n":len(h),"known_outcome_n":den,
                     "disability_risk":dis/den if den else np.nan,"death_risk":death/den if den else np.nan,
                     "combined_risk":(dis+death)/den if den else np.nan})
    pd.DataFrame(strat).to_csv(OUT / "trajectory_state_risk_stratification.csv", index=False)
    memo = f"""# IC--frailty trajectory/state analysis audit

Run timestamp (UTC): {datetime.now(timezone.utc).isoformat()}
Random seed: {SEED}

## Design
ELSA waves 6-8-9, CHARLS waves 1-3-4 and HRS waves 8-10-12. Baseline age >=50; baseline and intermediate common eight-item disability-free status; HRS proxy interviews excluded from IC state estimation. The common endpoint uses eight non-walking ADL/IADL items (dressing, bathing, eating, bed transfer, toileting, medication, shopping and meal preparation) to avoid direct locomotion overlap.

## Estimand
The state is defined from baseline-to-intermediate change in four observed-proxy IC domains and the eight-component outcome-disjoint deficit burden (six chronic conditions, self-rated health and BMI clinical-threshold deficit). The outcome window is intermediate-to-outcome incident disability or death. State labels are clinical patterns/risk strata, not validated biological phenotypes.

## Primary state rule
primary_sign: IC decline if the within-cohort baseline-standardised mean IC change is <0; deficit accumulation if FI change is >0. Sensitivity rules use within-cohort medians and 0.2/0.5 baseline-change SD thresholds.

## Missingness and mortality
Unknown/dropout outcome statuses are retained in the flow table and excluded from risk denominators. Death is retained as a competing outcome when the cohort status code identifies death. No mortality linkage was inferred. HAALSI is not included in this primary run because its official individual mortality linkage remains unaudited.

## Outputs
CSV outputs contain aggregate counts, Wilson 95% risk intervals, cause-specific logistic model coefficients, competing disability/death risk contrasts, standardisation parameters and sensitivity summaries. A multinomial model was screened but excluded because mortality separation occurred in a cohort with no observed deaths in the selected wave window. No participant identifiers or row-level predictions are written.
"""
    (OUT / "trajectory_state_method_memo.md").write_text(memo, encoding="utf-8")
    log = {"run": datetime.now(timezone.utc).isoformat(),
           "cohorts": {c: {"raw_n": len(all_raw[c]), "trajectory_n": int(sum(len(x) for x in all_z if x.cohort.iloc[0] == c))} for c in all_raw},
           "definitions": defs, "outputs": sorted(x.name for x in OUT.glob("*.csv"))}
    (OUT / "trajectory_state_analysis_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote outputs to", OUT, flush=True)

if __name__ == "__main__":
    main()
