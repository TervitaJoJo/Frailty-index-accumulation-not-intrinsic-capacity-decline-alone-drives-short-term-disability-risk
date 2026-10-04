"""Conditional analyses for the directional IC--frailty state manuscript.

The primary directional analysis has one outcome window ending at a cohort
wave.  This script adds bounded sensitivity analyses without changing the
primary state definition: expanded selection weighting, entrant/non-entrant
baseline comparisons, one-window Aalen--Johansen CIFs, trajectory-measure
multiple imputation, and explicit feasibility audits for CFA/DIF and Fine--Gray
models.  Only aggregate outputs are written.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

PKG = Path(r"PATH_TO_IC_FRAILTY_V2")
AUDIT = PKG / "analysis_audit" / "trajectory_states_20260930"
OUT = AUDIT / "conditional_analyses_20261002"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG / "scripts"))
import trajectory_states_analysis_20260930 as ts  # noqa: E402

SEED = 20261002
COHORTS = ["ELSA", "CHARLS", "HRS"]
STATE_ORDER = ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"]


def baseline_eligible(raw, cohort):
    s = ts.COHORT_SPEC[cohort]
    b, i, o = s["base"], s["intermediate"], s["outcome"]
    return (
        raw[f"age_{b}"].ge(50)
        & raw["alive_base"]
        & raw[f"proxy_{b}"]
        & raw[f"disability_{b}"].eq(0)
        & raw[f"disability_{b}"].notna()
    )


def outcome_label(raw, cohort):
    s = ts.COHORT_SPEC[cohort]
    o = s["outcome"]
    return pd.Series(
        np.select(
            [raw["death_outcome"], raw["alive_outcome"] & raw[f"proxy_{o}"] & raw[f"disability_{o}"].eq(1),
             raw["alive_outcome"] & raw[f"proxy_{o}"] & raw[f"disability_{o}"].eq(0)],
            ["death", "disability", "disability_free"],
            default="unknown",
        ),
        index=raw.index,
    )


def make_raw_objects():
    objects = {}
    for cohort in COHORTS:
        raw, _ = ts.load_cohort(cohort)
        raw["age_baseline"] = raw[f"age_{ts.COHORT_SPEC[cohort]['base']}"]
        raw["outcome_category"] = outcome_label(raw, cohort)
        raw["baseline_eligible"] = baseline_eligible(raw, cohort)
        raw["trajectory_gate"] = ts.prepare_analysis(raw)[1]
        raw["known_outcome"] = raw.outcome_category.ne("unknown")
        objects[cohort] = raw
    return objects


def selection_bias_and_extended_ipw(objects):
    profile_rows, model_rows, weight_rows = [], [], []
    selected_parts = []
    for cohort, raw in objects.items():
        base = raw.loc[raw.baseline_eligible].copy()
        base["selected"] = (base.trajectory_gate & base.known_outcome).astype(int)
        profile_vars = ["age_baseline", "male", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
        for selected, g in base.groupby("selected"):
            row = {"cohort": cohort, "selected": int(selected), "n": int(len(g))}
            for v in profile_vars:
                row[v + "_mean"] = float(g[v].mean()) if v in g else np.nan
                row[v + "_missing_pct"] = float(g[v].isna().mean() * 100) if v in g else np.nan
            profile_rows.append(row)

        # Expanded selection model: baseline capacity domains are included in
        # addition to the demographic and FI variables used in the basic IPW.
        fvars = ["age_baseline", "male", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
        d = base[["selected"] + fvars].copy().dropna()
        if d.selected.nunique() < 2:
            continue
        rhs = " + ".join(fvars)
        fit = smf.glm("selected ~ " + rhs, data=d, family=sm.families.Binomial()).fit()
        p = fit.predict(d).clip(0.01, 0.99)
        sw = float(d.selected.mean())
        d["weight"] = np.where(d.selected.eq(1), sw / p, np.nan)
        selected = d.loc[d.selected.eq(1)].copy()
        diagnostics = {
            "cohort": cohort,
            "baseline_eligible_n": int(len(d)),
            "selected_n": int(len(selected)),
            "selection_rate": float(selected.shape[0] / len(d)),
            "weight_mean_selected": float(selected.weight.mean()),
            "weight_p01_selected": float(selected.weight.quantile(.01)),
            "weight_p99_selected": float(selected.weight.quantile(.99)),
            "weight_max_selected": float(selected.weight.max()),
        }
        weight_rows.append(diagnostics)
        # Map weights back to the selected complete-case rows by index.  The
        # raw frame index is stable within this in-memory analysis.
        raw_sel = raw.loc[selected.index.intersection(raw.index)].copy()
        raw_sel["extended_weight"] = selected.weight
        selected_parts.append(raw_sel)

    pd.DataFrame(profile_rows).to_csv(OUT / "baseline_selection_profile.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT / "extended_selection_weight_diagnostics.csv", index=False)
    if not selected_parts:
        return
    q = pd.concat(selected_parts, ignore_index=True)
    q = q.loc[q.outcome_category.ne("unknown")].copy()
    q["disability_event"] = q.outcome_category.eq("disability").astype(int)
    q["death_event"] = q.outcome_category.eq("death").astype(int)
    rows = []
    for outcome in ["disability_event", "death_event"]:
        d = q if outcome == "disability_event" else q.loc[q.cohort.isin(["CHARLS", "HRS"])].copy()
        if d[outcome].sum() == 0:
            continue
        formula = f"{outcome} ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi + C(cohort)"
        # Attach state and baseline covariates using the same directional rule
        # as the primary model; only complete state rows enter this sensitivity.
        d = pd.concat([ts.assign_states(g.copy(), "primary_sign") for _, g in d.groupby("cohort")], ignore_index=True)
        d["age_baseline"] = d[[c for c in d.columns if c == "age_baseline"][0]] if "age_baseline" in d else np.nan
        # The raw objects carry cohort-specific age columns; harmonise here.
        for cohort in d.cohort.unique():
            b = ts.COHORT_SPEC[cohort]["base"]
            ix = d.cohort.eq(cohort)
            d.loc[ix, "age_baseline"] = d.loc[ix, f"age_{b}"]
        d = d.dropna(subset=["state", "age_baseline", "male", "education", "baseline_fi", "extended_weight"])
        fit = smf.glm(formula, data=d, family=sm.families.Binomial(), freq_weights=d.extended_weight).fit(cov_type="HC3")
        term = "C(state, Treatment(reference='preserved_low'))[T.coupled]"
        b, se = float(fit.params[term]), float(fit.bse[term])
        rows.append({"outcome": outcome, "n": int(fit.nobs), "events": int(d[outcome].sum()),
                     "OR": float(np.exp(b)), "OR_low": float(np.exp(b - 1.959964 * se)),
                     "OR_high": float(np.exp(b + 1.959964 * se)), "p": float(fit.pvalues[term])})
    pd.DataFrame(rows).to_csv(OUT / "extended_selection_weight_models.csv", index=False)


def one_window_aj_cif():
    src = pd.read_csv(AUDIT / "trajectory_state_outcomes.csv")
    src = src.loc[src.definition.eq("primary_sign") & src.cohort.isin(["Pooled", "CHARLS", "HRS"])].copy()
    rows = []
    for (cohort, state), g in src.groupby(["cohort", "state"], sort=False):
        den = int(g.denominator_known_outcome.iloc[0])
        dis = int(g.loc[g.outcome.eq("disability"), "n"].iloc[0])
        death = int(g.loc[g.outcome.eq("death"), "n"].iloc[0])
        rows.append({"population": cohort, "state": state, "risk_set_n": den,
                     "disability_events": dis, "death_events": death,
                     "cif_disability": dis / den, "cif_death": death / den,
                     "survival_free": 1 - (dis + death) / den,
                     "time_window": "single observed outcome window",
                     "method_note": "One-window Aalen-Johansen recursion; CIF equals the mutually exclusive endpoint proportion."})
    pd.DataFrame(rows).to_csv(OUT / "aalen_johansen_one_window_cif.csv", index=False)


def restricted_primary_models(objects):
    """Re-estimate primary pooled models with mortality restricted to CHARLS/HRS."""
    parts = []
    for cohort, raw in objects.items():
        z = raw.loc[raw.trajectory_gate & raw.known_outcome].copy()
        if len(z):
            z = ts.assign_states(z, "primary_sign")
            z["age_baseline"] = z[f"age_{ts.COHORT_SPEC[cohort]['base']}"]
            parts.append(z)
    q = pd.concat(parts, ignore_index=True)
    rows = []
    for outcome, subset in [("disability_event", q), ("death_event", q.loc[q.cohort.isin(["CHARLS", "HRS"])].copy())]:
        subset["disability_event"] = subset.outcome_category.eq("disability").astype(int)
        subset["death_event"] = subset.outcome_category.eq("death").astype(int)
        subset = subset.dropna(subset=["state", "age_baseline", "male", "education", "baseline_fi"])
        fit = smf.glm(f"{outcome} ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi + C(cohort)",
                      data=subset, family=sm.families.Binomial()).fit(cov_type="HC3")
        for term in ["C(state, Treatment(reference='preserved_low'))[T.ic_decline_only]",
                     "C(state, Treatment(reference='preserved_low'))[T.fi_accumulation_only]",
                     "C(state, Treatment(reference='preserved_low'))[T.coupled]"]:
            b, se = float(fit.params[term]), float(fit.bse[term])
            rows.append({"outcome": outcome, "death_population": "CHARLS+HRS" if outcome == "death_event" else "ELSA+CHARLS+HRS",
                         "term": term, "n": int(fit.nobs), "events": int(subset[outcome].sum()),
                         "OR": float(np.exp(b)), "OR_low": float(np.exp(b - 1.959964 * se)),
                         "OR_high": float(np.exp(b + 1.959964 * se)), "p": float(fit.pvalues[term])})
    pd.DataFrame(rows).to_csv(OUT / "restricted_primary_models.csv", index=False)


def leave_one_cohort_out_metrics(objects):
    """Recalculate disability metrics on the same known-outcome state ledger."""
    parts = []
    for cohort, raw in objects.items():
        z = raw.loc[raw.trajectory_gate & raw.known_outcome].copy()
        if not len(z):
            continue
        z = ts.assign_states(z, "primary_sign")
        z["age_baseline"] = z[f"age_{ts.COHORT_SPEC[cohort]['base']}"]
        z["disability_event"] = z.outcome_category.eq("disability").astype(int)
        parts.append(z)
    q = pd.concat(parts, ignore_index=True)
    rows = []
    base_formula = "disability_event ~ age_baseline + male + education + baseline_fi"
    aug_formula = base_formula + " + C(state, Treatment(reference='preserved_low'))"
    for held in COHORTS:
        train = q.loc[q.cohort.ne(held)].dropna(subset=["disability_event", "age_baseline", "male", "education", "baseline_fi"])
        test = q.loc[q.cohort.eq(held)].dropna(subset=["disability_event", "age_baseline", "male", "education", "baseline_fi"])
        if len(test) == 0:
            continue
        fit0 = smf.glm(base_formula, data=train, family=sm.families.Binomial()).fit()
        fit1 = smf.glm(aug_formula, data=train, family=sm.families.Binomial()).fit()
        y = test.disability_event.to_numpy(int)
        p0 = np.asarray(fit0.predict(test)); p1 = np.asarray(fit1.predict(test))
        rows.extend([
            {"held_out": held, "model": "base", "n": int(len(test)), "events": int(y.sum()),
             "auc": float(roc_auc_score(y, p0)), "brier": float(brier_score_loss(y, p0))},
            {"held_out": held, "model": "state_augmented", "n": int(len(test)), "events": int(y.sum()),
             "auc": float(roc_auc_score(y, p1)), "brier": float(brier_score_loss(y, p1))},
        ])
    out = pd.DataFrame(rows)
    wide = out.pivot(index="held_out", columns="model", values=["n", "events", "auc", "brier"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    wide["delta_auc"] = wide["auc_state_augmented"] - wide["auc_base"]
    wide["delta_brier"] = wide["brier_state_augmented"] - wide["brier_base"]
    out.to_csv(OUT / "leave_one_cohort_out_metrics_reconciled.csv", index=False)
    wide.to_csv(OUT / "leave_one_cohort_out_metrics_reconciled_wide.csv", index=False)


def longitudinal_change_mi(objects, m=10):
    # Impute missing trajectory change scores among baseline-eligible people
    # with an observed outcome. Outcomes and death labels are never imputed.
    estimates = []
    miss_rows = []
    for cohort, raw in objects.items():
        d = raw.loc[raw.baseline_eligible & raw.known_outcome].copy()
        d = ts.assign_states(d.loc[d.delta_ic.notna() & d.delta_fi.notna()].copy(), "primary_sign") if len(d) else d
        # Retain the broader eligible frame for MI; state is assigned after imputation.
        d = raw.loc[raw.baseline_eligible & raw.known_outcome].copy()
        d["age_baseline"] = d[f"age_{ts.COHORT_SPEC[cohort]['base']}"]
        cols = ["delta_ic", "delta_fi", "age_baseline", "male", "education", "baseline_fi", "cognition", "locomotion", "grip_vitality", "psychological"]
        cols = [c for c in cols if c in d.columns]
        x = d[cols].replace([np.inf, -np.inf], np.nan)
        for v in cols:
            miss_rows.append({"cohort": cohort, "variable": v, "n": int(len(x)), "missing": int(x[v].isna().sum()), "missing_pct": float(x[v].isna().mean() * 100)})
        if len(x) < 50 or x[["delta_ic", "delta_fi"]].notna().sum().min() == 0:
            continue
        for imp in range(m):
            imputer = IterativeImputer(random_state=SEED + imp + len(cohort), max_iter=20, sample_posterior=True, min_value=None)
            xi = pd.DataFrame(imputer.fit_transform(x), columns=cols, index=d.index)
            z = d.copy()
            for c in cols:
                z[c] = xi[c]
            z["ic_decline"] = z.delta_ic < 0
            z["fi_accumulation"] = z.delta_fi > 0
            z["state"] = np.select([
                ~z.ic_decline & ~z.fi_accumulation,
                z.ic_decline & ~z.fi_accumulation,
                ~z.ic_decline & z.fi_accumulation,
                z.ic_decline & z.fi_accumulation,
            ], STATE_ORDER, default="unknown")
            z["disability_event"] = z.outcome_category.eq("disability").astype(int)
            z["death_event"] = z.outcome_category.eq("death").astype(int)
            for outcome in ["disability_event", "death_event"]:
                dd = z if outcome == "disability_event" else z.loc[z.cohort.isin(["CHARLS", "HRS"])].copy()
                dd = dd.dropna(subset=["state", "age_baseline", "male", "education", "baseline_fi"])
                if dd[outcome].sum() == 0:
                    continue
                fit = smf.glm(f"{outcome} ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi",
                              data=dd, family=sm.families.Binomial()).fit(cov_type="HC3")
                term = "C(state, Treatment(reference='preserved_low'))[T.coupled]"
                b, se = float(fit.params[term]), float(fit.bse[term])
                estimates.append({"cohort": cohort, "imputation": imp + 1, "outcome": outcome,
                                  "log_OR": b, "variance": se ** 2, "n": int(fit.nobs), "events": int(dd[outcome].sum())})
    est = pd.DataFrame(estimates)
    pooled = []
    if len(est):
        # Pool within each cohort/outcome; then report the pooled development
        # estimate across cohorts as a transparent sensitivity summary.
        for (cohort, outcome), g in est.groupby(["cohort", "outcome"]):
            qbar = g.log_OR.mean(); ubar = g.variance.mean(); bvar = g.log_OR.var(ddof=1) if len(g) > 1 else 0.0
            total = ubar + (1 + 1 / len(g)) * bvar; se = np.sqrt(max(total, 0.0))
            pooled.append({"cohort": cohort, "outcome": outcome, "imputations": int(len(g)), "OR": float(np.exp(qbar)),
                           "OR_low": float(np.exp(qbar - 1.959964 * se)), "OR_high": float(np.exp(qbar + 1.959964 * se)),
                           "n_median": int(g.n.median()), "events_median": int(g.events.median())})
    est.to_csv(OUT / "longitudinal_change_mi_estimates.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "longitudinal_change_mi_pooled.csv", index=False)
    pd.DataFrame(miss_rows).to_csv(OUT / "longitudinal_change_mi_missingness.csv", index=False)


def measurement_audit_and_fine_gray_audit():
    src_dir = PKG / "tables" / "revision_20260929"
    fit = pd.read_csv(src_dir / "measurement_invariance_fit.csv")
    fit.to_csv(OUT / "multigroup_cfa_dif_fit.csv", index=False)
    partial = pd.read_csv(src_dir / "measurement_partial_scalar_fit.csv")
    partial.to_csv(OUT / "multigroup_cfa_dif_partial_scalar.csv", index=False)
    rows = []
    for _, r in fit.iterrows():
        rows.append({"model": r.model, "converged": bool(r.converged), "post_check": bool(r.post_check),
                     "CFI": r.cfi, "RMSEA": r.rmsea, "SRMR": r.srmr,
                     "interpretation": "configural/partial metric support" if r.model in ["configural", "metric_partial_core", "metric_partial_domain"] else "scalar comparability not supported"})
    pd.DataFrame(rows).to_csv(OUT / "multigroup_cfa_dif_interpretation.csv", index=False)
    # Fine--Gray requires individual event times and censoring times. The
    # frozen directional endpoint has only a terminal wave category.
    fg = pd.DataFrame([
        {"cohort": "ELSA", "individual_event_time_available": False, "death_time_audited": False, "fine_gray_estimable": False},
        {"cohort": "CHARLS", "individual_event_time_available": False, "death_time_audited": False, "fine_gray_estimable": False},
        {"cohort": "HRS", "individual_event_time_available": False, "death_time_audited": False, "fine_gray_estimable": False},
        {"cohort": "Pooled", "individual_event_time_available": False, "death_time_audited": False, "fine_gray_estimable": False},
    ])
    fg["reason"] = "The primary directional analysis records mutually exclusive status at the outcome wave; no validated individual failure time is used. A Fine--Gray subdistribution hazard would require event and censoring times."
    fg.to_csv(OUT / "fine_gray_feasibility_audit.csv", index=False)


def main():
    objects = make_raw_objects()
    selection_bias_and_extended_ipw(objects)
    one_window_aj_cif()
    restricted_primary_models(objects)
    leave_one_cohort_out_metrics(objects)
    longitudinal_change_mi(objects, m=10)
    measurement_audit_and_fine_gray_audit()
    manifest = {
        "run": pd.Timestamp.utcnow().isoformat(),
        "cohorts": COHORTS,
        "analyses": ["expanded_selection_weighting", "baseline_selection_profile", "one_window_aalen_johansen_cif", "restricted_primary_models", "reconciled_leave_one_cohort_out_metrics", "longitudinal_change_multiple_imputation", "multigroup_cfa_dif_boundary_audit", "fine_gray_feasibility_audit"],
        "fine_gray_status": "not estimable from the frozen endpoint-only directional data",
        "mi_status": "trajectory-measure MI sensitivity; outcome and death labels were retained as observed",
        "cfa_status": "reused audited harmonised-indicator multigroup fit and partial-scalar boundary outputs; no latent means compared",
    }
    (OUT / "conditional_analyses_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "conditional_analyses_method_note.md").write_text(
        """# Conditional analyses

The directional primary analysis has one terminal outcome window. Aalen--Johansen CIFs therefore reduce to the mutually exclusive endpoint proportions for that single window. A classical Fine--Gray model was not fitted because validated individual failure and censoring times are not part of this frozen endpoint definition.

The multiple-imputation sensitivity imputes missing longitudinal change scores among baseline-eligible participants with an observed outcome; it does not impute death, disability status, or missing outcome-wave follow-up.

The multi-group CFA/DIF outputs are a measurement-boundary audit of the harmonised IC indicator framework. Partial metric structure is usable for risk-pattern construction, whereas scalar comparability is not assumed for absolute latent means.
""", encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
