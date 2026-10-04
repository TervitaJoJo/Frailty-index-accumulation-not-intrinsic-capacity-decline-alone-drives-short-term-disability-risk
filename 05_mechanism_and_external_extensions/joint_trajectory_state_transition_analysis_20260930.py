"""Joint IC--frailty trajectory, state-transition and discrete-time event analysis.

This script is deliberately aggregate-output only. Participant-level panels are
constructed in memory and are not written to disk. The estimand is interval
specific transition to incident disability or death, with IC and outcome-
disjoint deficit burden updated at each observed wave. Continuous person-level
slopes are also estimated as a transparent joint-trajectory representation.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.mixture import GaussianMixture

PKG = Path(r"PATH_TO_IC_FRAILTY_V2")
OUT = PKG / "analysis_audit" / "joint_trajectory_transition_20260930"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG / "scripts"))
from revision_data import META, COMMON, SHARE14, predictors, items_outcome, legacy_outcome, numeric  # noqa: E402

SEED = 20260930
DOM = ["cognition", "locomotion", "grip_vitality", "psychological"]
ENDPOINT_ITEMS = [x for x in COMMON if x != "moneya"]


def avg(vals, minimum=1):
    z = pd.concat(vals, axis=1)
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= minimum)


def load_cols(path, cols):
    available = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    use = [c for c in dict.fromkeys(cols) if c in available]
    return pyreadstat.read_dta(path, usecols=use)[0], set(available)


def required_for_wave(cohort, waves):
    if cohort == "ELSA":
        raw = ["idauniqc", "wave", "inw", "iwstat", "iwindm", "iwindy", "hhidc", "ragender", "raeducl"]
        suffixes = ["agey", "shlt", "mbmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge",
                    "tcog_z_z", "imrc", "dlrc", "orient", "gripsum", "cesd", "walkra", "walk100a", *COMMON, "adltot6", "iadltot2_e"]
        raw += suffixes
        return raw
    if cohort == "CHARLS":
        raw = ["ID", "communityID", "ragender", "raeduc_c"]
        suffixes = ["iwstat", "iwm", "iwy", "chdeathe", "agey", "shlt", "shlta", "mbmi", "hibpe", "diabe",
                    "hearte", "stroke", "cancre", "arthre", "lunge", "orient", "tr20", "walk100a", "lgrip", "rgrip",
                    "cesd10", *COMMON, "adlwa", "iadla"]
        raw += [f"r{w}{s}" for w in waves for s in suffixes]
        return raw
    if cohort == "HRS":
        raw = ["hhidpn", "hhid", "ragender", "raeduc"]
        suffixes = ["iwstat", "iwmid", "iwmidf", "pexit", "proxy", "agey_m", "shlt", "bmi", "hibpe", "diabe",
                    "hearte", "stroke", "cancre", "arthre", "lunge", "imrc", "dlrc", "ser7", "orient", "tr20",
                    "walkra", "walk1a", "walksa", "grpl", "grpr", "cesd", "adl5a", "iadl5a"]
        raw += [f"r{w}{s}" for w in waves for s in suffixes]
        return raw
    if cohort == "SHARE":
        return ["mergeid", "wave", "hhidc", "country", "iwstat", "iwm", "iwy", "agey", "ragender", "raeducl",
                "shlt", "bmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge", "imrc", "dlrc",
                "orient", "walkra", "walk100a", "lgrip", "rgrip", "eurod", *COMMON, "adltot6", "iadltot2_e"]
    raise ValueError(cohort)


def make_wave_row(d, cohort, wave, available):
    p = "" if cohort in ["ELSA", "SHARE"] else f"r{wave}"
    # CHARLS wave 4 uses shlta and has no grip/BMI in the harmonised file; it is
    # kept in metadata but excluded from four-domain trajectory intervals.
    if cohort == "CHARLS":
        if f"{p}shlt" not in d.columns and f"{p}shlta" in d.columns:
            d[f"{p}shlt"] = d[f"{p}shlta"]
        for v in ["lgrip", "rgrip", "mbmi"]:
            if f"{p}{v}" not in d.columns:
                d[f"{p}{v}"] = np.nan
    x = predictors(d, cohort, p)
    if cohort == "HRS":
        x["disability"] = legacy_outcome(d, p, cohort)
        x["death"] = numeric(d[f"{p}iwstat"]).isin([2, 3, 5, 6]) if f"{p}iwstat" in d else False
        x["alive"] = numeric(d[f"{p}iwstat"]).isin([1, 4]) if f"{p}iwstat" in d else False
        x["proxy"] = numeric(d[f"{p}proxy"]).eq(0) if f"{p}proxy" in d else True
        x["time_year"] = np.nan
        x["time_month"] = numeric(d.get(f"{p}iwmid", pd.Series(index=d.index)))
    elif cohort == "ELSA":
        x["disability"] = items_outcome(d, p, cohort, ENDPOINT_ITEMS, len(ENDPOINT_ITEMS))
        x["death"] = numeric(d.get("iwstat", pd.Series(index=d.index))).eq(1)
        x["alive"] = numeric(d.get("iwstat", pd.Series(index=d.index))).eq(0)
        x["proxy"] = True
        x["time_year"] = numeric(d.get("iwindy", pd.Series(index=d.index)))
        x["time_month"] = numeric(d.get("iwindm", pd.Series(index=d.index)))
    elif cohort == "CHARLS":
        x["disability"] = items_outcome(d, p, cohort, ENDPOINT_ITEMS, len(ENDPOINT_ITEMS))
        s = numeric(d.get(f"{p}iwstat", pd.Series(index=d.index)))
        x["death"] = s.isin([5, 6])
        x["alive"] = s.isin([0, 1, 4])
        x["proxy"] = True
        x["time_year"] = numeric(d.get(f"{p}iwy", pd.Series(index=d.index)))
        x["time_month"] = numeric(d.get(f"{p}iwm", pd.Series(index=d.index)))
    else:  # SHARE
        x["disability"] = items_outcome(d, p, cohort, ENDPOINT_ITEMS, len(ENDPOINT_ITEMS))
        s = numeric(d.get("iwstat", pd.Series(index=d.index)))
        x["death"] = s.eq(1)
        x["alive"] = s.eq(0)
        x["proxy"] = True
        x["time_year"] = numeric(d.get("iwy", pd.Series(index=d.index)))
        x["time_month"] = numeric(d.get("iwm", pd.Series(index=d.index)))
    ident = {"ELSA": "idauniqc", "CHARLS": "ID", "HRS": "hhidpn", "SHARE": "mergeid"}[cohort]
    cluster = {"ELSA": "hhidc", "CHARLS": "communityID", "HRS": "hhid", "SHARE": "hhidc"}[cohort]
    x["id"] = d[ident].astype(str).values
    x["cluster"] = d[cluster].astype(str).values if cluster in d else "missing"
    x["wave"] = wave
    x["cohort"] = cohort
    return x


def panel_elsa(waves):
    d, avail = load_cols(META["ELSA"]["path"], required_for_wave("ELSA", waves))
    d = d.loc[numeric(d["wave"]).isin(waves) & numeric(d["inw"]).eq(1)].copy()
    frames = []
    for w in waves:
        g = d.loc[numeric(d["wave"]).eq(w)].copy()
        if len(g): frames.append(make_wave_row(g, "ELSA", w, avail))
    return pd.concat(frames, ignore_index=True)


def panel_wide(cohort, waves):
    d, avail = load_cols(META[cohort]["path"], required_for_wave(cohort, waves))
    frames = [make_wave_row(d.copy(), cohort, w, avail) for w in waves]
    return pd.concat(frames, ignore_index=True)


def panel_share(waves):
    d, avail = load_cols(META["SHARE"]["path"], required_for_wave("SHARE", waves))
    d = d.loc[numeric(d["wave"]).isin(waves)].copy()
    frames = [make_wave_row(d.loc[numeric(d.wave).eq(w)].copy(), "SHARE", w, avail) for w in waves]
    return pd.concat(frames, ignore_index=True)


def add_standardised_scores(panel):
    p = panel.copy()
    # Cohort-specific wave standardisation prevents the failed scalar-invariance
    # assumption from being silently converted into cross-national means.
    for c, g in p.groupby("cohort", sort=False):
        idx = g.index
        for v in DOM + ["baseline_fi"]:
            mu = p.loc[idx, v].mean(); sd = p.loc[idx, v].std(ddof=1)
            if not np.isfinite(sd) or sd <= 0: sd = 1.0
            p.loc[idx, v + "_z"] = (p.loc[idx, v] - mu) / sd
    p["has_ic"] = p[DOM].notna().sum(axis=1) >= 3
    p["has_fi"] = p["baseline_fi"].notna()
    p["ic_score"] = p[[v + "_z" for v in DOM]].mean(axis=1).where(p[DOM].notna().sum(axis=1) >= 3)
    return p


def make_fi(panel):
    # predictors() already computes the eight-component outcome-disjoint short
    # FI. Retain the observed denominator as an audit variable.
    p = panel.copy()
    p["fi_complete"] = p["baseline_fi"].notna()
    return p


def complete_intervals(panel, min_domains=3):
    p = panel.sort_values(["cohort", "id", "time_year", "wave"], na_position="last").copy()
    p["has_ic"] = p[DOM].notna().sum(axis=1) >= min_domains
    p["has_fi"] = p["baseline_fi"].notna()
    p["has_state_measure"] = p.has_ic & p.has_fi
    # Retain adjacent observed waves. The interval start must be disability
    # free and alive; the endpoint can be disability, death, or still free.
    p["next_wave"] = p.groupby(["cohort", "id"])["wave"].shift(-1)
    for v in ["ic_score", "baseline_fi", "disability", "alive", "death", "proxy", "time_year", "time_month", "has_state_measure"]:
        p["next_" + v] = p.groupby(["cohort", "id"])[v].shift(-1)
    p["interval_years"] = p["next_time_year"] - p["time_year"]
    # For HRS, only wave ordinal is reliable in the local product; use a
    # conservative nominal two-year interval when year is unavailable.
    p["interval_years"] = p["interval_years"].where(p["interval_years"].between(0.5, 8), 2.0)
    # A death interval is eligible even when the deceased participant has no
    # next-wave IC/FI measurements; otherwise competing mortality is silently
    # removed by complete-case gating.
    next_death_flag = p["next_death"].fillna(False).astype(bool)
    next_observed_flag = p.next_has_state_measure & p.next_proxy.fillna(False)
    p["interval_eligible"] = (p.has_state_measure & p.alive &
                               p.disability.eq(0) & p.disability.notna() & p.proxy &
                               (next_observed_flag | next_death_flag))
    p["delta_ic"] = p["next_ic_score"] - p["ic_score"]
    p["delta_fi"] = p["next_baseline_fi"] - p["baseline_fi"]
    death_next = p["next_death"].fillna(False).astype(bool).to_numpy()
    alive_next = p["next_alive"].fillna(False).astype(bool).to_numpy()
    disability_next = p["next_disability"].eq(1).to_numpy()
    no_disability_next = p["next_disability"].eq(0).to_numpy()
    p["event_type"] = np.select([
        death_next, alive_next & disability_next, alive_next & no_disability_next,
    ], ["death", "disability", "free"], default="unknown")
    return p


def state_label(ic, fi, ic_cut=0.0, fi_cut=0.0):
    a = pd.Series(ic < ic_cut, index=ic.index)
    b = pd.Series(fi > fi_cut, index=fi.index)
    return np.select([~a & ~b, a & ~b, ~a & b, a & b],
                     ["preserved_low", "ic_decline_only", "fi_accumulation_only", "coupled"], default="unknown")


def make_interval_states(p):
    q = p.loc[p.interval_eligible].copy()
    # Four transition states use joint change over each observed interval.
    q["start_state"] = state_label(q.delta_ic, q.delta_fi)
    q["end_state"] = state_label(q.next_delta_ic if "next_delta_ic" in q else q.delta_ic,
                                  q.next_delta_fi if "next_delta_fi" in q else q.delta_fi)
    # A level-based health state is required for a genuine transition matrix.
    # Thresholds are cohort-specific quartiles of the eligible baseline domain
    # scores; transition probabilities are reported within cohort and pooled.
    q["ic_low"] = False; q["fi_high"] = False
    for c, g in q.groupby("cohort", sort=False):
        ic_cut = g["ic_score"].median(); fi_cut = g["baseline_fi"].median()
        q.loc[g.index, "ic_low"] = q.loc[g.index, "ic_score"] < ic_cut
        q.loc[g.index, "fi_high"] = q.loc[g.index, "baseline_fi"] >= fi_cut
        q.loc[g.index, "ic_level_cut"] = ic_cut; q.loc[g.index, "fi_level_cut"] = fi_cut
    q["level_state"] = np.select([
        ~q.ic_low & ~q.fi_high, q.ic_low & ~q.fi_high, ~q.ic_low & q.fi_high, q.ic_low & q.fi_high
    ], ["highIC_lowFI", "lowIC_lowFI", "highIC_highFI", "lowIC_highFI"], default="unknown")
    # State at the end of an observed interval. Death/disability are retained
    # as absorbing endpoint labels; only intervals with a measured next state
    # receive a next level state.
    next_ic_low = q["next_ic_score"] < q["ic_level_cut"]
    next_fi_high = q["next_baseline_fi"] >= q["fi_level_cut"]
    q["next_level_state"] = np.select([
        ~next_ic_low & ~next_fi_high, next_ic_low & ~next_fi_high,
        ~next_ic_low & next_fi_high, next_ic_low & next_fi_high,
    ], ["highIC_lowFI", "lowIC_lowFI", "highIC_highFI", "lowIC_highFI"], default="unknown")
    q.loc[q.event_type.eq("disability"), "next_level_state"] = "disability"
    q.loc[q.event_type.eq("death"), "next_level_state"] = "death"
    return q


def slope_features(panel):
    rows = []
    usable = panel.loc[panel[DOM].notna().sum(axis=1).ge(3) & panel["baseline_fi"].notna()].copy()
    for (c, ident), g in usable.groupby(["cohort", "id"], sort=False):
        gg = g.sort_values(["time_year", "wave"])
        if len(gg) < 2: continue
        t = gg["time_year"].to_numpy(dtype=float)
        if not np.isfinite(t).all() or np.ptp(t) <= 0:
            t = gg["wave"].to_numpy(dtype=float) * 2.0
        rows.append({"cohort": c, "n_waves": len(gg), "ic_intercept": np.polyfit(t, gg.ic_score, 1)[1],
                     "ic_slope": np.polyfit(t, gg.ic_score, 1)[0], "fi_intercept": np.polyfit(t, gg.baseline_fi, 1)[1],
                     "fi_slope": np.polyfit(t, gg.baseline_fi, 1)[0], "baseline_age": gg.age.iloc[0],
                     "baseline_fi": gg.baseline_fi.iloc[0]})
    s = pd.DataFrame(rows)
    if len(s) == 0: return s
    for c in ["ic_intercept", "ic_slope", "fi_intercept", "fi_slope"]:
        s[c + "_z"] = s.groupby("cohort")[c].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) if x.std(ddof=1) > 0 else 1.0))
    return s


def fit_trajectory_classes(slopes, max_k=5):
    if len(slopes) < 100: return pd.DataFrame(), pd.DataFrame()
    vars_ = ["ic_slope_z", "fi_slope_z", "ic_intercept_z", "fi_intercept_z"]
    x = slopes[vars_].replace([np.inf, -np.inf], np.nan).dropna().copy()
    rows = []
    models = {}
    for k in range(2, min(max_k, len(x) - 1) + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED, n_init=20)
        gm.fit(x)
        models[k] = gm
        rows.append({"k": k, "bic": float(gm.bic(x)), "aic": float(gm.aic(x)), "min_class_n": int(np.bincount(gm.predict(x)).min()),
                     "min_class_pct": float(np.bincount(gm.predict(x)).min() / len(x))})
    metric = pd.DataFrame(rows)
    best_k = int(metric.loc[metric.bic.idxmin(), "k"])
    gm = models[best_k]
    out = slopes.loc[x.index].copy()
    out["trajectory_class"] = gm.predict(x) + 1
    return out, metric


def fit_discrete_time(q):
    if len(q) == 0: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = q.copy()
    d["disability_event"] = d.event_type.eq("disability").astype(int)
    d["death_event"] = d.event_type.eq("death").astype(int)
    # Use one interval per participant-wave, with death and disability as
    # competing interval outcomes. Cluster-robust SEs account for repeated rows.
    coef_rows = []; risk_rows = []; trans_rows = []
    model_base = d.loc[d.event_type.isin(["free", "disability", "death"])].copy()
    # For deaths first observed at the next wave, the next-wave IC/FI values
    # are structurally unavailable. Retain those competing events using an
    # explicit missing-next-measure indicator and neutralised change terms.
    model_base["next_measure_missing"] = (~model_base.next_has_state_measure.fillna(False)).astype(int)
    model_base["delta_ic_model"] = model_base.delta_ic.fillna(0.0)
    model_base["delta_fi_model"] = model_base.delta_fi.fillna(0.0)
    model_base = model_base.dropna(subset=["level_state", "age", "baseline_fi", "cohort", "wave"])
    for outcome in ["disability_event", "death_event"]:
        try:
            if outcome == "disability_event":
                formula = f"{outcome} ~ C(level_state) + delta_ic_model + delta_fi_model + age + baseline_fi + C(cohort) + C(wave)"
            else:
                # Death is observed through the next-wave status and therefore
                # has structurally missing next-wave measurements. Do not
                # condition on that measurement-missingness indicator when
                # estimating the state-associated death hazard.
                formula = f"{outcome} ~ C(level_state) + age + baseline_fi + C(cohort) + C(wave)"
            fit = smf.glm(formula, data=model_base, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": model_base.id})
            for term, b in fit.params.items():
                if term == "Intercept": continue
                se = fit.bse[term]
                coef_rows.append({"outcome": outcome, "term": term, "n": int(fit.nobs), "events": int(model_base[outcome].sum()),
                                  "estimate": float(b), "OR": float(np.exp(b)), "OR_low": float(np.exp(b-1.959964*se)),
                                  "OR_high": float(np.exp(b+1.959964*se)), "p": float(fit.pvalues[term])})
        except Exception as e:
            coef_rows.append({"outcome": outcome, "term": "MODEL_FAILED", "error": repr(e), "n": len(d)})
    for (c, st), g in d.groupby(["cohort", "level_state"], sort=True):
        risk_rows.append({"cohort": c, "level_state": st, "intervals": len(g), "disability_events": int(g.disability_event.sum()),
                          "death_events": int(g.death_event.sum()), "disability_risk": float(g.disability_event.mean()),
                          "death_risk": float(g.death_event.mean()), "free_next": float(g.event_type.eq("free").mean())})
    # Empirical transition matrix among level states; disability/death are
    # absorbing endpoint labels and are reported separately.
    for (c, st), g in d.groupby(["cohort", "level_state"], sort=True):
        for ev, h in g.groupby("event_type", sort=True):
            trans_rows.append({"cohort": c, "from_state": st, "to_event": ev, "n": len(h), "probability": len(h)/len(g)})
    for (c, st), g in d.groupby(["cohort", "level_state"], sort=True):
        for nxt, h in g.groupby("next_level_state", sort=True):
            trans_rows.append({"cohort": c, "from_state": st, "to_event": nxt, "n": len(h), "probability": len(h)/len(g), "transition_type": "level_or_absorbing"})
    return pd.DataFrame(coef_rows), pd.DataFrame(risk_rows), pd.DataFrame(trans_rows)


def main():
    configs = {
        "ELSA": [6, 8],                 # nurse grip available at both waves
        "CHARLS": [1, 2, 3],             # repeated four-domain measures
        "HRS": [8, 9, 10, 11, 12, 13],  # repeated grip/cognition/psychology
        "SHARE": [1, 2, 4, 5, 6, 8],     # wave 7 cognition excluded
    }
    panels = []
    logs = []
    for c, waves in configs.items():
        print("Loading", c, waves, flush=True)
        try:
            if c == "ELSA": p = panel_elsa(waves)
            elif c == "SHARE": p = panel_share(waves)
            else: p = panel_wide(c, waves)
            panels.append(p)
            logs.append({"cohort": c, "waves": ";".join(map(str, waves)), "rows": len(p), "participants": p.id.nunique(), "status": "ok"})
            print(c, len(p), p.id.nunique(), flush=True)
        except Exception as e:
            logs.append({"cohort": c, "waves": ";".join(map(str, waves)), "rows": 0, "participants": 0, "status": repr(e)})
            print(c, "FAILED", repr(e), flush=True)
    if not panels:
        raise RuntimeError("No panel could be built")
    panel = add_standardised_scores(pd.concat(panels, ignore_index=True))
    panel = make_fi(panel)
    # Keep outcome-free starts and build observed adjacent intervals.
    intervals = complete_intervals(panel)
    q = make_interval_states(intervals)
    q["disability_event"] = q.event_type.eq("disability").astype(int)
    q["death_event"] = q.event_type.eq("death").astype(int)
    slopes = slope_features(panel)
    slope_classes, class_metrics = fit_trajectory_classes(slopes)
    coef, risks, transitions = fit_discrete_time(q)
    # Aggregate-only exports.
    pd.DataFrame(logs).to_csv(OUT / "joint_panel_build_log.csv", index=False)
    panel.groupby(["cohort", "wave"], as_index=False).agg(rows=("id", "size"), participants=("id", "nunique"),
        ic_complete=("has_ic", "sum"), fi_complete=("has_fi", "sum"), disability_observed=("disability", lambda x: x.notna().sum()),
        disability_events=("disability", lambda x: (x == 1).sum()), deaths=("death", "sum")).to_csv(OUT / "joint_panel_wave_summary.csv", index=False)
    q.groupby(["cohort", "wave", "level_state"], as_index=False).agg(intervals=("id", "size"),
        disability_events=("disability_event", "sum"), death_events=("death_event", "sum")).to_csv(OUT / "interval_state_summary.csv", index=False)
    pd.DataFrame(coef).to_csv(OUT / "discrete_time_competing_event_models.csv", index=False)
    pd.DataFrame(risks).to_csv(OUT / "discrete_time_state_risks.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUT / "state_transition_probabilities.csv", index=False)
    slopes.groupby(["cohort", "n_waves"], as_index=False).agg(n=("cohort", "size"), ic_slope_mean=("ic_slope", "mean"),
        ic_slope_sd=("ic_slope", "std"), fi_slope_mean=("fi_slope", "mean"), fi_slope_sd=("fi_slope", "std")).to_csv(OUT / "joint_slope_summary.csv", index=False)
    if len(slope_classes):
        slope_classes.groupby(["cohort", "trajectory_class"], as_index=False).agg(n=("cohort", "size"),
            ic_slope_mean=("ic_slope", "mean"), fi_slope_mean=("fi_slope", "mean"), baseline_fi_mean=("baseline_fi", "mean")).to_csv(OUT / "joint_trajectory_class_summary.csv", index=False)
    class_metrics.to_csv(OUT / "joint_trajectory_class_fit_metrics.csv", index=False)
    memo = f"""# Joint IC--frailty trajectory and state-transition analysis

Run time (UTC): {datetime.now(timezone.utc).isoformat()}

The primary development panel uses ELSA waves 6 and 8 (the two ELSA waves
with the audited grip module), CHARLS waves 1--3, HRS waves 8--13, and SHARE
waves 1, 2, 4, 5, 6 and 8. Wave-specific observed-proxy IC domains and an
outcome-disjoint short FI are standardised within cohort. Adjacent observed
waves create person-period intervals; starts are alive, proxy-free, and free of
the common ADL/IADL disability endpoint. Disability and death are competing
interval outcomes.

The primary estimand is the interval probability of disability or death as a
function of the current joint IC/FI level state, recent continuous IC change,
recent FI change, age, baseline FI, cohort and survey wave. Repeated rows use
participant-clustered standard errors. This is a discrete-time survival model;
no continuous event date is imputed where only interval timing is available.

The joint-trajectory sensitivity represents each participant by empirical IC
and FI intercepts and slopes across all eligible waves and fits Gaussian
mixture classes with 2--{5} components selected by BIC. These classes are
descriptive trajectory phenotypes and are not treated as latent biological
subtypes without further validation.

No participant-level panel, identifiers, fitted predictions or mixture labels
are exported. HAALSI remains a separate three-wave disability replication
because its official individual mortality linkage is absent from the local
longitudinal file.
"""
    (OUT / "joint_trajectory_state_transition_method_memo.md").write_text(memo, encoding="utf-8")
    (OUT / "joint_trajectory_state_transition_run.json").write_text(json.dumps({"run": datetime.now(timezone.utc).isoformat(),
        "configs": configs, "aggregate_only": True, "n_panel_rows": int(len(panel)), "n_intervals": int(len(q)), "n_slope_people": int(len(slopes))}, indent=2), encoding="utf-8")
    print("Intervals", len(q), "slopes", len(slopes), "classes", len(slope_classes), flush=True)
    print(risks.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
