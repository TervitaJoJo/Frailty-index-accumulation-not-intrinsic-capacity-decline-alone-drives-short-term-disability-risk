"""Stage 30: missingness-IPW and survey-weighted longitudinal sensitivity.

This stage does not replace the primary ordinal WLSMV latent SEM. It asks a
narrower question: among the observed-proxy longitudinal analysis frames, do
locomotion and the incremental full-model fit change when (i) complete-case
selection is reweighted by a baseline-observable inclusion model, and (ii)
the cohort-specific anchor-wave survey/physical-module weight is used?

No individual-level derived data are written. Only aggregate coefficients,
sample sizes, diagnostics, and model-level flags are exported.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def load_stage25_definitions():
    """Load only imports and function definitions from Stage 25."""
    source = (OUT / "54_longitudinal_predictive_screen.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    allowed = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    ns = {"__name__": "stage25_definitions"}
    exec(compile(ast.Module(body=allowed, type_ignores=[]), str(OUT / "54_longitudinal_predictive_screen.py"), "exec"), ns)
    return ns


S25 = load_stage25_definitions()
num = S25["num"]
valid01 = S25["valid01"]
read_dta = S25["read_dta"]
fi_components = S25["fi_components"]
build_domains = S25["build_domains"]


def norm_key(x: pd.Series, width: int | None = None) -> pd.Series:
    z = x.astype("string").str.replace(r"\.0$", "", regex=True)
    return z.str.zfill(width) if width else z


def fi_parts(d: pd.DataFrame, cols: dict, direction: str) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for name, col in cols["disease"].items():
        out[name] = valid01(d[col]) if col in d else np.nan
    shlt = num(d[cols["shlt"]], 1, 5) if cols["shlt"] in d else pd.Series(np.nan, index=d.index)
    out["self_rated_health"] = (5 - shlt) / 4 if direction == "poor_is_low" else (shlt - 1) / 4
    bmi = num(d[cols["bmi"]], 0, 200) if cols["bmi"] in d else pd.Series(np.nan, index=d.index)
    out["bmi"] = bmi.notna().astype(float).where(bmi.notna())
    return out


def add_fi_and_counts(frame: pd.DataFrame, fi_cols: dict, direction: str) -> pd.DataFrame:
    parts = fi_parts(frame, fi_cols, direction)
    observed = parts.notna().sum(axis=1)
    frame = frame.copy()
    frame["fi"] = parts.sum(axis=1, min_count=1).div(observed).where(observed.ge(6))
    frame["fi_obs_count"] = observed
    frame["ic_obs_count"] = frame[["cognition", "locomotion", "grip_vitality", "psychological"]].notna().sum(axis=1)
    return frame


def standardize(x: pd.Series) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce")
    sd = z.std(ddof=1)
    return (z - z.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=x.index)


def add_weight_map(dataset: str, d: pd.DataFrame, id_col: str) -> pd.DataFrame:
    d = d.copy()
    d["survey_weight"] = np.nan
    d["module_flag"] = np.nan
    if dataset == "ELSA":
        p = ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_data_v2.dta"
        w = read_dta(p, ["idauniq", "w6xwgt"])
        w["key"] = norm_key(w["idauniq"])
        w = w.drop_duplicates("key")
        nurse_path = ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_nurse_data_v2.dta"
        nurse = read_dta(nurse_path, ["idauniq", "w6nurwt"])
        nurse["key"] = norm_key(nurse["idauniq"])
        nurse = nurse.drop_duplicates("key")
        w = w.merge(nurse[["key", "w6nurwt"]], on="key", how="left", validate="one_to_one")
        d["key"] = norm_key(d[id_col])
        d = d.merge(w[["key", "w6xwgt", "w6nurwt"]], on="key", how="left", validate="many_to_one")
        d["survey_weight"] = num(d["w6xwgt"], 0.01, np.inf)
        d["module_flag"] = num(d["w6nurwt"], 0.01, np.inf).notna().astype(float)
        return d.drop(columns=["key", "w6xwgt", "w6nurwt"])
    if dataset == "SHARE":
        p = next((ROOT / "SHARE").rglob("sharew6_rel9-0-0_gv_weights.dta"))
        w = read_dta(p, ["mergeid", "cciw_w6"])
        w["key"] = norm_key(w["mergeid"])
        w = w.drop_duplicates("key")
        d["key"] = norm_key(d[id_col])
        d = d.merge(w[["key", "cciw_w6"]], on="key", how="left", validate="many_to_one")
        d["survey_weight"] = num(d["cciw_w6"], 0.01, np.inf)
        return d.drop(columns=["key", "cciw_w6"])
    if dataset == "HRS":
        p = ROOT / "HRS" / "public_survey_data" / "Cross-Wave Tracker File" / "trk2022tr_r.csv"
        tr = pd.read_csv(p, usecols=["HHID", "PN", "MPMWGTR"], dtype={"HHID": "string", "PN": "string"})
        tr["key"] = norm_key(tr["HHID"], 6) + norm_key(tr["PN"], 3)
        tr = tr.drop_duplicates("key")
        d["key"] = norm_key(d[id_col])
        d = d.merge(tr[["key", "MPMWGTR"]], on="key", how="left", validate="one_to_one")
        d["module_flag"] = num(d["MPMWGTR"], 0.01, np.inf).notna().astype(float)
        # Physical-measures weight is the primary weight for a four-domain IC frame.
        d["survey_weight"] = num(d["MPMWGTR"], 0.01, np.inf)
        return d.drop(columns=["key", "MPMWGTR"])
    if dataset == "CHARLS":
        d["survey_weight"] = num(d["r3wtrespb"], 0.01, np.inf) if "r3wtrespb" in d else np.nan
        d["module_flag"] = num(d["r3gripcomp"], 0.0, np.inf).notna().astype(float) if "r3gripcomp" in d else np.nan
        return d
    raise ValueError(dataset)


def make_long_pairs(dataset, path, anchor, target, id_col, fi_spec, weight_source):
    cols = ["wave", id_col, "agey", "ragender", "imrc", "dlrc", "orient", "walkra", "walk100a", "lgrip", "rgrip", "gripsum", "gripcomp", "eurod", "cesd", fi_spec["shlt"], fi_spec["bmi"], *fi_spec["disease"].values()]
    d = read_dta(path, cols)
    d = d.loc[d["wave"].isin([anchor, target]) & d[id_col].notna()].copy()
    d = d.drop_duplicates([id_col, "wave"], keep="first")
    d = add_weight_map(dataset, d, id_col)
    d["age"] = num(d["agey"])
    d["sex"] = num(d["ragender"])
    d = pd.concat([d, build_domains(d, dataset)], axis=1)
    d = add_fi_and_counts(d, fi_spec, "poor_is_low")
    base = d.loc[d.wave.eq(anchor)].copy()
    fut = d.loc[d.wave.eq(target)].copy()
    bcols = [id_col, "fi", "age", "sex", "cognition", "locomotion", "grip_vitality", "psychological", "fi_obs_count", "ic_obs_count", "module_flag", "survey_weight"]
    fcols = [id_col, "fi", "fi_obs_count", "module_flag"]
    pairs = base[bcols].rename(columns={"fi": "baseline_fi", "cognition": "cognition_base", "locomotion": "locomotion_base", "grip_vitality": "grip_vitality_base", "psychological": "psychological_base", "fi_obs_count": "base_fi_obs_count", "ic_obs_count": "base_ic_obs_count", "module_flag": "base_module_flag", "survey_weight": "base_survey_weight"}).merge(fut[fcols].rename(columns={"fi": "future_fi", "fi_obs_count": "future_fi_obs_count", "module_flag": "future_module_flag"}), on=id_col, how="inner")
    pairs["complete"] = pairs[["baseline_fi", "future_fi", "age", "sex", "cognition_base", "locomotion_base", "grip_vitality_base", "psychological_base"]].notna().all(axis=1).astype(int)
    pairs["delta_fi"] = pairs["future_fi"] - pairs["baseline_fi"]
    return pairs


def make_wide_pairs(dataset, path, anchor, target, spec):
    cols = [spec["id"], spec["active"].format(w=anchor), spec["active"].format(w=target), spec["age"].format(w=anchor), spec["sex"]]
    if spec.get("proxy"):
        cols += [spec["proxy"].format(w=anchor), spec["proxy"].format(w=target)]
    for w in [anchor, target]:
        cols += [c.format(w=w) for c in spec["ic"]]
        cols += [c.format(w=w) for c in spec["fi"]]
        if spec.get("module"):
            cols.append(spec["module"].format(w=w))
    if dataset == "CHARLS":
        cols += ["r3wtrespb", "r3gripcomp"]
    d = read_dta(path, cols)
    d = add_weight_map(dataset, d, spec["id"])

    def wave_frame(w):
        out = pd.DataFrame(index=d.index)
        out["id"] = d[spec["id"]]
        out["active"] = num(d[spec["active"].format(w=w)]).eq(1)
        if spec.get("proxy"):
            out["active"] &= num(d[spec["proxy"].format(w=w)]).eq(0)
        age_col = spec["age"].format(w=w)
        out["age"] = num(d[age_col]) if age_col in d else np.nan
        out["sex"] = num(d[spec["sex"]]) if spec["sex"] in d else np.nan
        vals = {}
        for dom, colset in spec["ic_map"].items():
            arr = []
            for c in colset:
                cc = c.format(w=w)
                if cc in d:
                    x = num(d[cc])
                    if dom == "cognition":
                        if cc.endswith("imrc") or cc.endswith("dlrc"): x = x / 10
                        elif cc.endswith("orient"): x = x / 4
                        elif cc.endswith("ser7"): x = x / 5
                    elif dom == "locomotion": x = 1 - (x > 0).astype(float) if dataset == "HRS" else 1 - x
                    elif dom == "grip_vitality": x = x / 100
                    elif dom == "psychological": x = -x
                    arr.append(x)
            z = pd.concat(arr, axis=1) if arr else pd.DataFrame(index=d.index)
            if dom == "cognition": vals[dom] = z.mean(axis=1).where(z.notna().sum(axis=1).ge(2)) if arr else np.nan
            elif dom == "grip_vitality": vals[dom] = z.max(axis=1, skipna=True).where(z.notna().any(axis=1)) if arr else np.nan
            else: vals[dom] = z.mean(axis=1).where(z.notna().any(axis=1)) if arr else np.nan
        for k, v in vals.items(): out[k] = v
        fi_map = {"disease": {k: c.format(w=w) for k, c in spec["fi_map"]["disease"].items()}, "shlt": spec["fi_map"]["shlt"].format(w=w), "bmi": spec["fi_map"]["bmi"].format(w=w)}
        out = pd.concat([out, fi_components(d, fi_map, spec["shlt_direction"])], axis=1)
        out = add_fi_and_counts(out, {"disease": {}, "shlt": "__none__", "bmi": "__none__"}, "poor_is_low") if False else out
        parts = pd.DataFrame(index=d.index)
        for name, col in fi_map["disease"].items(): parts[name] = valid01(d[col]) if col in d else np.nan
        sh = num(d[fi_map["shlt"]], 1, 5) if fi_map["shlt"] in d else pd.Series(np.nan, index=d.index)
        parts["self_rated_health"] = (sh - 1) / 4 if spec["shlt_direction"] == "poor_is_high" else (5 - sh) / 4
        bmi = num(d[fi_map["bmi"]], 0, 200) if fi_map["bmi"] in d else pd.Series(np.nan, index=d.index)
        parts["bmi"] = bmi.notna().astype(float).where(bmi.notna())
        out["fi_obs_count"] = parts.notna().sum(axis=1)
        out["fi"] = parts.sum(axis=1, min_count=1).div(out["fi_obs_count"]).where(out["fi_obs_count"].ge(6))
        out["ic_obs_count"] = out[["cognition", "locomotion", "grip_vitality", "psychological"]].notna().sum(axis=1)
        out["module_flag"] = d["module_flag"] if "module_flag" in d else np.nan
        out["survey_weight"] = d["survey_weight"] if "survey_weight" in d else np.nan
        return out.loc[out["active"]].copy()

    base = wave_frame(anchor)
    fut = wave_frame(target)
    pairs = base[["id", "fi", "age", "sex", "cognition", "locomotion", "grip_vitality", "psychological", "fi_obs_count", "ic_obs_count", "module_flag", "survey_weight"]].rename(columns={"fi": "baseline_fi", "cognition": "cognition_base", "locomotion": "locomotion_base", "grip_vitality": "grip_vitality_base", "psychological": "psychological_base", "fi_obs_count": "base_fi_obs_count", "ic_obs_count": "base_ic_obs_count", "module_flag": "base_module_flag", "survey_weight": "base_survey_weight"}).merge(fut[["id", "fi", "fi_obs_count", "module_flag"]].rename(columns={"fi": "future_fi", "fi_obs_count": "future_fi_obs_count", "module_flag": "future_module_flag"}), on="id", how="inner")
    pairs["complete"] = pairs[["baseline_fi", "future_fi", "age", "sex", "cognition_base", "locomotion_base", "grip_vitality_base", "psychological_base"]].notna().all(axis=1).astype(int)
    pairs["delta_fi"] = pairs["future_fi"] - pairs["baseline_fi"]
    return pairs


def inclusion_weights(pairs: pd.DataFrame):
    covars = ["age", "sex", "base_fi_obs_count", "base_ic_obs_count", "future_fi_obs_count", "base_module_flag"]
    X = pairs[covars].copy()
    for c in covars:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        X[c] = X[c].fillna(med if np.isfinite(med) else 0.0)
    y = pairs["complete"].astype(int)
    if y.nunique() < 2:
        p = np.repeat(float(y.mean()), len(y))
        return p, {"model": "unavailable_single_class", "p_min": float(np.nanmin(p)), "p_max": float(np.nanmax(p))}
    model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 0.05, 0.99)
    sw = y.mean() / p
    sw = np.clip(sw, 0.1, 10.0)
    comp = sw[y.eq(1)]
    return sw, {"model": "logistic_baseline_observable_inclusion", "p_min": float(p.min()), "p_max": float(p.max()), "weight_mean": float(sw.mean()), "weight_p99": float(np.quantile(sw, 0.99)), "complete_weight_mean": float(comp.mean()), "complete_weight_sd": float(comp.std(ddof=1)), "complete_rate": float(y.mean())}


def fit_weighted(pairs, dataset, window, weights, label):
    complete = pairs.loc[pairs["complete"].eq(1)].copy()
    if len(complete) < 100:
        return pd.DataFrame([{"dataset": dataset, "window": window, "weighting": label, "term": "__model__", "n": len(complete), "estimate": np.nan, "se_hc3": np.nan, "p_hc3": np.nan, "std_beta": np.nan, "r2": np.nan, "delta_r2": np.nan, "error": "too_few_complete_cases"}])
    full = ["baseline_fi", "age", "sex", "cognition_base", "locomotion_base", "grip_vitality_base", "psychological_base"]
    base = ["baseline_fi", "age", "sex"]
    def design(cols):
        x = pd.DataFrame(index=complete.index)
        for c in cols: x[c] = complete[c].astype(float) if c == "sex" else standardize(complete[c])
        return sm.add_constant(x, has_constant="add")
    y = complete["future_fi"].astype(float)
    w = pd.Series(weights, index=pairs.index).loc[complete.index].astype(float)
    if w.notna().sum() == 0 or not np.isfinite(w).all() or (w <= 0).any():
        w = pd.Series(1.0, index=complete.index)
    xb, xf = design(base), design(full)
    mb = sm.WLS(y, xb, weights=w).fit(cov_type="HC3")
    mf = sm.WLS(y, xf, weights=w).fit(cov_type="HC3")
    out = []
    for term in mf.params.index:
        if term == "const": continue
        est = float(mf.params[term]); se = float(mf.bse[term]); ci = mf.conf_int().loc[term]
        out.append({"dataset": dataset, "window": window, "weighting": label, "term": term, "n": int(mf.nobs), "estimate": est, "se_hc3": se, "p_hc3": float(mf.pvalues[term]), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "std_beta": est / float(y.std(ddof=1)), "r2": float(mf.rsquared), "delta_r2": float(mf.rsquared - mb.rsquared), "error": ""})
    return pd.DataFrame(out)


def analyse(dataset, pairs, window):
    ipw, diag = inclusion_weights(pairs)
    survey = pd.to_numeric(pairs["base_survey_weight"], errors="coerce")
    survey = survey.where(survey > 0)
    survey = survey / survey.loc[pairs["complete"].eq(1)].mean() if survey.loc[pairs["complete"].eq(1)].notna().any() else survey
    survey = survey.fillna(1.0).clip(lower=0.1, upper=10.0)
    rows = [fit_weighted(pairs, dataset, window, np.ones(len(pairs)), "unweighted_complete_case"), fit_weighted(pairs, dataset, window, ipw, "complete_case_IPW"), fit_weighted(pairs, dataset, window, survey.to_numpy(), "survey_weight_only"), fit_weighted(pairs, dataset, window, ipw * survey.to_numpy(), "IPW_times_survey_weight")]
    for z in rows:
        z["complete_rate"] = diag.get("complete_rate", np.nan)
        z["ipw_p_min"] = diag.get("p_min", np.nan)
        z["ipw_p_max"] = diag.get("p_max", np.nan)
        z["ipw_p99"] = diag.get("weight_p99", np.nan)
        z["ipw_complete_weight_mean"] = diag.get("complete_weight_mean", np.nan)
        z["ipw_complete_weight_sd"] = diag.get("complete_weight_sd", np.nan)
    return pd.concat(rows, ignore_index=True), diag


def main():
    pairs_by_window = {}
    elsa_fi = {"id": "idauniqc", "disease": {"hypertension": "hibpe", "diabetes": "diabe", "heart_disease": "hearte", "stroke": "stroke", "cancer": "cancre", "arthritis": "arthre"}, "shlt": "shlt", "bmi": "mbmi"}
    charls_fi = {"disease": {"hypertension": "r{w}hibpe", "diabetes": "r{w}diabe", "heart_disease": "r{w}hearte", "stroke": "r{w}stroke", "cancer": "r{w}cancre", "arthritis": "r{w}arthre"}, "shlt": "r{w}shlt", "bmi": "r{w}mbmi"}
    hrs_fi = {"disease": {"hypertension": "r{w}hibpe", "diabetes": "r{w}diabe", "heart_disease": "r{w}hearte", "stroke": "r{w}stroke", "cancer": "r{w}cancre", "arthritis": "r{w}arthre"}, "shlt": "r{w}shlt", "bmi": "r{w}bmi"}
    share_fi = {"id": "mergeid", "disease": {"hypertension": "hibpe", "diabetes": "diabe", "heart_disease": "hearte", "stroke": "stroke", "cancer": "cancre", "arthritis": "arthre"}, "shlt": "shlt", "bmi": "bmi"}
    pairs_by_window["ELSA"] = make_long_pairs("ELSA", ROOT / "ELSA" / "Working_data" / "elsa.dta", 6, 7, "idauniqc", elsa_fi, "w6xwgt")
    pairs_by_window["SHARE"] = make_long_pairs("SHARE", next((ROOT / "SHARE").rglob("Working_data/share.dta")), 6, 8, "mergeid", share_fi, "cciw_w6")
    pairs_by_window["CHARLS"] = make_wide_pairs("CHARLS", ROOT / "CHARLS" / "Harmonized_CHARLS" / "H_CHARLS_D_Data.dta", 3, 4, {"id": "ID", "active": "inw{w}", "age": "r{w}agey", "sex": "ragender", "ic": ["r{w}imrc", "r{w}dlrc", "r{w}orient", "r{w}walk100a", "r{w}walk1kma", "r{w}lgrip1", "r{w}lgrip2", "r{w}rgrip1", "r{w}rgrip2", "r{w}cesd10"], "ic_map": {"cognition": ["r{w}imrc", "r{w}dlrc", "r{w}orient"], "locomotion": ["r{w}walk100a", "r{w}walk1kma"], "grip_vitality": ["r{w}lgrip1", "r{w}lgrip2", "r{w}rgrip1", "r{w}rgrip2"], "psychological": ["r{w}cesd10"]}, "fi": ["r{w}shlt", "r{w}mbmi", "r{w}hibpe", "r{w}diabe", "r{w}hearte", "r{w}stroke", "r{w}cancre", "r{w}arthre"], "fi_map": charls_fi, "shlt_direction": "poor_is_high"})
    pairs_by_window["HRS"] = make_wide_pairs("HRS", ROOT / "HRS" / "RAND HRS Data" / "Longitudinal and Cross-Wave Data Products" / "randhrs1992_2022v1.dta", 10, 11, {"id": "hhidpn", "active": "inw{w}", "proxy": "r{w}proxy", "age": "r{w}agey_m", "sex": "ragender", "ic": ["r{w}imrc", "r{w}dlrc", "r{w}ser7", "r{w}walkra", "r{w}walk1a", "r{w}walksa", "r{w}grpl", "r{w}grpr", "r{w}cesd"], "ic_map": {"cognition": ["r{w}imrc", "r{w}dlrc", "r{w}ser7"], "locomotion": ["r{w}walkra", "r{w}walk1a", "r{w}walksa"], "grip_vitality": ["r{w}grpl", "r{w}grpr"], "psychological": ["r{w}cesd"]}, "fi": ["r{w}shlt", "r{w}bmi", "r{w}hibpe", "r{w}diabe", "r{w}hearte", "r{w}stroke", "r{w}cancre", "r{w}arthre"], "fi_map": hrs_fi, "shlt_direction": "poor_is_high"})
    all_rows, diags = [], []
    for ds, pairs in pairs_by_window.items():
        rows, diag = analyse(ds, pairs, {"ELSA": "6->7", "CHARLS": "3->4", "HRS": "10->11", "SHARE": "6->8"}[ds])
        all_rows.append(rows)
        diags.append({"dataset": ds, "window": {"ELSA": "6->7", "CHARLS": "3->4", "HRS": "10->11", "SHARE": "6->8"}[ds], "n_pairs": len(pairs), **diag})
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(TAB / "stage30_missingness_weighted_longitudinal_coefficients.csv", index=False, encoding="utf-8-sig")
    diag_df = pd.DataFrame(diags)
    diag_df.to_csv(TAB / "stage30_missingness_weighted_longitudinal_diagnostics.csv", index=False, encoding="utf-8-sig")
    memo = """# Stage 30 missingness-IPW and survey-weighted longitudinal sensitivity

This stage supplements the primary latent WLSMV analysis with an observed-proxy longitudinal sensitivity. The complete-case inclusion indicator is modeled from baseline-observable age, sex, FI-component availability, IC-domain availability, future FI-component availability, and the physical-module flag where available. Stabilized inverse-probability weights are clipped to [0.1, 10]. This is an MAR-type sensitivity under the stated baseline-observable model; it does not recover structurally absent modules and is not multiple imputation.

Survey-weight sensitivity uses the cohort-specific anchor-wave person/physical-module weight audited in Stage 21: ELSA w6xwgt, CHARLS r3wtrespb, HRS MPMWGTR for the physical-measures IC frame, and SHARE cciw_w6. The weights are normalized and clipped for numerical stability. Because ELSA lacks a verified public PSU/stratum pair and CHARLS lacks a named stratum variable in the audited file, this stage reports weighted regression coefficients with robust HC3 covariance as a sensitivity, not design-based population inference or weighted SEM.

The primary estimand remains the future FI association conditional on baseline FI, age, sex, and four IC proxies. The complete-case IPW, survey-weight-only, and combined IPW-times-survey-weight estimates are interpreted as robustness diagnostics. No individual IDs or derived row-level data are written.
"""
    (OUT / "stage30_missingness_weighted_longitudinal_memo.md").write_text(memo, encoding="utf-8")
    info = {"stage": 30, "script": "63_missingness_weighted_longitudinal_sensitivity.py", "windows": {"ELSA": "6->7", "CHARLS": "3->4", "HRS": "10->11", "SHARE": "6->8"}, "estimands": ["future_fi_adjusted_baseline"], "ipw_model": "baseline-observable logistic inclusion model; stabilized weights clipped 0.1-10", "survey_weights": {"ELSA": "w6xwgt", "CHARLS": "r3wtrespb", "HRS": "MPMWGTR", "SHARE": "cciw_w6"}, "individual_level_output": False, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "n_rows": int(len(out))}
    (OUT / "stage30_missingness_weighted_longitudinal_run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.groupby(["dataset", "weighting"]).size().to_string())
    print(diag_df.to_string(index=False))


if __name__ == "__main__":
    main()
