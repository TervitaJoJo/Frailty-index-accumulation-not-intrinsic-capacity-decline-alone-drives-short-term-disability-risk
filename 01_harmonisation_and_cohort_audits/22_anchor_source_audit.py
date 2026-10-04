"""Source-audited aggregate anchor profiles; raw inputs remain read-only.

Replaces SHARE subjective memory and ELSA precomputed cognition, fixes HRS
serial-7 scaling, and separates cognitive component and sample sensitivities.
No identifiers or individual-level derived scores are written to disk.
"""
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
DOMAINS = ["cognition", "locomotion", "grip_vitality", "psychological"]
DISEASES = dict(hypertension="hibpe", diabetes="diabe", heart_disease="hearte",
                stroke="stroke", cancer="cancre", arthritis="arthre")
SOURCE_FILES = set()


def unique_path(pattern):
    matches = list(ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one source for {pattern}; found {len(matches)}")
    return matches[0]


def read(path, columns):
    SOURCE_FILES.add(Path(path))
    # Missing required columns must raise, rather than alter the construct.
    return pyreadstat.read_dta(str(path), usecols=list(dict.fromkeys(columns)),
                              apply_value_formats=False)[0]


def bounded(x, low, high):
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x.between(low, high)).astype(float)


def binary(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x.isin([0, 1])).astype(float)


def grip(df, columns):
    # Keep low valid values; no adjustment for age or sex in this audit.
    return df[columns].apply(pd.to_numeric, errors="coerce").where(lambda x: x.ge(0)).max(axis=1)


def deficit_components(df, disease, shlt, bmi, direction):
    s = pd.DataFrame({name: binary(df[col]) for name, col in disease.items()}, index=df.index)
    x = bounded(df[shlt], 1, 5)
    s["self_rated_health"] = (5-x)/4 if direction == "poor_is_low" else (x-1)/4
    x = pd.to_numeric(df[bmi], errors="coerce").where(lambda z: z.gt(0))
    s["bmi"] = (x.lt(18.5) | x.ge(30)).astype(float).where(x.notna())
    assert s.stack().dropna().between(0, 1).all()
    return s


def score_deficits(s, minimum=6):
    n = s.notna().sum(axis=1)
    return s.sum(axis=1, min_count=1).div(n.where(n.gt(0))).where(n.ge(minimum))


def exclusive_list_total(df, columns):
    """Parallel word lists are alternatives; ambiguous multiple lists stay missing."""
    scores = df[columns].apply(lambda x: bounded(x, 0, 10))
    n = scores.notna().sum(axis=1)
    return scores.sum(axis=1, min_count=1).where(n.eq(1)), n


def assemble(ic, parts, components, mobility):
    d = pd.DataFrame(ic)
    d["cognition"] = parts.mean(axis=1, skipna=False)
    d["cognition_available"] = parts.mean(axis=1, skipna=True)
    d["cognition_component_n"] = parts.notna().sum(axis=1)
    d["cognition_component_required"] = parts.shape[1]
    d["mobility_distance"] = mobility
    d["frailty"] = score_deficits(components, 6)
    d["fi_observed_n"] = components.notna().sum(axis=1)
    d["fi_complete8"] = score_deficits(components, 8)
    # Additional exploratory diagnostic only, minimum 6 of 7, not a new main FI.
    d["fi_without_srh"] = score_deficits(components.drop(columns="self_rated_health"), 6)
    return d


def load_anchors():
    """Return in-memory frames and aggregate evidence; do not export the frames."""
    evidence, frames = [], {}
    specs = {
        "ELSA": (ROOT/"ELSA/Working_data/elsa.dta", "wave", 6, "", "mbmi", "poor_is_low",
                 ["imrc", "dlrc", "orient", "tcog_z_z", "walkra", "walk100a", "gripsum", "cesd"]),
        "CHARLS": (ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta", "inw3", 1, "r3", "mbmi", "poor_is_high",
                   ["r3orient", "r3tr20", "r3walk100a", "r3lgrip", "r3rgrip", "r3cesd10"]),
        "HRS": (ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta", "inw10", 1, "r10", "bmi", "poor_is_high",
                ["r10imrc", "r10dlrc", "r10ser7", "r10walkra", "r10walk1a", "r10grpl", "r10grpr", "r10cesd", "r10agey_e", "ragender", "r10proxy"]),
        "LASI": (unique_path("LASI/**/Working_data/lasi.dta"), "r1iwy", None, "r1", "mbmi", "poor_is_low",
                 ["r1cog_total", "r1walk100a", "r1lgrip", "r1rgrip", "r1cesd10_l"]),
    }
    for cohort, (path, active_col, active_value, prefix, bmi, direction, extra) in specs.items():
        disease = {k: prefix+v for k, v in DISEASES.items()}
        df = read(path, [active_col, prefix+"shlt", prefix+bmi, *disease.values(), *extra])
        active = df[active_col].notna() if active_value is None else df[active_col].eq(active_value)
        df = df.loc[active].copy()
        if cohort == "ELSA":
            memory = (bounded(df.imrc, 0, 10)+bounded(df.dlrc, 0, 10))/20
            parts = pd.DataFrame({"recall": memory, "orientation": bounded(df.orient, 0, 4)/4})
            ic = dict(locomotion=1-binary(df.walkra), grip_vitality=grip(df, ["gripsum"]), psychological=1-bounded(df.cesd, 0, 8)/8)
            mobility = 1-binary(df.walk100a)
            raw_missing = df[["imrc", "dlrc", "orient"]].isna().all(axis=1)
            evidence.append(dict(cohort=cohort, check="legacy_total_valid_with_all_3_raw_tests_missing", n=int((raw_missing & df.tcog_z_z.notna()).sum()), detail="Precomputed total is not used in the revised profile."))
        elif cohort == "CHARLS":
            parts = pd.DataFrame({"recall": bounded(df.r3tr20, 0, 20)/20, "orientation": bounded(df.r3orient, 0, 4)/4})
            ic = dict(locomotion=1-binary(df.r3walk100a), grip_vitality=grip(df, ["r3lgrip", "r3rgrip"]), psychological=1-bounded(df.r3cesd10, 0, 30)/30)
            mobility = ic["locomotion"]
        elif cohort == "HRS":
            parts = pd.DataFrame({"immediate": bounded(df.r10imrc, 0, 10)/10, "delayed": bounded(df.r10dlrc, 0, 10)/10, "serial7": bounded(df.r10ser7, 0, 5)/5})
            ic = dict(locomotion=1-binary(df.r10walkra), grip_vitality=grip(df, ["r10grpl", "r10grpr"]), psychological=1-bounded(df.r10cesd, 0, 8)/8)
            mobility = 1-binary(df.r10walk1a)
            evidence.append(dict(cohort=cohort, check="serial7_range", n=int(df.r10ser7.notna().sum()), detail=f"Observed {df.r10ser7.min():g} to {df.r10ser7.max():g}; five subtractions, scale by 5, not 7."))
        else:
            parts = pd.DataFrame({"total": pd.to_numeric(df.r1cog_total, errors="coerce").where(lambda z: z.ge(0))})
            ic = dict(locomotion=1-binary(df.r1walk100a), grip_vitality=grip(df, ["r1lgrip", "r1rgrip"]), psychological=1-bounded(df.r1cesd10_l, 0, 10)/10)
            mobility = ic["locomotion"]
        components = deficit_components(df, disease, prefix+"shlt", prefix+bmi, direction)
        d = assemble(ic, parts, components, mobility)
        if cohort == "HRS":
            d["age"] = bounded(df.r10agey_e, 0, 120)
            d["male"] = df.ragender.eq(1).astype(float).where(df.ragender.isin([1, 2]))
            d["any_proxy"] = df.r10proxy.ne(0).astype(float).where(df.r10proxy.isin([0, 1, 2, 3]))
        frames[cohort] = d

    # Observed objective CF tests only; no supplied MI values in this profile.
    path = unique_path("SHARE/**/Working_data/share.dta")
    df = read(path, ["mergeid", "wave", "walkra", "walk100a", "lgrip", "rgrip", "eurod", "shlt", "bmi", *DISEASES.values()])
    df = df.loc[df.wave.eq(6)].copy()
    cfpath = unique_path("SHARE/**/*w6*cf.dta")
    orientation_cols = [f"cf{i:03d}_" for i in range(3, 7)]
    immediate_cols = [f"cf{i}tot" for i in range(104, 108)]
    delayed_cols = [f"cf{i}tot" for i in range(113, 117)]
    cf = read(cfpath, ["mergeid", "country", "cf103_", *orientation_cols, *immediate_cols, *delayed_cols])
    df = df.merge(cf, on="mergeid", how="left", validate="one_to_one")
    immediate, n_im = exclusive_list_total(df, immediate_cols)
    delayed, n_de = exclusive_list_total(df, delayed_cols)
    # Conservative observed-only rule: -1/-2 missing; category 2 explicitly
    # denotes incorrect/does not know and scores zero. Questionnaire review pending.
    orientation = df[orientation_cols].apply(lambda s: s.map({1: 1.0, 2: 0.0})).sum(axis=1, min_count=4)
    parts = pd.DataFrame({"recall": (immediate+delayed)/20, "orientation": orientation/4})
    components = deficit_components(df, DISEASES, "shlt", "bmi", "poor_is_low")
    d = assemble(dict(locomotion=1-binary(df.walkra), grip_vitality=grip(df, ["lgrip", "rgrip"]), psychological=1-bounded(df.eurod, 0, 12)/12), parts, components, 1-binary(df.walk100a))
    d["country"] = df.country
    frames["SHARE"] = d
    for label, counts in [("immediate", n_im), ("delayed", n_de)]:
        evidence.append(dict(cohort="SHARE", check=f"{label}_more_than_one_word_list", n=int(counts.gt(1).sum()), detail="Ambiguous alternative-list totals excluded from that test; no maximum or sum across lists."))
    # Implicat 1 is read solely to verify the legacy variable's meaning.
    imp = read(unique_path("SHARE/**/*w6*gv_imputations.dta"), ["mergeid", "implicat", "memory", "memory_f"])
    imp = imp.loc[imp.implicat.eq(1)].merge(cf[["mergeid", "cf103_"]], on="mergeid", how="left", validate="one_to_one")
    matched = imp.loc[imp.memory_f.eq(3) & imp.cf103_.between(1, 5), ["memory", "cf103_"]].dropna()
    evidence.append(dict(cohort="SHARE", check="memory_matches_self_rated_cf103", n=len(matched), detail=f"{matched.memory.eq(matched.cf103_).mean()*100:.2f}% exact match among regular observations; cf103 labels Excellent to Poor."))
    for cohort, d in frames.items():
        assert d["frailty"].dropna().between(0, 1).all()
        assert d["locomotion"].dropna().isin([0, 1]).all()
        assert d["psychological"].dropna().between(0, 1).all()
        assert d["cognition"].notna().sum() <= d["cognition_available"].notna().sum()
    return frames, pd.DataFrame(evidence)


def descriptives(cohort, d):
    rows = []
    for col in [*DOMAINS, "frailty", "cognition_available", "mobility_distance"]:
        x = d[col].dropna()
        rows.append(dict(dataset=cohort, construct=col, n_active=len(d), n_valid=len(x), pct_valid=100*len(x)/len(d),
                         mean=x.mean(), sd=x.std(), median=x.median(), p25=x.quantile(.25), p75=x.quantile(.75),
                         n_unique=x.nunique(), pct_at_min=100*x.eq(x.min()).mean(), pct_at_max=100*x.eq(x.max()).mean()))
    for name, cols in [("ic_complete_4_domains", DOMAINS), ("joint_ic_fi", [*DOMAINS, "frailty"])]:
        mask = d[cols].notna().all(axis=1)
        rows.append(dict(dataset=cohort, construct=name, n_active=len(d), n_valid=int(mask.sum()), pct_valid=100*mask.mean()))
    return rows


def correlations(cohort, d, sample="pairwise", pairs=None):
    rows = []
    pairs = pairs or list(combinations([*DOMAINS, "frailty"], 2))
    for a, b in pairs:
        p = d[[a, b]].dropna()
        rho = p[a].corr(p[b], method="spearman") if len(p)>2 and p[a].nunique()>1 and p[b].nunique()>1 else np.nan
        rows.append(dict(dataset=cohort, sample=sample, construct_a=a, construct_b=b, n_pairwise=len(p), spearman=rho))
    return rows


def main():
    TAB.mkdir(exist_ok=True)
    frames, evidence = load_anchors()
    summary, corr, sensitivity, missingness, variants, country_rows = [], [], [], [], [], []
    for cohort, d in frames.items():
        summary.extend(descriptives(cohort, d))
        corr.extend(correlations(cohort, d))
        for label, subset in [("all_available", d), ("grip_observed", d.loc[d.grip_vitality.notna()]),
                              ("grip_missing", d.loc[d.grip_vitality.isna()]),
                              ("joint_complete", d.dropna(subset=[*DOMAINS, "frailty"]))]:
            sensitivity.extend(correlations(cohort, subset, label))
        sensitivity.extend(correlations(cohort, d, "component_and_wording_sensitivity", [("cognition_available", "frailty"), ("mobility_distance", "frailty")]))
        # Match people for each comparison; separate definition from sample change.
        for domain in DOMAINS:
            for fi_variant in ["fi_complete8", "fi_without_srh"]:
                p = d[[domain, "frailty", fi_variant]].dropna()
                variants.append(dict(dataset=cohort, domain=domain, fi_variant=fi_variant, n_matched=len(p),
                                     primary_rho=p[domain].corr(p.frailty, method="spearman"),
                                     variant_rho=p[domain].corr(p[fi_variant], method="spearman")))
        for label, subset in [("grip_observed", d.loc[d.grip_vitality.notna()]), ("grip_missing", d.loc[d.grip_vitality.isna()])]:
            for variable in ["frailty", "cognition", "psychological", "locomotion", "age", "male", "any_proxy"]:
                if variable in subset:
                    x = subset[variable].dropna()
                    missingness.append(dict(dataset=cohort, group=label, variable=variable, n_group=len(subset), n_valid=len(x), mean=x.mean(), sd=x.std(), median=x.median()))
        if cohort == "SHARE":
            for country, g in d.groupby("country", dropna=False):
                country_rows.append(dict(country=country, n_active=len(g), cognition_pct=100*g.cognition.notna().mean(),
                                         grip_pct=100*g.grip_vitality.notna().mean(), joint_ic_fi_pct=100*g[[*DOMAINS, "frailty"]].notna().all(axis=1).mean()))
    outputs = {
        "anchor_construct_profile_summary.csv": summary,
        "anchor_construct_spearman.csv": corr,
        "anchor_construct_sensitivity.csv": sensitivity,
        "anchor_fi_definition_sensitivity.csv": variants,
        "anchor_grip_selection_summary.csv": missingness,
        "share_anchor_country_coverage.csv": country_rows,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).round(6).to_csv(TAB/name, index=False, encoding="utf-8-sig")
    evidence.to_csv(TAB/"anchor_source_corrections.csv", index=False, encoding="utf-8-sig")
    run = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
               pandas=pd.__version__, numpy=np.__version__, pyreadstat=pyreadstat.__version__,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               source_files=[dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(SOURCE_FILES)],
               design="unweighted aggregate exploratory screen; raw observed SHARE CF; no SEM, new imputation, or inference",
               cognition_rule="all selected normalized components required; available-component sensitivity separately reported",
               fi_rule="BMI<18.5 or >=30; minimum 6/8; observed denominator; short health-deficit comparator, validation pending")
    (OUT/"stage6_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"anchor_construct_profiles.md").write_text("""# Anchor construct profiles — source-audited v2

Supersedes the first stage-6 profile. Original files are retained in `archive/stage6_before_source_audit_20260923/`.

- ELSA cognition now uses observed immediate/delayed recall and orientation; the precomputed total is excluded.
- SHARE cognition uses raw objective CF recall (alternative word lists coalesced only when exactly one valid list exists) and orientation. The legacy `gv_imputations.memory` is self-rated memory and is excluded. No supplied MI values enter these revised profiles. Future use of supplied MI data still requires all five imputations and explicit flag handling.
- HRS serial-7 is divided by 5; this three-component score is not the 27-point total.
- The reference profile requires all selected cognitive components. Available-component cognition is separately tabulated.
- Legacy room-walking vs 100-metre wording is retained for a diagnostic reference only. `mobility_distance` uses 100 metres in ELSA/SHARE/CHARLS/LASI and one block in HRS; these are candidates, not established equivalent items.
- FI remains the user-selected eight-component score. It is a short health-deficit comparator with pending frailty construct validation, not a validated standard FI or a latent factor.

Spearman estimates are descriptive, unweighted, pairwise and unadjusted. No significance tests, confidence intervals, latent correlations, or invariance claims are made. See `stage6_discriminant_validity_memo.md` for interpretation.
""", encoding="utf-8")
    print("Wrote source-audited anchor profiles and sensitivity tables.")


if __name__ == "__main__":
    main()
