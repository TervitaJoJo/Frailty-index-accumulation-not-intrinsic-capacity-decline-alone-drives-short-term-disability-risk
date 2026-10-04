"""Audit raw indicators for within-cohort latent IC domains.

This is a pre-CFA audit for the selected scheme A. It exports only metadata,
aggregate coverage and pairwise correlations; no person-level file is written.
Indicators are retained as candidates until the source labels, ranges and
module-selection rules are frozen.
"""
from pathlib import Path
from itertools import combinations
import json
import platform
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"


def one(pattern):
    x = list(ROOT.glob(pattern))
    if len(x) != 1:
        raise ValueError(f"Expected one file for {pattern}; found {len(x)}")
    return x[0]


def elsa_wave6(filename):
    x = [p for p in ROOT.glob(f"ELSA/**/wave6/{filename}") if "_CN" not in p.name]
    if not x:
        raise ValueError(f"ELSA Wave 6 source not found: {filename}")
    # Prefer the UKDA release over a duplicate local copy when both exist.
    x.sort(key=lambda p: ("UKDA-5050" not in str(p), str(p)))
    return x[0]


def meta(path):
    return pyreadstat.read_dta(str(path), metadataonly=True)[1]


def read(path, cols):
    m = meta(path)
    available = [c for c in dict.fromkeys(cols) if c in m.column_names]
    return pyreadstat.read_dta(str(path), usecols=available, apply_value_formats=False)[0], m, available


def valid(x):
    x = pd.to_numeric(x, errors="coerce")
    # All selected raw indicators are non-negative measurement scales; negative
    # values are survey-specific missing/refusal codes (including ELSA -1).
    return x.mask(x < 0).astype(float)


def candidate(dataset, wave, domain, indicator, path, source_module, direction, role, note=""):
    return dict(dataset=dataset, anchor_wave=wave, domain=domain, indicator=indicator,
                source_module=source_module, direction=direction, role=role, note=note,
                source_path=str(path))


def manifest():
    rows = []
    def add(ds,w,domain,inds,path,module,direction="higher_better",role="primary",note=""):
        for ind in inds:
            rows.append(candidate(ds,w,domain,ind,path,module,direction,role,note))
    elsa = ROOT/"ELSA/Working_data/elsa.dta"
    elsa_raw = elsa_wave6("wave_6_elsa_data_v2.dta")
    elsa_nurse = elsa_wave6("wave_6_elsa_nurse_data_v2.dta")
    add("ELSA",6,"cognition",["imrc","dlrc","orient"],elsa,"working",note="Observed cognitive tests; tcog_z_z sensitivity only")
    add("ELSA",6,"cognition",["verbf"],elsa,"working","higher_better","sensitivity","No valid Wave 6 observations in current working file")
    add("ELSA",6,"locomotion",["walkra","walk100a"],elsa,"working","higher_worse","primary","Self-report wording candidates")
    add("ELSA",6,"locomotion",["wspeed","walkcomp"],elsa,"working","mixed","sensitivity","Objective speed and completion")
    add("ELSA",6,"grip_vitality",["gripsum","gripcomp"],elsa,"working","higher_better","sensitivity","Working-file summary and completion")
    add("ELSA",6,"grip_vitality",["mmgsd1","mmgsd2","mmgsd3","mmgsn1","mmgsn2","mmgsn3"],elsa_nurse,"raw_nurse","higher_better","primary","Raw dominant/non-dominant repeated measurements")
    add("ELSA",6,"psychological",["cesd"],elsa,"working","higher_worse","sensitivity","Working-file total score")
    add("ELSA",6,"psychological",["PScedA","PScedB","PScedC","PScedD","PScedE","PScedF","PScedG","PScedH"],elsa_raw,"raw_wave6","mixed","primary","Raw CES-D items; positive items need reverse coding")

    charls = ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"
    add("CHARLS",3,"cognition",["r3imrc","r3dlrc","r3orient","r3ser7"],charls,"working",note="Raw cognitive tests")
    add("CHARLS",3,"cognition",["r3tr20"],charls,"working","higher_better","sensitivity","Summary score; avoid treating as independent of recall components")
    add("CHARLS",3,"locomotion",["r3walk100a","r3walk1kma"],charls,"working","higher_worse","primary","Self-report distance indicators")
    add("CHARLS",3,"locomotion",["r3wspeed1","r3wspeed2","r3wspeed","r3walkcomp"],charls,"working","mixed","sensitivity","Objective walking module")
    add("CHARLS",3,"grip_vitality",["r3lgrip","r3rgrip","r3gripsum","r3gripcomp"],charls,"working","higher_better","primary","Left/right and summary grip")
    add("CHARLS",3,"psychological",["r3depresl","r3effortl","r3sleeprl","r3whappyl","r3flonel","r3botherl","r3goingl","r3mindtsl","r3fhopel","r3fearll"],charls,"working","mixed","primary","CES-D items; positive items need reverse coding")
    add("CHARLS",3,"psychological",["r3cesd10"],charls,"working","higher_worse","sensitivity","Derived total; not independent of items")

    hrs = ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
    add("HRS",10,"cognition",["r10imrc","r10dlrc","r10ser7","r10bwc20"],hrs,"working",note="Observed cognitive tests; cog27 sensitivity")
    add("HRS",10,"locomotion",["r10walkra","r10walk1a","r10walksa"],hrs,"working","higher_worse","primary","Self-report distance candidates")
    add("HRS",10,"locomotion",["r10timwlk","r10timwlka"],hrs,"working","mixed","sensitivity","Timed walk and aid flag")
    add("HRS",10,"grip_vitality",["r10grpl","r10grpr"],hrs,"working","higher_better","primary","Left/right grip; module selection remains structural")
    add("HRS",10,"psychological",["r10depres","r10effort","r10sleepr","r10whappy","r10flone","r10fsad","r10going","r10enlife"],hrs,"working","mixed","primary","CES-D items; positive items need reverse coding")
    add("HRS",10,"psychological",["r10cesd"],hrs,"working","higher_worse","sensitivity","Derived total; not independent of items")

    share = one("SHARE/**/Working_data/share.dta")
    cf = one("SHARE/**/*w6*cf.dta")
    gh = one("SHARE/**/*w6*gv_health.dta")
    mh = one("SHARE/**/*w6*mh.dta")
    ac = one("SHARE/**/*w6*ac.dta")
    add("SHARE",6,"cognition",["cf003_","cf004_","cf005_","cf006_"],cf,"cf",note="Date-orientation items")
    add("SHARE",6,"cognition",[f"cf{i}tot" for i in range(104,108)]+[f"cf{i}tot" for i in range(113,117)],cf,"cf",note="Word-list totals; alternate-list structure requires questionnaire review")
    add("SHARE",6,"cognition",["cf010_","cf012_","cf013_","cf014_","cf015_"],cf,"cf",note="Verbal fluency/numeracy candidates")
    add("SHARE",6,"locomotion",["walkra","walk100a"],share,"working","higher_worse","primary","Self-report wording candidates")
    add("SHARE",6,"locomotion",["wspeed","walkcomp"],share,"working","mixed","sensitivity","Objective walking module")
    add("SHARE",6,"grip_vitality",["lgrip","rgrip","maxgrip"],share,"working","higher_better","primary","Left/right and maximum grip")
    add("SHARE",6,"psychological",["euro1","euro2","euro5","euro11"],gh,"gv_health","mixed","primary","EURO-D item candidates; item direction requires labels")
    add("SHARE",6,"psychological",["eurod","eurodcat"],gh,"gv_health","mixed","sensitivity","Derived EURO-D total/caseness; not independent of items")
    add("SHARE",6,"psychological",["mh002_","mh007_","mh016_","mh037_"],mh,"mh","mixed","sensitivity","Alternative mental-health item set")
    add("SHARE",6,"grip_vitality",["ac023_"],ac,"ac","higher_better","sensitivity","Energy item as vitality sensitivity")

    lasi = one("LASI/**/Working_data/lasi.dta")
    add("LASI",1,"cognition",["r1imrc","r1dlrc","r1orient","r1orientp","r1bwc20a","r1bwc100a","r1ser7"],lasi,"working",note="Raw cognitive tests")
    add("LASI",1,"cognition",["r1tr20"],lasi,"working","higher_better","sensitivity","Summary score; avoid treating as independent of recall components")
    add("LASI",1,"locomotion",["r1walkra","r1walk100a"],lasi,"working","higher_worse","primary","Self-report distance indicators")
    add("LASI",1,"locomotion",["r1wspeed1","r1wspeed2","r1wspeed","r1walkcomp"],lasi,"working","mixed","sensitivity","Objective walking module")
    add("LASI",1,"grip_vitality",["r1lgrip1","r1lgrip2","r1rgrip1","r1rgrip2","r1lgrip","r1rgrip"],lasi,"working","higher_better","primary","Repeated left/right grip measures")
    add("LASI",1,"psychological",["r1cesd10_l"],lasi,"working","higher_worse","primary","CES-D total; item-level set not located")
    add("LASI",1,"psychological",["r1cesd10","r1cesd10dep"],lasi,"working","higher_worse","sensitivity","Derived CES-D variants; not independent items")
    return pd.DataFrame(rows)


def load_cohort(ds, m, frames):
    path = frames[ds]
    if ds == "ELSA":
        # The working file uses idauniqc; raw Wave 6 modules use idauniq.
        working = m.loc[m.source_module.eq("working"), "indicator"].tolist()
        base, _, available = read(path, ["idauniqc", "wave", *working])
        base = base.loc[base.wave.eq(6)].copy()
        base["__key"] = base.idauniqc.astype(str).str.strip()
        out = base[["__key", *[c for c in available if c not in ["idauniqc", "wave"]]]].copy()
        module_files = {"raw_wave6": (frames["ELSA_RAW"], "idauniq"), "raw_nurse": (frames["ELSA_NURSE"], "idauniq")}
        for mod, g in m.loc[~m.source_module.eq("working")].groupby("source_module"):
            mp, key = module_files[mod]
            cols = [key] + g.indicator.tolist()
            x, _, avail = read(mp, cols)
            x["__key"] = x[key].astype(str).str.strip()
            keep = ["__key"] + [c for c in avail if c != key]
            x = x[keep].drop_duplicates("__key")
            out = out.merge(x, on="__key", how="left", validate="one_to_one")
        return out
    if ds == "SHARE":
        base, bm, _ = read(path, ["mergeid","wave"])
        base = base.loc[base.wave.eq(6)].copy()
        module_files = {"cf": one("SHARE/**/*w6*cf.dta"), "gv_health": one("SHARE/**/*w6*gv_health.dta"), "mh": one("SHARE/**/*w6*mh.dta"), "ac": one("SHARE/**/*w6*ac.dta"), "working": path}
        out = base[["mergeid"]].copy()
        for mod, g in m.groupby("source_module"):
            mp = module_files[mod]
            cols = ["mergeid"] + g.indicator.tolist()
            if mod == "working":
                cols = ["mergeid", "wave"] + g.indicator.tolist()
            x, _, available = read(mp, cols)
            if mod == "working":
                x = x.loc[x.wave.eq(6)].drop(columns="wave")
                available = [c for c in available if c != "wave"]
            out = out.merge(x[available], on="mergeid", how="left", validate="one_to_one")
        return out
    active = {"ELSA": ("wave",6), "CHARLS": ("inw3",1), "HRS": ("inw10",1), "LASI": ("r1iwy",None)}[ds]
    cols = [active[0]] + m.indicator.tolist()
    x, _, available = read(path, cols)
    x = x.loc[x[active[0]].notna() if active[1] is None else x[active[0]].eq(active[1])].copy()
    return x[available]


def main():
    TAB.mkdir(exist_ok=True)
    man = manifest()
    frames = {"ELSA": ROOT/"ELSA/Working_data/elsa.dta", "ELSA_RAW": elsa_wave6("wave_6_elsa_data_v2.dta"), "ELSA_NURSE": elsa_wave6("wave_6_elsa_nurse_data_v2.dta"), "CHARLS": ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta", "HRS": ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta", "SHARE": one("SHARE/**/Working_data/share.dta"), "LASI": one("LASI/**/Working_data/lasi.dta")}
    coverage, cors, domain_summary = [], [], []
    for ds in man.dataset.unique():
        sub = man.loc[man.dataset.eq(ds)].copy()
        dat = load_cohort(ds, sub, frames)
        n = len(dat)
        for _, row in sub.iterrows():
            ind = row.indicator
            if ind not in dat.columns:
                coverage.append({**row.to_dict(), "n_active": n, "n_valid": 0, "pct_valid": 0, "mean": np.nan, "sd": np.nan, "n_unique": 0, "status": "not_found"})
                continue
            x = valid(dat[ind])
            y = x.dropna()
            coverage.append({**row.to_dict(), "n_active": n, "n_valid": len(y), "pct_valid": 100*len(y)/n if n else np.nan,
                             "mean": y.mean() if len(y) else np.nan, "sd": y.std() if len(y)>1 else np.nan,
                             "n_unique": y.nunique(), "status": "available" if len(y) else "empty"})
        for (domain, role), g in sub.groupby(["domain", "role"]):
            cols = [c for c in g.indicator if c in dat.columns]
            z = pd.DataFrame({c: valid(dat[c]) for c in cols})
            for a,b in combinations(cols,2):
                p = z[[a,b]].dropna()
                cors.append(dict(dataset=ds,domain=domain,indicator_a=a,indicator_b=b,n_pairwise=len(p),spearman=p[a].corr(p[b],method="spearman") if len(p)>2 and p[a].nunique()>1 and p[b].nunique()>1 else np.nan))
            usable = z.notna().sum(axis=1)
            for threshold in [1,2,3]:
                domain_summary.append(dict(dataset=ds,domain=domain,role=role,n_active=n,n_indicators_manifest=len(g),n_indicators_found=len(cols),n_at_least_threshold=int(usable.ge(threshold).sum()),pct_at_least_threshold=100*usable.ge(threshold).mean(),threshold=threshold))
    man.to_csv(TAB/"domain_indicator_manifest.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(coverage).round(6).to_csv(TAB/"domain_indicator_coverage.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(cors).round(6).to_csv(TAB/"domain_indicator_spearman.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(domain_summary).round(6).to_csv(TAB/"domain_indicator_domain_summary.csv",index=False,encoding="utf-8-sig")
    report = ["# Stage 7 domain indicator audit", "", "This is a pre-CFA audit for scheme A. It uses raw, source-audited candidate indicators within each anchor wave and writes only aggregate coverage and correlations.", "", "## Rules", "", "- Candidate indicators with missing source columns are retained in the manifest and marked `not_found`; they are not silently replaced.", "- Special negative codes are treated as missing for coverage and correlation summaries.", "- The audit does not choose item coding, estimate latent factors, perform MI, use survey weights or claim invariance.", "- SHARE modules are merged by `mergeid` after restricting the working file to wave 6; module-level one-to-one validation is enforced.", "", "## Outputs", "", "See `tables/domain_indicator_manifest.csv`, `domain_indicator_coverage.csv`, `domain_indicator_spearman.csv` and `domain_indicator_domain_summary.csv`. Domains with at least two defensible indicators should enter within-cohort CFA; single-indicator domains require a preregistered fixed-error or sensitivity treatment."]
    (OUT/"stage7_domain_indicator_audit.md").write_text("\n".join(report),encoding="utf-8")
    run = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyreadstat": pyreadstat.__version__, "cohorts": list(man.dataset.unique()), "scope": "anchor waves; aggregate-only pre-CFA audit"}
    (OUT/"stage7_run_info.json").write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Wrote Stage 7 domain indicator audit")


if __name__ == "__main__":
    main()
