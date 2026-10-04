from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

def num(x, lo=None, hi=None):
    z = pd.to_numeric(x, errors="coerce")
    z = z.mask(z < -90)
    if lo is not None:
        z = z.mask(z < lo)
    if hi is not None:
        z = z.mask(z > hi)
    return z

def bin01(x):
    z = num(x)
    return z.where(z.isin([0,1]))

def max_valid(df, cols):
    vals = [num(df[c]) for c in cols if c in df]
    if not vals:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(vals, axis=1).max(axis=1, skipna=True)

def fi_score(df, prefix_map, shlt_direction):
    out = pd.DataFrame(index=df.index)
    for name, col in prefix_map["disease"].items():
        out[name] = bin01(df[col]) if col in df else np.nan
    shlt = num(df[prefix_map["shlt"]], 1, 5) if prefix_map["shlt"] in df else pd.Series(np.nan,index=df.index)
    out["self_rated_health"] = ((5-shlt)/4 if shlt_direction=="poor_is_low" else (shlt-1)/4)
    bmi = num(df[prefix_map["bmi"]], 0, 200) if prefix_map["bmi"] in df else pd.Series(np.nan,index=df.index)
    out["bmi"] = (bmi.lt(18.5) | bmi.ge(30)).where(bmi.notna())
    out = out.apply(pd.to_numeric, errors="coerce")
    observed = out.notna().sum(axis=1)
    total = out.sum(axis=1, min_count=1)
    fi = total.div(observed).where(observed.ge(6))
    return fi, observed, out.notna().all(axis=1)

def domain_availability_long(df, dataset):
    if dataset == "ELSA":
        cog_cols = ["imrc","dlrc","orient"]
        loc_cols = ["walkra","walk100a"]
        grip = "gripsum" if "gripsum" in df else "gripcomp"
        psych = "cesd"
    else:
        cog_cols = [c for c in ["imrc","dlrc","orient","tr20"] if c in df]
        loc_cols = ["walkra","walk100a"]
        grip = "gripcomp" if "gripcomp" in df else "gripsum"
        psych = "eurod"
    cog_n = df[cog_cols].apply(lambda x: num(x).notna(), axis=0).sum(axis=1) if cog_cols else pd.Series(0,index=df.index)
    loc_n = df[loc_cols].apply(lambda x: num(x).notna(), axis=0).sum(axis=1)
    psych_ok = num(df[psych]).notna() if psych in df else pd.Series(False,index=df.index)
    return pd.DataFrame({
        "cognition_available": cog_n >= (2 if len(cog_cols)>=3 else 2),
        "locomotion_available": loc_n >= 1,
        "grip_vitality_available": num(df[grip]).notna() if grip in df else False,
        "psychological_available": psych_ok,
    }, index=df.index)

def read_long(path, cols):
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in dict.fromkeys(cols) if c in meta.column_names]
    d, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    return d

def read_wide(path, cols):
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in dict.fromkeys(cols) if c in meta.column_names]
    d, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    return d

def build_long_results(dataset, path, anchor_wave, target_waves, spec):
    cols = ["wave", spec["id"], *spec["ic"], *spec["fi"]]
    d = read_long(path, cols)
    d = d.loc[d["wave"].isin([anchor_wave,*target_waves]) & d[spec["id"]].notna()].copy()
    d["dataset"] = dataset
    d["anchor_wave"] = anchor_wave
    d["target_rule"] = "active_id_nonmissing"
    domain_cols = ["imrc","dlrc","orient","tr20","walkra","walk100a","gripsum","gripcomp","eurod","cesd"]
    dom = domain_availability_long(d, dataset)
    dom["all_four_ic"] = dom.all(axis=1)
    d = pd.concat([d,dom],axis=1)
    fi, observed, complete = fi_score(d, spec["fi_map"], spec["shlt_direction"])
    d["fi"] = fi
    d["fi_observed_n"] = observed
    d["fi_complete8"] = complete
    anchor = d.loc[d.wave.eq(anchor_wave)].copy()
    rows = []
    for target in target_waves:
        fut = d.loc[d.wave.eq(target)].copy()
        pairs = anchor[[spec["id"],"all_four_ic","fi"]].rename(columns={"all_four_ic":"baseline_all4_ic","fi":"baseline_fi"}).merge(
            fut[[spec["id"],"fi","fi_complete8"]].rename(columns={"fi":"future_fi","fi_complete8":"future_fi_complete8"}),
            on=spec["id"], how="inner")
        pairs["pair_complete"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi"]].notna().all(axis=1)
        pairs["pair_complete8_future"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi_complete8"]].notna().all(axis=1)
        rows.append({"dataset":dataset,"anchor_wave":anchor_wave,"target_wave":target,"target_rule":"active_id_nonmissing","n_anchor":len(anchor),"n_target":len(fut),"n_anchor_all4_ic":int(anchor["all_four_ic"].sum()),"n_anchor_fi":int(anchor["fi"].notna().sum()),"n_future_fi":int(fut["fi"].notna().sum()),"n_future_fi_complete8":int(fut["fi_complete8"].notna().sum()),"n_pair_baseline_ic_fi_future_fi":int(pairs["pair_complete"].sum()),"pct_anchor_pair":round(float(pairs["pair_complete"].mean()*100),2) if len(pairs) else np.nan,"n_pair_baseline_ic_fi_future_complete8":int(pairs["pair_complete8_future"].sum()),"pct_anchor_pair_complete8":round(float(pairs["pair_complete8_future"].mean()*100),2) if len(pairs) else np.nan})
    return pd.DataFrame(rows), d

def build_wide_results(dataset, path, anchor_wave, target_waves, spec):
    cols = [spec["id"]]
    for w in [anchor_wave,*target_waves]:
        cols += [spec["active"].format(w=w), spec["proxy"].format(w=w) if spec.get("proxy") else None]
        cols += [c.format(w=w) for c in spec["ic"]]
        cols += [c.format(w=w) for c in spec["fi"]]
    d = read_wide(path, cols)
    d = d.loc[d[spec["id"]].notna()].copy()
    base_active_col = spec["active"].format(w=anchor_wave)
    base_proxy_col = spec["proxy"].format(w=anchor_wave) if spec.get("proxy") else None
    anchor = pd.DataFrame(index=d.index)
    anchor["id"] = d[spec["id"]]
    anchor["active"] = num(d[base_active_col]).eq(1) if base_active_col in d else True
    anchor["proxy"] = num(d[base_proxy_col]).eq(1) if base_proxy_col and base_proxy_col in d else False
    anchor = anchor.loc[anchor["active"] & (~anchor["proxy"] if spec.get("exclude_proxy") else True)].copy()
    def make_wave(w):
        out = pd.DataFrame(index=anchor.index)
        out["id"] = d.loc[anchor.index,spec["id"]]
        act = spec["active"].format(w=w)
        prox = spec["proxy"].format(w=w) if spec.get("proxy") else None
        out["active"] = num(d.loc[anchor.index,act]).eq(1) if act in d else True
        out["proxy"] = num(d.loc[anchor.index,prox]).eq(1) if prox and prox in d else False
        ic_cols = [c.format(w=w) for c in spec["ic"]]
        ic = pd.DataFrame(index=anchor.index)
        for domain, cols2 in spec["ic_map"].items():
            vals = []
            for c in cols2:
                c2=c.format(w=w)
                if c2 in d: vals.append(num(d.loc[anchor.index,c2]))
            if domain=="cognition":
                ic[domain+"_available"] = pd.concat(vals,axis=1).notna().sum(axis=1).ge(2) if vals else False
            elif domain=="locomotion":
                ic[domain+"_available"] = pd.concat(vals,axis=1).notna().sum(axis=1).ge(1) if vals else False
            elif domain=="grip_vitality":
                ic[domain+"_available"] = pd.concat(vals,axis=1).notna().any(axis=1) if vals else False
            else:
                ic[domain+"_available"] = pd.concat(vals,axis=1).notna().any(axis=1) if vals else False
        ic["all_four_ic"] = ic.all(axis=1)
        fi_map = {"disease":{},"shlt":"","bmi":""}
        for name,c in spec["fi_map"]["disease"].items(): fi_map["disease"][name]=c.format(w=w)
        fi_map["shlt"]=spec["fi_map"]["shlt"].format(w=w)
        fi_map["bmi"]=spec["fi_map"]["bmi"].format(w=w)
        temp = d.loc[anchor.index].copy()
        fi, obs, complete = fi_score(temp, fi_map, spec["shlt_direction"])
        out["fi"] = fi; out["fi_complete8"] = complete; out["all_four_ic"] = ic["all_four_ic"]
        out["active"] = out["active"] & out["id"].notna()
        if spec.get("exclude_proxy"):
            out["active"] = out["active"] & (~out["proxy"])
        return out
    base = make_wave(anchor_wave)
    rows=[]
    for target in target_waves:
        fut=make_wave(target)
        pairs = pd.DataFrame({"id":base["id"],"baseline_all4_ic":base["all_four_ic"],"baseline_fi":base["fi"]}).merge(
            pd.DataFrame({"id":fut["id"],"future_fi":fut["fi"],"future_fi_complete8":fut["fi_complete8"],"future_active":fut["active"]}),on="id",how="inner")
        pairs["pair_complete"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi"]].notna().all(axis=1) & pairs["future_active"]
        pairs["pair_complete8_future"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi_complete8"]].notna().all(axis=1) & pairs["future_active"]
        rows.append({"dataset":dataset,"anchor_wave":anchor_wave,"target_wave":target,"target_rule":"nonproxy_anchor_and_target" if spec.get("exclude_proxy") else "active_anchor_and_target","n_anchor":int(base["active"].sum()),"n_target":int(fut["active"].sum()),"n_anchor_all4_ic":int((base["active"] & base["all_four_ic"]).sum()),"n_anchor_fi":int((base["active"] & base["fi"].notna()).sum()),"n_future_fi":int((fut["active"] & fut["fi"].notna()).sum()),"n_future_fi_complete8":int((fut["active"] & fut["fi_complete8"].notna()).sum()),"n_pair_baseline_ic_fi_future_fi":int(pairs["pair_complete"].sum()),"pct_anchor_pair":round(float(pairs["pair_complete"].mean()*100),2) if len(pairs) else np.nan,"n_pair_baseline_ic_fi_future_complete8":int(pairs["pair_complete8_future"].sum()),"pct_anchor_pair_complete8":round(float(pairs["pair_complete8_future"].mean()*100),2) if len(pairs) else np.nan})
    return pd.DataFrame(rows)

results=[]
# ELSA/SHARE working long files
elsa_path = ROOT/"ELSA/Working_data/elsa.dta"
elsa_spec = {"id":"idauniqc","ic":["imrc","dlrc","orient","walkra","walk100a","gripsum","cesd"],"fi":["shlt","mbmi","hibpe","diabe","hearte","stroke","cancre","arthre"],"fi_map":{"disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"},"shlt":"shlt","bmi":"mbmi"},"shlt_direction":"poor_is_low"}
r,_=build_long_results("ELSA",elsa_path,6,[7,8,9],elsa_spec); results.append(r)

share_path = next((ROOT/"SHARE").rglob("Working_data/share.dta"))
share_spec = {"id":"mergeid","ic":["imrc","dlrc","orient","tr20","walkra","walk100a","gripcomp","eurod"],"fi":["shlt","bmi","hibpe","diabe","hearte","stroke","cancre","arthre"],"fi_map":{"disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"},"shlt":"shlt","bmi":"bmi"},"shlt_direction":"poor_is_low"}
r,_=build_long_results("SHARE",share_path,6,[7,8],share_spec); results.append(r)

# CHARLS wide
charls_spec = {
    "id":"ID","active":"inw{w}","proxy":None,"exclude_proxy":False,
    "ic":["r{w}imrc","r{w}dlrc","r{w}orient","r{w}tr20","r{w}walk100a","r{w}walk1kma","r{w}lgrip1","r{w}lgrip2","r{w}rgrip1","r{w}rgrip2","r{w}cesd10"],
    "ic_map":{"cognition":["r{w}imrc","r{w}dlrc","r{w}orient","r{w}tr20"],"locomotion":["r{w}walk100a","r{w}walk1kma"],"grip_vitality":["r{w}lgrip1","r{w}lgrip2","r{w}rgrip1","r{w}rgrip2"],"psychological":["r{w}cesd10"]},
    "fi":["r{w}shlt","r{w}mbmi","r{w}hibpe","r{w}diabe","r{w}hearte","r{w}stroke","r{w}cancre","r{w}arthre"],
    "fi_map":{"disease":{"hypertension":"r{w}hibpe","diabetes":"r{w}diabe","heart_disease":"r{w}hearte","stroke":"r{w}stroke","cancer":"r{w}cancre","arthritis":"r{w}arthre"},"shlt":"r{w}shlt","bmi":"r{w}mbmi"},"shlt_direction":"poor_is_high"
}
charls_path=ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"
results.append(build_wide_results("CHARLS",charls_path,3,[4],charls_spec))

# HRS strict non-proxy target and anchor
hrs_spec = {
    "id":"hhidpn","active":"inw{w}","proxy":"r{w}proxy","exclude_proxy":True,
    "ic":["r{w}imrc","r{w}dlrc","r{w}ser7","r{w}bwc20","r{w}walkra","r{w}walk1a","r{w}walksa","r{w}grpl","r{w}grpr","r{w}cesd"],
    "ic_map":{"cognition":["r{w}imrc","r{w}dlrc","r{w}ser7","r{w}bwc20"],"locomotion":["r{w}walkra","r{w}walk1a","r{w}walksa"],"grip_vitality":["r{w}grpl","r{w}grpr"],"psychological":["r{w}cesd"]},
    "fi":["r{w}shlt","r{w}bmi","r{w}hibpe","r{w}diabe","r{w}hearte","r{w}stroke","r{w}cancre","r{w}arthre"],
    "fi_map":{"disease":{"hypertension":"r{w}hibpe","diabetes":"r{w}diabe","heart_disease":"r{w}hearte","stroke":"r{w}stroke","cancer":"r{w}cancre","arthritis":"r{w}arthre"},"shlt":"r{w}shlt","bmi":"r{w}bmi"},"shlt_direction":"poor_is_high"
}
hrs_path=ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
results.append(build_wide_results("HRS",hrs_path,10,[11,12,13],hrs_spec))

audit = pd.concat(results,ignore_index=True)
audit.to_csv(TAB/"stage24_longitudinal_predictive_feasibility.csv",index=False,encoding="utf-8-sig")
summary = audit.groupby(["dataset","target_rule"],as_index=False).agg(
    max_pair_n=("n_pair_baseline_ic_fi_future_fi","max"),
    min_pair_n=("n_pair_baseline_ic_fi_future_fi","min"),
    max_pair_pct=("pct_anchor_pair","max"),
    min_pair_pct=("pct_anchor_pair","min"),
    target_waves=("target_wave",lambda x:";".join(map(str,sorted(x))))
)
summary.to_csv(TAB/"stage24_longitudinal_predictive_feasibility_summary.csv",index=False,encoding="utf-8-sig")
run = {"stage":24,"date":"2026-09-23","script":"53_longitudinal_predictive_feasibility.py","outputs":["tables/stage24_longitudinal_predictive_feasibility.csv","tables/stage24_longitudinal_predictive_feasibility_summary.csv"],"anchor_waves":{"ELSA":6,"CHARLS":3,"HRS":10,"SHARE":6},"candidate_future_waves":{"ELSA":[7,8,9],"CHARLS":[4],"HRS":[11,12,13],"SHARE":[7,8]},"share_wave7_role":"audit only; excluded from primary cognition path","hrs_rule":"baseline and target non-proxy","fi_rule":"six chronic disease indicators + self-rated health + BMI; minimum 6/8; observed denominator; BMI<18.5 or >=30","output_level":"aggregate only; no individual IDs or scores written"}
(OUT/"stage24_longitudinal_predictive_feasibility_run_info.json").write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding="utf-8")
print(summary.to_string(index=False))


