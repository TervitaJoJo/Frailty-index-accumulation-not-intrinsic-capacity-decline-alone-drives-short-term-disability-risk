from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

def num(x, lo=None, hi=None):
    z = pd.to_numeric(x, errors="coerce")
    z = z.mask(z < -90)
    if lo is not None: z = z.mask(z < lo)
    if hi is not None: z = z.mask(z > hi)
    return z.astype(float)

def valid01(x):
    z = num(x)
    return z.where(z.isin([0,1]))

def read_dta(path, cols):
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in dict.fromkeys(cols) if c in meta.column_names]
    dat, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    return dat

def fi_components(d, cols, direction):
    out = pd.DataFrame(index=d.index)
    for name, col in cols["disease"].items():
        out[name] = valid01(d[col]) if col in d else np.nan
    shlt = num(d[cols["shlt"]], 1, 5) if cols["shlt"] in d else pd.Series(np.nan, index=d.index)
    out["self_rated_health"] = (5-shlt)/4 if direction=="poor_is_low" else (shlt-1)/4
    bmi = num(d[cols["bmi"]], 0, 200) if cols["bmi"] in d else pd.Series(np.nan, index=d.index)
    out["bmi"] = (bmi.lt(18.5) | bmi.ge(30)).where(bmi.notna())
    out = out.apply(pd.to_numeric, errors="coerce")
    observed = out.notna().sum(axis=1)
    total = out.sum(axis=1, min_count=1)
    fi = total.div(observed).where(observed.ge(6))
    return fi

def mean_min(df, cols, minimum=1, transforms=None):
    vals=[]
    for c in cols:
        if c in df:
            x = num(df[c])
            if transforms and c in transforms: x = transforms[c](x)
            vals.append(x.rename(c))
    if not vals: return pd.Series(np.nan,index=df.index)
    z=pd.concat(vals,axis=1)
    return z.mean(axis=1).where(z.notna().sum(axis=1).ge(minimum))

def max_valid(df, cols, transforms=None):
    vals=[]
    for c in cols:
        if c in df:
            x=num(df[c])
            if transforms and c in transforms: x=transforms[c](x)
            vals.append(x.rename(c))
    if not vals: return pd.Series(np.nan,index=df.index)
    z=pd.concat(vals,axis=1)
    return z.max(axis=1,skipna=True).where(z.notna().any(axis=1))

def build_domains(df, dataset):
    if dataset=="ELSA":
        cog = mean_min(df, ["imrc","dlrc","orient"], minimum=2, transforms={"imrc":lambda x:x/10,"dlrc":lambda x:x/10,"orient":lambda x:x/4})
        loc = mean_min(df, ["walkra","walk100a"], minimum=1, transforms={"walkra":lambda x:1-x,"walk100a":lambda x:1-x})
        grip = max_valid(df, ["lgrip","rgrip"], transforms={"lgrip":lambda x:x/100,"rgrip":lambda x:x/100})
        if grip.isna().all() and "gripsum" in df: grip=num(df["gripsum"])/100
        psych = -num(df["cesd"])
    elif dataset=="CHARLS":
        cog = mean_min(df, ["imrc","dlrc","orient"], minimum=2, transforms={"imrc":lambda x:x/10,"dlrc":lambda x:x/10,"orient":lambda x:x/4})
        loc = mean_min(df, ["walk100a","walk1kma"], minimum=1, transforms={"walk100a":lambda x:1-x,"walk1kma":lambda x:1-x})
        grip = max_valid(df, ["lgrip1","lgrip2","rgrip1","rgrip2"], transforms={c:lambda x:x/100 for c in ["lgrip1","lgrip2","rgrip1","rgrip2"]})
        psych = -num(df["cesd10"]) if "cesd10" in df else pd.Series(np.nan,index=df.index)
    elif dataset=="HRS":
        cog = mean_min(df, ["imrc","dlrc","ser7"], minimum=2, transforms={"imrc":lambda x:x/10,"dlrc":lambda x:x/10,"ser7":lambda x:x/5})
        loc = mean_min(df, ["walkra","walk1a","walksa"], minimum=1, transforms={"walkra":lambda x:1-(x>0).astype(float),"walk1a":lambda x:1-(x>0).astype(float),"walksa":lambda x:1-(x>0).astype(float)})
        grip = max_valid(df, ["grpl","grpr"], transforms={"grpl":lambda x:x/100,"grpr":lambda x:x/100})
        psych = -num(df["cesd"])
    elif dataset=="SHARE":
        cog = mean_min(df, ["imrc","dlrc","orient"], minimum=2, transforms={"imrc":lambda x:x/10,"dlrc":lambda x:x/10,"orient":lambda x:x/4})
        loc = mean_min(df, ["walkra","walk100a"], minimum=1, transforms={"walkra":lambda x:1-x,"walk100a":lambda x:1-x})
        grip = max_valid(df, ["lgrip","rgrip","gripcomp"], transforms={"lgrip":lambda x:x/100,"rgrip":lambda x:x/100,"gripcomp":lambda x:x/100})
        psych = -num(df["eurod"])
    else: raise ValueError(dataset)
    return pd.DataFrame({"cognition":cog,"locomotion":loc,"grip_vitality":grip,"psychological":psych},index=df.index)

def standardize(x):
    x=pd.to_numeric(x,errors="coerce")
    s=x.std(ddof=1)
    return (x-x.mean())/s if s and np.isfinite(s) and s>0 else pd.Series(np.nan,index=x.index)

def fit_model(dat, outcome, predictors, cohort, window, model_name):
    d=dat[[outcome]+predictors].dropna().copy()
    if len(d)<100:
        return pd.DataFrame([{"dataset":cohort,"window":window,"model":model_name,"term":"__model__","n":len(d),"estimate":np.nan,"se_hc3":np.nan,"p_hc3":np.nan,"ci_low":np.nan,"ci_high":np.nan,"std_beta":np.nan,"r2":np.nan,"delta_r2":np.nan,"error":"too_few_complete_cases"}])
    y=d[outcome]
    X=pd.DataFrame(index=d.index)
    for p in predictors: X[p]=d[p].astype(float) if p=="sex" else standardize(d[p])
    X=sm.add_constant(X,has_constant="add")
    model=sm.OLS(y,X,missing="drop").fit(cov_type="HC3")
    rows=[]
    for term in model.params.index:
        if term=="const": continue
        est=float(model.params[term]); se=float(model.bse[term]); p=float(model.pvalues[term]); ci=model.conf_int().loc[term]
        rows.append({"dataset":cohort,"window":window,"model":model_name,"term":term,"n":int(model.nobs),"estimate":est,"se_hc3":se,"p_hc3":p,"ci_low":float(ci[0]),"ci_high":float(ci[1]),"std_beta":est/float(y.std(ddof=1)),"r2":float(model.rsquared),"delta_r2":np.nan,"error":""})
    return pd.DataFrame(rows)

def regression_pair(pairs, dataset, window, outcome):
    basepred=["baseline_fi","age","sex"]
    fullpred=basepred+["cognition_base","locomotion_base","grip_vitality_base","psychological_base"]
    bres=fit_model(pairs,outcome,basepred,dataset,window,"base")
    fres=fit_model(pairs,outcome,fullpred,dataset,window,"future_fi_adjusted_baseline" if outcome=="future_fi" else "delta_fi")
    common=pairs[[outcome]+fullpred].dropna()
    if len(common)>=100:
        y=common[outcome]
        Xb=sm.add_constant(pd.DataFrame({p:standardize(common[p]) if p!="sex" else common[p].astype(float) for p in basepred}),has_constant="add")
        Xf=sm.add_constant(pd.DataFrame({p:standardize(common[p]) if p!="sex" else common[p].astype(float) for p in fullpred}),has_constant="add")
        mb=sm.OLS(y,Xb).fit(); mf=sm.OLS(y,Xf).fit()
        fres["delta_r2"]=float(mf.rsquared-mb.rsquared)
    return bres,fres

def prepare_long(dataset,path,anchor,target_waves,fi_spec):
    cols=["wave",fi_spec["id"],"agey","ragender","imrc","dlrc","orient","walkra","walk100a","lgrip","rgrip","gripsum","gripcomp","eurod","cesd",fi_spec["shlt"],fi_spec["bmi"],*fi_spec["disease"].values()]
    d=read_dta(path,cols)
    d=d.loc[d["wave"].isin([anchor,*target_waves]) & d[fi_spec["id"]].notna()].copy()
    d=d.drop_duplicates([fi_spec["id"],"wave"],keep="first")
    d["age"]=num(d["agey"]) if "agey" in d else np.nan
    d["sex"]=num(d["ragender"]) if "ragender" in d else np.nan
    d["fi"]=fi_components(d,fi_spec,"poor_is_low")
    d=pd.concat([d,build_domains(d,dataset)],axis=1)
    rows=[]; models=[]
    b=d.loc[d.wave.eq(anchor)].copy()
    for w in target_waves:
        f=d.loc[d.wave.eq(w)].copy()
        pairs=b[[fi_spec["id"],"fi","age","sex","cognition","locomotion","grip_vitality","psychological"]].rename(columns={"fi":"baseline_fi","cognition":"cognition_base","locomotion":"locomotion_base","grip_vitality":"grip_vitality_base","psychological":"psychological_base"}).merge(f[[fi_spec["id"],"fi"]].rename(columns={"fi":"future_fi"}),on=fi_spec["id"],how="inner")
        pairs["delta_fi"]=pairs["future_fi"]-pairs["baseline_fi"]
        pairs=pairs.dropna(subset=["baseline_fi","future_fi","cognition_base","locomotion_base","grip_vitality_base","psychological_base"])
        for outcome in ["future_fi","delta_fi"]: models.extend(regression_pair(pairs,dataset,str(anchor)+"->"+str(w),outcome))
        rows.append({"dataset":dataset,"anchor_wave":anchor,"target_wave":w,"n_anchor":len(b),"n_target":len(f),"n_complete_predictive":len(pairs),"baseline_fi_mean":float(b["fi"].mean()),"future_fi_mean":float(f["fi"].mean()),"delta_fi_mean":float(pairs["delta_fi"].mean())})
    return pd.DataFrame(rows),pd.concat(models,ignore_index=True)

def prepare_wide(dataset,path,anchor,target_waves,spec):
    cols=[spec["id"]]
    for w in [anchor,*target_waves]:
        cols += [spec["active"].format(w=w),spec["proxy"].format(w=w) if spec.get("proxy") else None,spec["age"].format(w=w),spec["sex"]]
        cols += [c.format(w=w) for c in spec["ic"]]
        cols += [c.format(w=w) for c in spec["fi"]]
    d=read_dta(path,cols); d=d.loc[d[spec["id"]].notna()].copy()
    def wave_frame(w):
        out=pd.DataFrame(index=d.index)
        out["id"]=d[spec["id"]]
        act=spec["active"].format(w=w); prox=spec["proxy"].format(w=w) if spec.get("proxy") else None
        out["active"]=num(d[act]).eq(1) if act in d else True
        if prox and prox in d: out["active"]=out["active"] & num(d[prox]).eq(0)
        agecol=spec["age"].format(w=w); out["age"]=num(d[agecol]) if agecol in d else np.nan
        out["sex"]=num(d[spec["sex"]]) if spec["sex"] in d else np.nan
        for dom, cols2 in spec["ic_map"].items():
            vals=[]
            for c in cols2:
                c2=c.format(w=w)
                if c2 in d:
                    x=num(d[c2])
                    if dom=="cognition":
                        if c2.endswith("imrc"): x=x/10
                        elif c2.endswith("dlrc"): x=x/10
                        elif c2.endswith("orient"): x=x/4
                        elif c2.endswith("ser7"): x=x/5
                    elif dom=="locomotion":
                        x=1-(x>0).astype(float) if dataset=="HRS" else 1-x
                    elif dom=="grip_vitality": x=x/100
                    elif dom=="psychological": x=-x
                    vals.append(x)
            z=pd.concat(vals,axis=1) if vals else pd.DataFrame(index=d.index)
            if dom=="cognition": out[dom]=z.mean(axis=1).where(z.notna().sum(axis=1).ge(2)) if vals else np.nan
            elif dom=="locomotion": out[dom]=z.mean(axis=1).where(z.notna().sum(axis=1).ge(1)) if vals else np.nan
            elif dom=="grip_vitality": out[dom]=z.max(axis=1,skipna=True).where(z.notna().any(axis=1)) if vals else np.nan
            else: out[dom]=z.mean(axis=1).where(z.notna().any(axis=1)) if vals else np.nan
        fi_map={"disease":{k:c.format(w=w) for k,c in spec["fi_map"]["disease"].items()},"shlt":spec["fi_map"]["shlt"].format(w=w),"bmi":spec["fi_map"]["bmi"].format(w=w)}
        out["fi"]=fi_components(d,fi_map,spec["shlt_direction"])
        return out
    base=wave_frame(anchor); base=base.loc[base["active"]].copy()
    rows=[]; models=[]
    for w in target_waves:
        fut=wave_frame(w); fut=fut.loc[fut["active"]].copy()
        pairs=base[["id","fi","age","sex","cognition","locomotion","grip_vitality","psychological"]].rename(columns={"fi":"baseline_fi","cognition":"cognition_base","locomotion":"locomotion_base","grip_vitality":"grip_vitality_base","psychological":"psychological_base"}).merge(fut[["id","fi"]].rename(columns={"fi":"future_fi"}),on="id",how="inner")
        pairs["delta_fi"]=pairs["future_fi"]-pairs["baseline_fi"]
        pairs=pairs.dropna(subset=["baseline_fi","future_fi","cognition_base","locomotion_base","grip_vitality_base","psychological_base"])
        for outcome in ["future_fi","delta_fi"]: models.extend(regression_pair(pairs,dataset,str(anchor)+"->"+str(w),outcome))
        rows.append({"dataset":dataset,"anchor_wave":anchor,"target_wave":w,"n_anchor":len(base),"n_target":len(fut),"n_complete_predictive":len(pairs),"baseline_fi_mean":float(base["fi"].mean()),"future_fi_mean":float(fut["fi"].mean()),"delta_fi_mean":float(pairs["delta_fi"].mean())})
    return pd.DataFrame(rows),pd.concat(models,ignore_index=True)

elsa_fi={"id":"idauniqc","disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"},"shlt":"shlt","bmi":"mbmi"}
charls_fi={"disease":{"hypertension":"r{w}hibpe","diabetes":"r{w}diabe","heart_disease":"r{w}hearte","stroke":"r{w}stroke","cancer":"r{w}cancre","arthritis":"r{w}arthre"},"shlt":"r{w}shlt","bmi":"r{w}mbmi"}
hrs_fi={"disease":{"hypertension":"r{w}hibpe","diabetes":"r{w}diabe","heart_disease":"r{w}hearte","stroke":"r{w}stroke","cancer":"r{w}cancre","arthritis":"r{w}arthre"},"shlt":"r{w}shlt","bmi":"r{w}bmi"}

summary=[]; coefficients=[]
r,c=prepare_long("ELSA",ROOT/"ELSA/Working_data/elsa.dta",6,[7,8,9],elsa_fi); summary.append(r); coefficients.append(c)
r,c=prepare_wide("CHARLS",ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",3,[4],{"id":"ID","active":"inw{w}","age":"r{w}agey","sex":"ragender","ic":["r{w}imrc","r{w}dlrc","r{w}orient","r{w}walk100a","r{w}walk1kma","r{w}lgrip1","r{w}lgrip2","r{w}rgrip1","r{w}rgrip2","r{w}cesd10"],"ic_map":{"cognition":["r{w}imrc","r{w}dlrc","r{w}orient"],"locomotion":["r{w}walk100a","r{w}walk1kma"],"grip_vitality":["r{w}lgrip1","r{w}lgrip2","r{w}rgrip1","r{w}rgrip2"],"psychological":["r{w}cesd10"]},"fi":["r{w}shlt","r{w}mbmi","r{w}hibpe","r{w}diabe","r{w}hearte","r{w}stroke","r{w}cancre","r{w}arthre"],"fi_map":charls_fi,"shlt_direction":"poor_is_high"}); summary.append(r); coefficients.append(c)
r,c=prepare_wide("HRS",ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",10,[11,12,13],{"id":"hhidpn","active":"inw{w}","proxy":"r{w}proxy","age":"r{w}agey_m","sex":"ragender","ic":["r{w}imrc","r{w}dlrc","r{w}ser7","r{w}walkra","r{w}walk1a","r{w}walksa","r{w}grpl","r{w}grpr","r{w}cesd"],"ic_map":{"cognition":["r{w}imrc","r{w}dlrc","r{w}ser7"],"locomotion":["r{w}walkra","r{w}walk1a","r{w}walksa"],"grip_vitality":["r{w}grpl","r{w}grpr"],"psychological":["r{w}cesd"]},"fi":["r{w}shlt","r{w}bmi","r{w}hibpe","r{w}diabe","r{w}hearte","r{w}stroke","r{w}cancre","r{w}arthre"],"fi_map":hrs_fi,"shlt_direction":"poor_is_high"}); summary.append(r); coefficients.append(c)
share_fi={"id":"mergeid","disease":{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"},"shlt":"shlt","bmi":"bmi"}
r,c=prepare_long("SHARE",next((ROOT/"SHARE").rglob("Working_data/share.dta")),6,[8],share_fi); summary.append(r); coefficients.append(c)

summary_df=pd.concat(summary,ignore_index=True); coef_df=pd.concat(coefficients,ignore_index=True)
summary_df.to_csv(TAB/"stage25_longitudinal_predictive_summary.csv",index=False,encoding="utf-8-sig")
coef_df.to_csv(TAB/"stage25_longitudinal_predictive_coefficients.csv",index=False,encoding="utf-8-sig")
coef_df.loc[coef_df["window"].isin(["6->7","3->4","10->11","6->8"])].to_csv(TAB/"stage25_longitudinal_predictive_main_window.csv",index=False,encoding="utf-8-sig")
memo = """# Stage 25 longitudinal predictive-validity screen

This stage tests whether baseline four-domain IC proxies predict later FI after adjustment for baseline FI, age, and sex. The primary outcome is future FI conditional on baseline FI (future_fi_adjusted_baseline); FI change (delta_fi) is secondary. Continuous predictors are standardized within each cohort-window; estimates are on the FI scale and HC3 robust standard errors are reported. The base model contains baseline FI, age, and sex; the full model adds baseline cognition, locomotion, grip/vitality, and psychological capacity proxies. delta_r2 is the full-model minus base-model R-squared on the common complete-case sample.

Primary windows are ELSA 6->7, CHARLS 3->4, and HRS 10->11. ELSA 6->8/9 and HRS 10->12/13 are temporal sensitivity windows. SHARE 6->8 is external predictive validation; wave 7 is not used as the primary cognitive follow-up because of structural cognitive missingness.

The proxy domains follow the frozen direction conventions: higher cognition, locomotion, grip/vitality, and psychological values indicate better capacity. They are pre-specified observed composites and do not replace the partial-metric latent SEM. Results are exploratory predictive associations, not causal effects. No individual IDs are written.
"""
(OUT/"stage25_longitudinal_predictive_validity_memo.md").write_text(memo,encoding="utf-8")
run={"stage":25,"date":"2026-09-24","script":"54_longitudinal_predictive_screen.py","primary_windows":{"ELSA":"6->7","CHARLS":"3->4","HRS":"10->11"},"sensitivity_windows":{"ELSA":["6->8","6->9"],"HRS":["10->12","10->13"],"SHARE":"6->8"},"primary_outcome":"future FI adjusted for baseline FI, age, sex","secondary_outcome":"FI change","estimator":"OLS with HC3 robust SE; no survey weights or imputation","output_level":"aggregate only; no IDs or scores written"}
(OUT/"stage25_longitudinal_predictive_run_info.json").write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding="utf-8")
print(summary_df.to_string(index=False))
print("coefficient rows",len(coef_df))



