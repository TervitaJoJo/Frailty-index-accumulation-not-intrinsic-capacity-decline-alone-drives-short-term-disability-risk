"""Aggregate anchor-wave IC domain profiles and IC--FI correlations."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

def valid(x):
    return pd.to_numeric(x, errors="coerce").mask(lambda z: z < -90)

def frailty(df, disease, shlt, bmi, direction):
    s = pd.DataFrame(index=df.index)
    for name, col in disease.items():
        x = valid(df[col]); s[name] = x.eq(1).where(x.notna())
    x = valid(df[shlt])
    s["self_rated_health"] = ((5 - x) / 4 if direction == "poor_is_low" else (x - 1) / 4).where(x.notna())
    x = valid(df[bmi]); s["bmi"] = (x.lt(18.5) | x.ge(30)).where(x.notna())
    s = s.astype(float); n = s.notna().sum(axis=1)
    return s.sum(axis=1, min_count=1).div(n).where(n.ge(6))

def profile(dataset, df, ic, fi):
    d = pd.DataFrame(ic, index=df.index).astype(float); d["frailty"] = fi
    d["ic_complete_4_domains"] = d[["cognition","locomotion","grip_vitality","psychological"]].notna().all(axis=1)
    out = []
    for col in ["cognition","locomotion","grip_vitality","psychological","frailty"]:
        x = d[col].dropna()
        out.append({"dataset":dataset,"construct":col,"n_active":len(d),"n_valid":len(x),"pct_valid":round(float(len(x)/len(d)*100),2),"mean":round(float(x.mean()),4) if len(x) else None,"sd":round(float(x.std(ddof=1)),4) if len(x)>1 else None,"median":round(float(x.median()),4) if len(x) else None,"p25":round(float(x.quantile(.25)),4) if len(x) else None,"p75":round(float(x.quantile(.75)),4) if len(x) else None})
    out.append({"dataset":dataset,"construct":"ic_complete_4_domains","n_active":len(d),"n_valid":int(d.ic_complete_4_domains.sum()),"pct_valid":round(float(d.ic_complete_4_domains.mean()*100),2)})
    c = d[["cognition","locomotion","grip_vitality","psychological","frailty"]].corr(method="spearman", min_periods=2)
    cor = []
    for a in c.index:
        for b in c.columns:
            if a < b:
                pair = d[[a,b]].dropna()
                cor.append({"dataset":dataset,"construct_a":a,"construct_b":b,"n_pairwise":len(pair),"spearman":round(float(c.loc[a,b]),4) if pd.notna(c.loc[a,b]) else None})
    return out, cor

profiles=[]; correlations=[]

# Dataset specifications for the selected anchor waves.
specs = {
    "ELSA": {"path":ROOT/"ELSA/Working_data/elsa.dta","active":("wave",6),"cols":["wave","tcog_z_z","walkra","gripsum","cesd","shlt","mbmi","hibpe","diabe","hearte","stroke","cancre","arthre"]},
    "CHARLS": {"path":ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta","active":("inw3",1),"cols":["inw3","r3orient","r3tr20","r3walk100a","r3lgrip","r3rgrip","r3cesd10","r3shlt","r3mbmi","r3hibpe","r3diabe","r3hearte","r3stroke","r3cancre","r3arthre"]},
    "HRS": {"path":ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta","active":("inw10",1),"cols":["inw10","r10imrc","r10dlrc","r10ser7","r10walkra","r10grpl","r10grpr","r10cesd","r10shlt","r10bmi","r10hibpe","r10diabe","r10hearte","r10stroke","r10cancre","r10arthre"]},
    "LASI": {"path":next(ROOT.glob("LASI/**/Working_data/lasi.dta")),"active_year":"r1iwy","cols":["r1iwy","r1cog_total","r1walk100a","r1lgrip","r1rgrip","r1cesd10_l","r1shlt","r1mbmi","r1hibpe","r1diabe","r1hearte","r1stroke","r1cancre","r1arthre"]},
}

for dataset, spec in specs.items():
    _, meta = pyreadstat.read_dta(str(spec["path"]), metadataonly=True)
    use=[c for c in spec["cols"] if c in meta.column_names]
    df,_=pyreadstat.read_dta(str(spec["path"]),usecols=use,apply_value_formats=False)
    if "active_year" in spec: active=df[spec["active_year"]].notna()
    else: active=valid(df[spec["active"][0]]).eq(spec["active"][1])
    df=df.loc[active].copy()
    if dataset == "ELSA":
        ic={"cognition":valid(df.tcog_z_z),"locomotion":1-valid(df.walkra),"grip_vitality":valid(df.gripsum),"psychological":1-valid(df.cesd)/8}; direction="poor_is_low"; shlt="shlt"; bmi="mbmi"; dis={"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"}
    elif dataset == "CHARLS":
        ic={"cognition":pd.concat([valid(df.r3orient)/4,valid(df.r3tr20)/20],axis=1).mean(axis=1,skipna=True),"locomotion":1-valid(df.r3walk100a),"grip_vitality":pd.concat([valid(df.r3lgrip),valid(df.r3rgrip)],axis=1).max(axis=1,skipna=True),"psychological":1-valid(df.r3cesd10)/30}; direction="poor_is_high"; shlt="r3shlt"; bmi="r3mbmi"; dis={"hypertension":"r3hibpe","diabetes":"r3diabe","heart_disease":"r3hearte","stroke":"r3stroke","cancer":"r3cancre","arthritis":"r3arthre"}
    elif dataset == "HRS":
        ic={"cognition":pd.concat([valid(df.r10imrc)/10,valid(df.r10dlrc)/10,valid(df.r10ser7)/7],axis=1).mean(axis=1,skipna=True),"locomotion":1-valid(df.r10walkra),"grip_vitality":pd.concat([valid(df.r10grpl),valid(df.r10grpr)],axis=1).max(axis=1,skipna=True),"psychological":1-valid(df.r10cesd)/8}; direction="poor_is_high"; shlt="r10shlt"; bmi="r10bmi"; dis={"hypertension":"r10hibpe","diabetes":"r10diabe","heart_disease":"r10hearte","stroke":"r10stroke","cancer":"r10cancre","arthritis":"r10arthre"}
    else:
        ic={"cognition":valid(df.r1cog_total),"locomotion":1-valid(df.r1walk100a),"grip_vitality":pd.concat([valid(df.r1lgrip),valid(df.r1rgrip)],axis=1).max(axis=1,skipna=True),"psychological":1-valid(df.r1cesd10_l)/10}; direction="poor_is_low"; shlt="r1shlt"; bmi="r1mbmi"; dis={"hypertension":"r1hibpe","diabetes":"r1diabe","heart_disease":"r1hearte","stroke":"r1stroke","cancer":"r1cancre","arthritis":"r1arthre"}
    fi=frailty(df,dis,shlt,bmi,direction); r,c=profile(dataset,df,ic,fi); profiles+=r; correlations+=c

# SHARE wave 6: merge raw cognition at implicat=1 for exploratory coverage.
path=next(ROOT.glob("SHARE/**/Working_data/share.dta")); _,meta=pyreadstat.read_dta(str(path),metadataonly=True)
cols=["mergeid","wave","walkra","lgrip","rgrip","eurod","shlt","bmi","hibpe","diabe","hearte","stroke","cancre","arthre"]
share,_=pyreadstat.read_dta(str(path),usecols=[c for c in cols if c in meta.column_names],apply_value_formats=False); share=share.loc[share.wave.eq(6)].copy()
raw_path=next((ROOT/"SHARE").rglob("sharew6_rel9-0-0_gv_imputations.dta")); raw,_=pyreadstat.read_dta(str(raw_path),usecols=["mergeid","implicat","orienti","memory","orienti_f","memory_f"],apply_value_formats=False); raw=raw.loc[raw.implicat.eq(1)].copy(); share=share.merge(raw,on="mergeid",how="left",validate="one_to_one")
ori=valid(share.orienti).where(pd.to_numeric(share.orienti_f,errors="coerce").between(3,13)); mem=((5-valid(share.memory))/4).where(pd.to_numeric(share.memory_f,errors="coerce").between(3,13)); cog=pd.concat([ori/4,mem],axis=1).mean(axis=1,skipna=True)
fi=frailty(share,{"hypertension":"hibpe","diabetes":"diabe","heart_disease":"hearte","stroke":"stroke","cancer":"cancre","arthritis":"arthre"},"shlt","bmi","poor_is_low")
ic={"cognition":cog,"locomotion":1-valid(share.walkra),"grip_vitality":pd.concat([valid(share.lgrip),valid(share.rgrip)],axis=1).max(axis=1,skipna=True),"psychological":1-valid(share.eurod)/12}; r,c=profile("SHARE",share,ic,fi); profiles+=r; correlations+=c

pd.DataFrame(profiles).to_csv(TAB/"anchor_construct_profile_summary.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(correlations).to_csv(TAB/"anchor_construct_spearman.csv",index=False,encoding="utf-8-sig")
(OUT/"anchor_construct_profiles.md").write_text("""# Anchor construct profiles\n\nThis exploratory audit direction-aligns IC domains so higher values indicate better capacity, computes the primary outcome-disjoint FI, and reports aggregate distributions and Spearman correlations. SHARE cognition uses implicat=1 only for this profile screen; final modeling must preserve all five imputations. A negative IC–FI association is expected because higher IC means better capacity and higher FI means greater deficit burden. These summaries are construct-validity diagnostics, not invariance or causal results.\n""",encoding="utf-8")
print("wrote anchor construct profiles")
