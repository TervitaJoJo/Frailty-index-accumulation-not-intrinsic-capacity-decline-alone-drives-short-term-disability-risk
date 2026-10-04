"""Stage 22 design-weighted descriptive and IC-FI sensitivity audit.

Only 0--1 capacity-oriented proxy summaries and weighted rank correlations are
written. No weighted CFA/SEM, variance estimate, or imputation is attempted.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"; TAB.mkdir(parents=True, exist_ok=True)

def read(path, cols):
    _, m = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in dict.fromkeys(cols) if c in m.column_names]
    return pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)[0]

def n(x, lo=-np.inf, hi=np.inf):
    y = pd.to_numeric(x, errors="coerce"); return y.where(y.between(lo, hi))

def b(x):
    y = pd.to_numeric(x, errors="coerce"); return y.where(y.isin([0, 1]))

def mean_parts(parts, minimum=1):
    z = pd.concat(parts, axis=1); return z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1).ge(minimum))

def fi(d, pref=""):
    names = ["hibpe", "diabe", "hearte", "stroke", "cancre", "arthre"]
    z = pd.DataFrame({k: pd.to_numeric(d[f"{pref}{k}"], errors="coerce").where(pd.to_numeric(d[f"{pref}{k}"], errors="coerce").isin([0,1])) for k in names})
    shlt = n(d[f"{pref}shlt"], 1, 5); z["self_rated_health"] = (5-shlt)/4
    bmi = n(d[f"{pref}bmi"], .01, 200); z["bmi"] = (bmi.lt(18.5) | bmi.ge(30)).astype(float).where(bmi.notna())
    k = z.notna().sum(axis=1); return z.sum(axis=1, min_count=1).div(k.where(k.ge(6)))

def weight(x): return n(x, .000001, np.inf)

def wmean(x, w):
    z = pd.DataFrame({"x": x, "w": weight(w)}).dropna()
    if len(z)==0: return np.nan, 0, np.nan
    return float(np.average(z.x, weights=z.w)), len(z), float(z.w.sum()**2/z.w.pow(2).sum())

def wrho(x, y, w):
    z = pd.DataFrame({"x":x,"y":y,"w":weight(w)}).dropna()
    if len(z)<3 or z.x.nunique()<2 or z.y.nunique()<2: return np.nan, len(z), np.nan
    rx=z.x.rank(method="average"); ry=z.y.rank(method="average"); ww=z.w
    mx=np.average(rx,weights=ww); my=np.average(ry,weights=ww)
    vx=np.average((rx-mx)**2,weights=ww); vy=np.average((ry-my)**2,weights=ww)
    r=np.average((rx-mx)*(ry-my),weights=ww)/np.sqrt(vx*vy) if vx>0 and vy>0 else np.nan
    return float(r),len(z),float(ww.sum()**2/ww.pow(2).sum())

def add(out, ds, d, w, role, wvar):
    for c in ["cognition","locomotion","grip_vitality","psychological","fi_primary"]:
        x=pd.to_numeric(d[c],errors="coerce"); u=x.dropna(); mu, nw, ess=wmean(x,w)
        out["desc"].append({"dataset":ds,"weight_role":role,"weight_variable":wvar,"construct":c,"n_unweighted":len(u),"unweighted_mean":u.mean() if len(u) else np.nan,"weighted_n":nw,"weighted_mean":mu,"kish_effective_n":ess})
    for c in ["cognition","locomotion","grip_vitality","psychological"]:
        z=pd.DataFrame({"x":d[c],"y":d.fi_primary,"w":weight(w)}).dropna()
        uw=z.x.corr(z.y,method="spearman") if len(z)>2 and z.x.nunique()>1 and z.y.nunique()>1 else np.nan
        rw,npair,ess=wrho(d[c],d.fi_primary,w)
        out["assoc"].append({"dataset":ds,"weight_role":role,"weight_variable":wvar,"construct":c,"outcome":"fi_primary","n_pairwise":len(z),"unweighted_spearman":uw,"weighted_spearman":rw,"kish_effective_n":ess})

def key(x,w=None):
    y=x.astype("string").str.replace(r"\.0$","",regex=True); return y.str.zfill(w) if w else y

def elsa():
    p=ROOT/"ELSA/Working_data/elsa.dta"; r=ROOT/"ELSA/Raw_data/wave6/wave_6_elsa_data_v2.dta"
    cols=["idauniqc","wave","imrc","dlrc","orient","walkra","walk100a","lgrip","rgrip","cesd","shlt","mbmi","hibpe","diabe","hearte","stroke","cancre","arthre"]
    d=read(p,cols); d=d[d.wave.eq(6)].copy(); d["idauniqc"]=key(d.idauniqc)
    w=read(r,["idauniq","w6xwgt"]); w["idauniq"]=key(w.idauniq); d=d.merge(w,left_on="idauniqc",right_on="idauniq",how="left",validate="one_to_one")
    d["cognition"]=mean_parts([n(d.imrc,0,10)/10,n(d.dlrc,0,10)/10,n(d.orient,0,4)/4],2)
    d["locomotion"]=mean_parts([1-b(d.walkra),1-b(d.walk100a)],2); d["grip_vitality"]=pd.concat([n(d.lgrip,0,100),n(d.rgrip,0,100)],axis=1).max(axis=1,skipna=True)/100
    d["psychological"]=1-n(d.cesd,0,8)/8; d["bmi"]=d.mbmi; d["fi_primary"]=fi(d)
    return d,d.w6xwgt,"w6xwgt"

def charls():
    p=ROOT/"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"
    cols=["inw3","r3imrc","r3dlrc","r3orient","r3ser7","r3walk100a","r3walk1kma","r3lgrip1","r3lgrip2","r3rgrip1","r3rgrip2","r3depresl","r3effortl","r3sleeprl","r3whappyl","r3shlt","r3mbmi","r3hibpe","r3diabe","r3hearte","r3stroke","r3cancre","r3arthre","r3wtrespb"]
    d=read(p,cols); d=d[d.inw3.eq(1)].copy(); d["cognition"]=mean_parts([n(d.r3imrc,0,10)/10,n(d.r3dlrc,0,10)/10,n(d.r3orient,0,4)/4,n(d.r3ser7,0,5)/5],2)
    d["locomotion"]=mean_parts([1-b(d.r3walk100a),1-b(d.r3walk1kma)],2); d["grip_vitality"]=pd.concat([n(d.r3lgrip1,0,100),n(d.r3lgrip2,0,100),n(d.r3rgrip1,0,100),n(d.r3rgrip2,0,100)],axis=1).max(axis=1,skipna=True)/100
    sy=pd.concat([n(d.r3depresl,1,4),n(d.r3effortl,1,4),n(d.r3sleeprl,1,4),5-n(d.r3whappyl,1,4)],axis=1); d["psychological"]=1-(sy.mean(axis=1)-1)/3
    d["bmi"]=d.r3mbmi; d["hibpe"]=d.r3hibpe; d["diabe"]=d.r3diabe; d["hearte"]=d.r3hearte; d["stroke"]=d.r3stroke; d["cancre"]=d.r3cancre; d["arthre"]=d.r3arthre; d["shlt"]=d.r3shlt; d["fi_primary"]=fi(d)
    return d,d.r3wtrespb,"r3wtrespb"

def hrs():
    p=ROOT/"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"
    cols=["hhid","pn","inw10","r10proxy","r10imrc","r10dlrc","r10ser7","r10walkra","r10walk1a","r10grpl","r10grpr","r10depres","r10effort","r10sleepr","r10whappy","r10shlt","r10bmi","r10hibpe","r10diabe","r10hearte","r10stroke","r10cancre","r10arthre","r10wtresp"]
    d=read(p,cols); d=d[d.inw10.eq(1)&d.r10proxy.eq(0)].copy(); tp=ROOT/"HRS/public_survey_data/Cross-Wave Tracker File/trk2022TR_R.csv"; t=pd.read_csv(tp,usecols=["HHID","PN","MPMWGTR"],dtype="string")
    d["hhid"]=key(d.hhid,6); d["pn"]=key(d.pn,3); t["HHID"]=key(t.HHID,6); t["PN"]=key(t.PN,3); d=d.merge(t,left_on=["hhid","pn"],right_on=["HHID","PN"],how="left",validate="many_to_one")
    d["cognition"]=mean_parts([n(d.r10imrc,0,10)/10,n(d.r10dlrc,0,10)/10,n(d.r10ser7,0,5)/5],2); d["locomotion"]=mean_parts([1-n(d.r10walkra,0,2)/2,1-n(d.r10walk1a,0,2)/2],2)
    d["grip_vitality"]=pd.concat([n(d.r10grpl,0,100),n(d.r10grpr,0,100)],axis=1).max(axis=1,skipna=True)/100; d["psychological"]=1-pd.concat([b(d.r10depres),b(d.r10effort),b(d.r10sleepr),b(d.r10whappy)],axis=1).mean(axis=1)
    for a,z in [("hibpe","r10hibpe"),("diabe","r10diabe"),("hearte","r10hearte"),("stroke","r10stroke"),("cancre","r10cancre"),("arthre","r10arthre"),("shlt","r10shlt"),("bmi","r10bmi")]: d[a]=d[z]
    d["fi_primary"]=fi(d); return d,d.r10wtresp,d.MPMWGTR,"r10wtresp","MPMWGTR"

def share():
    p=OUT/"_share_ascii_real/Working_data/share.dta"; d=read(p,["mergeid","wave","walkra","walk100a","lgrip","rgrip","shlt","bmi","hibpe","diabe","hearte","stroke","cancre","arthre"]); d=d[d.wave.eq(6)].copy(); d["mergeid"]=key(d.mergeid)
    cf=next((OUT/"_share_ascii_real/Raw_data").rglob("*w6*cf.dta")); gh=next((OUT/"_share_ascii_real/Raw_data").rglob("*w6*gv_health.dta")); cc=["mergeid","cf003_","cf004_","cf005_","cf006_","cf010_"]+[f"cf{i}tot" for i in range(104,108)]+[f"cf{i}tot" for i in range(113,117)]
    c=read(cf,cc); g=read(gh,["mergeid","euro1","euro2","euro5","euro11"]); c["mergeid"]=key(c.mergeid); g["mergeid"]=key(g.mergeid); d=d.merge(c,on="mergeid",how="left",validate="one_to_one").merge(g,on="mergeid",how="left",validate="one_to_one")
    o=d[["cf003_","cf004_","cf005_","cf006_"]].apply(lambda x:n(x,1,2).map({1:1.,2:0.})).mean(axis=1); ix=[f"cf{i}tot" for i in range(104,108)]; dx=[f"cf{i}tot" for i in range(113,117)]; iv=d[ix].apply(lambda x:n(x,0,10)); dv=d[dx].apply(lambda x:n(x,0,10)); imm=iv.sum(axis=1,min_count=1).div(10).where(iv.notna().sum(axis=1).eq(1)); de=dv.sum(axis=1,min_count=1).div(10).where(dv.notna().sum(axis=1).eq(1)); d["cognition"]=mean_parts([imm,de,o,n(d.cf010_,0,100)/100],3)
    sev=pd.Series(np.nan,index=d.index); a=b(d.walk100a); q=b(d.walkra); sev.loc[a.eq(0)]=0; sev.loc[a.eq(1)&q.eq(0)]=1; sev.loc[q.eq(1)]=2; d["locomotion"]=1-sev/2; d["grip_vitality"]=pd.concat([n(d.lgrip,0,100),n(d.rgrip,0,100)],axis=1).max(axis=1,skipna=True)/100; d["psychological"]=1-pd.concat([b(d.euro1),b(d.euro2),b(d.euro5),b(d.euro11)],axis=1).mean(axis=1,skipna=False); d["fi_primary"]=fi(d)
    wp=next((ROOT/"SHARE").rglob("sharew6_rel9-0-0_gv_weights.dta")); w=read(wp,["mergeid","cciw_w6"]); w["mergeid"]=key(w.mergeid); d=d.merge(w,on="mergeid",how="left",validate="one_to_one"); return d,d.cciw_w6,"cciw_w6"

def main():
    out={"desc":[],"assoc":[]}; e,ew,ev=elsa(); add(out,"ELSA",e,ew,"general_cross_sectional",ev); c,cw,cv=charls(); add(out,"CHARLS",c,cw,"general_cross_sectional",cv); h,hw,hm,hv,hmv=hrs(); add(out,"HRS",h,hw,"general_cross_sectional",hv); hp=h[h.grip_vitality.notna()&h.fi_primary.notna()]; add(out,"HRS",hp,hm.loc[hp.index],"physical_module",hmv); s,sw,sv=share(); add(out,"SHARE",s,sw,"external_cross_sectional",sv)
    desc=pd.DataFrame(out["desc"]).sort_values(["dataset","weight_role","construct"]).round(6); assoc=pd.DataFrame(out["assoc"]).sort_values(["dataset","weight_role","construct"]).round(6); desc.to_csv(TAB/"stage22_weighted_descriptives.csv",index=False,encoding="utf-8-sig"); assoc.to_csv(TAB/"stage22_weighted_associations.csv",index=False,encoding="utf-8-sig")
    counts={"ELSA":len(e),"CHARLS":len(c),"HRS":len(h),"SHARE":len(s)}; memo=f"""# Stage 22 design-weighted descriptive and IC–FI sensitivity\n\n本阶段仅比较 0–1 capacity-oriented proxy 的未加权/加权均值和 IC–FI 加权秩相关，不替代 Stage 15–17 的 partial-metric 主模型，不提供复杂抽样置信区间，也不做插补。\n\n- ELSA 使用 `w6xwgt`；CHARLS 使用 `r3wtrespb`；HRS 一般结构使用 `r10wtresp`，另以 `MPMWGTR` 做 physical-measures 敏感性；SHARE 使用 `cciw_w6`。权重只保留正值。\n- ELSA 没有审计到的官方 PSU/stratum，CHARLS 缺少明确命名的 stratum，因此结果仅是设计影响敏感性，不能宣称四队列统一的复杂抽样方差校正。\n- FI 按 6/8 observed-component denominator、BMI<18.5 或 >=30；所有相关均为 pairwise complete，不进行 MI。\n\n锚定样本量：ELSA {counts['ELSA']:,}，CHARLS {counts['CHARLS']:,}，HRS 非 proxy {counts['HRS']:,}，SHARE {counts['SHARE']:,}。\n\n加权均值见 `tables/stage22_weighted_descriptives.csv`，IC–FI 相关见 `tables/stage22_weighted_associations.csv`。若加权与未加权方向一致，只能支持结构方向对样本构成加权不太敏感；不能写成测量等值性、总体因果效应或潜在均值比较。\n"""; (OUT/"stage22_design_weighted_sensitivity_memo.md").write_text(memo,encoding="utf-8")
    info={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"stage":"22 design-weighted descriptive and IC-FI sensitivity","aggregate_only":True,"counts":counts,"outputs":["tables/stage22_weighted_descriptives.csv","tables/stage22_weighted_associations.csv","stage22_design_weighted_sensitivity_memo.md"],"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; (OUT/"stage22_design_weighted_sensitivity_run_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8"); print("Wrote Stage 22 design-weighted sensitivity outputs.")

if __name__=="__main__": main()
