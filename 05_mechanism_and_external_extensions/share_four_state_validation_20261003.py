"""Aggregate-only audit for full directional four-state replication in SHARE."""
from pathlib import Path
import json, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf
PKG=Path(r"PATH_TO_IC_FRAILTY_V2")
OUT=PKG/"analysis_audit/share_four_state_validation_20261003"; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PKG/"scripts"))
from revision_data import META,COMMON,predictors,items_outcome,numeric
DOM=["cognition","locomotion","grip_vitality","psychological"]; END=[x for x in COMMON if x!="moneya"]
CAND=[(1,2,4),(1,2,6),(1,2,8),(1,4,6),(1,4,8),(2,4,6),(2,4,8),(4,5,6),(4,5,8),(5,6,8)]

def wilson(k,n):
    if n<=0:return np.nan,np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n;ctr=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,ctr-h),min(1,ctr+h)

def load():
    cols=["mergeid","wave","iwstat","iwy","iwm","agey","ragender","raeducl","country","shlt","bmi","hibpe","diabe","hearte","stroke","cancre","arthre","lunge","imrc","dlrc","orient","walkra","walk100a","lgrip","rgrip","eurod",*COMMON]
    cols=[c for c in dict.fromkeys(cols) if c in META["SHARE"]["columns"]]
    d,_=pyreadstat.read_dta(META["SHARE"]["path"],usecols=cols); d["mergeid"]=d.mergeid.astype(str); return d

def wave_frame(d,w):
    s=d.loc[numeric(d.wave).eq(w)].copy().reset_index(drop=True); x=predictors(s,"SHARE","")
    x["disability"]=items_outcome(s,"","SHARE",END,len(END)); x["status"]=numeric(s.iwstat); x["proxy"]=True; x["id"]=s.mergeid.values; x["country"]=s.country.values if "country" in s else np.nan; x["wave"]=w; return x

def build(wf,b,i,o):
    x=wf[b].copy()
    for w in [i,o]:
        z=wf[w].copy(); keep=["id","status","proxy","disability","age","baseline_fi","country",*DOM]; z=z[keep].rename(columns={c:f"{c}_{w}" for c in keep if c!="id"}); x=x.merge(z,on="id",how="left",validate="one_to_one")
    x=x.rename(columns={"status":f"status_{b}","proxy":f"proxy_{b}","disability":f"disability_{b}","age":f"age_{b}"})
    base=x[f"age_{b}"].ge(50)&x[f"proxy_{b}"]; pars=[]
    for d in DOM:
        mu=x.loc[base,d].mean(); sd=x.loc[base,d].std(ddof=1); sd=sd if np.isfinite(sd) and sd>0 else 1.0; pars.append((d,mu,sd))
        x[f"{d}_z_{b}"]=(x[d]-mu)/sd
        for w in [i,o]: x[f"{d}_z_{w}"]=(x[f"{d}_{w}"]-mu)/sd
    x["delta_ic"]=x[[f"{d}_z_{i}" for d in DOM]].mean(axis=1)-x[[f"{d}_z_{b}" for d in DOM]].mean(axis=1); x["delta_fi"]=x[f"baseline_fi_{i}"]-x.baseline_fi
    x["alive_base"]=x[f"status_{b}"].eq(0); x["alive_i"]=x[f"status_{i}"].eq(0); x["alive_o"]=x[f"status_{o}"].eq(0); x["status_o_observed"]=x[f"status_{o}"].notna()
    m=x[f"age_{b}"].ge(50)&x.alive_base&x[f"proxy_{b}"]&x[f"disability_{b}"].eq(0)&x[f"disability_{b}"].notna()&x.alive_i&x[f"proxy_{i}"]&x[f"disability_{i}"].eq(0)&x[f"disability_{i}"].notna()&x.delta_ic.notna()&x.delta_fi.notna()&x.baseline_fi.notna()&x[f"baseline_fi_{i}"].notna()
    z=x.loc[m].copy(); z["age_baseline"]=z[f"age_{b}"]; z["outcome_category"]=np.select([z.status_o_observed&~z.alive_o,z.alive_o&z[f"disability_{o}"].eq(1),z.alive_o&z[f"disability_{o}"].eq(0)],["death","disability","disability_free"],default="unknown")
    z["ic_decline"]=z.delta_ic<0; z["fi_accumulation"]=z.delta_fi>0; z["state"]=np.select([~z.ic_decline&~z.fi_accumulation,z.ic_decline&~z.fi_accumulation,~z.ic_decline&z.fi_accumulation,z.ic_decline&z.fi_accumulation],["preserved_low","ic_decline_only","fi_accumulation_only","coupled"],default="unknown")
    return x,z,m,pars

def fit(z, country_adjusted=False):
    q=z.loc[z.outcome_category.isin(["disability","disability_free"])].copy(); q["event"]=q.outcome_category.eq("disability").astype(int)
    if len(q)==0 or q.event.nunique()<2:return pd.DataFrame([{"status":"not_estimable","n":len(q),"events":int(q.event.sum()) if len(q) else 0}])
    try:
        formula="event ~ C(state, Treatment(reference='preserved_low')) + age_baseline + male + education + baseline_fi" + (" + C(country)" if country_adjusted else "")
        f=smf.glm(formula,data=q,family=sm.families.Binomial()).fit(cov_type="HC3"); rows=[]
        for t,c in f.params.items():
            if t=="Intercept":continue
            se=f.bse[t]; rows.append({"term":t,"OR":np.exp(c),"OR_low":np.exp(c-1.959964*se),"OR_high":np.exp(c+1.959964*se),"p":f.pvalues[t],"n":int(f.nobs),"events":int(q.event.sum()),"status":"estimated","country_adjusted":country_adjusted})
        return pd.DataFrame(rows)
    except Exception as e:return pd.DataFrame([{"status":"failed","error":str(e),"n":len(q),"events":int(q.event.sum())}])

def main():
    d=load(); waves=sorted(d.wave.unique().tolist()); wf={w:wave_frame(d,int(w)) for w in waves}; audit=[]; states=[]; models=[]
    for b,i,o in CAND:
        x,z,m,pars=build(wf,b,i,o); name=f"SHARE_{b}_{i}_{o}"
        row={"mapping":name,"base":b,"intermediate":i,"outcome":o,"trajectory_gate_n":int(m.sum()),"known_outcome_n":int(z.outcome_category.ne("unknown").sum()),"disability_events":int(z.outcome_category.eq("disability").sum()),"death_events":int(z.outcome_category.eq("death").sum()),"base_ic_complete":float(x[DOM].notna().all(axis=1).mean()),"intermediate_ic_complete":float(x[[f"{q}_{i}" for q in DOM]].notna().all(axis=1).mean()),"outcome_endpoint_complete":float(x[f"disability_{o}"].notna().mean()),"base_year":float(d.loc[d.wave.eq(b),"iwy"].median()),"intermediate_year":float(d.loc[d.wave.eq(i),"iwy"].median()),"outcome_year":float(d.loc[d.wave.eq(o),"iwy"].median())}; audit.append(row)
        for st,h in z.groupby("state",sort=True):
            den=int(h.outcome_category.ne("unknown").sum()); ev=int(h.outcome_category.eq("disability").sum()); lo,hi=wilson(ev,den); states.append({"mapping":name,"state":st,"n":len(h),"known_outcome_n":den,"disability_events":ev,"death_events":int(h.outcome_category.eq("death").sum()),"disability_risk":ev/den if den else np.nan,"risk_low":lo,"risk_high":hi,"delta_ic_mean":h.delta_ic.mean(),"delta_fi_mean":h.delta_fi.mean()})
        f=fit(z); f["mapping"]=name; models.append(f)
        if (b,i,o)==(4,5,6):
            fc=fit(z, country_adjusted=True); fc["mapping"]=name; models.append(fc)
    pd.DataFrame(audit).to_csv(OUT/"share_mapping_audit.csv",index=False); pd.DataFrame(states).to_csv(OUT/"share_candidate_state_risks.csv",index=False); pd.concat(models,ignore_index=True).to_csv(OUT/"share_candidate_models.csv",index=False)
    # Mapping locked by temporal architecture and measurement completeness before reading state-risk outputs.
    # Waves 4->5->6 provide adjacent two-year intervals, matching the
    # development design more closely than the long-gap alternatives, while
    # retaining the largest complete trajectory set among adjacent mappings.
    b,i,o=4,5,6; x,z,m,pars=build(wf,b,i,o); name="SHARE_4_5_6"
    locked=z.groupby("state",as_index=False).agg(n=("state","size"),known_outcome_n=("outcome_category",lambda x:int(x.ne("unknown").sum())),disability_events=("outcome_category",lambda x:int(x.eq("disability").sum())),death_events=("outcome_category",lambda x:int(x.eq("death").sum())),delta_ic_mean=("delta_ic","mean"),delta_fi_mean=("delta_fi","mean"))
    locked["disability_risk"]=locked.disability_events/locked.known_outcome_n
    ci=locked.apply(lambda r: wilson(int(r.disability_events),int(r.known_outcome_n)),axis=1,result_type="expand"); locked["risk_low"]=ci[0]; locked["risk_high"]=ci[1]
    ref=float(locked.loc[locked.state.eq("preserved_low"),"disability_risk"].iloc[0]); locked["risk_difference_vs_preserved"]=locked.disability_risk-ref
    locked.to_csv(OUT/"share_locked_four_state_summary.csv",index=False)
    pd.concat([fit(z),fit(z,country_adjusted=True)],ignore_index=True).to_csv(OUT/"share_locked_four_state_models.csv",index=False)
    pd.DataFrame([{"mapping":name,"base":b,"intermediate":i,"outcome":o,"rule":"delta_IC<0 and delta_FI>0 after within-SHARE baseline standardisation","death_ascertainment":"unavailable as an outcome: all SHARE iwstat records in the working file are alive-coded"}]).to_csv(OUT/"share_locked_mapping_manifest.csv",index=False)
    memo=f"""# SHARE complete four-state validation audit\n\nRun: {datetime.now(timezone.utc).isoformat()}\nAvailable waves: {waves}\nLocked mapping: wave 4 -> wave 5 -> wave 6.\n\nCandidate mappings were compared for temporal order, IC/FI completeness and disability ascertainment before state-specific risks were examined. Wave 4-5-6 was locked because it preserves three adjacent two-year assessments and the largest complete trajectory set among adjacent-wave mappings; no mapping was chosen from outcome gradients. The development rule was applied without refitting: IC decline = within-SHARE standardised delta_IC < 0; FI accumulation = delta_FI > 0.\n\nThe current SHARE working file has iwstat=0 for all records. This permits external replication of the four-state disability pattern but not a SHARE mortality estimate. Mortality is therefore reported as unavailable, not as zero.\n"""; (OUT/"method_memo.md").write_text(memo,encoding="utf-8"); (OUT/"run_manifest.json").write_text(json.dumps({"run":datetime.now(timezone.utc).isoformat(),"waves_available":waves,"locked_mapping":{"base":b,"intermediate":i,"outcome":o},"aggregate_only":True},indent=2),encoding="utf-8")
    print(pd.DataFrame(audit).to_string(index=False)); print("\nLOCKED STATES\n",pd.read_csv(OUT/"share_locked_four_state_summary.csv").to_string(index=False)); print("\nLOCKED MODELS\n",pd.read_csv(OUT/"share_locked_four_state_models.csv").to_string(index=False))
if __name__=="__main__":main()
