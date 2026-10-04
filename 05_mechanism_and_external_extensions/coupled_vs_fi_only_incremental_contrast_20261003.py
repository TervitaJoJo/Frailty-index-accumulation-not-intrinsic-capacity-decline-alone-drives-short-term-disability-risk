from pathlib import Path
import sys, json, os, hashlib, platform
from datetime import datetime, timezone
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
PKG=Path(r'PATH_TO_IC_FRAILTY_V2')
AUDIT=PKG/'analysis_audit'
OUT=AUDIT/'trajectory_states_20260930'/'conditional_analyses_20261002'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PKG/'scripts'))
import trajectory_states_analysis_20260930 as ts
sys.path.insert(0,str(AUDIT/'share_four_state_validation_20261003'))
import share_four_state_validation_20261003 as sh
DEFS=['primary_sign','cohort_median','meaningful_0.2SD','meaningful_0.5SD']
STATES=['preserved_low','ic_decline_only','fi_accumulation_only','coupled']
BASE=['age_baseline','male','education','baseline_fi']
Z=1.959963984540054

def model_sample(q):
    return q.loc[q.outcome_category.isin(['disability','disability_free','death'])].copy()

def risk_stats(q):
    rows={}
    for s in ['fi_accumulation_only','coupled']:
        h=q.loc[q.state.eq(s)]; n=len(h); e=int(h.outcome_category.eq('disability').sum()); rows[s]=(n,e,e/n)
    nf,ef,pf=rows['fi_accumulation_only']; nc,ec,pc=rows['coupled']; rd=pc-pf
    se=np.sqrt(pc*(1-pc)/nc+pf*(1-pf)/nf)
    return {'fi_only_n':nf,'fi_only_events':ef,'fi_only_risk':pf,'coupled_n':nc,'coupled_events':ec,'coupled_risk':pc,'risk_difference':rd,'risk_difference_low':rd-Z*se,'risk_difference_high':rd+Z*se}

def fit(q, model):
    q=model_sample(q); q['event']=q.outcome_category.eq('disability').astype(int)
    extra={'covariate_adjusted':[],'cohort_adjusted':['cohort'],'country_adjusted':['country']}[model]
    g=q.dropna(subset=['state']+BASE+extra).copy()
    rhs="event ~ C(state, Treatment(reference='fi_accumulation_only')) + "+' + '.join(BASE)
    if extra: rhs+=' + '+' + '.join(f'C({x})' for x in extra)
    f=smf.glm(rhs,data=g,family=sm.families.Binomial()).fit(cov_type='HC3')
    term=next(x for x in f.params.index if x.endswith('[T.coupled]'))
    b=float(f.params[term]); se=float(f.bse[term])
    out={'model':model,'n':int(f.nobs),'events':int(g.event.sum()),'known_outcome_n':len(q),'known_disability_events':int(q.event.sum()),'covariate_missing_excluded_n':len(q)-len(g),'OR_coupled_vs_FI_only':float(np.exp(b)),'OR_low':float(np.exp(b-Z*se)),'OR_high':float(np.exp(b+Z*se)),'p':float(f.pvalues[term]),'formula':rhs,'converged':bool(f.converged)}
    out.update(risk_stats(q)); return out

def state_rows(q,pop,definition):
    known=model_sample(q); rows=[]
    for s in STATES:
        h=q.loc[q.state.eq(s)]; k=known.loc[known.state.eq(s)]; n=len(k); e=int(k.outcome_category.eq('disability').sum()); lo,hi=ts.wilson(e,n)
        rows.append({'population':pop,'definition':definition,'state':s,'trajectory_gate_n':len(h),'trajectory_gate_proportion':len(h)/len(q),'known_outcome_n':n,'known_outcome_proportion':n/len(known),'unknown_outcome_n':len(h)-n,'disability_events':e,'disability_risk':e/n,'risk_low':lo,'risk_high':hi})
    return rows

def make(z,definition): return ts.assign_states(z,definition)

def threshold_rows(z,pop,definition):
    q=make(z,definition); p=make(z,'primary_sign')
    return {'population':pop,'definition':definition,'trajectory_gate_n':len(z),'ic_cut':float(q.ic_cut.iloc[0]),'fi_cut':float(q.fi_cut.iloc[0]),'delta_ic_sd':float(z.delta_ic.std(ddof=1)),'delta_fi_sd':float(z.delta_fi.std(ddof=1)),'delta_ic_median':float(z.delta_ic.median()),'delta_fi_median':float(z.delta_fi.median()),'reclassified_n_vs_sign':int(q.state.ne(p.state).sum()),'reclassified_fraction_vs_sign':float(q.state.ne(p.state).mean())}

def main():
    dev={}
    for c in ['ELSA','CHARLS','HRS']:
        raw,_=ts.load_cohort(c); dev[c],_=ts.prepare_analysis(raw)
    d=sh.load(); wf={w:sh.wave_frame(d,w) for w in [4,5,6]}; _,share,_,_=sh.build(wf,4,5,6)
    results=[]; risks=[]; cuts=[]
    for definition in DEFS:
        parts=[]
        for c,z in dev.items():
            q=make(z,definition); parts.append(q); risks+=state_rows(q,c,definition); cuts.append(threshold_rows(z,c,definition))
        pooled=pd.concat(parts,ignore_index=True); risks+=state_rows(pooled,'ELSA+CHARLS+HRS',definition)
        x=fit(pooled,'cohort_adjusted'); x.update(population='ELSA+CHARLS+HRS',definition=definition,outcome='disability',mapping='ELSA 6-8-9; CHARLS 1-3-4; HRS 8-10-12'); results.append(x)
        q=make(share,definition); risks+=state_rows(q,'SHARE',definition); cuts.append(threshold_rows(share,'SHARE',definition))
        for m in ['covariate_adjusted','country_adjusted']:
            x=fit(q,m); x.update(population='SHARE',definition=definition,outcome='disability',mapping='SHARE 4-5-6'); results.append(x)
    r=pd.DataFrame(results); r['p_holm_12']=multipletests(r.p,method='holm')[1]
    r.to_csv(OUT/'coupled_vs_fi_only_disability_contrast_all_thresholds.csv',index=False)
    r.loc[r.definition.eq('primary_sign')].to_csv(OUT/'coupled_vs_fi_only_disability_contrast.csv',index=False)
    pd.DataFrame(risks).to_csv(OUT/'coupled_vs_fi_only_state_risks_all_thresholds.csv',index=False)
    pd.DataFrame(cuts).to_csv(OUT/'coupled_vs_fi_only_threshold_definitions.csv',index=False)
    plan={'run_utc':datetime.now(timezone.utc).isoformat(),'definitions':DEFS,'primary_rule':'delta_IC < 0 and delta_FI > 0','sensitivity_rules':'within-cohort median, 0.2 SD and 0.5 SD of the observed change; strict inequalities','development_model':'age + sex + education + baseline FI + cohort fixed effects','share_models':'age + sex + education + baseline FI, with and without country fixed effects','risk_difference':'unadjusted fixed-window disability risk; normal-approximation 95% CI','p_values':'two-sided nominal P values; Holm-adjusted P values across all 12 contrasts','threshold_selection':'all rules reported; no threshold selected by outcome or P value','estimand':'disability endpoint proportion; coded death treated as a non-disability endpoint in this contrast','nnt':'not estimated because this observational contrast is not an intervention effect','aggregate_only':True}
    (OUT/'coupled_vs_fi_only_disability_contrast_manifest.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    print(r[['population','definition','model','n','events','OR_coupled_vs_FI_only','OR_low','OR_high','p','p_holm_12','risk_difference']].to_string(index=False))
if __name__=='__main__': main()
