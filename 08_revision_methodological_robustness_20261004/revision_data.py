"""Canonical, source-label-based prediction data. Source files are read only.

Person-level frames stay in memory except anonymous MI matrices in a separate
restricted local analysis folder outside the submission package.
"""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
import pyreadstat

PKG=Path(os.environ.get('IC_FRAILTY_V2_ROOT', 'PATH_TO_IC_FRAILTY_V2'))
AUDIT=Path(os.environ.get('IC_FRAILTY_SOURCE_METADATA', str(PKG/'analysis_audit/statistical_revision_20260929')))
OUT=Path(__file__).resolve().parent/'outputs'
OUT.mkdir(exist_ok=True,parents=True)
META=json.loads((AUDIT/'source_metadata.json').read_text(encoding='utf8'))
if os.environ.get('IC_FRAILTY_DATA_ROOT'):
    original=Path(os.environ.get('IC_FRAILTY_ORIGINAL_ROOT', 'PATH_TO_STATISTICAL_MODELING'))
    data_root=Path(os.environ['IC_FRAILTY_DATA_ROOT'])
    for c,m in META.items():
        source=Path(m['path'])
        try:
            relative=source.relative_to(original)
        except ValueError:
            relative=Path(source.name)
        m['path']=str(data_root/relative)
PRIVATE=Path(os.environ.get('IC_FRAILTY_PRIVATE',str(Path(__file__).resolve().parent/'private_outputs')))
DOM=['cognition','locomotion','grip_vitality','psychological']
BASE=['age','male','education','baseline_fi']
DISEASES=['hibpe','diabe','hearte','stroke','cancre','arthre']
COMMON=['dressa','batha','eata','beda','toilta','medsa','moneya','shopa','mealsa']
SHARE14=['walkra',*COMMON,'phonea','mapa','leavhsa','laundrya']
SCALE=['age','education','baseline_fi',*DOM,'bmi','srh']

def numeric(x,lo=None,hi=None):
    z=pd.to_numeric(x,errors='coerce').astype(float)
    if lo is not None:z=z.where(z>=lo)
    if hi is not None:z=z.where(z<=hi)
    return z

def binary(x,hrs=False):
    z=numeric(x)
    return z.map({0:0.,1:1.,2:1.} if hrs else {0:0.,1:1.})

def health_deficit(x,cohort):
    z=numeric(x,1,5)
    return (5-z)/4 if cohort in ['ELSA','SHARE'] else (z-1)/4

def avg(vals,minimum=1):
    z=pd.concat(vals,axis=1)
    return z.mean(axis=1).where(z.notna().sum(axis=1)>=minimum)

def read(cohort,cols):
    missing=sorted(set(cols)-set(META[cohort]['columns']))
    if missing:raise ValueError(f'{cohort}: missing required columns {missing}')
    return pyreadstat.read_dta(META[cohort]['path'],usecols=list(dict.fromkeys(cols)))[0]

def items_outcome(d,prefix,cohort,items,minimum):
    z=pd.concat([binary(d[prefix+x],cohort=='HRS') for x in items],axis=1)
    return (z.max(axis=1)>0).astype(float).where(z.notna().sum(axis=1)>=minimum)

def legacy_outcome(d,prefix,cohort):
    if cohort=='SHARE':return items_outcome(d,prefix,cohort,SHARE14,10)
    cols={'ELSA':['adltot6','iadltot2_e'],'CHARLS':['adlwa','iadla'],'HRS':['adl5a','iadl5a']}[cohort]
    a=numeric(d[prefix+cols[0]],0,30); b=numeric(d[prefix+cols[1]],0,30)
    return (a+b>0).astype(float).where(a.notna() & b.notna())

def predictors(d,c,p=''):
    o=pd.DataFrame(index=d.index)
    col=lambda s:d[p+s]
    o['age']=numeric(col('agey_m' if c=='HRS' else 'agey'),18,115)
    o['male']=numeric(d.ragender).map({0:0.,1:1.} if c in ['ELSA','SHARE'] else {1:1.,2:0.})
    ed={'ELSA':('raeducl',3),'SHARE':('raeducl',3),'CHARLS':('raeduc_c',10),'HRS':('raeduc',5)}[c]
    o['education']=numeric(d[ed[0]],1,ed[1])
    for s in [*DISEASES,'lunge']:o[s]=binary(col(s))
    o['srh']=health_deficit(col('shlt'),c)
    o['bmi']=numeric(col('mbmi' if c in ['ELSA','CHARLS'] else 'bmi'),10,80)
    o['bmi_deficit']=((o.bmi<18.5)|(o.bmi>=30)).astype(float).where(o.bmi.notna())
    z=o[[*DISEASES,'srh','bmi_deficit']]
    o['fi_observed']=z.notna().sum(axis=1)
    o['baseline_fi']=z.mean(axis=1).where(o.fi_observed>=6)
    if c=='ELSA':
        o['cognition']=numeric(d.tcog_z_z,-15,15)
        o['locomotion']=avg([1-binary(d.walkra),1-binary(d.walk100a)])
        o['grip_vitality']=numeric(d.gripsum,0,100)
        o['psychological']=1-numeric(d.cesd,0,8)/8
    elif c=='CHARLS':
        o['cognition']=avg([numeric(col('orient'),0,4)/4,numeric(col('tr20'),0,20)/20],2)
        o['locomotion']=1-binary(col('walk100a'))
        o['grip_vitality']=pd.concat([numeric(col('lgrip'),0,100),numeric(col('rgrip'),0,100)],axis=1).max(axis=1)
        o['psychological']=1-numeric(col('cesd10'),0,30)/30
    elif c=='HRS':
        o['cognition']=avg([numeric(col('imrc'),0,10)/10,numeric(col('dlrc'),0,10)/10,numeric(col('ser7'),0,5)/5],2)
        o['locomotion']=avg([1-binary(col(s),True) for s in ['walkra','walk1a','walksa']])
        o['grip_vitality']=pd.concat([numeric(col('grpl'),0,100),numeric(col('grpr'),0,100)],axis=1).max(axis=1)
        o['psychological']=1-numeric(col('cesd'),0,8)/8
    else:
        o['cognition']=avg([numeric(col('imrc'),0,10)/10,numeric(col('dlrc'),0,10)/10,numeric(col('orient'),0,4)/4],2)
        o['locomotion']=avg([1-binary(col(s)) for s in ['walkra','walk100a']])
        o['grip_vitality']=pd.concat([numeric(col('lgrip'),0,100),numeric(col('rgrip'),0,100)],axis=1).max(axis=1)
        o['psychological']=1-numeric(col('eurod'),0,12)/12
    o['raw_cognition']=avg([numeric(col('imrc'),0,10)/10,numeric(col('dlrc'),0,10)/10,numeric(col('orient'),0,4)/4],2) if c in ['ELSA','SHARE'] else o.cognition
    return o

def load_pair(c,target=None,raw_share=False):
    long=c in ['ELSA','SHARE']
    bw={'ELSA':6,'SHARE':6,'CHARLS':3,'HRS':10}[c]
    fw=target or bw+1
    p='' if long else f'r{bw}'; q='' if long else f'r{fw}'
    ident={'ELSA':'idauniqc','SHARE':'mergeid','CHARLS':'ID','HRS':'hhidpn'}[c]
    cluster={'ELSA':'hhidc','SHARE':'hhidc','CHARLS':'communityID','HRS':'hhid'}[c]
    ed={'ELSA':'raeducl','SHARE':'raeducl','CHARLS':'raeduc_c','HRS':'raeduc'}[c]
    specific={'ELSA':['tcog_z_z','gripsum','cesd','walkra','walk100a','imrc','dlrc','orient'],
              'SHARE':['imrc','dlrc','orient','walkra','walk100a','lgrip','rgrip','eurod'],
              'CHARLS':['orient','tr20','walk100a','lgrip','rgrip','cesd10'],
              'HRS':['imrc','dlrc','ser7','walkra','walk1a','walksa','grpl','grpr','cesd']}[c]
    sums={'ELSA':['adltot6','iadltot2_e'],'CHARLS':['adlwa','iadla'],'HRS':['adl5a','iadl5a'],'SHARE':SHARE14}[c]
    cols=[ident,cluster,'ragender',ed]+[p+s for s in [*specific,*DISEASES,'lunge','shlt','agey_m' if c=='HRS' else 'agey','mbmi' if c in ['ELSA','CHARLS'] else 'bmi']]
    cols+=[pr+s for pr in set([p,q]) for s in [*COMMON,*sums]]
    if long:cols+=['wave','iwstat']
    else:cols += [f'inw{bw}',f'inw{fw}',q+'iwstat']
    if c=='HRS':cols +=[p+'proxy',q+'proxy']
    if c=='SHARE':cols+=['country']
    d=read(c,cols)
    if long:
        b=d.loc[d.wave.eq(bw)].copy();f=d.loc[d.wave.eq(fw)].copy()
        # Duplicate keys must not be silently discarded.
        assert not b[ident].duplicated().any() and not f[ident].duplicated().any()
        f=f.set_index(ident).reindex(b[ident]).set_axis(b.index)
        present=b[ident].isin(d.loc[d.wave.eq(fw),ident])
        alive=numeric(f.iwstat).eq(0)
        active=pd.Series(True,index=b.index)
        proxy=active
        target_ok=present & alive
        death=numeric(f.iwstat).eq(1)
    else:
        b=d.loc[numeric(d[f'inw{bw}']).eq(1)].copy(); f=b
        active=pd.Series(True,index=b.index)
        proxy=numeric(b[p+'proxy']).eq(0) if c=='HRS' else active
        present=numeric(b[f'inw{fw}']).eq(1)
        target_ok=present & (numeric(b[q+'proxy']).eq(0) if c=='HRS' else True)
        death=numeric(b[q+'iwstat']).isin([3,5,6])
    o=predictors(b,c,p)
    if c=='SHARE' and raw_share:
        rawpath=next(Path(META[c]['path']).parents[1].rglob('sharew6_rel9-0-0_gv_health.dta'))
        raw=pyreadstat.read_dta(str(rawpath),usecols=['mergeid','cf008tot','cf016tot','orienti','eurod','maxgrip','bmi','sphus'])[0]
        assert not raw.mergeid.duplicated().any()
        raw=raw.set_index('mergeid').reindex(b[ident]).set_axis(b.index)
        o['cognition']=avg([numeric(raw.cf008tot,0,10)/10,numeric(raw.cf016tot,0,10)/10,numeric(raw.orienti,0,4)/4],2)
        o['psychological']=1-numeric(raw.eurod,0,12)/12
        o['grip_vitality']=numeric(raw.maxgrip,0,100)
        o['bmi']=numeric(raw.bmi,10,80)
        o['bmi_deficit']=((o.bmi<18.5)|(o.bmi>=30)).astype(float).where(o.bmi.notna())
        o['srh']=(numeric(raw.sphus,1,5)-1)/4
        zz=o[[*DISEASES,'srh','bmi_deficit']]
        o['fi_observed']=zz.notna().sum(axis=1)
        o['baseline_fi']=zz.mean(axis=1).where(o.fi_observed>=6)
    o['baseline_event']=legacy_outcome(b,p,c)
    o['event']=legacy_outcome(f,q,c).where(target_ok)
    o['baseline_common']=items_outcome(b,p,c,COMMON,9)
    o['event_common']=items_outcome(f,q,c,COMMON,9).where(target_ok)
    o['baseline_self']=proxy
    o['followup_interview']=present
    o['followup_eligible']=target_ok
    o['death']=death
    # Factorized cluster keys remain in memory only; never export original keys.
    keys=b[cluster].astype('string').fillna(pd.Series(['missing_'+str(i) for i in range(len(b))],index=b.index))
    o['cluster']=c+'_'+pd.Series(pd.factorize(keys)[0],index=b.index).astype(str)
    o['cohort']=c;o['window']=f'{bw}-{fw}';o['years']=3 if c=='CHARLS' else (fw-bw)*2
    if c=='SHARE':o['country']=b.country
    return o.reset_index(drop=True)

def select_frame(d,endpoint='legacy'):
    b='baseline_event' if endpoint=='legacy' else 'baseline_common'
    f='event' if endpoint=='legacy' else 'event_common'
    o=d.loc[d.baseline_self & d[b].eq(0)].copy()
    if endpoint!='legacy':o['event']=o[f]
    return o

def standardize(d,ref=None):
    o=d.copy();pars=[]
    if ref is None:ref=d
    for c in SCALE:
        mu=ref[c].mean(); sd=ref[c].std(ddof=1)
        if not np.isfinite(sd) or sd==0:sd=1.
        o[c+'_z']=(o[c]-mu)/sd
        pars.append({'cohort':d.cohort.iloc[0],'window':d.window.iloc[0],'variable':c,'mean':mu,'sd':sd,'reference_n':len(ref),'observed_n':int(ref[c].notna().sum())})
    for c in ['age','bmi','srh']:o[c+'_z2']=o[c+'_z']**2
    return o,pd.DataFrame(pars)

def export_flow(frames):
    rows=[];miss=[]
    for key,d in frames.items():
        for endpoint in ['legacy','common9']:
            b='baseline_event' if endpoint=='legacy' else 'baseline_common'
            f='event' if endpoint=='legacy' else 'event_common'
            masks=[('baseline_interview',pd.Series(True,index=d.index)),('baseline_self',d.baseline_self)]
            masks += [('baseline_outcome_observed',d.baseline_self & d[b].notna()),('baseline_disability_free',d.baseline_self & d[b].eq(0))]
            eligible=masks[-1][1]
            masks += [('followup_interview',eligible & d.followup_interview),('followup_eligible',eligible & d.followup_eligible),('followup_outcome_observed',eligible & d[f].notna()),('complete_predictors',eligible & d[f].notna() & d[BASE+DOM].notna().all(axis=1))]
            for label,mask in masks:rows.append({'sample':key,'endpoint':endpoint,'stage':label,'n':int(mask.sum()),'events':int(d.loc[mask,f].sum()) if 'followup' in label or label=='complete_predictors' else np.nan})
            rows.append({'sample':key,'endpoint':endpoint,'stage':'known_deaths_among_eligible','n':int((eligible & d.death).sum()),'events':np.nan})
            for v in BASE+DOM+['event']:
                x=d.loc[eligible,v if v!='event' else f]
                miss.append({'sample':key,'endpoint':endpoint,'variable':v,'eligible':len(x),'missing':int(x.isna().sum()),'missing_pct':100*x.isna().mean()})
    pd.DataFrame(rows).to_csv(OUT/'participant_flow.csv',index=False)
    pd.DataFrame(miss).to_csv(OUT/'missingness.csv',index=False)

def build_all():
    frames={}
    for c in ['ELSA','CHARLS','HRS','SHARE']:
        print('Reading',c,flush=True);frames[c]=load_pair(c)
    frames['SHARE4y']=load_pair('SHARE',8)
    export_flow(frames)
    return frames

if __name__=='__main__':
    frames=build_all()
    for k,d in frames.items():
        e=select_frame(d)
        print(k,'baseline',len(d),'eligible',len(e),'CC',len(e.dropna(subset=BASE+DOM+['event'])),'common9 eligible',len(select_frame(d,'common9')),flush=True)
