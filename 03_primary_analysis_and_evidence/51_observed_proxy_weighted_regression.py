"""Stage 22b observed-proxy multivariable sensitivity.

Four 0--1 capacity-oriented proxy scores are entered simultaneously to compare
unweighted OLS and positive-weight WLS. No standard errors or causal claims.
"""
from pathlib import Path
import importlib.util, hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

OUT=Path(r"PATH_TO_IC_FRAILTY"); TAB=OUT/"tables"; TAB.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("stage22",OUT/"50_design_weighted_sensitivity.py"); stage22=importlib.util.module_from_spec(spec); spec.loader.exec_module(stage22)
PRED=["cognition","locomotion","grip_vitality","psychological"]

def fit(d,w):
    z=d[PRED+["fi_primary"]].copy(); ww=stage22.weight(w); keep=z.notna().all(axis=1)&ww.notna(); z=z.loc[keep]; ww=ww.loc[keep]
    if len(z)<=len(PRED): return None
    X=np.column_stack([np.ones(len(z)),z[PRED].to_numpy(float)]); y=z.fi_primary.to_numpy(float)
    def beta(q):
        sw=np.sqrt(q.to_numpy(float)); Xw=X*sw[:,None]; yw=y*sw; return np.linalg.pinv(Xw.T@Xw)@(Xw.T@yw)
    return z,ww,beta(pd.Series(np.ones(len(z)),index=z.index)),beta(ww)

def main():
    e,ew,ev=stage22.elsa(); c,cw,cv=stage22.charls(); h,hw,hm,hv,hmv=stage22.hrs(); s,sw,sv=stage22.share()
    jobs=[("ELSA","general_cross_sectional",e,ew,ev),("CHARLS","general_cross_sectional",c,cw,cv),("HRS","general_cross_sectional",h,hw,hv)]
    hp=h[h.grip_vitality.notna()&h.fi_primary.notna()]; jobs.append(("HRS","physical_module",hp,hm.loc[hp.index],hmv)); jobs.append(("SHARE","external_cross_sectional",s,sw,sv))
    rows=[]
    for ds,role,d,w,wvar in jobs:
        res=fit(d,w)
        if res is None: continue
        z,ww,bu,bw=res
        for i,term in enumerate(["intercept"]+PRED): rows.append({"dataset":ds,"weight_role":role,"weight_variable":wvar,"term":term,"n_complete":len(z),"weighted_n":len(ww),"ols_estimate":bu[i],"wls_estimate":bw[i],"wls_minus_ols":bw[i]-bu[i]})
    pd.DataFrame(rows).round(8).to_csv(TAB/"stage22_weighted_proxy_regression.csv",index=False,encoding="utf-8-sig")
    (OUT/"stage22_weighted_proxy_regression_memo.md").write_text("# Stage 22b observed-proxy multivariable sensitivity\n\nThis auxiliary table regresses the observed FI proxy on four 0–1 capacity-oriented domain proxies simultaneously. OLS and positive-weight WLS estimates are shown without standard errors, design-based variance, causal interpretation, or latent-variable meaning. It is a stability audit, not a replacement for the partial-metric latent SEM.\n\nOutput: `tables/stage22_weighted_proxy_regression.csv`.\n",encoding="utf-8")
    info={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"aggregate_only":True,"outputs":["tables/stage22_weighted_proxy_regression.csv","stage22_weighted_proxy_regression_memo.md"],"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; (OUT/"stage22_weighted_proxy_regression_run_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8"); print("Wrote Stage 22b observed-proxy OLS/WLS sensitivity outputs.")

if __name__=="__main__": main()
