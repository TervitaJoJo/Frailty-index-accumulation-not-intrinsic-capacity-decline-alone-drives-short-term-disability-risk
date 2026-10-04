from pathlib import Path
p=Path(r'PATH_TO_IC_FRAILTY\53_longitudinal_predictive_feasibility.py')
s=p.read_text(encoding='utf-8')
old='''    observed = out.notna().sum(axis=1)
    fi = out.sum(axis=1, min_count=1).div(observed).where(observed.ge(6))
'''
new='''    out = out.apply(pd.to_numeric, errors="coerce")
    observed = out.notna().sum(axis=1)
    total = out.sum(axis=1, min_count=1)
    fi = total.div(observed).where(observed.ge(6))
'''
assert old in s
p.write_text(s.replace(old,new), encoding='utf-8')
print('patched')

