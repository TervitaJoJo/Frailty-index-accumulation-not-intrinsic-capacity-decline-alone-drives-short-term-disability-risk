from pathlib import Path
p=Path(r'PATH_TO_IC_FRAILTY\53_longitudinal_predictive_feasibility.py')
s=p.read_text(encoding='utf-8')
old='''        pairs["pair_complete"] = pairs[["baseline_all4_ic","baseline_fi","future_fi"]].notna().all(axis=1)
        pairs["pair_complete8_future"] = pairs[["baseline_all4_ic","baseline_fi","future_fi_complete8"]].notna().all(axis=1)
'''
new='''        pairs["pair_complete"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi"]].notna().all(axis=1)
        pairs["pair_complete8_future"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi_complete8"]].notna().all(axis=1)
'''
assert old in s
s=s.replace(old,new)
old2='''        pairs["pair_complete"] = pairs[["baseline_all4_ic","baseline_fi","future_fi"]].notna().all(axis=1) & pairs["future_active"]
        pairs["pair_complete8_future"] = pairs[["baseline_all4_ic","baseline_fi","future_fi_complete8"]].notna().all(axis=1) & pairs["future_active"]
'''
new2='''        pairs["pair_complete"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi"]].notna().all(axis=1) & pairs["future_active"]
        pairs["pair_complete8_future"] = pairs["baseline_all4_ic"].eq(True) & pairs[["baseline_fi","future_fi_complete8"]].notna().all(axis=1) & pairs["future_active"]
'''
assert old2 in s
p.write_text(s.replace(old2,new2),encoding='utf-8')
print('patched bool requirement')

