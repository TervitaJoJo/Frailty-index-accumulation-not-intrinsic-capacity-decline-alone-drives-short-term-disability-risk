from pathlib import Path
import pyreadstat, re, os, json
files = [
Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Raw_data\UKDA-5050-stata\stata\stata13_se\wave6\wave_6_elsa_data_v2.dta'),
Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Raw_data\UKDA-5050-stata\stata\stata13_se\wave6\wave_6_elsa_nurse_data_v2.dta'),
Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Raw_data\UKDA-5050-stata\stata\stata13_se\wave7\wave_7_elsa_data.dta'),
Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Raw_data\UKDA-5050-stata\stata\stata13_se\wave8\wave_8_elsa_data_eul_v2.dta'),
Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Raw_data\UKDA-5050-stata\stata\stata13_se\wave9\wave_9_elsa_data_eul_v1.dta'),
Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Raw_data\elsa_nurse_w8w9_data_eul.dta'),
]
patterns = r'ida|wave|shlt|mbmi|hibpe|diabe|hearte|stroke|cancre|arthre|imrc|dlrc|orient|walk|grip|mmg|cesd|depres|effort|sleep|happy|flone|sced|psced|iadl|adl|inw|mort|death'
for p in files:
    _,m=pyreadstat.read_dta(str(p),metadataonly=True)
    cols=[c for c in m.column_names if re.search(patterns,c,re.I)]
    print('\n---',p.name, len(m.column_names),'candidate',len(cols),'---')
    print(', '.join(cols[:400]))

