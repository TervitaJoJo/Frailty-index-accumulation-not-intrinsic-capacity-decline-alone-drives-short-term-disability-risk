from pathlib import Path
import pyreadstat, re
p=Path(r'PATH_TO_STATISTICAL_MODELING\ELSA\Working_data\elsa.dta')
_,m=pyreadstat.read_dta(str(p),metadataonly=True)
cols=m.column_names
print('n',len(cols))
for pat in ['ida','wave','shlt','mbmi','hibpe','diabe','hearte','stroke','cancre','arthre','imrc','dlrc','orient','walk','grip','cesd','depres','effort','sleep','happy','sced','cog','frail','adl','iadl','inw']:
  c=[x for x in cols if re.search(pat,x,re.I)]
  print(pat, len(c), c[:200])

