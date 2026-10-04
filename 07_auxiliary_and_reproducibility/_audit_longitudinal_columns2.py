from pathlib import Path
import pyreadstat, re, os
paths = {
 'SHARE': next(Path(r'PATH_TO_STATISTICAL_MODELING\SHARE').rglob('Working_data/share.dta')),
 'CHARLS': Path(r'PATH_TO_STATISTICAL_MODELING\CHARLS\Harmonized_CHARLS\H_CHARLS_D_Data.dta'),
 'HRS': Path(r'PATH_TO_STATISTICAL_MODELING\HRS\RAND HRS Data\Longitudinal and Cross-Wave Data Products\randhrs1992_2022v1.dta'),
 'KLoSA': next(Path(r'PATH_TO_STATISTICAL_MODELING\KLoSA').rglob('Working_data/klosa.dta')),
 'MHAS': next(Path(r'PATH_TO_STATISTICAL_MODELING\MHAS').rglob('Working_data/mhas.dta')),
}
for ds,p in paths.items():
 _,m=pyreadstat.read_dta(str(p),metadataonly=True); cols=m.column_names
 print('\n---',ds,len(cols),p.name,'---')
 pats=['mergeid','wave','id','shlt','mbmi','bmi','hibpe','diabe','hearte','stroke','cancre','arthre','imrc','dlrc','orient','tr20','cog','walk','grip','grpl','grpr','cesd','depres','effort','sleep','happy','euro','frail','adl','iadl','inw','proxy']
 for pat in pats:
  c=[x for x in cols if re.search(pat,x,re.I)]
  if c: print(pat,len(c),c[:120])

