import pyreadstat
import re
from pathlib import Path

files = {
    "ELSA": Path(r"PATH_TO_STATISTICAL_MODELING\ELSA\Working_data\elsa.dta"),
    "CHARLS": Path(r"PATH_TO_STATISTICAL_MODELING\CHARLS\Harmonized_CHARLS\H_CHARLS_D_Data.dta"),
    "HRS": Path(r"PATH_TO_STATISTICAL_MODELING\HRS\RAND HRS Data\Longitudinal and Cross-Wave Data Products\randhrs1992_2022v1.dta"),
}
patterns = {
    "ELSA": r"(?i)(adl|iadl|dressa|batha|eata|beda|walkra|walk100a|mob)",
    "CHARLS": r"(?i)r[34].*(adl|iadl|walk|dress|bath|eat|bed|toil|shop|money|meal)",
    "HRS": r"(?i)r(10|11).*(adl|iadl|walk|dress|bath|eat|bed|toil|shop|money|meal)",
}
for cohort, path in files.items():
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    hits = [name for name in meta.column_names if re.search(patterns[cohort], name)]
    print(f"\n{cohort} ({len(hits)} matches)")
    print(" ".join(hits[:300]))
