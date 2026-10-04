import pyreadstat
from pathlib import Path
import pandas as pd

specs = {
    "ELSA": (Path(r"PATH_TO_STATISTICAL_MODELING\ELSA\Working_data\elsa.dta"), ["wave", "walkra", "walk100a", "dressa", "batha", "eata", "beda", "adltot6", "iadltot2_e"]),
    "CHARLS": (Path(r"PATH_TO_STATISTICAL_MODELING\CHARLS\Harmonized_CHARLS\H_CHARLS_D_Data.dta"), ["inw3", "inw4", "r3dressa", "r4dressa", "r3batha", "r4batha", "r3eata", "r4eata", "r3beda", "r4beda", "r3toilta", "r4toilta", "r3moneya", "r4moneya", "r3shopa", "r4shopa", "r3mealsa", "r4mealsa", "r3adlwa", "r4adlwa", "r3iadla", "r4iadla"]),
    "HRS": (Path(r"PATH_TO_STATISTICAL_MODELING\HRS\RAND HRS Data\Longitudinal and Cross-Wave Data Products\randhrs1992_2022v1.dta"), ["inw10", "inw11", "r10proxy", "r11proxy", "r10dressa", "r11dressa", "r10batha", "r11batha", "r10eata", "r11eata", "r10beda", "r11beda", "r10toilta", "r11toilta", "r10moneya", "r11moneya", "r10shopa", "r11shopa", "r10mealsa", "r11mealsa", "r10adl5a", "r11adl5a", "r10iadl5a", "r11iadl5a"]),
}
for cohort, (path, cols) in specs.items():
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    use = [c for c in cols if c in meta.column_names]
    d, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    print(f"\n{cohort}")
    for c in use:
        x = pd.to_numeric(d[c], errors="coerce")
        print(c, "missing=", round(float(x.isna().mean()), 4), "values=", x.value_counts(dropna=False).head(12).to_dict())
        idx = meta.column_names.index(c)
        print("  label=", meta.column_labels[idx])
