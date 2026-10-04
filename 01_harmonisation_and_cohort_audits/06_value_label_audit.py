"""Export a compact value-label audit for high-impact candidate variables."""
from pathlib import Path
import pandas as pd
import pyreadstat

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
files = {
    "ELSA": ROOT / "ELSA/Working_data/elsa.dta",
    "SHARE": next(ROOT.glob("SHARE/**/Working_data/share.dta")),
    "KLoSA": next(ROOT.glob("KLoSA/**/Working_data/klosa.dta")),
    "MHAS": next(ROOT.glob("MHAS/**/Working_data/mhas.dta")),
    "CHARLS": ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta",
    "HRS": ROOT / "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta",
    "LASI": next(ROOT.glob("LASI/**/Working_data/lasi.dta")),
}
vars_by_dataset = {
    "ELSA": ["sight", "hearing", "walkra", "wspeed", "gripsum", "gripcomp", "cesd", "frailty", "inw", "iwstat", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge", "shlt", "mbmi", "fall1y", "painlv"],
    "SHARE": ["nsight", "hearing", "walkra", "wspeed", "lgrip", "rgrip", "eurod", "frailtyb", "iwstat", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge", "kidneye", "shlt", "bmi", "fall_s", "painlv", "hosp1y"],
    "KLoSA": ["sighta", "hearinga", "lgrip", "rgrip", "cesd10b", "iwstat", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge", "shlt", "bmi", "fall", "painlv_1"],
    "MHAS": ["sight", "hearing", "walkra", "wspeed", "lgrip", "rgrip", "cesd_m", "proxy", "iwstat", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre", "lunge", "shlt", "bmi", "fall", "painlv", "hosp1y"],
    "CHARLS": ["inw1", "r1walk100a", "r1wspeed", "r1tr20", "r1lgrip", "r1rgrip", "r1cesd10", "r1iwstat", "r1hibpe", "r1diabe", "r1hearte", "r1stroke", "r1cancre", "r1arthre", "r1lunge", "r1shlt", "r1mbmi", "r1hosp1y"],
    "HRS": ["inw8", "r8walkra", "r8timwlk", "r8cogtot", "r8grpl", "r8grpr", "r8cesd", "r8iwstat", "r8hibpe", "r8diabe", "r8hearte", "r8stroke", "r8cancre", "r8arthre", "r8lunge", "r8shlt", "r8bmi", "r8hosp"],
    "LASI": ["r1walk100a", "r1wspeed", "r1cog_total", "r1lgrip", "r1rgrip", "r1cesd10_l", "r1iwy", "r1hibpe", "r1diabe", "r1hearte", "r1stroke", "r1cancre", "r1arthre", "r1lunge", "r1shlt", "r1mbmi", "r1fall", "r1hosp1y"],
}
rows = []
for dataset, path in files.items():
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    available = set(meta.column_names)
    selected = [v for v in vars_by_dataset[dataset] if v in available]
    data, _ = pyreadstat.read_dta(str(path), usecols=selected, apply_value_formats=False)
    for var in vars_by_dataset[dataset]:
        label_name = meta.variable_to_label.get(var, "") if var in available else ""
        value_map = meta.value_labels.get(label_name, {}) if label_name else {}
        x = data[var] if var in data else pd.Series(dtype="float64")
        observed = int(x.notna().sum())
        vals = []
        for value, label in list(value_map.items())[:100]:
            vals.append(f"{value}={label}")
        rows.append({
            "dataset": dataset,
            "variable": var,
            "available": var in available,
            "variable_label": meta.column_labels[meta.column_names.index(var)] if var in available else "",
            "value_label_name": label_name,
            "value_label_mapping_first100": " | ".join(vals),
            "n_rows_observed_raw": observed,
            "audit_note": "Direction must be verified from labels; raw negative special codes are not treated as measurements in the main audit." if var in available else "Variable not present in current working file.",
        })
out = pd.DataFrame(rows)
out.to_csv(TAB / "candidate_value_label_audit.csv", index=False, encoding="utf-8-sig")
print("wrote", TAB / "candidate_value_label_audit.csv", "rows", len(out))
