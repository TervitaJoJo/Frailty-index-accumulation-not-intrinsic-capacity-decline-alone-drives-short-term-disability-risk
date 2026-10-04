"""Audit the construction of existing ELSA/SHARE frailty variables."""
from pathlib import Path
import pandas as pd

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

records = [
    {
        "dataset": "ELSA",
        "indicator": "frailty",
        "source_file": "ELSA/Dofiles/数据合并.do",
        "source_lines": "1282-1287",
        "formula": "egen r{i}frailty=rowtotal(30 items),mi; replace = /30*100",
        "n_components": 30,
        "components": "hibpe;diabe;hearte;stroke;cancre;arthre;lunge;psyche;memory1;sight1;hearing1;shlt1;dressa;batha;eata;beda;toilta;moneya;medsa;shopa;mealsa;walk100a;chaira;climsa;lifta;dimea;stoopa;armsa;depressive;cogition",
        "chronic_conditions": "hibpe,diabe,hearte,stroke,cancre,arthre,lunge,psyche",
        "sensory_or_self_health": "sight1,hearing1,shlt1",
        "adl_iadl": "dressa,batha,eata,beda,toilta,moneya,medsa,shopa,mealsa",
        "mobility_function": "walk100a,chaira,climsa,lifta,dimea,stoopa,armsa",
        "psychological": "depressive",
        "cognition": "memory1,cogition",
        "death": "none explicit",
        "waves_in_formula": "7-9",
        "audit_conclusion": "Not outcome-disjoint: includes cognition and ADL/IADL; also overlaps mobility and psychological IC domains.",
    },
    {
        "dataset": "SHARE",
        "indicator": "frailtyb",
        "source_file": "SHARE/SHARE_欧洲/Dofiles/no.9_数据合并.do",
        "source_lines": "897-902",
        "formula": "egen r{i}frailtyb=rowtotal(30 items),mi; replace = /30*100",
        "n_components": 30,
        "components": "hibpe;diabe;hearte;stroke;cancre;arthre;lunge;psyche;alzdeme;sight2;hear2;shlt2;dressa;batha;eata;beda;toilta;moneya;medsa;shopa;mealsa;walk100a;chaira;climsa;lifta;dimea;stoopa;armsa;depression;cogition",
        "chronic_conditions": "hibpe,diabe,hearte,stroke,cancre,arthre,lunge,psyche,alzdeme",
        "sensory_or_self_health": "sight2,hear2,shlt2",
        "adl_iadl": "dressa,batha,eata,beda,toilta,moneya,medsa,shopa,mealsa",
        "mobility_function": "walk100a,chaira,climsa,lifta,dimea,stoopa,armsa",
        "psychological": "depression",
        "cognition": "alzdeme,cogition",
        "death": "none explicit",
        "waves_in_formula": "4-8",
        "audit_conclusion": "Not outcome-disjoint: includes dementia/cognition and ADL/IADL; also overlaps mobility and psychological IC domains.",
    },
]

df = pd.DataFrame(records)
df.to_csv(TAB / "existing_frailty_component_audit.csv", index=False, encoding="utf-8-sig")

lines = [
    "# Existing frailty component audit",
    "",
    "This audit is based on the two source do-files that create the composite variables.",
    "",
    "| Dataset | Indicator | Components | Cognition overlap | ADL/IADL overlap | Sensory/self-health | Psychological | Explicit death item |",
    "|---|---|---:|---|---|---|---|---|",
]
for r in records:
    lines.append(f"| {r['dataset']} | {r['indicator']} | {r['n_components']} | {r['cognition']} | {r['adl_iadl']} | {r['sensory_or_self_health']} | {r['psychological']} | {r['death']} |")
lines += [
    "",
    "Interpretation: neither existing composite can serve as an independent frailty outcome in a construct-comparison model that includes cognition, ADL/IADL, mobility, sensory or psychological domains. A new outcome-disjoint, domain-transparent FI should be reconstructed from prespecified items; the existing variables can remain descriptive anchors or sensitivity outcomes.",
]
(OUT / "existing_frailty_component_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", TAB / "existing_frailty_component_audit.csv")
