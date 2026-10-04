"""Phase-0 metadata and candidate-variable audit for the IC-frailty project.

This script is intentionally read-only with respect to source datasets. It uses
Stata metadata and variable labels to identify candidate IC/frailty indicators;
it does not silently choose final operational definitions or alter raw data.
"""
from __future__ import annotations

import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyreadstat


PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT_ROOT = Path(r"PATH_TO_IC_FRAILTY")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


DATASET_PATTERNS = {
    "CHARLS": ["CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"],
    "HRS": ["HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"],
    "ELSA": ["ELSA/Working_data/elsa.dta"],
    "SHARE": ["SHARE/**/Working_data/share.dta"],
    "KLoSA": ["KLoSA/**/Working_data/klosa.dta"],
    "MHAS": ["MHAS/**/Working_data/mhas.dta"],
    "HAALSI": ["HAALSI/Wave3/HAALSI W1-3 Longitudinal Dataset.dta"],
    "LASI": ["LASI/**/Working_data/lasi.dta"],
}


KEYWORDS = {
    "identity_time": ["wave", "year", "age", "id", "person", "respondent", "proxy", "death", "mortality"],
    "cognition": ["cog", "memory", "recall", "orientation", "orient", "animal", "serial", "math", "calculation", "cognitive"],
    "locomotion": ["walk", "gait", "speed", "mobility", "locomot", "stand", "balance", "chair", "movement"],
    "vitality": ["vital", "grip", "strength", "weight", "bmi", "nutrition", "nutri", "fatigue", "exhaust", "energy"],
    "sensory": ["vision", "visual", "hearing", "hear", "sensory", "sight", "audio"],
    "psychological": ["psych", "depress", "cesd", "eurod", "mood", "wellbeing", "well-being", "anx", "lonely", "sad"],
    "frailty": ["frail", "frailty", "deficit", "phenotype", "frail index", "frailty index"],
    "function": ["adl", "iadl", "daily living", "disab", "depend", "functional"],
}


def resolve(pattern: str) -> Path | None:
    direct = PROJECT_ROOT / pattern
    if direct.exists():
        return direct
    matches = list(PROJECT_ROOT.glob(pattern))
    return matches[0] if matches else None


def categories(name: str, label: str) -> list[str]:
    text = f"{name} {label}".lower()
    return [cat for cat, kws in KEYWORDS.items() if any(k in text for k in kws)]


def main() -> None:
    manifest: list[dict] = []
    candidates: list[dict] = []

    for dataset, patterns in DATASET_PATTERNS.items():
        source = next((resolve(p) for p in patterns if resolve(p)), None)
        if source is None:
            manifest.append({"dataset": dataset, "status": "missing", "patterns": patterns})
            continue
        try:
            _, meta = pyreadstat.read_dta(str(source), metadataonly=True)
            labels = meta.column_labels or [""] * len(meta.column_names)
            manifest.append(
                {
                    "dataset": dataset,
                    "status": "read_metadata",
                    "source_relative": str(source.relative_to(PROJECT_ROOT)),
                    "file_size_bytes": source.stat().st_size,
                    "n_rows": int(meta.number_rows),
                    "n_columns": int(len(meta.column_names)),
                    "file_encoding": getattr(meta, "file_encoding", None),
                }
            )
            for name, label in zip(meta.column_names, labels):
                cats = categories(str(name), str(label))
                if cats:
                    candidates.append(
                        {
                            "dataset": dataset,
                            "source_relative": str(source.relative_to(PROJECT_ROOT)),
                            "variable": str(name),
                            "label": str(label),
                            "candidate_domains": ";".join(cats),
                        }
                    )
        except Exception as exc:
            manifest.append({"dataset": dataset, "status": "error", "source": str(source), "error_type": type(exc).__name__})

    run_info = {
        "analysis": "IC-frailty exploratory metadata audit",
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_root": str(PROJECT_ROOT),
        "output_root": str(OUT_ROOT),
        "read_only_source": True,
        "note": "Candidate domains are label/name keyword hits, not validated operational definitions.",
    }
    (OUT_ROOT / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(candidates).to_csv(OUT_ROOT / "candidate_variables.csv", index=False, encoding="utf-8-sig")

    # Compact per-dataset domain inventory for quick review.
    rows = []
    cdf = pd.DataFrame(candidates)
    for dataset in DATASET_PATTERNS:
        sub = cdf[cdf.dataset.eq(dataset)] if not cdf.empty else pd.DataFrame()
        for domain in KEYWORDS:
            vars_ = sub.loc[sub.candidate_domains.fillna("").str.contains(domain, regex=False), "variable"].tolist() if not sub.empty else []
            rows.append({"dataset": dataset, "domain": domain, "n_candidate_variables": len(vars_), "examples": " | ".join(vars_[:20])})
    pd.DataFrame(rows).to_csv(OUT_ROOT / "candidate_domain_inventory.csv", index=False, encoding="utf-8-sig")

    print(json.dumps({"manifest": manifest, "candidate_rows": len(candidates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
