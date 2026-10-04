"""Stage 21: survey-weight inventory and IC/FI missingness mechanism audit.

The audit is aggregate-only. Source Stata files are read in memory and no
individual-level derived file, identifier, or row-level diagnostic is written.
The output is intended to decide whether later survey-weighted sensitivity
models and multiple-imputation analyses are scientifically defensible.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pyreadstat


ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def read_dta(path: Path, columns: list[str] | None = None) -> tuple[pd.DataFrame, object]:
    """Read only the requested columns and preserve Stata missing values as NaN."""
    if columns is None:
        return pyreadstat.read_dta(str(path), apply_value_formats=False)
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    present = [c for c in dict.fromkeys(columns) if c in meta.column_names]
    return pyreadstat.read_dta(
        str(path), usecols=present, apply_value_formats=False
    )


def metadata_rows(path: Path, names: list[str]) -> dict[str, str]:
    """Return variable labels without printing raw metadata or values."""
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    labels = meta.column_names_to_labels or {}
    return {n: str(labels.get(n, "")) for n in names if n in meta.column_names}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def num(series: pd.Series, low: float = -np.inf, high: float = np.inf) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.between(low, high))


def valid_binary(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.isin([0, 1])


def valid_num(series: pd.Series, low: float, high: float) -> pd.Series:
    return num(series, low, high).notna()


def normalize_key(series: pd.Series, width: int | None = None) -> pd.Series:
    x = series.astype("string").str.replace(r"\.0$", "", regex=True)
    if width:
        x = x.str.zfill(width)
    return x


def source_weight_row(
    rows: list[dict],
    dataset: str,
    wave: int,
    variable: str,
    role: str,
    label: str,
    source: Path,
    active: pd.Series | int,
    values: pd.Series | None,
    notes: str,
    design_component: str = "weight",
) -> None:
    n_active = int(active if isinstance(active, int) else active.sum())
    if values is None:
        n_nonmissing = n_positive = n_unique = 0
        present = False
    else:
        raw = pd.Series(values)
        x = pd.to_numeric(raw, errors="coerce")
        n_nonmissing = int(raw.notna().sum())
        n_positive = int(x.gt(0).sum())
        n_unique = int(raw.nunique(dropna=True))
        present = True
    rows.append(
        {
            "dataset": dataset,
            "wave": wave,
            "design_component": design_component,
            "variable": variable,
            "role": role,
            "label": label,
            "source_file": relpath(source),
            "variable_present": present,
            "n_active": n_active,
            "n_nonmissing": n_nonmissing,
            "n_positive": n_positive,
            "pct_positive": 100 * n_positive / n_active if n_active else np.nan,
            "n_unique": n_unique,
            "notes": notes,
        }
    )


def source_design_row(
    rows: list[dict],
    dataset: str,
    wave: int,
    variable: str,
    role: str,
    label: str,
    source: Path,
    active: pd.Series | int,
    values: pd.Series | None,
    notes: str,
) -> None:
    source_weight_row(
        rows,
        dataset,
        wave,
        variable,
        role,
        label,
        source,
        active,
        values,
        notes,
        design_component="design_variable",
    )


def add_missingness(
    rows: list[dict],
    dataset: str,
    wave: int,
    sample: str,
    construct: str,
    indicator: str,
    active: pd.Series,
    valid: pd.Series,
    structural: pd.Series | None,
    rule: str,
    mechanism_note: str,
) -> None:
    active = active.fillna(False).astype(bool)
    valid = valid.fillna(False).astype(bool)
    valid_active = active & valid
    missing = active & ~valid_active
    n_active = int(active.sum())
    n_valid = int(valid_active.sum())
    n_missing = int(missing.sum())
    if structural is None:
        n_structural = 0
        n_item = 0
        n_unresolved = n_missing
        classification = "unresolved_missingness"
    else:
        structural = structural.fillna(False).astype(bool) & active
        structural_missing = missing & structural
        n_structural = int(structural_missing.sum())
        n_item = n_missing - n_structural
        n_unresolved = 0
        classification = "module_flag_based"
    rows.append(
        {
            "dataset": dataset,
            "wave": wave,
            "sample": sample,
            "construct": construct,
            "indicator": indicator,
            "n_active": n_active,
            "n_valid": n_valid,
            "n_missing": n_missing,
            "pct_valid": 100 * n_valid / n_active if n_active else np.nan,
            "n_structural_module": n_structural,
            "n_item_or_refusal": n_item,
            "n_unresolved_missing": n_unresolved,
            "missingness_classification": classification,
            "validity_rule": rule,
            "mechanism_note": mechanism_note,
        }
    )


def add_domain_complete(
    rows: list[dict],
    dataset: str,
    wave: int,
    sample: str,
    construct: str,
    indicators: dict[str, pd.Series],
    active: pd.Series,
    note: str,
) -> None:
    complete = pd.concat(indicators, axis=1).notna().all(axis=1)
    add_missingness(
        rows,
        dataset,
        wave,
        sample,
        construct,
        "all_core_indicators",
        active,
        complete,
        None,
        "; ".join(indicators),
        note,
    )


def fi_parts(d: pd.DataFrame, prefix: str = "") -> dict[str, pd.Series]:
    names = {
        "hypertension": f"{prefix}hibpe",
        "diabetes": f"{prefix}diabe",
        "heart_disease": f"{prefix}hearte",
        "stroke": f"{prefix}stroke",
        "cancer": f"{prefix}cancre",
        "arthritis": f"{prefix}arthre",
    }
    parts = {k: num(d[v], 0, 1).where(pd.to_numeric(d[v], errors="coerce").isin([0, 1])) for k, v in names.items()}
    parts["self_rated_health"] = num(d[f"{prefix}shlt"], 1, 5)
    bmi = num(d[f"{prefix}bmi"], 0.01, 200)
    parts["bmi"] = bmi.notna().astype(float).where(bmi.notna())
    return parts


def add_fi_rows(
    rows: list[dict],
    dataset: str,
    wave: int,
    sample: str,
    active: pd.Series,
    parts: dict[str, pd.Series],
    structural_bmi: pd.Series | None,
    bmi_rule: str,
    note: str,
) -> pd.Series:
    for name, value in parts.items():
        structural = structural_bmi if name == "bmi" else None
        rule = "binary 0/1" if name not in {"self_rated_health", "bmi"} else ("1-5" if name == "self_rated_health" else bmi_rule)
        add_missingness(
            rows,
            dataset,
            wave,
            sample,
            "frailty_component",
            name,
            active,
            value.notna(),
            structural,
            rule,
            note,
        )
    observed = pd.concat(parts, axis=1).notna().sum(axis=1)
    fi = pd.concat(parts, axis=1).sum(axis=1, min_count=1).div(observed.where(observed.gt(0)))
    fi = fi.where(observed.ge(6))
    add_missingness(
        rows,
        dataset,
        wave,
        sample,
        "frailty_index",
        "fi_primary_min6of8",
        active,
        fi.notna(),
        None,
        "6 of 8 observed; denominator equals observed component count; BMI<18.5 or >=30",
        "FI availability combines ordinary component missingness; no released module flag fully separates all mechanisms.",
    )
    return fi


def audit_elsa(weight_rows: list[dict], miss_rows: list[dict]) -> dict:
    work_path = ROOT / "ELSA" / "Working_data" / "elsa.dta"
    raw_path = ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_data_v2.dta"
    nurse_path = ROOT / "ELSA" / "Raw_data" / "wave6" / "wave_6_elsa_nurse_data_v2.dta"
    cols = [
        "idauniqc", "wave", "imrc", "dlrc", "orient", "walkra", "walk100a", "lgrip", "rgrip",
        "cesd", "shlt", "mbmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre",
    ]
    d, _ = read_dta(work_path, cols)
    d = d[d["wave"].eq(6)].copy()
    d["idauniqc"] = normalize_key(d["idauniqc"])
    nurse_cols = ["idauniq", "w6nurwt", "mmgswil", "mmgssta", "mmgsd1", "mmgsn1", "mmgsd2", "mmgsn2"]
    nurse, _ = read_dta(nurse_path, nurse_cols)
    nurse["idauniq"] = normalize_key(nurse["idauniq"])
    nurse = nurse.drop_duplicates("idauniq")
    d = d.merge(nurse, left_on="idauniqc", right_on="idauniq", how="left", validate="one_to_one")
    active = pd.Series(True, index=d.index)
    nurse_available = num(d["w6nurwt"], 0.01, np.inf).notna()
    labels_main = metadata_rows(raw_path, ["w6xwgt", "w6lwgt", "idahhw6"])
    raw, _ = read_dta(raw_path, ["idauniq", "w6xwgt", "w6lwgt", "idahhw6"])
    raw["idauniq"] = normalize_key(raw["idauniq"])
    raw = raw.drop_duplicates("idauniq")
    raw_active = raw["idauniq"].notna()
    for v, role, note in [
        ("w6xwgt", "wave-6 person cross-sectional weight", "Use for anchor-wave population-sensitive sensitivity; positive values only."),
        ("w6lwgt", "wave-6 longitudinal weight", "Available only for the released longitudinal base; do not treat as a universal cross-sectional weight."),
    ]:
        source_weight_row(weight_rows, "ELSA", 6, v, role, labels_main.get(v, ""), raw_path, raw_active, raw[v], note)
    labels_nurse = metadata_rows(nurse_path, ["w6nurwt"])
    source_weight_row(
        weight_rows, "ELSA", 6, "w6nurwt", "nurse/physical-measures module weight", labels_nurse.get("w6nurwt", ""),
        nurse_path, active, d["w6nurwt"], "Use to identify the released nurse module; it is not interchangeable with w6xwgt.",
    )
    source_design_row(weight_rows, "ELSA", 6, "idahhw6", "household cluster proxy", labels_main.get("idahhw6", ""), raw_path, raw_active, raw["idahhw6"], "Household serial is available; official ELSA PSU/stratum variables were not found in the audited public wave-6 files.")
    source_design_row(weight_rows, "ELSA", 6, "official_psu_stratum", "official PSU/stratum", "", work_path, int(active.sum()), None, "Not released in audited working/raw wave-6 files; household serial is only a clustering proxy.")

    indicators = {
        "immediate_recall": valid_num(d["imrc"], 0, 10),
        "delayed_recall": valid_num(d["dlrc"], 0, 10),
        "orientation": valid_num(d["orient"], 0, 4),
        "locomotion_room": valid_binary(d["walkra"]),
        "locomotion_100m": valid_binary(d["walk100a"]),
        "grip_left": valid_num(d["lgrip"], 0, 100),
        "grip_right": valid_num(d["rgrip"], 0, 100),
        "psychological_cesd": valid_num(d["cesd"], 0, 8),
    }
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "cognition", "immediate_recall", active, indicators["immediate_recall"], None, "0-10", "Core cognitive test; no separate item/nonresponse flag in working file.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "cognition", "delayed_recall", active, indicators["delayed_recall"], None, "0-10", "Core cognitive test; no separate item/nonresponse flag in working file.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "cognition", "orientation", active, indicators["orientation"], None, "0-4", "Core cognitive test; no separate item/nonresponse flag in working file.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "locomotion", "room_plus_100m_complete", active, indicators["locomotion_room"] & indicators["locomotion_100m"], None, "both binary items valid", "Self-reported mobility; no module selection flag.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "grip_vitality", "any_handgrip", active, indicators["grip_left"] | indicators["grip_right"], ~nurse_available, "at least one handgrip 0-100", "Nurse weight missing is used as a structural module-absence flag; positive nurse weight with no grip is residual item/refusal/ability missingness.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "grip_vitality", "left_handgrip", active, indicators["grip_left"], ~nurse_available, "0-100 kg", "Same nurse-module classification as above.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "grip_vitality", "right_handgrip", active, indicators["grip_right"], ~nurse_available, "0-100 kg", "Same nurse-module classification as above.")
    add_missingness(miss_rows, "ELSA", 6, "wave-6 active", "psychological", "cesd_total", active, indicators["psychological_cesd"], None, "0-8", "Score-level missingness; item-level CES-D responses are not retained in the working file.")
    parts = fi_parts(d.rename(columns={"mbmi": "bmi"}))
    add_fi_rows(miss_rows, "ELSA", 6, "wave-6 active", active, parts, ~nurse_available, "BMI<18.5 or >=30", "Nurse module weight is used only for the measured BMI structural-absence screen.")
    return {"n_active": int(active.sum()), "n_proxy_excluded": 0}


def audit_charls(weight_rows: list[dict], miss_rows: list[dict]) -> dict:
    path = ROOT / "CHARLS" / "Harmonized_CHARLS" / "H_CHARLS_D_Data.dta"
    cols = [
        "inw3", "r3imrc", "r3dlrc", "r3orient", "r3ser7", "r3walk100a", "r3walk1kma",
        "r3lgrip1", "r3lgrip2", "r3rgrip1", "r3rgrip2", "r3gripcomp", "r3wtcomp", "r3htcomp",
        "r3depresl", "r3effortl", "r3sleeprl", "r3whappyl", "r3shlt", "r3mbmi", "r3hibpe",
        "r3diabe", "r3hearte", "r3stroke", "r3cancre", "r3arthre", "r3wtresp", "r3wtrespa",
        "r3wtrespb", "r3wtrespl", "communityID",
    ]
    d, _ = read_dta(path, cols)
    active = d["inw3"].eq(1)
    d = d[active].copy()
    active = pd.Series(True, index=d.index)
    labels = metadata_rows(path, [c for c in cols if c in d.columns])
    for v, role, note in [
        ("r3wtresp", "wave-3 individual weight without NR adjustment", "Retained as a source-audit candidate only."),
        ("r3wtrespa", "wave-3 individual weight with household NR adjustment", "Retained as a source-audit candidate only."),
        ("r3wtrespb", "wave-3 individual weight with household/individual NR adjustment", "Preferred anchor-wave respondent weight for sensitivity analyses."),
        ("r3wtrespl", "wave-3 individual longitudinal weight", "Not available in the released wave-3 harmonized file; only cross-sectional weights are documented."),
    ]:
        source_weight_row(weight_rows, "CHARLS", 3, v, role, labels.get(v, ""), path, active, d[v] if v in d else None, note)
    source_design_row(weight_rows, "CHARLS", 3, "communityID", "community PSU/cluster", labels.get("communityID", ""), path, active, d["communityID"], "Community ID is available in the harmonized file and is the usable PSU/cluster key for sensitivity variance estimation.")
    psu_path = ROOT / "CHARLS" / "2013" / "PSU.dta"
    psu, _ = read_dta(psu_path, ["communityID", "urban_nbs", "areatype"])
    psu["communityID"] = psu["communityID"].astype("string")
    d["communityID"] = d["communityID"].astype("string")
    joined = d[["communityID"]].merge(psu, on="communityID", how="left", validate="many_to_one")
    source_design_row(weight_rows, "CHARLS", 3, "urban_nbs", "urban/rural design auxiliary", "Urban or Rural according to NBS", psu_path, active, joined["urban_nbs"], "Available only in the separate PSU file; the harmonized file does not retain an explicit named stratum variable.")
    source_design_row(weight_rows, "CHARLS", 3, "explicit_stratum", "explicit stratum", "", path, int(active.sum()), None, "No explicit stratum variable found in the audited harmonized wave-3 file or PSU auxiliary file.")

    ind = {
        "immediate_recall": valid_num(d["r3imrc"], 0, 10),
        "delayed_recall": valid_num(d["r3dlrc"], 0, 10),
        "orientation": valid_num(d["r3orient"], 0, 4),
        "executive": valid_num(d["r3ser7"], 0, 5),
        "locomotion_100m": valid_binary(d["r3walk100a"]),
        "locomotion_1km": valid_binary(d["r3walk1kma"]),
        "grip_left": valid_num(d["r3lgrip1"], 0, 100) | valid_num(d["r3lgrip2"], 0, 100),
        "grip_right": valid_num(d["r3rgrip1"], 0, 100) | valid_num(d["r3rgrip2"], 0, 100),
        "psych_depressed": valid_num(d["r3depresl"], 1, 4),
        "psych_effort": valid_num(d["r3effortl"], 1, 4),
        "psych_sleep": valid_num(d["r3sleeprl"], 1, 4),
        "psych_happy": valid_num(d["r3whappyl"], 1, 4),
    }
    grip_module = d["r3gripcomp"].notna()
    bmi_module = d["r3wtcomp"].notna() | d["r3htcomp"].notna()
    for name, key, rule in [
        ("immediate_recall", "immediate_recall", "0-10"), ("delayed_recall", "delayed_recall", "0-10"),
        ("orientation", "orientation", "0-4"), ("executive", "executive", "0-5"),
        ("locomotion_100m", "locomotion_100m", "binary 0/1"), ("locomotion_1km", "locomotion_1km", "binary 0/1"),
        ("psych_depressed", "psych_depressed", "1-4"), ("psych_effort", "psych_effort", "1-4"),
        ("psych_sleep", "psych_sleep", "1-4"), ("psych_happy", "psych_happy", "1-4"),
    ]:
        construct = "cognition" if key in {"immediate_recall", "delayed_recall", "orientation", "executive"} else ("locomotion" if key.startswith("locomotion") else "psychological")
        add_missingness(miss_rows, "CHARLS", 3, "wave-3 active", construct, name, active, ind[key], None, rule, "No complete item/module flag is available for this self-report or cognitive indicator.")
    add_missingness(miss_rows, "CHARLS", 3, "wave-3 active", "grip_vitality", "left_handgrip_any_measure", active, ind["grip_left"], ~grip_module, "at least one valid left-hand attempt", "r3gripcomp presence identifies the physical-measures module; positive module flag with missing measurement is residual refusal/ability/item missingness.")
    add_missingness(miss_rows, "CHARLS", 3, "wave-3 active", "grip_vitality", "right_handgrip_any_measure", active, ind["grip_right"], ~grip_module, "at least one valid right-hand attempt", "Same physical-module classification as above.")
    parts = fi_parts(d.rename(columns={"r3hibpe": "hibpe", "r3diabe": "diabe", "r3hearte": "hearte", "r3stroke": "stroke", "r3cancre": "cancre", "r3arthre": "arthre", "r3shlt": "shlt", "r3mbmi": "bmi"}), prefix="")
    add_fi_rows(miss_rows, "CHARLS", 3, "wave-3 active", active, parts, ~bmi_module, "BMI<18.5 or >=30", "Physical-measures completion flags are used for measured BMI; disease and self-rated health missingness remains unresolved at item level.")
    return {"n_active": int(active.sum()), "n_proxy_excluded": 0}


def audit_hrs(weight_rows: list[dict], miss_rows: list[dict]) -> dict:
    path = ROOT / "HRS" / "RAND HRS Data" / "Longitudinal and Cross-Wave Data Products" / "randhrs1992_2022v1.dta"
    cols = [
        "hhid", "pn", "inw10", "r10proxy", "r10imrc", "r10dlrc", "r10ser7", "r10walkra", "r10walk1a",
        "r10grpl", "r10grpr", "r10depres", "r10effort", "r10sleepr", "r10whappy", "r10flone", "r10fsad",
        "r10going", "r10enlife", "r10shlt", "r10bmi", "r10hibpe", "r10diabe", "r10hearte", "r10stroke",
        "r10cancre", "r10arthre", "r10wtresp", "r10wtr_nh", "raestrat", "raehsamp",
    ]
    d, _ = read_dta(path, cols)
    active_all = d["inw10"].eq(1)
    proxy_excluded = active_all & d["r10proxy"].eq(1)
    active = active_all & d["r10proxy"].eq(0)
    d = d[active].copy()
    active = pd.Series(True, index=d.index)
    labels = metadata_rows(path, [c for c in cols if c in d.columns])
    for v, role, note in [
        ("r10wtresp", "wave-10 person-level analysis weight", "Preferred general anchor-wave respondent weight for HRS sensitivity."),
        ("r10wtr_nh", "wave-10 nursing-home resident weight", "Not used for the community-dwelling IC primary sample."),
    ]:
        source_weight_row(weight_rows, "HRS", 10, v, role, labels.get(v, ""), path, active, d[v], note)
    for v, role, note in [
        ("raestrat", "sampling-error stratum", "Use jointly with raehsamp/SECU."),
        ("raehsamp", "sampling-error computation unit (SECU)", "SECU is not an independent geography; pair with raestrat."),
    ]:
        source_design_row(weight_rows, "HRS", 10, v, role, labels.get(v, ""), path, active, d[v], note)
    tracker_path = ROOT / "HRS" / "public_survey_data" / "Cross-Wave Tracker File" / "trk2022TR_R.csv"
    tracker = pd.read_csv(tracker_path, usecols=["HHID", "PN", "MPMWGTR"], dtype={"HHID": "string", "PN": "string"})
    tracker["HHID"] = normalize_key(tracker["HHID"], 6)
    tracker["PN"] = normalize_key(tracker["PN"], 3)
    d["hhid"] = normalize_key(d["hhid"], 6)
    d["pn"] = normalize_key(d["pn"], 3)
    d = d.merge(tracker, left_on=["hhid", "pn"], right_on=["HHID", "PN"], how="left", validate="many_to_one")
    # The merge resets the row index; recreate the active mask before any
    # aligned Boolean operations used by the missingness summaries.
    active = pd.Series(True, index=d.index)
    mpm_positive = num(d["MPMWGTR"], 0.01, np.inf).notna()
    source_weight_row(weight_rows, "HRS", 10, "MPMWGTR", "2010 physical-measures respondent weight", "Respondent weight for the 2010 physical measures subsample", tracker_path, active, d["MPMWGTR"], "Use for the physical-measures module; do not multiply blindly by r10wtresp.")
    source_design_row(weight_rows, "HRS", 10, "raestrat_x_raehsamp", "joint STRATUM x SECU key", "", path, active, d["raestrat"].astype("string") + ":" + d["raehsamp"].astype("string"), "Joint design key is complete in the non-proxy anchor sample.")

    ind = {
        "immediate_recall": valid_num(d["r10imrc"], 0, 10),
        "delayed_recall": valid_num(d["r10dlrc"], 0, 10),
        "executive": valid_num(d["r10ser7"], 0, 5),
        "locomotion_room": valid_num(d["r10walkra"], 0, 2),
        "locomotion_100m": valid_num(d["r10walk1a"], 0, 2),
        "grip_left": valid_num(d["r10grpl"], 0, 100),
        "grip_right": valid_num(d["r10grpr"], 0, 100),
        "psych_depressed": valid_binary(d["r10depres"]),
        "psych_effort": valid_binary(d["r10effort"]),
        "psych_sleep": valid_binary(d["r10sleepr"]),
        "psych_happy": valid_binary(d["r10whappy"]),
    }
    for name, key, rule in [
        ("immediate_recall", "immediate_recall", "0-10"), ("delayed_recall", "delayed_recall", "0-10"), ("executive", "executive", "0-5"),
        ("locomotion_room", "locomotion_room", "0-2"), ("locomotion_100m", "locomotion_100m", "0-2"),
        ("psych_depressed", "psych_depressed", "binary 0/1"), ("psych_effort", "psych_effort", "binary 0/1"),
        ("psych_sleep", "psych_sleep", "binary 0/1"), ("psych_happy", "psych_happy", "binary 0/1"),
    ]:
        construct = "cognition" if key in {"immediate_recall", "delayed_recall", "executive"} else ("locomotion" if key.startswith("locomotion") else "psychological")
        add_missingness(miss_rows, "HRS", 10, "wave-10 active non-proxy", construct, name, active, ind[key], None, rule, "Released RAND file does not provide a complete item-level reason code for this indicator.")
    add_missingness(miss_rows, "HRS", 10, "wave-10 active non-proxy", "grip_vitality", "left_handgrip", active, ind["grip_left"], ~mpm_positive, "0-100 kg", "MPMWGTR positive identifies the physical-measures subsample; residual missingness within that module is not automatically imputed.")
    add_missingness(miss_rows, "HRS", 10, "wave-10 active non-proxy", "grip_vitality", "right_handgrip", active, ind["grip_right"], ~mpm_positive, "0-100 kg", "Same physical-module classification as above.")
    rename = {"r10hibpe": "hibpe", "r10diabe": "diabe", "r10hearte": "hearte", "r10stroke": "stroke", "r10cancre": "cancre", "r10arthre": "arthre", "r10shlt": "shlt", "r10bmi": "bmi"}
    parts = fi_parts(d.rename(columns=rename))
    add_fi_rows(miss_rows, "HRS", 10, "wave-10 active non-proxy", active, parts, ~mpm_positive, "BMI<18.5 or >=30", "HRS measured BMI and grip share the physical-measures subsample; proxy respondents remain outside IC measurement.")
    return {"n_active": int(active.sum()), "n_proxy_excluded": int(proxy_excluded.sum())}


def exclusive_sum(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = df[columns].apply(lambda x: num(x, 0, 10))
    n = values.notna().sum(axis=1)
    return values.sum(axis=1, min_count=1).where(n.eq(1))


def audit_share(weight_rows: list[dict], miss_rows: list[dict]) -> dict:
    base_path = OUT / "_share_ascii_real" / "Working_data" / "share.dta"
    base_cols = [
        "mergeid", "wave", "walkra", "walk100a", "lgrip", "rgrip", "gripcomp", "shlt", "bmi",
        "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre",
    ]
    d, _ = read_dta(base_path, base_cols)
    d = d[d["wave"].eq(6)].copy()
    d["mergeid"] = normalize_key(d["mergeid"])
    cf_path = next((OUT / "_share_ascii_real" / "Raw_data").rglob("*w6*cf.dta"))
    gh_path = next((OUT / "_share_ascii_real" / "Raw_data").rglob("*w6*gv_health.dta"))
    cf_cols = ["mergeid", "cf003_", "cf004_", "cf005_", "cf006_", "cf010_"] + [f"cf{i}tot" for i in range(104, 108)] + [f"cf{i}tot" for i in range(113, 117)]
    cf, _ = read_dta(cf_path, cf_cols)
    gh, _ = read_dta(gh_path, ["mergeid", "euro1", "euro2", "euro5", "euro11"])
    for frame in (cf, gh):
        frame["mergeid"] = normalize_key(frame["mergeid"])
    d = d.merge(cf, on="mergeid", how="left", validate="one_to_one").merge(gh, on="mergeid", how="left", validate="one_to_one")
    active = pd.Series(True, index=d.index)
    cf_labels = metadata_rows(cf_path, cf_cols)
    weight_root = ROOT / "SHARE"
    weight_path = next(weight_root.rglob("sharew6_rel9-0-0_gv_weights.dta"))
    weights, _ = read_dta(weight_path, ["mergeid", "dw_w6", "cchw_w6", "cciw_w6", "stratum1", "stratum2", "psu", "ssu"])
    weights["mergeid"] = normalize_key(weights["mergeid"])
    d = d.merge(weights, on="mergeid", how="left", validate="one_to_one")
    labels = metadata_rows(weight_path, ["dw_w6", "cchw_w6", "cciw_w6", "stratum1", "stratum2", "psu", "ssu"])
    for v, role, note in [
        ("dw_w6", "wave-6 design weight", "Base design weight; use only when the estimand and release documentation match."),
        ("cchw_w6", "wave-6 calibrated household weight", "Household-level calibration weight."),
        ("cciw_w6", "wave-6 calibrated individual weight", "Preferred person-level anchor-wave sensitivity weight."),
    ]:
        source_weight_row(weight_rows, "SHARE", 6, v, role, labels.get(v, ""), weight_path, active, d[v], note)
    for v, role, note in [
        ("stratum1", "primary stratum", "Released SHARE design variable."),
        ("stratum2", "secondary stratum", "Released SHARE design variable where applicable."),
        ("psu", "primary sampling unit", "Released SHARE design variable."),
        ("ssu", "secondary sampling unit", "Released SHARE design variable where applicable."),
    ]:
        source_design_row(weight_rows, "SHARE", 6, v, role, labels.get(v, ""), weight_path, active, d[v], note)

    orient = d[["cf003_", "cf004_", "cf005_", "cf006_"]].apply(lambda x: num(x, 1, 2).map({1: 1.0, 2: 0.0}))
    orientation = orient.sum(axis=1, min_count=4) / 4
    immediate = exclusive_sum(d, [f"cf{i}tot" for i in range(104, 108)]) / 10
    delayed = exclusive_sum(d, [f"cf{i}tot" for i in range(113, 117)]) / 10
    verbal = num(d["cf010_"], 0, 100) / 100
    walk100 = valid_binary(d["walk100a"])
    walkra = valid_binary(d["walkra"])
    grip = pd.concat([num(d["lgrip"], 0, 100), num(d["rgrip"], 0, 100)], axis=1).max(axis=1, skipna=True)
    grip_valid = grip.notna()
    psych = {k: valid_binary(d[k]) for k in ["euro1", "euro2", "euro5", "euro11"]}
    for name, val, rule in [
        ("immediate_recall", immediate.notna(), "one valid word-list score, 0-10"),
        ("delayed_recall", delayed.notna(), "one valid word-list score, 0-10"),
        ("orientation", orientation.notna(), "four binary orientation items"),
        ("verbal_fluency", verbal.notna(), "0-100 scaled score"),
        ("locomotion_100m", walk100, "binary 0/1"),
        ("locomotion_room", walkra, "binary 0/1"),
    ]:
        construct = "cognition" if name in {"immediate_recall", "delayed_recall", "orientation", "verbal_fluency"} else "locomotion"
        add_missingness(miss_rows, "SHARE", 6, "wave-6 active", construct, name, active, val, None, rule, "SHARE wave-6 module/item missingness; no individual-level imputation is applied in this audit.")
    grip_module = d["gripcomp"].notna()
    add_missingness(miss_rows, "SHARE", 6, "wave-6 active", "grip_vitality", "any_handgrip", active, grip_valid, ~grip_module, "max(left,right) in 0-100 kg", "gripcomp presence identifies the physical module; residual missingness within the module remains item/refusal/ability missingness.")
    for name, val in psych.items():
        add_missingness(miss_rows, "SHARE", 6, "wave-6 active", "psychological", name, active, val, None, "binary EURO-D symptom indicator", "Item-level EURO-D missingness is not separated further in the reduced working file.")
    parts = fi_parts(d.rename(columns={"hibpe": "hibpe", "diabe": "diabe", "hearte": "hearte", "stroke": "stroke", "cancre": "cancre", "arthre": "arthre", "shlt": "shlt", "bmi": "bmi"}))
    add_fi_rows(miss_rows, "SHARE", 6, "wave-6 active", active, parts, None, "BMI<18.5 or >=30", "SHARE measured BMI is broadly available; no structural BMI flag is imposed.")
    return {"n_active": int(active.sum()), "n_proxy_excluded": 0}


def build_memo(weight_df: pd.DataFrame, miss_df: pd.DataFrame, counts: dict) -> str:
    lines = [
        "# Stage 21 survey-weight and missingness audit",
        "",
        "本阶段只输出聚合审计，不进行调查加权 SEM、多重插补或因果推断。四个队列的主锚定样本规则沿用已冻结方案；HRS IC 主样本排除 proxy respondents，SHARE wave 6 作为外部验证。",
        "",
        "## 设计变量结论",
        "",
        "- **ELSA wave 6**：原始主文件提供 `w6xwgt` 横断面权重和 `w6lwgt` 纵向权重；护士文件提供 `w6nurwt`，可用于护士/体格模块选择与权重敏感性。审计到的公开文件没有显式官方 PSU/stratum；`idahhw6` 只能作为家庭聚类代理，不能写成官方 PSU。",
        "- **CHARLS wave 3**：`r3wtrespb` 是带家庭和个人非应答调整的个人权重，适合作为锚定波次敏感性权重。`communityID` 在 harmonized 文件中保留，可作为 community PSU/cluster；`urban_nbs` 仅在独立 PSU 辅助文件中，未形成一个明确命名的 stratum 变量。wave 3 发布文件未提供可直接用于本项目的个体纵向权重。",
        "- **HRS wave 10**：`r10wtresp` 是人层面分析权重；`raestrat` 与 `raehsamp` 分别对应 STRATUM 和 SECU，必须联合使用。握力/BMI 属于 2010 physical-measures subsample，应审慎使用 tracker 中的 `MPMWGTR`；它不能机械地与 `r10wtresp` 相乘。",
        "- **SHARE wave 6**：完整权重文件提供 `cciw_w6` 校准个人权重、`stratum1/2`、`psu` 和 `ssu`。SHARE 是外部结构验证，若做加权敏感性，优先使用 `cciw_w6`，并显式报告校准权重和 PSU/分层处理。",
        "",
        "## 缺失机制结论",
        "",
        "结构性缺失与普通项目缺失不能用一个统一规则覆盖。ELSA 护士权重、CHARLS `r3gripcomp`/体格完成标志、HRS `MPMWGTR`、SHARE `gripcomp` 可用于识别体格模块未进入或未完成；其余认知、心理、自评健康和疾病组件缺乏足够一致的释放原因码，应标记为 unresolved/item-level missingness，不应自动宣称 MAR。",
        "",
        "HRS 的 proxy 排除必须在所有 IC 测量模型中保持；proxy 可保留在 FI 描述性或敏感性分析。SHARE wave 7 的认知结构性缺失仍按既定方案排除主纵向认知模型并作为敏感性波，不能通过普通插补把结构性模块缺失改造成可比认知测量。",
        "",
        "## 后续分析建议",
        "",
        "1. 锚定波次主模型继续保留未加权 partial-metric 结构比较，把设计加权作为预先规定的敏感性分析；对不同队列分别使用上述人层面权重和相应设计变量。",
        "2. 不在当前阶段直接做 MI。后续若进行 MI，应把普通项目缺失、体格模块结构性缺失和 HRS proxy 排除分开处理，并纳入年龄、性别、教育、队列/国家、疾病组件、完整的 IC 指标、模块完成标志和权重相关变量；结构性缺失需增加 pattern-mixture 或 inverse-probability sensitivity，而不是单纯补值。",
        "3. 对 HRS physical-measures 子样本，先比较 `MPMWGTR` 有效样本与一般 `r10wtresp` 有效样本的构成，再决定是否使用模块权重；不把模块权重与普通人层面权重直接相乘。",
        "",
        "## 聚合证据",
        "",
        f"锚定样本量：ELSA {counts['ELSA']['n_active']:,}，CHARLS {counts['CHARLS']['n_active']:,}，HRS 非 proxy {counts['HRS']['n_active']:,}（排除 proxy {counts['HRS']['n_proxy_excluded']:,}），SHARE {counts['SHARE']['n_active']:,}。",
        "权重可用率和每个 IC/FI 指标的有效率、结构性缺失率、未解析缺失率分别见 `tables/stage21_weight_inventory.csv` 和 `tables/stage21_missingness_summary.csv`。",
        "",
        "本阶段结果仍属于探索性设计审计；在权重与缺失规则通过预先锁定后，才进入设计加权或多重插补敏感性分析。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    weight_rows: list[dict] = []
    miss_rows: list[dict] = []
    counts = {
        "ELSA": audit_elsa(weight_rows, miss_rows),
        "CHARLS": audit_charls(weight_rows, miss_rows),
        "HRS": audit_hrs(weight_rows, miss_rows),
        "SHARE": audit_share(weight_rows, miss_rows),
    }
    weight_df = pd.DataFrame(weight_rows)
    miss_df = pd.DataFrame(miss_rows)
    weight_df.round(6).to_csv(TAB / "stage21_weight_inventory.csv", index=False, encoding="utf-8-sig")
    miss_df.round(6).to_csv(TAB / "stage21_missingness_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "stage21_weight_missingness_memo.md").write_text(build_memo(weight_df, miss_df, counts), encoding="utf-8")
    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "21 survey weight and missingness audit",
        "aggregate_only": True,
        "counts": counts,
        "outputs": ["tables/stage21_weight_inventory.csv", "tables/stage21_missingness_summary.csv", "stage21_weight_missingness_memo.md"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rules": {
            "fi": "six diseases + self-rated health + BMI; minimum 6/8; observed denominator; BMI<18.5 or >=30",
            "hrs_ic": "inw10==1 and r10proxy==0",
            "share": "wave 6 external validation; wave 7 structural cognitive exclusion retained",
        },
    }
    (OUT / "stage21_weight_missingness_run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote Stage 21 aggregate survey-weight and missingness audit outputs.")


if __name__ == "__main__":
    main()
