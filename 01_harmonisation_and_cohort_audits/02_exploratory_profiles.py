"""Exploratory profiles for the IC-frailty construct-comparison project.

The output is a feasibility audit, not a final harmonization or confirmatory
analysis. Raw source files are opened read-only. Values are summarized only;
no rows, IDs, or direct identifiers are exported.
"""
from __future__ import annotations

import json
import platform
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns


PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT_ROOT = Path(r"PATH_TO_IC_FRAILTY")
FIG_ROOT = OUT_ROOT / "figures"
TAB_ROOT = OUT_ROOT / "tables"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TAB_ROOT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 100)


def resolve(pattern: str) -> Path:
    direct = PROJECT_ROOT / pattern
    if direct.exists():
        return direct
    matches = list(PROJECT_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def existing_columns(path: Path) -> list[str]:
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    return list(meta.column_names)


def safe_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    # Common Stata negative missing/special values are not measurements.
    out = out.mask(out < -90)
    return out


def read_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in dict.fromkeys(columns) if c]
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    available = set(meta.column_names)
    use = [c for c in cols if c in available]
    df, _ = pyreadstat.read_dta(str(path), usecols=use, apply_value_formats=False)
    for c in use:
        if c not in df:
            df[c] = np.nan
    return df


def unique_id_count(path: Path, id_column: str) -> int:
    """Count unique IDs in memory without exporting ID values."""
    df = read_selected(path, [id_column])
    if id_column not in df:
        return 0
    return int(df[id_column].dropna().nunique())


def add_missing_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df


def collapse_grip(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    vals = [safe_numeric(df[c]) for c in cols if c in df]
    return pd.concat(vals, axis=1).max(axis=1, skipna=True) if vals else pd.Series(np.nan, index=df.index)


def build_long(dataset: str, path: Path, spec: dict) -> pd.DataFrame:
    vars_: list[str] = [spec["id"], spec["wave"]]
    for value in spec["vars"].values():
        if isinstance(value, list):
            vars_.extend(value)
        elif value:
            vars_.append(value)
    df = read_selected(path, vars_)
    add_missing_columns(df, vars_)
    out = pd.DataFrame(index=df.index)
    out["dataset"] = dataset
    out["id_present"] = df[spec["id"]].notna() if spec["id"] in df else False
    # Long working files are already respondent/person-wave records.  The
    # participation flags in these files are not consistently retained (and
    # in several files are constant placeholders), so retain all rows here
    # and document them as active records for this exploratory audit.
    out["active_wave"] = out["id_present"]
    out["wave"] = df[spec["wave"]] if spec["wave"] in df else np.nan
    out["age"] = safe_numeric(df[spec["vars"].get("age")]) if spec["vars"].get("age") in df else np.nan
    out["sex"] = df[spec["vars"].get("sex")] if spec["vars"].get("sex") in df else np.nan
    for domain, value in spec["vars"].items():
        if domain in {"age", "sex"}:
            continue
        if isinstance(value, list):
            # For item lists, retain completeness only; item-level raw codes
            # are not assumed to share direction or scoring across surveys.
            vals = [safe_numeric(df[c]) for c in value if c in df]
            out[domain + "_n_items_observed"] = pd.concat(vals, axis=1).notna().sum(axis=1) if vals else 0
            out[domain] = np.nan
        elif value and value in df:
            out[domain] = safe_numeric(df[value])
        else:
            out[domain] = np.nan
    if "grip_left" in spec["vars"] or "grip_right" in spec["vars"]:
        grips = []
        for key in ["grip_left", "grip_right"]:
            val = spec["vars"].get(key)
            if isinstance(val, list):
                grips.extend([c for c in val if c in df])
            elif val:
                grips.append(val)
        out["vitality_grip"] = collapse_grip(df, grips)
    return out


def build_wide(dataset: str, path: Path, spec: dict) -> pd.DataFrame:
    available = set(existing_columns(path))
    columns = [spec["id"]]
    if spec.get("static"):
        columns.extend(spec["static"].values())
    wave_vars: dict[int, dict[str, list[str] | str]] = {}
    for w in spec["waves"]:
        wave_vars[w] = {}
        for domain, template in spec["templates"].items():
            if isinstance(template, list):
                vals = [x.format(w=w) for x in template]
                wave_vars[w][domain] = [x for x in vals if x in available]
                columns.extend(wave_vars[w][domain])
            else:
                val = template.format(w=w)
                wave_vars[w][domain] = val if val in available else ""
                if val in available:
                    columns.append(val)
        active_template = spec.get("active_template")
        if active_template:
            active_val = active_template.format(w=w)
            wave_vars[w]["_active"] = active_val if active_val in available else ""
            if active_val in available:
                columns.append(active_val)
    columns = [c for c in dict.fromkeys(columns) if c in available]
    df = read_selected(path, columns)
    rows = []
    for w in spec["waves"]:
        out = pd.DataFrame(index=df.index)
        out["dataset"] = dataset
        out["id_present"] = df[spec["id"]].notna() if spec["id"] in df else False
        active_val = wave_vars[w].get("_active", "")
        if active_val and active_val in df:
            # CHARLS/HRS use 1 for a respondent alive/participating in the
            # wave; keep only this flag as the structural active-wave marker.
            out["active_wave"] = safe_numeric(df[active_val]).eq(1) & out["id_present"]
        else:
            # LASI is a single baseline working file; a non-missing interview
            # year is the closest available participation marker.
            year_template = spec.get("active_year_template")
            year_val = year_template.format(w=w) if year_template else ""
            if year_val and year_val in df:
                out["active_wave"] = df[year_val].notna() & out["id_present"]
            else:
                out["active_wave"] = out["id_present"]
        out["wave"] = w
        static = spec.get("static", {})
        out["age"] = safe_numeric(df[static["age"]]) if static.get("age") in df else np.nan
        out["sex"] = df[static["sex"]] if static.get("sex") in df else np.nan
        for domain in spec["templates"]:
            value = wave_vars[w][domain]
            if isinstance(value, list):
                out[domain + "_n_items_observed"] = df[value].notna().sum(axis=1) if value else 0
                out[domain] = np.nan
            elif value and value in df:
                out[domain] = safe_numeric(df[value])
            else:
                out[domain] = np.nan
        grips = []
        for key in ["grip_left", "grip_right"]:
            value = wave_vars[w].get(key)
            if isinstance(value, list):
                grips.extend(value)
            elif value:
                grips.append(value)
        out["vitality_grip"] = collapse_grip(df, grips)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def quantile_summary(s: pd.Series) -> dict:
    x = safe_numeric(s).dropna()
    if x.empty:
        return {"n_observed": 0, "pct_observed": 0.0}
    q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "n_observed": int(x.size),
        "pct_observed": round(float(x.size / len(s) * 100), 2),
        "mean": round(float(x.mean()), 4),
        "sd": round(float(x.std(ddof=1)), 4) if x.size > 1 else np.nan,
        "p01": round(float(q.loc[0.01]), 4),
        "p25": round(float(q.loc[0.25]), 4),
        "median": round(float(q.loc[0.5]), 4),
        "p75": round(float(q.loc[0.75]), 4),
        "p99": round(float(q.loc[0.99]), 4),
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
    }


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    # Align directions for an exploratory "higher worse" view. Sensory
    # indicators are excluded until value-label direction is audited.
    work = pd.DataFrame(index=df.index)
    for source, target, sign in [
        ("cognition", "cognitive_impairment", -1),
        ("locomotion", "locomotion_impairment", 1),
        ("vitality_grip", "vitality_impairment", -1),
        ("psychological", "psychological_impairment", 1),
        ("frailty", "frailty", 1),
    ]:
        if source in df:
            x = safe_numeric(df[source])
            if x.notna().sum() >= 20 and x.nunique(dropna=True) > 1:
                work[target] = sign * x
    if work.shape[1] < 2:
        return pd.DataFrame()
    return work.corr(method="spearman", min_periods=30).round(4)


def make_plots(wave_availability: pd.DataFrame, corr_by_dataset: dict[str, pd.DataFrame]) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    if not wave_availability.empty:
        plot = wave_availability.pivot_table(index="dataset", columns="domain", values="pct_observed", aggfunc="mean")
        plt.figure(figsize=(14, 7))
        sns.heatmap(plot, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, cbar_kws={"label": "% observed"})
        plt.title("Exploratory domain availability across waves")
        plt.xlabel("Candidate domain/indicator")
        plt.ylabel("")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_ROOT / "domain_availability_heatmap.png", dpi=220)
        plt.close()
    for dataset, corr in corr_by_dataset.items():
        if corr.empty:
            continue
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="vlag", square=True)
        plt.title(f"Exploratory Spearman correlations: {dataset}")
        plt.tight_layout()
        plt.savefig(FIG_ROOT / f"spearman_{dataset.lower()}.png", dpi=220)
        plt.close()


def main() -> None:
    specs_long = {
        "ELSA": {
            "path": "ELSA/Working_data/elsa.dta", "id": "idauniqc", "wave": "wave",
            "vars": {"age": "agey", "sex": "ragender", "cognition": "tcog_z_z", "cognition_memory": "memory_z", "locomotion": "wspeed", "locomotion_selfreport": "walkra", "vitality_grip": "gripsum", "sensory_vision": "sight", "sensory_hearing": "hearing", "psychological": "cesd", "frailty": "frailty", "function_adl": "adltot6", "function_iadl": "iadltot2_e"},
        },
        "SHARE": {
            "path": "SHARE/**/Working_data/share.dta", "id": "mergeid", "wave": "wave",
            "vars": {"age": "agey", "sex": "ragender", "cognition": "orient", "cognition_math": "raccmathperf", "locomotion": "wspeed", "locomotion_selfreport": "walkra", "grip_left": ["lgrip"], "grip_right": ["rgrip"], "sensory_vision": "dsight", "sensory_hearing": "hearing", "psychological": "eurod", "frailty": "frailtyb", "function_adl_items": ["walkra", "dressa", "batha", "eata", "beda", "toilta"], "function_iadl_items": ["phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "housewka", "leavhsa", "laundrya"]},
        },
        "KLoSA": {
            "path": "KLoSA/**/Working_data/klosa.dta", "id": "pid", "wave": "wave",
            "vars": {"age": "agey", "sex": "ragender", "cognition": "cog_total", "locomotion": "", "locomotion_selfreport": "", "grip_left": ["lgrip"], "grip_right": ["rgrip"], "sensory_vision": "sighta", "sensory_hearing": "hearinga", "psychological": "cesd10b", "frailty": "", "function_adl_items": ["dressb", "bathb", "eatb", "toiltb", "bedb_k", "brushb", "urinb"], "function_iadl_items": ["mealsb", "shopb", "medsb", "moneyb", "phoneb", "transb", "gooutb", "laundryb", "housewkb", "groomb"]},
        },
        "MHAS": {
            "path": "MHAS/**/Working_data/mhas.dta", "id": "rahhidnp", "wave": "wave",
            "vars": {"age": "agey", "sex": "ragender", "cognition": "orient_m", "locomotion": "wspeed", "locomotion_selfreport": "walkra", "grip_left": ["lgrip"], "grip_right": ["rgrip"], "sensory_vision": "sight", "sensory_hearing": "hearing", "psychological": "cesd_m", "frailty": "", "function_adl": "adltot6", "function_iadl": "iadlfour"},
        },
    }
    specs_wide = {
        "CHARLS": {
            "path": "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta", "id": "ID", "waves": [1, 2, 3, 4],
            "static": {},
            "active_template": "inw{w}",
            "templates": {"age": "r{w}agey", "cognition": "r{w}tr20", "locomotion": "r{w}wspeed", "locomotion_selfreport": "r{w}walk100a", "grip_left": ["r{w}lgrip"], "grip_right": ["r{w}rgrip"], "psychological": "r{w}cesd10", "frailty": "", "function_adl": "r{w}adltot6", "function_iadl": "r{w}iadlza"},
        },
        "HRS": {
            "path": "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta", "id": "hhidpn", "waves": [8, 9, 10, 11, 12, 13],
            "static": {"age": "", "sex": "ragender"},
            "active_template": "inw{w}",
            "templates": {"age": "r{w}agey_m", "cognition": "r{w}cogtot", "locomotion": "r{w}timwlk", "locomotion_selfreport": "r{w}walkra", "grip_left": ["r{w}grpl"], "grip_right": ["r{w}grpr"], "sensory_vision": "r{w}eyert", "sensory_hearing": "r{w}earrt", "psychological": "r{w}cesd", "frailty": "", "function_adl": "r{w}adl6a", "function_iadl": "r{w}iadl5a"},
        },
        "LASI": {
            "path": "LASI/**/Working_data/lasi.dta", "id": "hhid", "waves": [1],
            "static": {},
            "active_year_template": "r{w}iwy",
            "templates": {"age": "r{w}agey", "cognition": "r{w}cog_total", "locomotion": "r{w}wspeed", "locomotion_selfreport": "r{w}walk100a", "grip_left": ["r{w}lgrip"], "grip_right": ["r{w}rgrip"], "sensory_vision": "r{w}dsighta", "sensory_hearing": "r{w}hearcnde", "psychological": "r{w}cesd10_l", "frailty": "", "function_adl": "r{w}adltot6", "function_iadl": "r{w}iadltot_l"},
        },
    }

    all_data: dict[str, pd.DataFrame] = {}
    source_rows = []
    for dataset, spec in specs_long.items():
        path = resolve(spec["path"])
        df = build_long(dataset, path, spec)
        all_data[dataset] = df
        source_rows.append({"dataset": dataset, "layout": "long", "source_relative": str(path.relative_to(PROJECT_ROOT)), "n_rows": len(df), "n_unique_ids": unique_id_count(path, spec["id"])})
    for dataset, spec in specs_wide.items():
        path = resolve(spec["path"])
        df = build_wide(dataset, path, spec)
        all_data[dataset] = df
        source_rows.append({"dataset": dataset, "layout": "wide_to_long_exploratory", "source_relative": str(path.relative_to(PROJECT_ROOT)), "n_rows": len(df), "n_unique_ids": unique_id_count(path, spec["id"])})

    domain_cols = ["cognition", "cognition_memory", "cognition_math", "locomotion", "locomotion_selfreport", "vitality_grip", "sensory_vision", "sensory_hearing", "psychological", "frailty", "function_adl", "function_iadl"]
    profile_rows = []
    availability_rows = []
    corr_by_dataset: dict[str, pd.DataFrame] = {}
    corr_rows = []
    for dataset, df in all_data.items():
        df.to_pickle(OUT_ROOT / f"_tmp_{dataset}.pkl")
        active = df["active_wave"].fillna(False).astype(bool)
        profile_rows.append({"dataset": dataset, "n_person_wave_rows": len(df), "n_active_wave_rows": int(active.sum()), "pct_rows_active": round(float(active.mean() * 100), 2), "n_waves": int(df["wave"].nunique(dropna=True)), "wave_values": ";".join(map(str, sorted(df["wave"].dropna().unique().tolist()))) if df["wave"].notna().any() else "", "n_rows_with_id": int(df["id_present"].sum()), "age_observed_pct_all_rows": round(float(df["age"].notna().mean() * 100), 2), "age_observed_pct_active": round(float(df.loc[active, "age"].notna().mean() * 100), 2) if active.any() else np.nan, "age_median_active": round(float(safe_numeric(df.loc[active, "age"]).median()), 2) if active.any() and df.loc[active, "age"].notna().any() else np.nan})
        for wave, sub in df.groupby("wave", dropna=False):
            for domain in domain_cols:
                if domain in sub:
                    s = sub[domain]
                else:
                    s = pd.Series(np.nan, index=sub.index)
                summ = quantile_summary(s)
                active_sub = sub.loc[sub["active_wave"].fillna(False).astype(bool)]
                active_series = active_sub[domain] if domain in active_sub else pd.Series(np.nan, index=active_sub.index)
                active_summ = quantile_summary(active_series) if len(active_sub) else {"n_observed": 0, "pct_observed": np.nan}
                availability_rows.append({"dataset": dataset, "wave": wave, "domain": domain, "n_rows_all": len(sub), "n_rows_active": len(active_sub), **{f"all_{k}": v for k, v in summ.items()}, **{f"active_{k}": v for k, v in active_summ.items()}})
        corr = corr_table(df.loc[df["active_wave"].fillna(False).astype(bool)].copy())
        corr_by_dataset[dataset] = corr
        if not corr.empty:
            corr.to_csv(TAB_ROOT / f"spearman_{dataset.lower()}.csv", encoding="utf-8-sig")
            for a in corr.index:
                for b in corr.columns:
                    if a < b:
                        corr_rows.append({"dataset": dataset, "domain_a": a, "domain_b": b, "spearman_rho": corr.loc[a, b]})

    source_df = pd.DataFrame(source_rows)
    profile_df = pd.DataFrame(profile_rows)
    avail_df = pd.DataFrame(availability_rows)
    corr_df = pd.DataFrame(corr_rows)
    source_df.to_csv(TAB_ROOT / "source_profile.csv", index=False, encoding="utf-8-sig")
    profile_df.to_csv(TAB_ROOT / "dataset_profile.csv", index=False, encoding="utf-8-sig")
    avail_df.to_csv(TAB_ROOT / "wave_domain_availability_and_summary.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(TAB_ROOT / "exploratory_pairwise_spearman_long.csv", index=False, encoding="utf-8-sig")

    # Existing composite availability and basic distribution by dataset.
    existing = []
    for dataset, df in all_data.items():
        for col in ["frailty", "cognition", "locomotion", "vitality_grip", "sensory_vision", "sensory_hearing", "psychological"]:
            if col not in df:
                continue
            x = safe_numeric(df[col])
            active = df["active_wave"].fillna(False).astype(bool)
            xa = x.loc[active]
            existing.append({"dataset": dataset, "indicator": col, "n_all_rows": len(x), "n_active_rows": int(active.sum()), "n": int(x.notna().sum()), "pct_rows_observed_all": round(float(x.notna().mean() * 100), 2), "pct_rows_observed_active": round(float(xa.notna().mean() * 100), 2) if len(xa) else np.nan, "n_unique_nonmissing": int(x.nunique(dropna=True)), "median_active": round(float(xa.median()), 4) if xa.notna().any() else np.nan, "p25_active": round(float(xa.quantile(.25)), 4) if xa.notna().any() else np.nan, "p75_active": round(float(xa.quantile(.75)), 4) if xa.notna().any() else np.nan})
    pd.DataFrame(existing).to_csv(TAB_ROOT / "indicator_distribution_overview.csv", index=False, encoding="utf-8-sig")

    # Person-wave completeness for candidate constructs. These rates are
    # descriptive only and do not define the final analysis sample.
    eligibility_rows = []
    for dataset, df in all_data.items():
        for wave, sub in df.groupby("wave", dropna=False):
            base = ["cognition", "locomotion", "vitality_grip", "psychological"]
            selfreport_base = ["cognition", "locomotion_selfreport", "vitality_grip", "psychological"]
            full_five = base + ["sensory_vision", "sensory_hearing"]
            full_five_selfreport = selfreport_base + ["sensory_vision", "sensory_hearing"]
            for label, cols in [("non_sensory_four_domain_objective_locomotion", base), ("non_sensory_four_domain_selfreport_locomotion", selfreport_base), ("five_domain_with_vision_hearing", full_five), ("five_domain_with_selfreport_locomotion", full_five_selfreport)]:
                present = [c for c in cols if c in sub]
                # A missing column means the construct definition is not
                # available in that source file; do not silently downgrade a
                # five-domain definition to a four-domain definition.
                complete = sub[present].notna().all(axis=1) if len(present) == len(cols) else pd.Series(False, index=sub.index)
                active = sub["active_wave"].fillna(False).astype(bool)
                active_complete = complete.loc[active]
                eligibility_rows.append({"dataset": dataset, "wave": wave, "eligibility_definition": label, "n_rows_all": len(sub), "n_rows_active": int(active.sum()), "n_complete_all": int(complete.sum()), "pct_complete_all": round(float(complete.mean() * 100), 2), "n_complete_active": int(active_complete.sum()), "pct_complete_active": round(float(active_complete.mean() * 100), 2) if len(active_complete) else np.nan})
    pd.DataFrame(eligibility_rows).to_csv(TAB_ROOT / "exploratory_construct_completeness.csv", index=False, encoding="utf-8-sig")

    plot_avail = avail_df.groupby(["dataset", "domain"], as_index=False)["active_pct_observed"].mean().rename(columns={"active_pct_observed": "pct_observed"})
    make_plots(plot_avail, corr_by_dataset)

    # Remove temporary files after all summaries are written; source files remain untouched.
    for dataset in all_data:
        tmp = OUT_ROOT / f"_tmp_{dataset}.pkl"
        if tmp.exists():
            tmp.unlink()

    run = {
        "analysis": "IC-frailty exploratory profiles",
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_root": str(PROJECT_ROOT),
        "output_root": str(OUT_ROOT),
        "read_only_source": True,
        "scope": "candidate indicator coverage, wave-level missingness, robust distribution summaries, pairwise Spearman correlations",
        "direction_rule": "exploratory impairment orientation: cognition and grip multiplied by -1; locomotion, psychological, frailty retained as higher-worse; sensory excluded from correlations until value labels are audited",
        "special_value_rule": "numeric values below -90 treated as missing for summaries; no imputation or row deletion",
        "warning": "Candidate variables are not final harmonized measures; item coding, units, survey weights, proxy status, and measurement invariance require a separate data dictionary audit.",
    }
    (OUT_ROOT / "exploratory_run_info.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable first report.
    lines = [
        "# IC–frailty 跨国潜在构念比较：探索性数据审计",
        "",
        f"运行时间（UTC）：{run['analysis_date_utc']}",
        "",
        "本轮仅做探索性资料审计，未进行正式测量等值性检验、插补、加权推断或因果分析。原始 Stata 文件只读；输出均为聚合结果。",
        "",
        "## 数据文件和覆盖",
        "",
        "见 `tables/source_profile.csv` 和 `tables/dataset_profile.csv`。宽格式 CHARLS、HRS、LASI 被展开为 person-wave 形式；CHARLS/HRS 使用逐波次 `inw` 旗标识别活跃波次，完整率和相关结构按活跃记录汇总。这不是最终分析数据集。",
        "",
        "## 当前可行性信号",
        "",
        "- ELSA 和 SHARE 具有现成 frailty 指数（分别为 `frailty` 和 `frailtyb`），可作为现有构念锚点。",
        "- CHARLS、HRS、KLoSA、MHAS 和 LASI 能找到认知、运动、握力/活力和心理候选指标，但需要自行构造 outcome-disjoint frailty 指标。",
        "- CHARLS 当前 harmonized 文件的候选标签中未发现明确的视觉或听觉变量；不能预先假设 CHARLS 可以参与完整五域 IC 的 exact comparison。",
        "- KLoSA 工作文件没有明确的步速候选变量；MHAS 只有一个简化的定向指标；这两个队列更适合作为外部验证或四域/简化构念的敏感性分析，除非回到原始模块补齐指标。",
        "- HAALSI 有 3 个波次；波次 2–3 的认知、步行困难和 CES-D 候选项覆盖较好，但波次 1 的认知仍是原始题目格式，当前作为补充验证队列。",
        "- 视觉、听觉变量的数值方向和类别编码尚未经过 value-label 审计，因此本轮没有把它们纳入相关矩阵。ELSA/SHARE 现成 frailty 的组成项目审计见 `tables/existing_frailty_component_audit.csv`。",
        "",
        "## 变量方向和统计规则",
        "",
        "本轮的 exploratory impairment orientation 将认知分数和握力乘以 -1，步行测试时间、抑郁/心理分数和现成 frailty 指数保持高值代表较差；这只是为了查看相关结构，不是最终 IC 或 frailty 评分规则。低于 -90 的数值按特殊缺失处理；没有自动插补、删异常值或应用调查权重。",
        "",
        "## 输出文件",
        "",
        "- `01_metadata_audit.py`：源文件元数据和候选变量审计。",
        "- `02_exploratory_profiles.py`：波次覆盖、缺失、分布和相关结构。",
        "- `candidate_variables.csv`、`candidate_domain_inventory.csv`：候选变量清单。",
        "- `tables/wave_domain_availability_and_summary.csv`：逐队列逐波次候选域可用率和稳健分布摘要，含全部展开行与活跃波次行。",
        "- `tables/indicator_distribution_overview.csv`：核心指标的整体分布概览。",
        "- `tables/exploratory_construct_completeness.csv`：四域和五域候选构念的 person-wave 完整率。",
        "- `tables/exploratory_pairwise_spearman_long.csv`：探索性相关结构。",
        "- `tables/candidate_value_label_audit.csv`：高影响候选变量的 value-label、原始观测数和方向审计。",
        "- `tables/existing_frailty_component_audit.csv`：ELSA/SHARE 现成 frailty 组成项目审计。",
        "- `05_haalSI_supplement.py` 与 `tables/haalSI_candidate_coverage.csv`：HAALSI 三波次的有限候选域覆盖审计。",
        "- `figures/domain_availability_heatmap.png`：域可用率热图；各队列的 Spearman 相关图位于同一目录。",
        "",
        "## 下一步 Go/No-Go",
        "",
        "1. 先完成七队列的逐变量 data dictionary、value-label 和单位核对；尤其是视听、步速时间/速度、握力、抑郁和代理回答。",
        "2. 以 CHARLS、HRS、ELSA 的共同四域/五域候选集合计算完整病例和可接受缺失阈值下的样本量；SHARE、KLoSA、MHAS 作为扩展验证。",
        "3. 冻结两套定义：`full_IC`（仅在五域可比队列）和 `non-sensory_IC`/`four-domain_IC`（跨更多队列）；同时冻结不包含认知、ADL/IADL 和死亡的 FI。",
        "4. 只有在逐项编码和样本量通过后，才进入多组 CFA/IRT、alignment 和潜在构念比较。",
    ]
    (OUT_ROOT / "exploratory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"datasets": list(all_data), "source_profiles": source_df.to_dict(orient="records"), "output": str(OUT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
