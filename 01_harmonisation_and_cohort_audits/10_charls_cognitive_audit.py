"""Audit CHARLS harmonized orientation and recall scores by active wave.

This is an exploratory, read-only audit. It does not impute, weight, or fit a
latent model.  Outputs are aggregate tables only; no IDs or person-level
records are written.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT_ROOT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT_ROOT / "tables"

DATA = PROJECT_ROOT / "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"
WAVES = [1, 2, 3, 4]


def valid_numeric(x: pd.Series) -> pd.Series:
    """Convert numeric data and remove negative/special missing codes."""
    y = pd.to_numeric(x, errors="coerce")
    return y.mask(y < -90)


def summary(x: pd.Series, prefix: str) -> dict[str, object]:
    y = valid_numeric(x).dropna()
    return {
        f"{prefix}_n_observed": int(y.size),
        f"{prefix}_pct_observed_active": None,
        f"{prefix}_mean": round(float(y.mean()), 4) if len(y) else None,
        f"{prefix}_sd": round(float(y.std(ddof=1)), 4) if len(y) > 1 else None,
        f"{prefix}_median": round(float(y.median()), 4) if len(y) else None,
        f"{prefix}_min": round(float(y.min()), 4) if len(y) else None,
        f"{prefix}_max": round(float(y.max()), 4) if len(y) else None,
    }


_, meta = pyreadstat.read_dta(str(DATA), metadataonly=True)
needed = ["ID"]
for wave in WAVES:
    needed.extend([f"inw{wave}", f"r{wave}orient", f"r{wave}tr20"])
needed = [c for c in needed if c in meta.column_names]
df, meta = pyreadstat.read_dta(str(DATA), usecols=needed, apply_value_formats=False)

rows: list[dict[str, object]] = []
for wave in WAVES:
    active = valid_numeric(df[f"inw{wave}"]).eq(1)
    orient = valid_numeric(df[f"r{wave}orient"])
    recall = valid_numeric(df[f"r{wave}tr20"])
    joint = active & orient.notna() & recall.notna()
    row: dict[str, object] = {
        "wave": wave,
        "n_rows_scanned": int(len(df)),
        "n_active": int(active.sum()),
        "orient_n_observed": int(orient.loc[active].notna().sum()),
        "orient_pct_observed_active": round(float(orient.loc[active].notna().mean() * 100), 2) if active.any() else None,
        "tr20_n_observed": int(recall.loc[active].notna().sum()),
        "tr20_pct_observed_active": round(float(recall.loc[active].notna().mean() * 100), 2) if active.any() else None,
        "joint_n_observed": int(joint.sum()),
        "joint_pct_observed_active": round(float(joint.loc[active].mean() * 100), 2) if active.any() else None,
        "orient_tr20_spearman_active": round(float(orient.loc[active].corr(recall.loc[active], method="spearman")), 4)
        if joint.sum() > 1 else None,
    }
    for key, value in summary(orient.loc[active], "orient").items():
        if key.endswith("pct_observed_active"):
            continue
        row[key] = value
    for key, value in summary(recall.loc[active], "tr20").items():
        if key.endswith("pct_observed_active"):
            continue
        row[key] = value
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(TAB / "charls_cognitive_domain_audit.csv", index=False, encoding="utf-8-sig")

label_rows = []
for wave in WAVES:
    for var in [f"r{wave}orient", f"r{wave}tr20"]:
        label_rows.append({
            "wave": wave,
            "variable": var,
            "variable_label": meta.column_names_to_labels.get(var, ""),
            "value_label_name": meta.variable_value_labels.get(var, ""),
        })
pd.DataFrame(label_rows).to_csv(TAB / "charls_cognitive_value_labels.csv", index=False, encoding="utf-8-sig")

lines = [
    "# CHARLS cognition audit",
    "",
    "本审计读取 CHARLS harmonized 文件中的 `r{wave}orient`（四项定向汇总）和 `r{wave}tr20`（即时+延迟词语回忆汇总），按 `inw{wave}=1` 限定 active person-wave。",
    "仅报告覆盖、分布和两指标的探索性 Spearman 相关；未进行插补、权重、测量等值性检验或正式推断。负值按特殊缺失处理。",
    "",
    "## 解释规则",
    "",
    "- `orient` 与 `tr20` 应被视为两个认知观测指标，而不是把 `tr20` 当作完整 cognition composite。",
    "- 联合覆盖率决定 CHARLS 在双指标 cognition 域中的可用 person-wave 范围。",
    "- 下一步正式模型应在方向统一、代理访谈标记和波次可比性审计后，使用这两个指标进行潜变量建模或敏感性比较。",
    "",
    "详见 `tables/charls_cognitive_domain_audit.csv` 和 `tables/charls_cognitive_value_labels.csv`。",
]
(OUT_ROOT / "charls_cognitive_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", TAB / "charls_cognitive_domain_audit.csv")
