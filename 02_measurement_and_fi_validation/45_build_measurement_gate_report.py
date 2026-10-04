"""Build a clean UTF-8 measurement/structural gate report from aggregate outputs."""
from pathlib import Path
import csv

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"

def read_csv(name):
    with (TAB / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")

fit15 = read_csv("stage15_raw_invariance_fit.csv")
fit15b = read_csv("stage15b_partial_scalar_fit.csv")
fit16 = read_csv("stage16_cross_cohort_partial_metric_sem_fit.csv")
lrt16 = read_csv("stage16_cross_cohort_partial_metric_sem_lrt.csv")
coef16 = read_csv("stage16_cross_cohort_partial_metric_sem_coefficients.csv")
fit17 = read_csv("stage17_cross_cohort_fi_sensitivity_fit.csv")
coef17 = read_csv("stage17_cross_cohort_fi_sensitivity_coefficients.csv")
share_corr = read_csv("stage20_share_external_spearman.csv")
share_desc = read_csv("stage20_share_external_descriptives.csv")

def row_by(rows, key, value):
    return next(r for r in rows if r.get(key) == value)

cfg = row_by(fit15, "model", "configural")
metric = row_by(fit15, "model", "metric")
partial = row_by(fit15, "model", "metric_partial_domain")
scalar = row_by(fit15b, "model", "scalar_partial_domain")
free = row_by(fit16, "model", "partial_metric_free")
eq_all = row_by(fit16, "model", "partial_metric_equal_all_slopes")
eq_loc = row_by(fit16, "model", "partial_metric_equal_locomotion")

def coeff(model, group, rhs):
    r = next(x for x in coef16 if x["model"] == model and x["group"] == str(group) and x["rhs"] == rhs)
    return f(r, "capacity_oriented_std_all")

def sens(outcome, group):
    r = next(x for x in coef17 if x["outcome"] == outcome and x["group"] == str(group) and x["rhs"] == "locomotion")
    return f(r, "capacity_oriented_std_all")

def pval_for(comparison):
    rows = [r for r in lrt16 if r.get("comparison") == comparison and r.get("model") == "b"]
    return f(rows[0], "Pr(>Chisq)") if rows else float("nan")

share_loc = next(r for r in share_corr if r["sample"] == "primary_full_component_composite" and r["construct_a"] == "locomotion_hierarchical" and r["construct_b"] == "fi_primary")
share_joint = next(r for r in share_desc if r["construct"] == "joint_four_domain_fi_complete")

lines = [
    "# IC–frailty 跨队列构念比较：Stage 15–17 测量与结构闸门报告",
    "",
    "本报告只使用聚合输出，不包含个体级 ID、因子分数或原始数据。主队列为 ELSA wave 6、CHARLS wave 3、HRS wave 10；HRS IC 样本排除代理访谈者。主 FI 为 6 项慢病 + self-rated health + BMI，至少观测 6/8 项并按实际观测项计分，BMI 缺陷采用 <18.5 或 ≥30。",
    "",
    "## 1. 测量闸门",
    "",
    f"- configural：CFI {f(cfg,'cfi'):.3f}，RMSEA {f(cfg,'rmsea'):.3f}，SRMR {f(cfg,'srmr'):.3f}；三队列共享四域结构的证据较好。",
    f"- full metric：CFI {f(metric,'cfi'):.3f}，RMSEA {f(metric,'rmsea'):.3f}，且 post-check 未通过；完全载荷等值不应作为主模型。",
    f"- prespecified partial metric：CFI {f(partial,'cfi'):.3f}，RMSEA {f(partial,'rmsea'):.3f}，SRMR {f(partial,'srmr'):.3f}，post-check 通过。释放 cognition–executive、grip–right、locomotion–100m、psychological–happy 四个载荷后，模型恢复稳定。",
    f"- partial scalar：CFI {f(scalar,'cfi'):.3f}，RMSEA {f(scalar,'rmsea'):.3f}，post-check 未通过；HRS grip–right 的标准化载荷约 1.04，提示握力模块覆盖和参数化仍是限制。",
    "- threshold-only 模型的阈值标签确实在组间共享，但在混合连续/有序指标的均值结构识别下，相对 partial metric 的自由度变化为 0。因此不把 threshold-only 的零自由度比较称为阈值等值性证据；正式标量闸门采用 thresholds + continuous intercepts，结果不支持跨队列潜在均值比较。",
    "",
    "测量结论：论文主分析应明确写作“partial-metric measurement comparability / audit”，并比较结构系数；不报告跨队列 IC 潜在均值，不宣称建立 universal general IC。",
    "",
    "## 2. 部分 metric 下的 IC–FI 结构比较",
    "",
    f"自由斜率模型拟合：CFI {f(free,'cfi'):.3f}，RMSEA {f(free,'rmsea'):.3f}，SRMR {f(free,'srmr'):.3f}。四个结构系数均允许队列特异。",
    f"把四个斜率全部约束为相等后，CFI {f(eq_all,'cfi'):.3f}，RMSEA {f(eq_all,'rmsea'):.3f}；相对自由斜率模型的 robust LRT Δχ²≈385.3（8 df，p<10⁻⁷⁷）。",
    f"只约束 locomotion 斜率相等后，CFI {f(eq_loc,'cfi'):.3f}，RMSEA {f(eq_loc,'rmsea'):.3f}；Δχ²≈244.7（2 df，p<10⁻⁵³）。因此 locomotion 的方向可迁移，但效应大小不能视为跨队列相等。",
    "",
    "自由斜率模型的标准化 capacity-oriented 系数（ELSA / CHARLS / HRS）为：",
    f"- cognition：{coeff('partial_metric_free',1,'cognition'):.3f} / {coeff('partial_metric_free',2,'cognition'):.3f} / {coeff('partial_metric_free',3,'cognition'):.3f}；",
    f"- locomotion：{coeff('partial_metric_free',1,'locomotion'):.3f} / {coeff('partial_metric_free',2,'locomotion'):.3f} / {coeff('partial_metric_free',3,'locomotion'):.3f}；",
    f"- grip/vitality：{coeff('partial_metric_free',1,'grip_vitality'):.3f} / {coeff('partial_metric_free',2,'grip_vitality'):.3f} / {coeff('partial_metric_free',3,'grip_vitality'):.3f}；",
    f"- psychological：{coeff('partial_metric_free',1,'psychological'):.3f} / {coeff('partial_metric_free',2,'psychological'):.3f} / {coeff('partial_metric_free',3,'psychological'):.3f}。",
    "",
    "解释上，locomotion 是目前最清晰的可迁移结构信号：三队列均为负，说明更好的行动能力对应较低 FI；CHARLS 的幅度较小。cognition、grip/vitality 和 psychological 的符号或幅度存在队列差异，宜作为异质性结果而不是共同效应。",
    "",
    "## 3. FI 规则敏感性",
    "",
    f"locomotion 的标准化系数在主 FI / complete-8 / omit-self-rated-health 三个版本下分别为：ELSA {sens('fi_primary',1):.3f}、{sens('fi_complete8',1):.3f}、{sens('fi_without_srh',1):.3f}；CHARLS {sens('fi_primary',2):.3f}、{sens('fi_complete8',2):.3f}、{sens('fi_without_srh',2):.3f}；HRS {sens('fi_primary',3):.3f}、{sens('fi_complete8',3):.3f}、{sens('fi_without_srh',3):.3f}。",
    "三种 FI 定义下方向一致，支持 locomotion 结果不是由实际分母或 self-rated health 单一组件造成；但 FI 仍是短版、outcome-disjoint comparator，不能写成与 IC 完全独立的金标准 frailty。",
    "",
    "## 4. SHARE 外部结构验证",
    "",
    f"SHARE wave 6 采用预先限定的 hierarchical mobility composite 后，locomotion–FI Spearman ρ = {f(share_loc,'spearman'):.3f}（n={int(float(share_loc['n_pairwise'])):,}）；四域+FI 联合可用率为 {int(float(share_joint['n_valid'])):,}（{f(share_joint,'pct_valid'):.1f}%）。cognition、grip/vitality 和 psychological 与 FI 的方向同样为负。",
    "简单 mobility sum 敏感性与 hierarchical composite 几乎一致，因此 SHARE 支持 locomotion–FI 负向关系的外部结构方向。原始两项 mobility CFA 的 Heywood 问题仍需在论文中保留，SHARE composite 不等价于跨队列潜变量 scalar invariance。",
    "",
    "## 5. 可投稿的主张边界",
    "",
    "可以主张：跨队列四域 IC 的 configural 结构较稳定；完全 metric/scalar 可比性不足，但预先限定的 partial metric 模型可用于结构系数比较；locomotion–FI 关系方向跨队列稳定、幅度异质；其他域体现队列特异的测量/结构差异；outcome-disjoint FI 和 SHARE 外部验证共同构成 IC–frailty 区分效度审计。",
    "",
    "不能主张：首次发现 IC 与 frailty 相关；建立了跨国通用 IC 总分；完全测量等值性；跨队列潜在均值差异；斜率差异就是文化效应；outcome-disjoint FI 已经完全消除了构念重叠。",
    "",
    "查新边界仍以 `literature/novelty_boundary_memo.md` 和 `tables/novelty_evidence_matrix.csv` 为准。Stage 15–17 的正式输出分别见 `tables/stage15_raw_invariance_fit.csv`、`tables/stage15b_partial_scalar_fit.csv`、`tables/stage16_cross_cohort_partial_metric_sem_coefficients.csv` 和 `tables/stage17_cross_cohort_fi_sensitivity_coefficients.csv`；SHARE 外部验证见 `tables/stage20_share_external_spearman.csv`。",
]
(OUT / "stage15_17_measurement_structural_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
# Compact machine-readable index for manuscript tables and later plotting.
key_rows = []
for r in fit15:
    if r.get("model") in {"configural", "metric", "metric_partial_domain", "scalar_full", "scalar_partial_domain"}:
        key_rows.append({"stage": "15", "model": r.get("model"), "outcome": "measurement", "group": "all", "cfi": r.get("cfi"), "rmsea": r.get("rmsea"), "srmr": r.get("srmr"), "post_check": r.get("post_check")})
for r in fit16:
    key_rows.append({"stage": "16", "model": r.get("model"), "outcome": "fi_primary", "group": "all", "cfi": r.get("cfi"), "rmsea": r.get("rmsea"), "srmr": r.get("srmr"), "post_check": r.get("post_check")})
for r in fit17:
    key_rows.append({"stage": "17", "model": "partial_metric_free", "outcome": r.get("outcome"), "group": "all", "cfi": r.get("cfi"), "rmsea": r.get("rmsea"), "srmr": r.get("srmr"), "post_check": r.get("post_check")})
with (TAB / "stage15_17_key_results.csv").open("w", encoding="utf-8-sig", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=["stage", "model", "outcome", "group", "cfi", "rmsea", "srmr", "post_check"])
    writer.writeheader(); writer.writerows(key_rows)
print("Wrote stage15_17_measurement_structural_gate_report.md")
