"""Generate the eight final supplementary figures for the Scientific Reports package.

All plotted values are aggregate results already reported in the manuscript or in the
submission code bundle.  No individual-level cohort data are read by this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#173F5F",
    "blue": "#20639B",
    "teal": "#3CAEA3",
    "orange": "#F6A21A",
    "red": "#C84C4C",
    "ink": "#263238",
    "grid": "#D9E2E8",
    "light": "#F4F7F9",
    "green": "#3B7A57",
}
STATE_COLORS = [COLORS["navy"], COLORS["blue"], COLORS["orange"], COLORS["red"]]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.edgecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["ink"],
    "xtick.color": COLORS["ink"],
    "ytick.color": COLORS["ink"],
    "text.color": COLORS["ink"],
    "savefig.facecolor": "white",
})


def finish(fig, name):
    fig.tight_layout()
    for ext in ("png", "tif", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def forest(ax, labels, est, lo, hi, title, xlabel, xlim, colors=None):
    y = np.arange(len(labels))[::-1]
    colors = colors or [COLORS["blue"]] * len(labels)
    for i, yy in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yy - 0.42, yy + 0.42, color=COLORS["light"], zorder=0)
    ax.axvline(1, color="#7F8C8D", linestyle="--", linewidth=1, zorder=1)
    for e, l, h, yy, col in zip(est, lo, hi, y, colors):
        ax.plot([l, h], [yy, yy], color=COLORS["ink"], lw=1.7, zorder=2)
        ax.plot([l, l], [yy - 0.10, yy + 0.10], color=COLORS["ink"], lw=1.0, zorder=2)
        ax.plot([h, h], [yy - 0.10, yy + 0.10], color=COLORS["ink"], lw=1.0, zorder=2)
        ax.plot(e, yy, "o", ms=6.8, color=col, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.015, yy, f"{e:.2f} ({l:.2f}-{h:.2f})", va="center", fontsize=7.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_s1():
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    forest(
        ax,
        ["ELSA", "CHARLS", "HRS", "Random-effects pooled"],
        [1.86, 1.33, 1.38, 1.47],
        [1.37, 1.07, 1.11, 1.22],
        [2.51, 1.67, 1.73, 1.76],
        "S1  Cohort-specific primary disability associations",
        "Adjusted odds ratio: coupled vs preserved/low accumulation",
        (0.9, 2.75),
        [COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["red"]],
    )
    fig.text(0.01, 0.01, "Models adjust for age, sex, education and baseline FI; pooled estimate uses inverse-variance random-effects pooling.", fontsize=7.8)
    finish(fig, "SupplementaryFigureS1_primary_cohort_OR")


def fig_s2():
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.2), sharey=True)
    labels = ["Primary sign", "Cohort median", "0.2 SD", "0.5 SD*"]
    forest(axes[0], labels, [1.07, 1.10, 1.14, 1.27], [0.94, 0.97, 1.00, 1.09], [1.21, 1.26, 1.29, 1.47], "S2A  Development cohorts", "Adjusted OR: coupled vs FI-only", (0.85, 1.62), [COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["red"]])
    forest(axes[1], labels, [1.10, 1.19, 1.19, 1.27], [0.94, 1.03, 1.02, 1.07], [1.29, 1.38, 1.38, 1.50], "S2B  SHARE", "Country-adjusted OR: coupled vs FI-only", (0.85, 1.62), [COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["red"]])
    fig.text(0.5, 0.01, "*Distribution-based sensitivity rule; no rule establishes a clinical minimal important change. Primary sign rule remains the principal estimand.", ha="center", fontsize=7.8)
    finish(fig, "SupplementaryFigureS2_threshold_sensitivity")


def fig_s3():
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    labels = ["Preserved / low", "IC decline only", "FI accumulation only", "Coupled"]
    risk = [5.10, 5.20, 9.51, 10.25]
    lo = [3.96, 4.12, 8.06, 8.96]
    hi = [6.55, 6.55, 11.19, 11.69]
    x = np.arange(4)
    ax.bar(x, risk, color=STATE_COLORS, alpha=0.84, width=0.62)
    ax.errorbar(x, risk, yerr=[np.array(risk) - np.array(lo), np.array(hi) - np.array(risk)], fmt="none", ecolor=COLORS["ink"], capsize=3, lw=1.2)
    for xx, rr in zip(x, risk): ax.text(xx, rr + 0.38, f"{rr:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Incident disability risk (%)")
    ax.set_ylim(0, 13)
    ax.set_title("S3  Disability risk in participants with baseline FI <0.125", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.01, 0.01, "n=5,561 with known outcomes and 440 disability events. Bars show endpoint-window proportions; whiskers show Wilson 95% CIs.", fontsize=7.8)
    finish(fig, "SupplementaryFigureS3_low_FI_risk")


def fig_s4():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["Preserved / low", "IC decline only", "FI accumulation only", "Coupled"]
    death = [4.8, 6.6, 4.9, 8.0]
    combined = [14.7, 19.0, 17.7, 21.3]
    for ax, vals, title, ylim in [(axes[0], death, "S4A  Death risk", 10), (axes[1], combined, "S4B  Combined disability/death risk", 25)]:
        x = np.arange(4)
        ax.bar(x, vals, color=STATE_COLORS, alpha=0.84, width=0.62)
        for xx, vv in zip(x, vals): ax.text(xx, vv + ylim * 0.03, f"{vv:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Endpoint-window risk (%)")
        ax.set_ylim(0, ylim); ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.01, "CHARLS/HRS only; death and disability were treated as mutually exclusive terminal-window outcomes.", ha="center", fontsize=7.8)
    finish(fig, "SupplementaryFigureS4_mortality_combined_risk")


def fig_s5():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    labels = ["Held-out ELSA", "Held-out CHARLS", "Held-out HRS", "Optimism-corrected"]
    auc_base = [0.6740, 0.6413, 0.6669, np.nan]
    auc_aug = [0.6853, 0.6460, 0.6646, np.nan]
    auc_delta = [0.0113, 0.0047, -0.0023, -0.000024]
    brier_delta = [-0.00057, -0.00064, 0.00083, -0.000032]
    x = np.arange(3)
    w = 0.34
    axes[0].bar(x - w/2, auc_delta[:3], width=w, color=COLORS["blue"], label="Leave-one-cohort-out")
    axes[0].axhline(0, color=COLORS["ink"], lw=0.9)
    axes[0].bar(3, auc_delta[3], width=w, color=COLORS["red"], label="400-resample corrected")
    axes[0].set_xticks(np.arange(4)); axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("AUC difference")
    axes[0].set_title("S5A  Incremental discrimination", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].bar(x, brier_delta[:3], width=0.58, color=COLORS["blue"])
    axes[1].axhline(0, color=COLORS["ink"], lw=0.9)
    axes[1].bar(3, brier_delta[3], width=0.58, color=COLORS["red"])
    axes[1].set_xticks(np.arange(4)); axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Brier-score difference")
    axes[1].set_title("S5B  Probability error", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.7)
    for ax in axes: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.01, "The corrected estimates come from the nested FI model versus FI plus IC terms; bootstrap intervals are reported in Supplementary Table S9.", ha="center", fontsize=7.8)
    finish(fig, "SupplementaryFigureS5_prediction_performance")


def fig_s6():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fit_names = ["Configural", "Partial metric", "Partial scalar"]
    cfi = [0.988, 0.986, 0.970]
    rmsea = [0.038, 0.039, 0.054]
    axes[0].plot(fit_names, cfi, "o-", color=COLORS["blue"], lw=2, ms=6)
    axes[0].axhline(0.95, color=COLORS["ink"], ls="--", lw=0.9)
    axes[0].set_ylim(0.94, 1.00); axes[0].set_ylabel("CFI"); axes[0].set_title("S6A  Measurement fit", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=COLORS["grid"], ls="--", lw=0.7)
    axes[1].bar(fit_names, rmsea, color=[COLORS["blue"], COLORS["teal"], COLORS["red"]], alpha=0.85)
    axes[1].axhline(0.05, color=COLORS["ink"], ls="--", lw=0.9)
    axes[1].set_ylim(0, 0.07); axes[1].set_ylabel("RMSEA"); axes[1].set_title("S6B  Scalar constraint check", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color=COLORS["grid"], ls="--", lw=0.7)
    for ax in axes: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(axis="x", rotation=25)
    fig.text(0.5, 0.01, "Configural and partial metric models were supported; partial scalar constraints did not pass the post-check, so latent means were not compared.", ha="center", fontsize=7.8)
    finish(fig, "SupplementaryFigureS6_measurement_audit")


def fig_s7():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = ["CHARLS log-CRP", "CHARLS HbA1c", "ELSA log-CRP", "ELSA HbA1c"]
    est = [0.214, 0.185, 0.170, 0.164]
    lo = [0.066, 0.072, 0.0, 0.0]
    hi = [0.361, 0.297, 0.0, 0.0]
    y = np.arange(4)[::-1]
    ax.axvline(0, color=COLORS["ink"], ls="--", lw=0.9)
    for i, (yy, ee) in enumerate(zip(y, est)):
        if i % 2 == 0: ax.axhspan(yy - .42, yy + .42, color=COLORS["light"], zorder=0)
        if lo[i] == 0 and hi[i] == 0:
            ax.plot(ee, yy, "o", color=COLORS["teal"], ms=7, label="Directionally concordant" if i == 2 else None)
        else:
            ax.plot([lo[i], hi[i]], [yy, yy], color=COLORS["ink"], lw=1.7)
            ax.plot(ee, yy, "o", color=COLORS["blue"] if i < 2 else COLORS["teal"], ms=7)
        ax.text(0.39, yy, f"{ee:.3f} SD", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Coupled vs preserved/low contrast (standardised biomarker units)")
    ax.set_xlim(-0.04, 0.43); ax.set_title("S7  Baseline inflammatory-metabolic correlates", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], ls="--", lw=0.7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.01, 0.01, "CHARLS whiskers show 95% CIs; ELSA markers show concordant point estimates from response-weighted analyses. Biomarkers are secondary context, not mediation evidence.", fontsize=7.8)
    finish(fig, "SupplementaryFigureS7_biomarker_contrasts")


def fig_s8():
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    labels = ["Primary", "Selection-weighted", "Survey-weighted", "Trajectory-change MI", "FI content sensitivity"]
    est = [1.46, 1.49, 1.38, 1.46, 1.15]
    lo = [1.27, 1.26, 1.19, 1.22, 0.89]
    hi = [1.68, 1.77, 1.59, 1.75, 1.48]
    forest(ax, labels, est, lo, hi, "S8  Robustness of the coupled-state gradient", "OR: coupled vs preserved/low accumulation", (0.85, 2.05), [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["red"]])
    fig.text(0.01, 0.01, "The FI-content row represents the CHARLS five-of-six disease-only sensitivity; cohort-specific definitions and estimates are in Supplementary Table S12.", fontsize=7.8)
    finish(fig, "SupplementaryFigureS8_selection_robustness")


if __name__ == "__main__":
    fig_s1(); fig_s2(); fig_s3(); fig_s4(); fig_s5(); fig_s6(); fig_s7(); fig_s8()
    print(f"Generated final supplementary figures in {OUT}")
