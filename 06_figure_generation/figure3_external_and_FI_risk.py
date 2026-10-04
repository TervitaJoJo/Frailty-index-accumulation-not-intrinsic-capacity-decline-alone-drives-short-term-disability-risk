# Figure 3: SHARE external disability reproducibility and threshold sensitivity.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _utils.plot_utils import COLORS, PALETTE, _lighten, save_fig, setup_style

setup_style()
OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def forest(ax, labels, est, lo, hi, title, xlim):
    y = np.arange(len(labels))[::-1].astype(float)
    for i, yy in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yy - 0.42, yy + 0.42, color=COLORS["bg_box"], zorder=0)
    ax.axvline(1, color=COLORS["ref_line"], ls="--", lw=1.25, zorder=1)
    for i, (e, l, h, yy) in enumerate(zip(est, lo, hi, y)):
        col = [PALETTE[1], PALETTE[1], PALETTE[2], PALETTE[3]][i]
        ax.plot([l, h], [yy, yy], color=COLORS["text"], lw=1.8, zorder=2)
        ax.plot([l, l], [yy - 0.10, yy + 0.10], color=COLORS["text"], lw=1.15, zorder=2)
        ax.plot([h, h], [yy - 0.10, yy + 0.10], color=COLORS["text"], lw=1.15, zorder=2)
        ax.plot(e, yy, "o", ms=7, markerfacecolor=col, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.text(xlim[1] - 0.01, yy, f"{e:.2f} ({l:.2f}-{h:.2f})", ha="right", va="center", fontsize=8.0, color=COLORS["text"], clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLORS["text"])
    ax.set_xlabel("Country-adjusted odds ratio: coupled vs FI-only", fontsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15, ls="--")
    ax.tick_params(axis="both", labelsize=8.3)


fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15]})

# SHARE state-specific risks from the external reproducibility analysis.
states = ["Preserved /\nlow", "IC decline\nonly", "FI accumulation\nonly", "Coupled\nchange"]
values = [6.1, 6.5, 8.0, 8.8]
cols = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
for i, (value, col) in enumerate(zip(values, cols)):
    axs[0].bar(i, value, color=_lighten(col, 0.35), edgecolor=col, lw=1.5, width=0.62, zorder=2)
    axs[0].text(i, value + 0.20, f"{value:.1f}", ha="center", va="bottom", fontsize=8.8, color=COLORS["text"], fontweight="bold")
axs[0].set_xticks(np.arange(4))
axs[0].set_xticklabels(states, fontsize=7.9)
axs[0].set_ylabel("Incident disability risk (%)", fontsize=8.8)
axs[0].set_ylim(0, 10.5)
axs[0].set_title("A  SHARE disability gradient", loc="left", fontsize=11, fontweight="bold", color=COLORS["text"])
axs[0].grid(axis="y", alpha=0.16, ls="--")
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)

# Formal incremental contrast under the same four rules, with country fixed effects.
forest(
    axs[1],
    ["Primary sign", "Cohort median", "0.2 SD", "0.5 SD*"],
    [1.10, 1.19, 1.19, 1.27],
    [0.94, 1.03, 1.02, 1.07],
    [1.29, 1.38, 1.38, 1.50],
    "B  SHARE threshold sensitivity",
    (0.86, 1.85),
)

fig.text(
    0.5,
    0.015,
    "Panel A: SHARE waves 4–5–6, within-cohort standardisation. Panel B: country fixed-effects estimates. *Sensitivity rules support future clinical threshold evaluation.",
    ha="center",
    va="bottom",
    fontsize=8.1,
    color=COLORS["text"],
)
fig.subplots_adjust(left=0.08, right=0.91, top=0.91, bottom=0.18, wspace=0.48)
for ext in ("png", "tif"):
    fig.savefig(OUT / f"Figure3_external_and_FI_risk.{ext}", dpi=350, facecolor="white")
save_fig(fig, str(OUT / "Figure3_external_and_FI_risk.pdf"))

