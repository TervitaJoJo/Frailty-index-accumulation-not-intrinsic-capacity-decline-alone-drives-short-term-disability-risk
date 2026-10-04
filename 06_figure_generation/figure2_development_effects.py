# Figure 2: development-cohort disability associations and threshold sensitivity.
# The panels answer two linked questions: where do the four states sit on the
# disability gradient, and how much separation remains when FI accumulation only
# is used as the reference across the prespecified sensitivity rules?
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _utils.plot_utils import COLORS, PALETTE, save_fig, setup_style

setup_style()
OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def forest(ax, labels, est, lo, hi, title, xlabel, xlim, colors):
    y = np.arange(len(labels))[::-1].astype(float)
    for i, yy in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yy - 0.42, yy + 0.42, color=COLORS["bg_box"], zorder=0)
    ax.axvline(1, color=COLORS["ref_line"], ls="--", lw=1.25, zorder=1)
    for e, l, h, yy, col in zip(est, lo, hi, y, colors):
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
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15, ls="--")
    ax.tick_params(axis="both", labelsize=8.3)


fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=False)

# Pooled development-cohort state associations from the primary disability model.
forest(
    axs[0],
    ["Preserved / low", "IC decline only", "FI accumulation only", "Coupled change"],
    [1.00, 1.17, 1.37, 1.46],
    [1.00, 1.01, 1.18, 1.27],
    [1.00, 1.35, 1.60, 1.68],
    "A  Development disability gradient",
    "Adjusted odds ratio vs preserved / low",
    (0.90, 2.10),
    [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]],
)

# Formal incremental contrast, using FI accumulation only as the reference.
forest(
    axs[1],
    ["Primary sign", "Cohort median", "0.2 SD", "0.5 SD*"],
    [1.07, 1.10, 1.14, 1.27],
    [0.94, 0.97, 1.00, 1.09],
    [1.21, 1.26, 1.29, 1.47],
    "B  Incremental IC separation",
    "Adjusted odds ratio: coupled vs FI-only",
    (0.86, 1.85),
    [PALETTE[1], PALETTE[1], PALETTE[2], PALETTE[3]],
)

fig.text(
    0.5,
    0.015,
    "Points show adjusted odds ratios; horizontal lines show 95% CIs. *The 0.5-SD rule is a distribution-based sensitivity benchmark and does not establish a clinical threshold.",
    ha="center",
    va="bottom",
    fontsize=8.1,
    color=COLORS["text"],
)
fig.subplots_adjust(left=0.16, right=0.91, top=0.91, bottom=0.18, wspace=0.48)
for ext in ("png", "tif"):
    fig.savefig(OUT / f"Figure2_development_effects.{ext}", dpi=350, facecolor="white")
save_fig(fig, str(OUT / "Figure2_development_effects.pdf"))

