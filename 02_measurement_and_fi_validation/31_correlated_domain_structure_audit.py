"""Summarize cross-cohort stability of the correlated four-domain IC structure.

This reads only the aggregate Stage 8 latent-correlation table. It writes no
individual-level data. The one-factor Stage 9 alignment model produced
Heywood cases, so this audit treats the four domains as correlated constructs
and reports pairwise sign/range stability instead of forcing a general IC.
"""
from pathlib import Path
import pandas as pd

OUT = Path(r"PATH_TO_IC_FRAILTY")
TAB = OUT / "tables"
src = TAB / "stage8_domain_latent_correlations.csv"
df = pd.read_csv(src)
domains = ["cognition", "locomotion", "grip_vitality", "psychological"]
df = df[df.domain_a.isin(domains) & df.domain_b.isin(domains) & (df.domain_a != df.domain_b)].copy()
df["a_order"] = df["domain_a"].map({x: i for i, x in enumerate(domains)})
df["b_order"] = df["domain_b"].map({x: i for i, x in enumerate(domains)})
df = df[df.a_order < df.b_order].copy()
rows = []
for (a, b), g in df.groupby(["domain_a", "domain_b"], sort=False):
    x = g["latent_r"].astype(float)
    rows.append({
        "domain_a": a,
        "domain_b": b,
        "n_cohorts": int(x.notna().sum()),
        "min_r": float(x.min()),
        "max_r": float(x.max()),
        "range_r": float(x.max() - x.min()),
        "mean_r": float(x.mean()),
        "all_same_sign": bool((x > 0).all() or (x < 0).all()),
        "interpretation": "stable sign; magnitude differs" if ((x > 0).all() or (x < 0).all()) else "sign differs; alignment concern",
    })
out = pd.DataFrame(rows)
out.to_csv(TAB / "stage9_correlated_domain_structure.csv", index=False, encoding="utf-8-sig")

memo = [
    "# Stage 9 correlated-domain structure audit",
    "",
    "The four first-order IC domains are retained as correlated constructs. The Stage 9 one-factor domain alignment model was not accepted because it produced negative locomotion residual variances and standardized loadings above one in the configural model, with further fit deterioration under metric constraints.",
    "",
    "This table summarizes whether the signs and approximate magnitudes of pairwise latent-domain correlations are stable across ELSA, CHARLS, and HRS. Stable signs support a common multidimensional IC structure; ranges still reflect cohort-specific measurement, item composition, and missingness.",
    "",
    "The result is a structural-comparability diagnostic. It does not establish scalar invariance or permit direct comparison of latent means.",
]
(OUT / "stage9_correlated_domain_structure_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
print("Wrote correlated-domain structure audit.")
