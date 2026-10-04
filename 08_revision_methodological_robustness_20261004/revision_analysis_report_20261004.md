# Methodological robustness analyses

Run date: 2026-10-04  
Seed: 20261004  
Development cohorts: ELSA, CHARLS and HRS  
Analysis outputs: aggregate counts, model estimates and bootstrap summaries only

## Prespecified role of the analyses

The primary estimand remains the disability-risk ordering across four directional states. The secondary incremental estimand is the adjusted coupled-versus-FI-accumulation-only contrast under the primary sign rule. The analyses in this folder were used to determine whether the IC component adds information beyond FI accumulation and whether that conclusion is sensitive to model parameterisation, survey weights or FI content.

## Results

### Nested incremental state model

The model with FI accumulation, age, sex, education, baseline FI and cohort was compared with a model that additionally included IC decline and the IC-decline-by-FI-accumulation interaction. The complete analytic sample contained 18,525 participants and 2,097 disability events. The likelihood-ratio statistic was 5.63 on 2 degrees of freedom (P=0.060). The implied coupled-versus-FI-only contrast was OR=1.067 (95% CI 0.938-1.214; P=0.325). This provides no strong evidence that the IC terms materially improve the primary FI-accumulation model.

### Continuous change and interaction

Within-cohort standardised IC and FI changes were entered as continuous predictors with their product term. Per-SD IC change was associated with lower disability odds (OR=0.867, 95% CI 0.825-0.912; P<0.001), whereas per-SD FI change was associated with higher odds (OR=1.189, 95% CI 1.133-1.247; P<0.001). The interaction was close to the null (OR=0.985, 95% CI 0.944-1.027; P=0.467). The continuous analysis therefore supports separate main effects but not a strong multiplicative synergy.

### Optimism-corrected incremental performance

For the same nested models, the apparent AUC difference was 0.00037 and the apparent Brier-score difference was -0.000059. In 400 cohort-stratified bootstrap samples, the optimism-corrected differences were -0.000024 for AUC (bootstrap 95% interval -0.00032 to 0.00045) and -0.000032 for Brier score (bootstrap 95% interval -0.000064 to -0.000007). These values indicate negligible incremental discrimination and only a very small change in overall probability error.

### Survey-weight sensitivity

Normalised cohort-specific response weights were used as frequency weights in a pooled logistic model. This sensitivity retained the coupled-versus-preserved ordering (OR=1.377, 95% CI 1.189-1.594; P<0.001; n=19,782; weighted events=1,851). Because a common PSU/stratum specification was not available across all three cohorts, this is a weight-sensitivity analysis rather than a full complex-survey variance estimate, and it is not interpreted as eliminating selection bias.

### FI content sensitivity

The eight-component FI was replaced by a disease-only FI that excluded self-rated health and BMI, requiring at least five of six disease components or all six components. The coupled-versus-alternative-FI-only estimates were directionally compatible with the primary result: ELSA OR=1.117 (95% CI 0.726-1.719) and CHARLS OR=1.148 (0.890-1.480) under the five-of-six rule; the complete-six rule in CHARLS gave OR=1.263 (0.966-1.651). HRS gave OR=1.067 (0.793-1.434). These estimates do not provide evidence for a large, content-independent incremental IC effect.

## Interpretation and limits

The combined evidence supports a restrained interpretation: FI accumulation carries the main risk signal, the coupled state identifies the highest observed disability-risk group, and the incremental IC contrast is modest under the primary rule and more apparent under larger distribution-based thresholds. The threshold analyses remain sensitivity analyses and do not establish a minimally important clinical change. The survey analysis is limited by the absence of harmonised PSU and stratum variables. The performance analysis is internal and optimism corrected; it is not a clinical prediction-model validation. The endpoint-window estimand also does not replace time-to-event competing-risk analysis when exact event dates are available.
