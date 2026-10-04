# Blind analysis-code bundle

This bundle contains the executable analysis and reproducibility materials for the study: cohort harmonisation and data audits, measurement and FI validation, primary models, sensitivity analyses, external validation, mechanism extensions, and figure generation. It contains no manuscript-drafting scripts, claim registries, literature-search scripts, author metadata, or individual-level cohort data.

The cohort microdata are not redistributed. Scripts that read cohort files use configurable environment variables and require authorised access to ELSA, CHARLS, HRS and SHARE. Author-machine paths have been replaced with placeholders. Configure these variables locally before execution:

- `IC_FRAILTY_STATISTICAL_MODELING`: authorised data and statistical-model inputs;
- `IC_FRAILTY_ANALYSIS_ROOT`: analysis workspace and aggregate outputs;
- `IC_FRAILTY_CODE_ROOT`: this code directory;
- `IC_FRAILTY_V2_ROOT`, `IC_FRAILTY_SOURCE_METADATA` and `IC_FRAILTY_DATA_ROOT`: cohort-specific inputs where required.

The supplied scripts write aggregate summaries only. They do not redistribute individual-level records. Output files already included here are aggregate analysis results, run metadata, or reproducibility specifications; they are not manuscript text.

The revision folder `08_revision_methodological_robustness_20261004` contains the aggregate-only robustness analyses: nested incremental modelling, continuous-change interaction, optimism-corrected AUC/Brier comparison, survey-weight sensitivity, FI-content sensitivity, and the exploratory low-baseline-FI subgroup analysis. The low-FI analysis uses baseline FI <0.125 as a descriptive band and reports the global state-by-FI-band interaction test.

The figure-generation scripts create the main and supplementary figures from aggregate values. They do not read or redistribute individual-level data.
