#!/usr/bin/env Rscript
# Stage 11 exploratory latent-domain structural audit.
#
# Fits the four-domain measurement model and regresses the observed
# outcome-disjoint FI on the latent domains within each cohort. This carries
# domain measurement error into the structural coefficients and complements
# the Stage 10 factor-score regression. Person-level data stay in memory.

suppressPackageStartupMessages({
  library(lavaan)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local = .GlobalEnv)

domains <- c("cognition", "locomotion", "grip_vitality", "psychological")
cohort_data <- list(ELSA = elsa, CHARLS = charls, HRS = hrs)

make_syntax <- function(cohort, outcome) {
  grip_items <- if (cohort == "HRS") c("grip_1", "grip_2") else paste0("grip_", 1:4)
  cognition_items <- switch(cohort,
    ELSA = c("imrc", "dlrc", "orient"),
    CHARLS = c("imrc", "dlrc", "orient", "verbf"),
    HRS = c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20"))
  locomotion_items <- switch(cohort,
    ELSA = c("loc_1", "loc_2"),
    CHARLS = c("loc_1", "loc_2"),
    HRS = c("loc_1", "loc_2", "loc_3"))
  psych_items <- switch(cohort,
    ELSA = paste0("psych_", 1:8),
    CHARLS = paste0("psych_", 1:10),
    HRS = paste0("psych_", 1:8))
  paste(
    paste("cognition =~", paste(cognition_items, collapse = " + ")),
    paste("locomotion =~", paste(locomotion_items, collapse = " + ")),
    paste("grip_vitality =~", paste(grip_items, collapse = " + ")),
    paste("psychological =~", paste(psych_items, collapse = " + ")),
    "cognition ~~ locomotion + grip_vitality + psychological",
    "locomotion ~~ grip_vitality + psychological",
    "grip_vitality ~~ psychological",
    paste(outcome, "~ cognition + locomotion + grip_vitality + psychological"),
    sep = "\n")
}

ordered_for <- function(cohort) {
  switch(cohort,
    ELSA = c("loc_1", "loc_2", paste0("psych_", 1:8)),
    CHARLS = c("loc_1", "loc_2", paste0("psych_", 1:10)),
    HRS = c("loc_1", "loc_2", "loc_3", paste0("psych_", 1:8)))
}

fit_one <- function(cohort, d, outcome) {
  syntax <- make_syntax(cohort, outcome)
  fit <- lavaan::sem(syntax, data = d, ordered = ordered_for(cohort),
                     estimator = "WLSMV", std.lv = TRUE, missing = "pairwise")
  fm <- lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE)
  reg <- pe[pe$op == "~" & pe$lhs == outcome,
            c("lhs", "rhs", "est", "se", "pvalue", "std.all")]
  reg$dataset <- cohort
  reg$outcome <- outcome
  capacity_sign <- c(cognition = 1, locomotion = -1,
                     grip_vitality = 1, psychological = -1)
  reg$orientation <- ifelse(reg$rhs %in% c("locomotion", "psychological"),
                            "higher_capacity_after_sign_reversal",
                            "higher_capacity_as_fitted")
  reg$capacity_oriented_est <- reg$est * unname(capacity_sign[reg$rhs])
  reg$capacity_oriented_std_all <- reg$std.all * unname(capacity_sign[reg$rhs])
  rsq <- lavaan::inspect(fit, "r2")
  r2 <- if (!is.null(rsq[[outcome]])) as.numeric(rsq[[outcome]]) else NA_real_
  fit_row <- data.frame(dataset = cohort, outcome = outcome,
                        n = nrow(d), n_outcome_valid = sum(!is.na(d[[outcome]])),
                        converged = lavaan::lavInspect(fit, "converged"),
                        post_check = lavaan::lavInspect(fit, "post.check"),
                        r2 = r2, t(as.data.frame(fm)), check.names = FALSE)
  list(fit = fit, fit_row = fit_row, reg = reg, syntax = syntax)
}

outcomes <- c("fi_primary", "fi_complete8", "fi_without_srh")
fits <- list()
for (cohort in names(cohort_data)) {
  for (outcome in outcomes) {
    fits[[paste(cohort, outcome, sep = "_")]] <- fit_one(cohort, cohort_data[[cohort]], outcome)
  }
}

fit_table <- do.call(rbind, lapply(fits, `[[`, "fit_row"))
reg_table <- do.call(rbind, lapply(fits, `[[`, "reg"))
syntax_lines <- unlist(lapply(names(fits), function(nm) c(paste0("# ", nm), fits[[nm]]$syntax, "")))

write.csv(fit_table, file.path(TAB, "stage11_latent_domain_fi_sem_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(reg_table, file.path(TAB, "stage11_latent_domain_fi_sem_coefficients.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(syntax_lines, file.path(OUT, "stage11_latent_domain_fi_sem_syntax.txt"))

memo <- c(
  "# Stage 11 latent-domain IC–FI structural audit", "",
  "This exploratory SEM fits the previously audited four-domain CFA within each cohort and regresses the observed outcome-disjoint FI on the latent domains. WLSMV with pairwise available records is used for ordered indicators; FI is treated as a continuous bounded outcome. The coefficients therefore account for first-order domain measurement error more directly than Stage 10 factor-score regression.", "",
  "The primary outcome is `fi_primary` (minimum 6/8 observed components, observed denominator, BMI <18.5 or >=30). `fi_complete8` and `fi_without_srh` are sensitivity outcomes. The fitted locomotion and psychological factors retain their raw higher-worse direction; the coefficient table therefore includes `capacity_oriented_est` and `capacity_oriented_std_all`, which reverse those two domains so that negative values consistently indicate higher capacity with lower FI. Coefficients are standardized (`std.all`) and should be interpreted as cross-sectional structural associations, not causal effects.", "",
  "Use the fit table to check convergence/post-check and the coefficient table to compare domain directions across FI definitions. If HRS grip or CHARLS FI missingness changes the coefficient pattern, retain that limitation in the main analysis rather than treating the factor-score model as definitive.", "",
  "Outputs: `tables/stage11_latent_domain_fi_sem_fit.csv`, `tables/stage11_latent_domain_fi_sem_coefficients.csv`, and `stage11_latent_domain_fi_sem_syntax.txt`. No individual-level scores or IDs are written."
)
writeLines(memo, file.path(OUT, "stage11_latent_domain_fi_sem_memo.md"))
writeLines(c(paste0("timestamp=", format(Sys.time(), tz = "UTC")), paste0("R=", R.version.string),
             paste0("lavaan=", as.character(packageVersion("lavaan"))),
             "estimator=WLSMV; missing=pairwise; FI treated as continuous outcome",
             "design=exploratory within-cohort latent-domain SEM; person-level data memory-only"),
           file.path(OUT, "stage11_latent_domain_fi_sem_run_info.txt"))
cat("Wrote Stage 11 latent-domain FI SEM aggregate outputs.\n")
