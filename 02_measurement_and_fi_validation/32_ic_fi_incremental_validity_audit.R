#!/usr/bin/env Rscript
# Stage 10 exploratory incremental-validity audit.
# Factor scores and person-level FI are held in memory only; aggregate model
# coefficients, robust SEs, fit summaries, and partial R2 are written.

suppressPackageStartupMessages({
  library(sandwich)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local = .GlobalEnv)

domains <- c("cognition", "locomotion", "grip_vitality", "psychological")
cohort_data <- list(ELSA = elsa, CHARLS = charls, HRS = hrs)

robust_coef <- function(fit, cohort, n_used, outcome) {
  V <- sandwich::vcovHC(fit, type = "HC3")
  b <- stats::coef(fit)
  se <- sqrt(diag(V))
  z <- b / se
  p <- 2 * stats::pnorm(abs(z), lower.tail = FALSE)
  ci_l <- b - stats::qnorm(.975) * se
  ci_u <- b + stats::qnorm(.975) * se
  data.frame(dataset = cohort, outcome = outcome, term = names(b), estimate = unname(b),
             robust_se = unname(se), z = unname(z), p_value = unname(p),
             ci95_low = unname(ci_l), ci95_high = unname(ci_u), n_used = n_used,
             stringsAsFactors = FALSE)
}

fit_cohort <- function(cohort, d) {
  s <- as.data.frame(results[[cohort]]$scores)
  names(s) <- domains
  d2 <- cbind(s, fi_primary = d$fi_primary, fi_complete8 = d$fi_complete8,
              fi_without_srh = d$fi_without_srh)
  d2 <- d2[complete.cases(d2[, c(domains, "fi_primary")]), , drop = FALSE]
  # Standardize domains within cohort for interpretable coefficient comparison.
  d2[domains] <- lapply(d2[domains], function(x) as.numeric(scale(x)))
  fit <- lm(fi_primary ~ cognition + locomotion + grip_vitality + psychological, data = d2)
  fit_reduced <- lapply(domains, function(drop) {
    keep <- setdiff(domains, drop)
    lm(reformulate(keep, response = "fi_primary"), data = d2)
  })
  names(fit_reduced) <- domains
  coef <- robust_coef(fit, cohort, nrow(d2), "fi_primary")
  partial <- do.call(rbind, lapply(names(fit_reduced), function(drop) {
    red <- fit_reduced[[drop]]
    full_sse <- sum(stats::residuals(fit)^2)
    red_sse <- sum(stats::residuals(red)^2)
    data.frame(dataset = cohort, omitted_domain = drop, n_used = nrow(d2),
               full_r2 = summary(fit)$r.squared, reduced_r2 = summary(red)$r.squared,
               partial_r2 = max(0, (red_sse - full_sse) / red_sse),
               delta_r2 = summary(fit)$r.squared - summary(red)$r.squared,
               stringsAsFactors = FALSE)
  }))
  fit_summary <- data.frame(dataset = cohort, n_used = nrow(d2),
                            r2 = summary(fit)$r.squared, adj_r2 = summary(fit)$adj.r.squared,
                            residual_sd = summary(fit)$sigma,
                            stringsAsFactors = FALSE)
  list(coef = coef, partial = partial, summary = fit_summary)
}

cohort_results <- Map(fit_cohort, names(cohort_data), cohort_data)
coef_tab <- do.call(rbind, lapply(cohort_results, `[[`, "coef"))
partial_tab <- do.call(rbind, lapply(cohort_results, `[[`, "partial"))
fit_tab <- do.call(rbind, lapply(cohort_results, `[[`, "summary"))

# Pooled interaction model: domain scores are centered/scaled within cohort,
# while FI remains on its common 0-1 deficit scale.
pooled <- do.call(rbind, lapply(names(cohort_data), function(cohort) {
  d <- cohort_data[[cohort]]
  s <- as.data.frame(results[[cohort]]$scores); names(s) <- domains
  s$dataset <- cohort
  s$fi_primary <- d$fi_primary
  s <- s[complete.cases(s[, c(domains, "fi_primary")]), , drop = FALSE]
  s[domains] <- lapply(s[domains], function(x) as.numeric(scale(x)))
  s
}))
pooled$dataset <- factor(pooled$dataset, levels = names(cohort_data))
pooled_fit <- lm(fi_primary ~ dataset * (cognition + locomotion + grip_vitality + psychological), data = pooled)
pooled_coef <- robust_coef(pooled_fit, "pooled_interaction", nrow(pooled), "fi_primary")
pooled_summary <- data.frame(dataset = "pooled_interaction", n_used = nrow(pooled),
                             r2 = summary(pooled_fit)$r.squared, adj_r2 = summary(pooled_fit)$adj.r.squared,
                             residual_sd = summary(pooled_fit)$sigma)

write.csv(coef_tab, file.path(TAB, "stage10_ic_fi_incremental_coefficients.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(partial_tab, file.path(TAB, "stage10_ic_fi_incremental_partial_r2.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(rbind(fit_tab, pooled_summary), file.path(TAB, "stage10_ic_fi_incremental_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(pooled_coef, file.path(TAB, "stage10_ic_fi_incremental_pooled_interactions.csv"), row.names = FALSE, fileEncoding = "UTF-8")

memo <- c(
  "# Stage 10 IC–FI incremental-validity audit", "",
  "This exploratory audit estimates the independent association of the four capacity-oriented IC domains with the primary outcome-disjoint FI. Domain scores are standardized within cohort; FI remains on the common 0–1 deficit scale. Coefficients use HC3 robust standard errors. No causal interpretation is intended.", "",
  "The main evidence is the sign and magnitude of the simultaneous domain coefficients, the leave-one-domain-out partial R2, and whether the pooled domain-by-cohort interactions are material. These results complement, rather than replace, the pairwise Spearman audit.", "",
  "Because first-order factor scores are estimated with cohort-specific indicator sets and measurement error, this is an incremental-validity screen. A final paper should carry factor-score uncertainty or use a latent-variable outcome model before making formal cross-cohort effect comparisons.", "",
  "Outputs: `tables/stage10_ic_fi_incremental_coefficients.csv`, `tables/stage10_ic_fi_incremental_partial_r2.csv`, `tables/stage10_ic_fi_incremental_fit.csv`, and `tables/stage10_ic_fi_incremental_pooled_interactions.csv`. Individual-level rows and IDs are not written."
)
writeLines(memo, file.path(OUT, "stage10_ic_fi_incremental_validity_memo.md"))
writeLines(c(paste0("timestamp=", format(Sys.time(), tz = "UTC")), paste0("R=", R.version.string),
             paste0("sandwich=", as.character(packageVersion("sandwich"))),
             "model=OLS with HC3 robust SE; domains z-scored within cohort; FI primary 6/8 observed denominator",
             "design=exploratory aggregate audit; scores and person-level rows memory-only"),
           file.path(OUT, "stage10_ic_fi_incremental_run_info.txt"))
cat("Wrote Stage 10 IC-FI incremental-validity aggregate outputs.\n")
