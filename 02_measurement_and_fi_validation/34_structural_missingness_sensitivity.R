#!/usr/bin/env Rscript
# Stage 12 structural-missingness sensitivity for the latent IC--FI SEM.
# No individual rows or IDs are exported.

suppressPackageStartupMessages(library(lavaan))

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)
source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local = .GlobalEnv)

domains_all <- c("cognition", "locomotion", "grip_vitality", "psychological")
cohort_data <- list(ELSA = elsa, CHARLS = charls, HRS = hrs)

items_for <- function(cohort, domains) {
  x <- list(
    cognition = switch(cohort, ELSA = c("imrc", "dlrc", "orient"), CHARLS = c("imrc", "dlrc", "orient", "verbf"), HRS = c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20")),
    locomotion = switch(cohort, ELSA = c("loc_1", "loc_2"), CHARLS = c("loc_1", "loc_2"), HRS = c("loc_1", "loc_2", "loc_3")),
    grip_vitality = if (cohort == "HRS") c("grip_1", "grip_2") else paste0("grip_", 1:4),
    psychological = switch(cohort, ELSA = paste0("psych_", 1:8), CHARLS = paste0("psych_", 1:10), HRS = paste0("psych_", 1:8))
  )
  x[domains]
}

ordered_for <- function(cohort) {
  switch(cohort,
         ELSA = c("loc_1", "loc_2", paste0("psych_", 1:8)),
         CHARLS = c("loc_1", "loc_2", paste0("psych_", 1:10)),
         HRS = c("loc_1", "loc_2", "loc_3", paste0("psych_", 1:8)))
}

make_syntax <- function(cohort, domains, outcome) {
  im <- items_for(cohort, domains)
  lines <- c(vapply(domains, function(g) paste(g, "=~", paste(im[[g]], collapse = " + ")), character(1)))
  pairs <- combn(domains, 2, simplify = FALSE)
  lines <- c(lines, vapply(pairs, function(p) paste(p[1], "~~", p[2]), character(1)),
              paste(outcome, "~", paste(domains, collapse = " + ")))
  paste(lines, collapse = "\n")
}

fit_sensitivity <- function(cohort, d, label, domains, outcome = "fi_primary") {
  syntax <- make_syntax(cohort, domains, outcome)
  fit <- lavaan::sem(syntax, data = d, ordered = intersect(ordered_for(cohort), names(d)),
                     estimator = "WLSMV", std.lv = TRUE, missing = "pairwise")
  fm <- lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE)
  reg <- pe[pe$op == "~" & pe$lhs == outcome,
            c("lhs", "rhs", "est", "se", "pvalue", "std.all")]
  sign_map <- c(cognition = 1, locomotion = -1, grip_vitality = 1, psychological = -1)
  reg$capacity_oriented_est <- reg$est * unname(sign_map[reg$rhs])
  reg$capacity_oriented_std_all <- reg$std.all * unname(sign_map[reg$rhs])
  reg$dataset <- cohort; reg$sample <- label; reg$outcome <- outcome
  r2all <- lavaan::inspect(fit, "r2")
  r2 <- if (!is.null(r2all[[outcome]])) as.numeric(r2all[[outcome]]) else NA_real_
  fit_row <- data.frame(dataset = cohort, sample = label, outcome = outcome,
                        n = nrow(d), n_outcome_valid = sum(!is.na(d[[outcome]])),
                        n_complete_domain_set = sum(complete.cases(d[, unlist(items_for(cohort, domains)), drop = FALSE])),
                        converged = lavaan::lavInspect(fit, "converged"), post_check = lavaan::lavInspect(fit, "post.check"),
                        r2 = r2, t(as.data.frame(fm)), check.names = FALSE)
  list(reg = reg, fit = fit_row, fit_obj = fit, syntax = syntax)
}

all_reg <- list(); all_fit <- list(); all_bad <- list(); syntaxes <- list()
for (cohort in names(cohort_data)) {
  d <- cohort_data[[cohort]]
  grip_items <- unlist(items_for(cohort, "grip_vitality"), use.names = FALSE)
  full_items <- unlist(items_for(cohort, domains_all), use.names = FALSE)
  variants <- list(
    primary_all_available = list(data = d, domains = domains_all),
    fi_complete8_sample = list(data = d[!is.na(d$fi_complete8), , drop = FALSE], domains = domains_all),
    grip_complete_sample = list(data = d[complete.cases(d[, grip_items, drop = FALSE]), , drop = FALSE], domains = domains_all),
    omit_grip_domain = list(data = d, domains = setdiff(domains_all, "grip_vitality"))
  )
  for (label in names(variants)) {
    z <- fit_sensitivity(cohort, variants[[label]]$data, label, variants[[label]]$domains)
    all_reg[[length(all_reg) + 1L]] <- z$reg
    all_fit[[length(all_fit) + 1L]] <- z$fit
    pe_bad <- lavaan::parameterEstimates(z$fit_obj, standardized = TRUE)
    all_bad[[length(all_bad) + 1L]] <- pe_bad[pe_bad$op == "~~" & pe_bad$lhs == pe_bad$rhs & pe_bad$est < 0,
      c("lhs", "est", "se", "std.all")]
    if (nrow(all_bad[[length(all_bad)]])) {
      all_bad[[length(all_bad)]]$dataset <- cohort
      all_bad[[length(all_bad)]]$sample <- label
    }
    syntaxes[[length(syntaxes) + 1L]] <- c(paste0("# ", cohort, " / ", label), z$syntax, "")
  }
}

reg <- do.call(rbind, all_reg)
fit <- do.call(rbind, all_fit)
write.csv(reg, file.path(TAB, "stage12_structural_missingness_coefficients.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(fit, file.path(TAB, "stage12_structural_missingness_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
bad <- do.call(rbind, all_bad)
if (is.null(bad) || nrow(bad) == 0) bad <- data.frame(dataset = character(), sample = character(), lhs = character(), est = numeric(), se = numeric(), std.all = numeric())
write.csv(bad, file.path(TAB, "stage12_structural_missingness_negative_variances.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(unlist(syntaxes), file.path(OUT, "stage12_structural_missingness_syntax.txt"))

memo <- c(
  "# Stage 12 structural-missingness sensitivity", "",
  "This audit repeats the latent-domain FI SEM under three prespecified sample/model perturbations: (1) restrict to complete 8-component FI; (2) restrict to respondents with all grip/vitality indicators observed; and (3) omit the grip/vitality domain from the structural model. The primary all-available four-domain model is included as the reference.", "",
  "The most important comparison is whether the capacity-oriented locomotion coefficient remains negative and of similar magnitude. HRS grip-module selection is structural, so the complete-grip analysis is a selected-subpopulation sensitivity rather than a replacement estimate. CHARLS complete-FI analysis isolates the effect of its lower FI denominator coverage.", "",
  "All models use WLSMV with pairwise available records and treat FI as a continuous outcome. Results remain exploratory, unweighted, and cross-sectional.", "",
  "Models with a failed lavaan post-check are retained only as diagnostics; their negative residual-variance locations are listed in `tables/stage12_structural_missingness_negative_variances.csv`.", "",
  "Outputs: `tables/stage12_structural_missingness_coefficients.csv`, `tables/stage12_structural_missingness_fit.csv`, `tables/stage12_structural_missingness_negative_variances.csv`, and `stage12_structural_missingness_syntax.txt`."
)
writeLines(memo, file.path(OUT, "stage12_structural_missingness_memo.md"))
writeLines(c(paste0("timestamp=", format(Sys.time(), tz = "UTC")), paste0("R=", R.version.string),
             paste0("lavaan=", as.character(packageVersion("lavaan"))),
             "estimator=WLSMV; missing=pairwise; FI continuous",
             "variants=primary, complete8 FI, complete grip module, omit grip domain"),
           file.path(OUT, "stage12_structural_missingness_run_info.txt"))
cat("Wrote Stage 12 structural-missingness sensitivity outputs.\n")
