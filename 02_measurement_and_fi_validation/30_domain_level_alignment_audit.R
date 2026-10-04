#!/usr/bin/env Rscript
# Stage 9 exploratory domain-level alignment audit.
#
# This script first sources the Stage 8 in-memory CFA pipeline, then fits a
# one-factor multi-group model to the four capacity-oriented domain scores.
# These domain scores are used only in memory and are not exported. Because
# lavaan factor scores are centered within cohort when the first-order CFA is
# fit with std.lv=TRUE, scalar/intercept results are calibration diagnostics;
# they must not be interpreted as cross-country latent-mean estimates.

suppressPackageStartupMessages({
  library(lavaan)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

# Stage 8 defines cohort-specific preparation, CFA, and FI functions and
# stores the in-memory factor scores in `results`. It writes only aggregates.
source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local = .GlobalEnv)

domains <- c("cognition", "locomotion", "grip_vitality", "psychological")

score_frames <- lapply(names(results), function(cohort) {
  s <- as.data.frame(results[[cohort]]$scores)
  s <- s[, domains, drop = FALSE]
  s$dataset <- cohort
  s
})
domain_scores <- do.call(rbind, score_frames)
domain_scores$dataset <- factor(domain_scores$dataset, levels = names(results))
domain_scores <- domain_scores[rowSums(!is.na(domain_scores[, domains, drop = FALSE])) > 0, , drop = FALSE]

alignment_syntax <- paste("general_ic =~", paste(domains, collapse = " + "))

fit_group <- function(equal = character(0), partial = character(0)) {
  lavaan::cfa(alignment_syntax, data = domain_scores, group = "dataset",
              estimator = "MLR", missing = "fiml", meanstructure = TRUE,
              std.lv = TRUE, group.equal = equal,
              group.partial = partial)
}

cfg <- fit_group()
met <- fit_group("loadings")
sca <- fit_group(c("loadings", "intercepts"))

metric_partial <- fit_group("loadings", "general_ic=~psychological")
scalar_partial <- fit_group(c("loadings", "intercepts"),
                            c("general_ic=~psychological", "cognition~1"))

model_list <- list(configural = cfg, metric = met, scalar = sca,
                   metric_partial_psych_loading = metric_partial,
                   scalar_partial_psych_loading_cognition_intercept = scalar_partial)

fit_table <- do.call(rbind, lapply(names(model_list), function(label) {
  f <- model_list[[label]]
  fm <- lavaan::fitMeasures(f, c("chisq", "df", "cfi", "tli", "rmsea", "srmr", "aic", "bic"))
  data.frame(model = label, converged = lavaan::lavInspect(f, "converged"),
             post_check = lavaan::lavInspect(f, "post.check"),
             t(as.data.frame(fm)), check.names = FALSE)
}))

extract_parameters <- function(label, f) {
  pe <- lavaan::parameterEstimates(f, standardized = TRUE)
  keep <- pe$op %in% c("=~", "~1", "~~") &
    ((pe$op == "=~" & pe$lhs == "general_ic") |
     (pe$op == "~1" & pe$lhs %in% domains) |
     (pe$op == "~~" & pe$lhs %in% domains & pe$lhs == pe$rhs))
  out <- pe[keep, c("lhs", "op", "rhs", "group", "est", "se", "pvalue", "std.all")]
  out$model <- label
  out$cohort <- ifelse(out$group == 1, levels(domain_scores$dataset)[1],
                       ifelse(out$group == 2, levels(domain_scores$dataset)[2], levels(domain_scores$dataset)[3]))
  out
}
parameters <- do.call(rbind, Map(extract_parameters, names(model_list), model_list))

safe_lrt <- function(a, b, label) {
  z <- tryCatch(lavaan::lavTestLRT(a, b), error = function(e) NULL)
  if (is.null(z)) return(data.frame(comparison = label, error = "lavTestLRT failed", stringsAsFactors = FALSE))
  zz <- as.data.frame(z)
  zz$model <- rownames(zz)
  zz$comparison <- label
  rownames(zz) <- NULL
  zz
}
lrt <- rbind(safe_lrt(cfg, met, "configural_vs_metric"),
             safe_lrt(met, sca, "metric_vs_scalar"),
             safe_lrt(met, metric_partial, "metric_vs_metric_partial"),
             safe_lrt(metric_partial, scalar_partial, "metric_partial_vs_scalar_partial"))

coverage <- aggregate(domain_scores[, domains], list(dataset = domain_scores$dataset),
                      function(x) mean(!is.na(x)) * 100)
names(coverage)[-1] <- paste0(names(coverage)[-1], "_pct_score_available")
n_by_group <- as.data.frame(table(domain_scores$dataset), stringsAsFactors = FALSE)
names(n_by_group) <- c("dataset", "n_nonempty_domain_score_rows")
coverage <- merge(coverage, n_by_group, by = "dataset", sort = FALSE)

# A simple descriptive scale check: within each cohort, the score origins and
# dispersions are reported only to document why scalar means are not used.
descriptive <- do.call(rbind, lapply(levels(domain_scores$dataset), function(g) {
  d <- domain_scores[domain_scores$dataset == g, domains, drop = FALSE]
  do.call(rbind, lapply(domains, function(v) data.frame(dataset = g, domain = v,
    n = sum(!is.na(d[[v]])), mean = mean(d[[v]], na.rm = TRUE), sd = sd(d[[v]], na.rm = TRUE))))
}))

write.csv(fit_table, file.path(TAB, "stage9_domain_alignment_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(parameters, file.path(TAB, "stage9_domain_alignment_parameters.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(lrt, file.path(TAB, "stage9_domain_alignment_lrt.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(coverage, file.path(TAB, "stage9_domain_alignment_coverage.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(descriptive, file.path(TAB, "stage9_domain_score_descriptives.csv"), row.names = FALSE, fileEncoding = "UTF-8")

memo <- c(
  "# Stage 9 domain-level alignment audit", "",
  "This exploratory audit fits a one-factor multi-group CFA to the four capacity-oriented first-order factor scores from Stage 8. It compares configural, metric, scalar, and limited partial-invariance specifications across ELSA, CHARLS, and HRS.", "",
  "The scores are generated with cohort-specific CFA models using `std.lv=TRUE`; their origins are therefore cohort-centered and their scales inherit cohort-specific indicator sets. Consequently, the scalar/intercept model is a calibration diagnostic for the four-domain pattern, not a valid estimate of country differences in latent IC means. A formal mean comparison requires a calibrated domain-score model with identified intercept/threshold anchors or a model-based alignment procedure that carries first-order measurement error.", "",
  "The primary gate is configural/metric stability of the general IC pattern. Partial models release the psychological loading and, for the scalar diagnostic, the cognition intercept. Fit changes must be interpreted with the domain-score coverage and HRS grip module audit.", "",
  "Outputs: `tables/stage9_domain_alignment_fit.csv`, `tables/stage9_domain_alignment_parameters.csv`, `tables/stage9_domain_alignment_lrt.csv`, `tables/stage9_domain_alignment_coverage.csv`, and `tables/stage9_domain_score_descriptives.csv`. No person-level score file is written."
)
writeLines(memo, file.path(OUT, "stage9_domain_alignment_memo.md"))
writeLines(c(
  paste0("timestamp=", format(Sys.time(), tz = "UTC")), paste0("R=", R.version.string),
  paste0("lavaan=", as.character(packageVersion("lavaan"))),
  "estimator=MLR; missing=FIML; scores=Stage 8 in-memory factor scores",
  "interpretation=metric/configural diagnostic; scalar means not interpreted"
), file.path(OUT, "stage9_domain_alignment_run_info.txt"))
cat("Wrote Stage 9 domain-level alignment aggregate outputs.\n")
