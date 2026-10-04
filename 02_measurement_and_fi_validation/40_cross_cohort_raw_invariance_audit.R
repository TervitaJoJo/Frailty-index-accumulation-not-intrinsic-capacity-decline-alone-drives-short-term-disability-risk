#!/usr/bin/env Rscript
# Stage 15: raw-indicator multi-group CFA invariance audit.
#
# This is a bounded measurement audit for the primary ELSA/CHARLS/HRS anchor
# waves. It uses a deliberately small set of cross-cohort analogue indicators
# so that the same observed variables can be fitted in a single multi-group
# model. It does not export person-level data or scores.

suppressPackageStartupMessages({
  library(haven)
  library(lavaan)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

# Reuse the audited source preparation and coding rules. Objects `elsa`,
# `charls`, and `hrs` are created in memory by this script and are never saved.
source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local = .GlobalEnv)

max_na <- function(...) {
  z <- cbind(...)
  out <- apply(z, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE))
  as.numeric(out)
}

common_frame <- function(dat, cohort) {
  if (cohort == "ELSA") {
    cog_imrc <- dat$imrc; cog_dlrc <- dat$dlrc; cog_exec <- dat$orient
    loc_room <- ifelse(is.na(dat$loc_1), NA_real_, 1 - dat$loc_1)
    loc_100m <- ifelse(is.na(dat$loc_2), NA_real_, 1 - dat$loc_2)
    grip_left <- max_na(dat$grip_1, dat$grip_2)
    grip_right <- max_na(dat$grip_3, dat$grip_4)
  } else if (cohort == "CHARLS") {
    cog_imrc <- dat$imrc; cog_dlrc <- dat$dlrc; cog_exec <- dat$orient
    loc_room <- ifelse(is.na(dat$loc_1), NA_real_, 1 - dat$loc_1)
    loc_100m <- ifelse(is.na(dat$loc_2), NA_real_, 1 - dat$loc_2)
    grip_left <- max_na(dat$grip_1, dat$grip_2)
    grip_right <- max_na(dat$grip_3, dat$grip_4)
  } else if (cohort == "HRS") {
    cog_imrc <- dat$cog_imrc; cog_dlrc <- dat$cog_dlrc
    # The HRS executive score is capped at 4 to align its observed range with
    # the ELSA/CHARLS orientation score used for this audit.
    cog_exec <- pmin(dat$cog_ser7, 4)
    loc_room <- ifelse(is.na(dat$loc_1), NA_real_, 1 - as.numeric(dat$loc_1 > 0))
    loc_100m <- ifelse(is.na(dat$loc_2), NA_real_, 1 - as.numeric(dat$loc_2 > 0))
    grip_left <- dat$grip_1; grip_right <- dat$grip_2
  } else stop("unknown cohort")

  # The first four psychological indicators are conceptually shared across
  # the three surveys. Convert CHARLS four-level responses to a common
  # symptom-present threshold (>=3); ELSA/HRS are already binary. Higher
  # values point in the same item direction as the Stage 8 CFA.
  psych_binary <- function(x) {
    x <- as.numeric(x)
    if (cohort == "CHARLS") {
      ifelse(is.na(x), NA_real_, as.numeric(x >= 3))
    } else {
      ifelse(is.na(x), NA_real_, as.numeric(x >= 1))
    }
  }
  out <- data.frame(
    cog_imrc = as.numeric(cog_imrc),
    cog_dlrc = as.numeric(cog_dlrc),
    cog_exec = as.numeric(cog_exec),
    loc_room = as.numeric(loc_room),
    loc_100m = as.numeric(loc_100m),
    grip_left = as.numeric(grip_left),
    grip_right = as.numeric(grip_right),
    psych_dep = psych_binary(dat$psych_1),
    psych_effort = psych_binary(dat$psych_2),
    psych_sleep = psych_binary(dat$psych_3),
    psych_happy = psych_binary(dat$psych_4),
    cohort = cohort,
    stringsAsFactors = FALSE
  )
  out
}

harm <- do.call(rbind, list(
  common_frame(elsa, "ELSA"),
  common_frame(charls, "CHARLS"),
  common_frame(hrs, "HRS")
))
harm$cohort <- factor(harm$cohort, levels = c("ELSA", "CHARLS", "HRS"))

items <- c("cog_imrc", "cog_dlrc", "cog_exec", "loc_room", "loc_100m",
           "grip_left", "grip_right", "psych_dep", "psych_effort",
           "psych_sleep", "psych_happy")
ordered_items <- c("loc_room", "loc_100m", "psych_dep", "psych_effort",
                   "psych_sleep", "psych_happy")

# Convert only binary mobility/psychological indicators to ordered categories;
# cognition counts and grip remain continuous in this cross-cohort audit.
for (v in ordered_items) harm[[v]] <- as.ordered(harm[[v]])

syntax <- paste(
  "cognition =~ cog_imrc + cog_dlrc + cog_exec",
  "locomotion =~ loc_room + loc_100m",
  "grip_vitality =~ grip_left + grip_right",
  "psychological =~ psych_dep + psych_effort + psych_sleep + psych_happy",
  "cognition ~~ locomotion + grip_vitality + psychological",
  "locomotion ~~ grip_vitality + psychological",
  "grip_vitality ~~ psychological",
  sep = "\n"
)

fit_safe <- function(equal = character(0), partial = character(0)) {
  tryCatch(
    lavaan::cfa(syntax, data = harm, group = "cohort", ordered = ordered_items,
                estimator = "WLSMV", parameterization = "theta", std.lv = TRUE,
                meanstructure = TRUE, missing = "pairwise", group.equal = equal,
                group.partial = partial),
    error = function(e) structure(list(error = conditionMessage(e)), class = "fit_error")
  )
}

partial_core <- c("cognition=~cog_exec", "grip_vitality=~grip_right")
partial_domain <- c(partial_core, "locomotion=~loc_100m", "psychological=~psych_happy")
models <- list(
  configural = fit_safe(),
  metric = fit_safe(c("loadings")),
  metric_partial_core = fit_safe(c("loadings"), partial_core),
  metric_partial_domain = fit_safe(c("loadings"), partial_domain),
  # For mixed continuous/ordered indicators, threshold-only and metric models
  # have the same df under lavaan's mean-structure identification. Keep the
  # threshold candidate for parameter-label auditing, but use the scalar model
  # (thresholds + continuous intercepts) for the formal next gate.
  threshold = fit_safe(c("loadings", "thresholds")),
  threshold_partial_domain = fit_safe(c("loadings", "thresholds"), partial_domain),
  scalar_full = fit_safe(c("loadings", "thresholds", "intercepts")),
  scalar_partial_domain = fit_safe(c("loadings", "thresholds", "intercepts"), partial_domain)
)

fit_row <- function(label, fit) {
  if (inherits(fit, "fit_error")) {
    return(data.frame(model = label, converged = FALSE, post_check = FALSE,
                      error = fit$error, stringsAsFactors = FALSE))
  }
  fm <- tryCatch(lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr")),
                 error = function(e) setNames(rep(NA_real_, 6), c("chisq", "df", "cfi", "tli", "rmsea", "srmr")))
  data.frame(model = label,
             converged = lavaan::lavInspect(fit, "converged"),
             post_check = tryCatch(lavaan::lavInspect(fit, "post.check"), error = function(e) NA),
             t(as.data.frame(fm)), error = NA_character_, check.names = FALSE)
}

fits <- do.call(rbind, Map(fit_row, names(models), models))

parameter_rows <- lapply(names(models), function(label) {
  fit <- models[[label]]
  if (inherits(fit, "fit_error")) return(NULL)
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE)
  keep <- (pe$op == "=~" & pe$lhs %in% c("cognition", "locomotion", "grip_vitality", "psychological")) |
    (pe$op %in% c("|", "~1") & pe$lhs %in% ordered_items)
  if (!any(keep)) return(NULL)
  out <- pe[keep, c("lhs", "op", "rhs", "group", "est", "se", "pvalue", "std.all")]
  out$model <- label
  out
})
parameters <- do.call(rbind, parameter_rows)

safe_lrt <- function(a, b, label) {
  if (inherits(a, "fit_error") || inherits(b, "fit_error"))
    return(data.frame(comparison = label, error = "one model failed", stringsAsFactors = FALSE))
  z <- tryCatch(lavaan::lavTestLRT(a, b), error = function(e) NULL)
  if (is.null(z)) return(data.frame(comparison = label, error = "lavTestLRT failed", stringsAsFactors = FALSE))
  zz <- as.data.frame(z); zz$model <- rownames(zz); zz$comparison <- label; rownames(zz) <- NULL; zz
}

lrt <- rbind(
  safe_lrt(models$configural, models$metric, "configural_vs_metric"),
  safe_lrt(models$metric, models$metric_partial_core, "metric_vs_metric_partial_core"),
  safe_lrt(models$metric_partial_core, models$metric_partial_domain, "metric_partial_core_vs_domain"),
  safe_lrt(models$metric_partial_domain, models$threshold_partial_domain, "metric_partial_domain_vs_threshold_partial_domain_zero_df_audit"),
  safe_lrt(models$threshold_partial_domain, models$scalar_partial_domain, "threshold_partial_domain_vs_scalar_partial_domain"),
  safe_lrt(models$metric_partial_domain, models$scalar_partial_domain, "metric_partial_domain_vs_scalar_partial_domain")
)

# Explicitly document whether threshold labels are shared across groups. This
# guards against confusing the zero-df threshold-only comparison with a failed
# implementation of group.equal = "thresholds".
constraint_audit <- do.call(rbind, lapply(names(models), function(label) {
  fit <- models[[label]]
  if (inherits(fit, "fit_error")) return(NULL)
  pt <- lavaan::parTable(fit)
  th <- pt[pt$op == "|" & pt$lhs %in% ordered_items, c("lhs", "group", "free", "label", "plabel")]
  data.frame(model = label,
             n_threshold_rows = nrow(th),
             n_unique_nonempty_labels = length(unique(th$label[nchar(th$label) > 0])),
             threshold_labels_shared = if (nrow(th) == 0) NA else all(tapply(th$label, th$lhs, function(z) length(unique(z[nchar(z) > 0])) <= 1)),
             threshold_df = lavaan::fitMeasures(fit, "df"),
             npar = lavaan::fitMeasures(fit, "npar"),
             stringsAsFactors = FALSE)
}))

coverage <- do.call(rbind, lapply(levels(harm$cohort), function(g) {
  d <- harm[harm$cohort == g, items, drop = FALSE]
  data.frame(cohort = g, n = nrow(d),
             indicator = items,
             n_valid = vapply(d, function(x) sum(!is.na(x)), integer(1)),
             pct_valid = vapply(d, function(x) 100 * mean(!is.na(x)), numeric(1)),
             stringsAsFactors = FALSE)
}))

group_n <- as.data.frame(table(harm$cohort), stringsAsFactors = FALSE)
names(group_n) <- c("cohort", "n")

write.csv(fits, file.path(TAB, "stage15_raw_invariance_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(parameters, file.path(TAB, "stage15_raw_invariance_parameters.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(lrt, file.path(TAB, "stage15_raw_invariance_lrt.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(constraint_audit, file.path(TAB, "stage15_raw_invariance_constraint_audit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(coverage, file.path(TAB, "stage15_raw_invariance_coverage.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(group_n, file.path(TAB, "stage15_raw_invariance_group_n.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(syntax, file.path(OUT, "stage15_raw_invariance_syntax.txt"))

memo <- c(
  "# Stage 15 raw-indicator cross-cohort invariance audit", "",
  "This is an exploratory multi-group CFA audit across ELSA wave 6, CHARLS wave 3, and HRS wave 10 non-proxy respondents. It is a measurement gate and does not estimate cross-cohort latent means.", "",
  "To make a single multi-group model identifiable, the audit uses cross-cohort analogue indicators: immediate recall, delayed recall and an executive/orientation marker for cognition; room and 100-metre mobility for locomotion; left/right maximum grip for grip/vitality; and four common CES-D concepts for psychological capacity. The HRS serial-7 executive score is capped at four to align the observed range with orientation scores. CHARLS psychological responses are dichotomized at >=3 while ELSA/HRS binary items are retained as binary. These choices are transparent approximations and must be reported as an audit model, not as proof that the source instruments are identical.", "",
  "The primary comparison is configural -> metric -> prespecified partial metric -> partial threshold -> partial scalar (thresholds for ordered indicators plus intercepts for continuous indicators). With mixed continuous/ordered indicators, lavaan's threshold-only model can have the same degrees of freedom as the metric model because of mean-structure identification; this is recorded as a parameter-label audit and is not interpreted as a threshold LRT. Ordered indicators are fitted with WLSMV and theta parameterization; grip indicators remain continuous. Fit change, item coverage and parameter instability are interpreted jointly. If metric or scalar constraints fail, the next step is targeted partial invariance based on prespecified domain/item differences, followed by alignment only if the partial model is clinically interpretable.", "",
  "Outputs: `tables/stage15_raw_invariance_fit.csv`, `tables/stage15_raw_invariance_parameters.csv`, `tables/stage15_raw_invariance_lrt.csv`, `tables/stage15_raw_invariance_constraint_audit.csv`, `tables/stage15_raw_invariance_coverage.csv`, and `tables/stage15_raw_invariance_group_n.csv`. No person-level file is written."
)
writeLines(memo, file.path(OUT, "stage15_raw_invariance_memo.md"))
writeLines(c(
  paste0("timestamp=", format(Sys.time(), tz = "UTC")),
  paste0("R=", R.version.string), paste0("lavaan=", as.character(packageVersion("lavaan"))),
  "design=exploratory raw-indicator multi-group CFA; ELSA wave 6, CHARLS wave 3, HRS wave 10 non-proxy",
  paste0("ordered_items=", paste(ordered_items, collapse = ",")),
  "no_person_level_output=true"
), file.path(OUT, "stage15_raw_invariance_run_info.txt"))
cat("Wrote Stage 15 raw-indicator invariance audit outputs.\n")
