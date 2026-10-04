#!/usr/bin/env Rscript
# Scheme A CFA template. This file defines, checks, and prints model syntax;
# it does not read or export person-level data by default.

suppressPackageStartupMessages({
  if (!requireNamespace("lavaan", quietly = TRUE)) stop("Install lavaan first")
  library(lavaan)
})

OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")

MODEL_SPEC <- list(
  ELSA = list(
    cognition = c("imrc", "dlrc", "orient"),
    locomotion = c("loc_room", "loc_100m"),
    grip_vitality = c("grip_dom_1", "grip_dom_2", "grip_dom_3", "grip_non_1", "grip_non_2", "grip_non_3"),
    psychological = c("psych_dep", "psych_effort", "psych_sleep", "psych_happy", "psych_lonely", "psych_enjoy", "psych_sad", "psych_going")
  ),
  CHARLS = list(
    cognition = c("imrc", "dlrc", "orient", "ser7"),
    locomotion = c("loc_100m", "loc_1km"),
    grip_vitality = c("grip_left_1", "grip_left_2", "grip_right_1", "grip_right_2"),
    psychological = c("psych_dep", "psych_effort", "psych_sleep", "psych_happy", "psych_lonely", "psych_bother", "psych_going", "psych_mind", "psych_hope", "psych_fear")
  ),
  HRS = list(
    cognition = c("imrc", "dlrc", "ser7", "bwc20"),
    locomotion = c("loc_room", "loc_1block", "loc_sevblocks"),
    grip_vitality = c("grip_left", "grip_right", "grip_max_sensitivity"),
    psychological = c("psych_dep", "psych_effort", "psych_sleep", "psych_happy", "psych_lonely", "psych_sad", "psych_going", "psych_enjoy")
  ),
  SHARE = list(
    cognition = c("orientation_total", "verbal_fluency", "immediate_recall_alt", "delayed_recall_alt"),
    locomotion = c("loc_room", "loc_100m"),
    grip_vitality = c("grip_left", "grip_right", "grip_max_sensitivity"),
    psychological = c("psych_dep", "psych_pessimism", "psych_sleep", "psych_enjoy")
  )
)

make_domain_syntax <- function(spec) {
  stopifnot(length(spec) >= 2)
  paste(vapply(names(spec), function(domain) {
    sprintf("%s =~ %s", domain, paste(spec[[domain]], collapse = " + "))
  }, character(1)), collapse = "\n")
}

make_four_domain_syntax <- function(spec, general = FALSE) {
  syntax <- make_domain_syntax(spec)
  if (general) {
    syntax <- paste(syntax, "IC_general =~ cognition + locomotion + grip_vitality + psychological", sep = "\n")
  } else {
    syntax <- paste(syntax, "cognition ~~ locomotion + grip_vitality + psychological\nlocomotion ~~ grip_vitality + psychological\ngrip_vitality ~~ psychological", sep = "\n")
  }
  syntax
}

fit_scheme_a_cfa <- function(data, spec, ordered = character(), estimator = "WLSMV") {
  # The caller must supply a cleaned, cohort-specific data frame. This helper
  # intentionally does not recode survey special values or write data files.
  needed <- unique(unlist(spec, use.names = FALSE))
  missing <- setdiff(needed, names(data))
  if (length(missing)) stop("Missing model indicators: ", paste(missing, collapse = ", "))
  fit <- lavaan::cfa(make_four_domain_syntax(spec), data = data,
                     estimator = estimator, ordered = intersect(ordered, names(data)),
                     std.lv = TRUE, missing = if (estimator == "WLSMV") "pairwise" else "fiml")
  fit
}

if (identical(environment(), globalenv())) {
  cat("lavaan version:", as.character(packageVersion("lavaan")), "\n")
  for (cohort in names(MODEL_SPEC)) {
    cat("\n---", cohort, "---\n", make_four_domain_syntax(MODEL_SPEC[[cohort]]), "\n")
  }
  cat("\nTemplate check completed; no person-level data were read or exported.\n")
}
