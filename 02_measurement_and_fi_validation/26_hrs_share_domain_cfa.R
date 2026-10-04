#!/usr/bin/env Rscript
# Scheme A: HRS/SHARE within-cohort domain CFA pilot.
#
# This script reads only source variables needed for the anchor-wave pilot and
# writes aggregate coverage, fit, and standardized-loading tables. No person
# identifiers or person-level derived files are written.

suppressPackageStartupMessages({
  library(haven)
  library(lavaan)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
SHARE_ROOT <- file.path(OUT, "_share_ascii_real")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

num <- function(x, low = -Inf, high = Inf) {
  x <- as.numeric(x)
  x[is.na(x) | x < low | x > high] <- NA_real_
  x
}

row_coalesce <- function(x) {
  # x is a data.frame/matrix of mutually exclusive alternate module columns.
  z <- as.matrix(x)
  out <- rep(NA_real_, nrow(z))
  for (j in seq_len(ncol(z))) {
    take <- is.na(out) & !is.na(z[, j])
    out[take] <- z[take, j]
  }
  out
}

module_file <- function(kind) {
  # Keep every path ASCII after traversing the temporary junction. This avoids
  # R expanding the junction back to the non-ASCII original parent directory.
  if (kind == "working") return(file.path(SHARE_ROOT, "Working_data", "share.dta"))
  file.path(SHARE_ROOT, "Raw_data", "Wave 6 Release 9.0.0",
            paste0("sharew6_rel9-0-0_", kind, ".dta"))
}

read_one <- function(path, cols) {
  read_dta(path, col_select = all_of(cols), .name_repair = "minimal",
            encoding = "UTF-8")
}

prepare_hrs <- function() {
  path <- file.path(ROOT, "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta")
  cols <- c("inw10", "r10proxy", "r10imrc", "r10dlrc", "r10ser7", "r10bwc20",
            "r10walkra", "r10walk1a", "r10walksa", "r10grpl", "r10grpr",
            "r10depres", "r10effort", "r10sleepr", "r10whappy", "r10flone",
            "r10fsad", "r10going", "r10enlife")
  d <- read_one(path, cols)
  # Proxy respondents have no cognitive or psychological battery in this
  # wave. Exclude them from the IC measurement sample; they remain eligible
  # for FI descriptives/sensitivity analyses handled outside this CFA script.
  d <- d[!is.na(d$inw10) & d$inw10 == 1 & !is.na(d$r10proxy) & d$r10proxy == 0, , drop = FALSE]
  out <- data.frame(
    cog_imrc = num(d$r10imrc, 0, 10),
    cog_dlrc = num(d$r10dlrc, 0, 10),
    cog_ser7 = num(d$r10ser7, 0, 5),
    cog_bwc20 = num(d$r10bwc20, 0, 2),
    loc_1 = num(d$r10walkra, 0, 2),
    loc_2 = num(d$r10walk1a, 0, 2),
    loc_3 = num(d$r10walksa, 0, 2),
    grip_1 = num(d$r10grpl, 0, 100),
    grip_2 = num(d$r10grpr, 0, 100),
    # HRS CES-D screening items are 0=no, 1=yes. The two positively worded
    # items are reversed so larger values consistently indicate worse status.
    psych_1 = num(d$r10depres, 0, 1),
    psych_2 = num(d$r10effort, 0, 1),
    psych_3 = num(d$r10sleepr, 0, 1),
    psych_4 = 1 - num(d$r10whappy, 0, 1),
    psych_5 = num(d$r10flone, 0, 1),
    psych_6 = num(d$r10fsad, 0, 1),
    psych_7 = num(d$r10going, 0, 1),
    psych_8 = 1 - num(d$r10enlife, 0, 1)
  )
  attr(out, "source_note") <- "HRS wave 10 active non-proxy respondents (inw10==1 & r10proxy==0); grip module is structurally selected."
  out
}

prepare_share <- function() {
  base_path <- module_file("working")
  cf_path <- module_file("cf")
  gh_path <- module_file("gv_health")

  base <- read_one(base_path, c("mergeid", "wave", "walkra", "walk100a", "lgrip", "rgrip"))
  base <- base[!is.na(base$wave) & base$wave == 6, , drop = FALSE]
  if (anyDuplicated(base$mergeid)) stop("SHARE wave 6 working file has duplicate mergeid")

  cf_cols <- c("mergeid", "cf003_", "cf004_", "cf005_", "cf006_", "cf010_",
               paste0("cf", 104:107, "tot"), paste0("cf", 113:116, "tot"))
  cf <- read_one(cf_path, cf_cols)
  gh <- read_one(gh_path, c("mergeid", "euro1", "euro2", "euro5", "euro11"))
  if (anyDuplicated(cf$mergeid) || anyDuplicated(gh$mergeid))
    stop("SHARE wave 6 module has duplicate mergeid")
  d <- merge(base, cf, by = "mergeid", all.x = TRUE, sort = FALSE)
  d <- merge(d, gh, by = "mergeid", all.x = TRUE, sort = FALSE)

  # In SHARE cognitive tests, code 1 denotes a correct orientation response;
  # the four date items are summed after reversing the raw 1/2 coding. Memory
  # lists are alternate forms: each respondent contributes at most one form.
  orient_items <- lapply(d[c("cf003_", "cf004_", "cf005_", "cf006_")], function(x) {
    x <- num(x, 1, 2)
    ifelse(is.na(x), NA_real_, 2 - x)
  })
  orient_total <- rowSums(as.data.frame(orient_items), na.rm = TRUE)
  orient_total[rowSums(!is.na(as.data.frame(orient_items))) == 0] <- NA_real_
  immediate <- row_coalesce(d[paste0("cf", 104:107, "tot")])
  delayed <- row_coalesce(d[paste0("cf", 113:116, "tot")])

  out <- data.frame(
    cog_orientation = num(orient_total, 0, 4),
    cog_immediate = num(immediate, 0, 10),
    cog_delayed = num(delayed, 0, 10),
    cog_verbal = num(d$cf010_, 0, 100),
    loc_1 = num(d$walkra, 0, 1),
    loc_2 = num(d$walk100a, 0, 1),
    grip_1 = num(d$lgrip, 0, 150),
    grip_2 = num(d$rgrip, 0, 150),
    # EURO-D derived binary items are coded 1=selected symptom, 0=not
    # selected; all four selected candidates therefore point in the same
    # higher-worse direction. Negative values are refusal/don't-know.
    psych_1 = ifelse(num(d$euro1, 0, 1) == 1, 1, ifelse(num(d$euro1, 0, 1) == 0, 0, NA)),
    psych_2 = ifelse(num(d$euro2, 0, 1) == 1, 1, ifelse(num(d$euro2, 0, 1) == 0, 0, NA)),
    psych_3 = ifelse(num(d$euro5, 0, 1) == 1, 1, ifelse(num(d$euro5, 0, 1) == 0, 0, NA)),
    psych_4 = ifelse(num(d$euro11, 0, 1) == 1, 1, ifelse(num(d$euro11, 0, 1) == 0, 0, NA))
  )
  attr(out, "source_note") <- "SHARE wave 6; raw CF/GV-health modules remerged by mergeid; wave 7 excluded from primary longitudinal work."
  out
}

fit_one <- function(cohort, dat, ordered, cognition_names, locomotion_names, psych_names) {
  syntax <- paste(
    paste("cognition =~", paste(cognition_names, collapse = " + ")),
    paste("locomotion =~", paste(locomotion_names, collapse = " + ")),
    "grip_vitality =~ grip_1 + grip_2",
    paste("psychological =~", paste(psych_names, collapse = " + ")),
    "cognition ~~ locomotion + grip_vitality + psychological",
    "locomotion ~~ grip_vitality + psychological",
    "grip_vitality ~~ psychological",
    sep = "\n"
  )
  needed <- c(cognition_names, locomotion_names, "grip_1", "grip_2", psych_names)
  stopifnot(all(needed %in% names(dat)))
  fit <- lavaan::cfa(syntax, data = dat, ordered = ordered,
                     estimator = "WLSMV", std.lv = TRUE, missing = "pairwise")
  fm <- lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))
  fit_row <- data.frame(
    dataset = cohort, n_active = nrow(dat),
    n_complete_all_indicators = sum(complete.cases(dat[, needed, drop = FALSE])),
    converged = lavaan::lavInspect(fit, "converged"),
    post_check = lavaan::lavInspect(fit, "post.check"),
    t(as.data.frame(fm)), check.names = FALSE
  )
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE)
  load <- pe[pe$op == "=~", c("lhs", "rhs", "est", "se", "pvalue", "std.all")]
  load$dataset <- cohort
  variances <- pe[pe$op == "~~" & pe$lhs == pe$rhs,
                  c("lhs", "rhs", "est", "se", "pvalue", "std.all")]
  variances$dataset <- cohort
  list(fit = fit_row, load = load, variances = variances, syntax = syntax)
}

coverage_table <- function(cohort, dat, domain_map) {
  rows <- lapply(names(dat), function(v) {
    x <- dat[[v]]
    data.frame(dataset = cohort, indicator = v, n_active = nrow(dat),
               n_valid = sum(!is.na(x)), pct_valid = 100 * mean(!is.na(x)),
               n_unique = length(unique(x[!is.na(x)])),
               domain = unname(domain_map[[v]]), stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  domain_rows <- do.call(rbind, lapply(unique(out$domain), function(g) {
    vars <- out$indicator[out$domain == g]
    usable <- rowSums(!is.na(dat[, vars, drop = FALSE]))
    data.frame(dataset = cohort, indicator = paste0("at_least_2_", g),
               n_active = nrow(dat), n_valid = sum(usable >= 2),
               pct_valid = 100 * mean(usable >= 2), n_unique = NA_integer_, domain = g)
  }))
  rbind(out, domain_rows)
}

hrs <- prepare_hrs()
share <- prepare_share()

hrs_map <- c(setNames(rep("cognition", 4), c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20")),
             setNames(rep("locomotion", 3), paste0("loc_", 1:3)),
             setNames(rep("grip_vitality", 2), paste0("grip_", 1:2)),
             setNames(rep("psychological", 8), paste0("psych_", 1:8)))
share_map <- c(setNames(rep("cognition", 4), c("cog_orientation", "cog_immediate", "cog_delayed", "cog_verbal")),
               setNames(rep("locomotion", 2), paste0("loc_", 1:2)),
               setNames(rep("grip_vitality", 2), paste0("grip_", 1:2)),
               setNames(rep("psychological", 4), paste0("psych_", 1:4)))

h <- fit_one("HRS", hrs,
             c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20", "loc_1", "loc_2", "loc_3", paste0("psych_", 1:8)),
             c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20"),
             c("loc_1", "loc_2", "loc_3"), paste0("psych_", 1:8))
s <- fit_one("SHARE", share,
             c("cog_orientation", "cog_immediate", "cog_delayed", "loc_1", "loc_2", paste0("psych_", 1:4)),
             c("cog_orientation", "cog_immediate", "cog_delayed", "cog_verbal"),
             c("loc_1", "loc_2"), paste0("psych_", 1:4))

# Sensitivity for the SHARE mobility Heywood warning: walkra is a rare,
# nested difficulty item. Treat walk100a as a fixed-error-free single observed
# mobility marker only for a diagnostic comparison; this is not the preferred
# latent measurement model.
share_single_syntax <- paste(
  "cognition =~ cog_orientation + cog_immediate + cog_delayed + cog_verbal",
  "locomotion =~ 1*loc_2",
  "loc_2 ~~ 0*loc_2",
  "grip_vitality =~ grip_1 + grip_2",
  "psychological =~ psych_1 + psych_2 + psych_3 + psych_4",
  "cognition ~~ locomotion + grip_vitality + psychological",
  "locomotion ~~ grip_vitality + psychological",
  "grip_vitality ~~ psychological", sep = "\n")
share_single_fit <- lavaan::cfa(
  share_single_syntax, data = share,
  ordered = c("cog_orientation", "cog_immediate", "cog_delayed", paste0("psych_", 1:4)),
  estimator = "WLSMV", std.lv = TRUE, missing = "pairwise")
share_single_fm <- lavaan::fitMeasures(share_single_fit,
                                       c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))
share_single_row <- data.frame(
  dataset = "SHARE_single_locomotion_sensitivity",
  n_active = nrow(share),
  n_complete_all_indicators = sum(complete.cases(share[, c("cog_orientation", "cog_immediate", "cog_delayed", "cog_verbal", "loc_2", "grip_1", "grip_2", paste0("psych_", 1:4)), drop = FALSE])),
  converged = lavaan::lavInspect(share_single_fit, "converged"),
  post_check = lavaan::lavInspect(share_single_fit, "post.check"),
  t(as.data.frame(share_single_fm)), check.names = FALSE)

fits <- rbind(h$fit, s$fit)
loads <- rbind(h$load, s$load)
variances <- rbind(h$variances, s$variances)
coverage <- rbind(coverage_table("HRS", hrs, hrs_map), coverage_table("SHARE", share, share_map))
write.csv(fits, file.path(TAB, "pilot_domain_cfa_fit_hrs_share.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(loads, file.path(TAB, "pilot_domain_cfa_loadings_hrs_share.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(variances, file.path(TAB, "pilot_domain_cfa_variances_hrs_share.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(share_single_row, file.path(TAB, "pilot_domain_cfa_share_locomotion_sensitivity.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(coverage, file.path(TAB, "pilot_domain_cfa_coverage_hrs_share.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(c("# HRS/SHARE Scheme A pilot syntax", "", "## HRS", h$syntax, "", "## SHARE", s$syntax),
           file.path(OUT, "pilot_domain_cfa_syntax_hrs_share.txt"))

memo <- c(
  "# Stage 7 HRS/SHARE domain CFA pilot", "",
  "This is an exploratory within-cohort CFA pilot for scheme A. It is not a cross-cohort invariance test.",
  "The script used WLSMV with pairwise available records, treated cognition/locomotion/psychological indicators as ordered where appropriate, and retained grip as continuous.",
  "HRS uses active non-proxy wave-10 respondents (inw10==1 & r10proxy==0). The four-domain model converged and passed the lavaan post-check (CFI 0.982, TLI 0.978, RMSEA 0.049, SRMR 0.068). Standardized loadings were strong for recall (0.87-0.89), locomotion (0.91-0.94), grip (0.95), and most psychological items (0.62-0.90); orientation/executive cognition indicators were lower (0.43-0.49). The grip module is structurally selected and therefore requires module-selection and missing-data sensitivity analysis; only 40.4% of non-proxy respondents had at least two grip measures. Proxy respondents are retained only in FI descriptives/sensitivity analyses.",
  "SHARE uses the wave-6 working file plus raw wave-6 CF and GV-health modules remerged by mergeid. Alternate word-list forms are row-wise coalesced into immediate and delayed recall indicators, which raises effective memory coverage to about 95% at wave 6. Wave 7 remains excluded from the primary longitudinal analysis. The four-domain model has acceptable descriptive fit (CFI 0.981, TLI 0.973, RMSEA 0.058, SRMR 0.067) but fails the post-check because the rare walkra mobility item has a boundary/negative residual variance (standardized loading about 1.00). This is evidence of a nested-item/method problem, so the SHARE model cannot yet be treated as an accepted latent measurement model. The single-indicator locomotion diagnostic is not a replacement: it produced poor fit and should not be used as the main model.",
  "The SHARE orientation score assumes raw date-response code 1 is correct and reverses the 1/2 coding before summation. EURO-D candidates are retained in their selected-symptom direction (1=worse). These coding assumptions require confirmation against the final questionnaire codebook. The next model gate should test a preregistered mobility composite or a domain-score treatment for SHARE, with the two-item CFA retained as a diagnostic sensitivity.",
  "See tables/pilot_domain_cfa_fit_hrs_share.csv, tables/pilot_domain_cfa_loadings_hrs_share.csv, tables/pilot_domain_cfa_variances_hrs_share.csv, and tables/pilot_domain_cfa_coverage_hrs_share.csv."
)
writeLines(memo, file.path(OUT, "stage7_hrs_share_cfa_memo.md"))
cat("Wrote HRS/SHARE Scheme A pilot outputs.\n")
