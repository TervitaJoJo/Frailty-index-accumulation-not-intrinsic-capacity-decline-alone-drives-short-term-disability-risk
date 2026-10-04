#!/usr/bin/env Rscript
# Stage 8 exploratory audit: domain reliability, discriminant validity, and
# aggregate IC--FI associations for the primary ELSA/CHARLS/HRS comparison.
#
# Individual-level factor scores are held in memory only. The script writes
# aggregate tables and an auditable memo; source Stata files remain read-only.

suppressPackageStartupMessages({
  library(haven)
  library(lavaan)
  library(semTools)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

num <- function(x, low = -Inf, high = Inf) {
  x <- as.numeric(x)
  x[is.na(x) | x < low | x > high] <- NA_real_
  x
}

bin01 <- function(x) {
  x <- as.numeric(x)
  out <- rep(NA_real_, length(x))
  out[x %in% c(0, 1)] <- x[x %in% c(0, 1)]
  out
}

fi_score <- function(shlt, bmi, disease, shlt_direction) {
  components <- as.data.frame(lapply(disease, bin01), check.names = FALSE)
  shlt <- num(shlt, 1, 5)
  components$self_rated_health <- if (shlt_direction == "poor_is_low") {
    (5 - shlt) / 4
  } else {
    (shlt - 1) / 4
  }
  bmi <- num(bmi, 0, 200)
  components$bmi <- ifelse(is.na(bmi), NA_real_,
                           as.numeric(bmi < 18.5 | bmi >= 30))
  observed <- rowSums(!is.na(components))
  primary <- rowSums(components, na.rm = TRUE) / observed
  primary[observed < 6] <- NA_real_
  complete8 <- rowSums(components, na.rm = TRUE) / 8
  complete8[observed != 8] <- NA_real_
  without_srh <- components[, setdiff(names(components), "self_rated_health"), drop = FALSE]
  n7 <- rowSums(!is.na(without_srh))
  fi_without_srh <- rowSums(without_srh, na.rm = TRUE) / n7
  fi_without_srh[n7 < 6] <- NA_real_
  list(primary = primary, complete8 = complete8,
       without_srh = fi_without_srh, observed_n = observed,
       components = components)
}

prepare_elsa <- function() {
  work_path <- file.path(ROOT, "ELSA/Working_data/elsa.dta")
  raw_path <- file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_data_v2.dta")
  nurse_path <- file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_nurse_data_v2.dta")
  wcols <- c("idauniqc", "wave", "imrc", "dlrc", "orient", "walkra", "walk100a",
             "shlt", "mbmi", "hibpe", "diabe", "hearte", "stroke", "cancre", "arthre")
  w <- read_dta(work_path, col_select = wcols)
  w <- w[w$wave == 6, , drop = FALSE]
  r <- read_dta(raw_path, col_select = c("idauniq", paste0("PSced", LETTERS[1:8])))
  n <- read_dta(nurse_path, col_select = c("idauniq", "mmgsd1", "mmgsd2", "mmgsd3", "mmgsn1", "mmgsn2", "mmgsn3"))
  w$.__key <- as.character(w$idauniqc)
  r$.__key <- as.character(r$idauniq)
  n$.__key <- as.character(n$idauniq)
  d <- merge(w, r[, c(".__key", paste0("PSced", LETTERS[1:8]))], by = ".__key", all.x = TRUE, sort = FALSE)
  d <- merge(d, n[, c(".__key", "mmgsd1", "mmgsd2", "mmgsd3", "mmgsn1", "mmgsn2", "mmgsn3")], by = ".__key", all.x = TRUE, sort = FALSE)
  out <- data.frame(
    imrc = num(d$imrc, 0, 10), dlrc = num(d$dlrc, 0, 10), orient = num(d$orient, 0, 4),
    loc_1 = num(d$walkra, 0, 1), loc_2 = num(d$walk100a, 0, 1),
    grip_1 = num(d$mmgsd1, 0, 100), grip_2 = num(d$mmgsd2, 0, 100),
    grip_3 = num(d$mmgsn1, 0, 100), grip_4 = num(d$mmgsn2, 0, 100),
    psych_1 = ifelse(num(d$PScedA, 1, 2) == 1, 1, ifelse(num(d$PScedA, 1, 2) == 2, 0, NA)),
    psych_2 = ifelse(num(d$PScedB, 1, 2) == 1, 1, ifelse(num(d$PScedB, 1, 2) == 2, 0, NA)),
    psych_3 = ifelse(num(d$PScedC, 1, 2) == 1, 1, ifelse(num(d$PScedC, 1, 2) == 2, 0, NA)),
    psych_4 = ifelse(num(d$PScedD, 1, 2) == 1, 0, ifelse(num(d$PScedD, 1, 2) == 2, 1, NA)),
    psych_5 = ifelse(num(d$PScedE, 1, 2) == 1, 1, ifelse(num(d$PScedE, 1, 2) == 2, 0, NA)),
    psych_6 = ifelse(num(d$PScedF, 1, 2) == 1, 0, ifelse(num(d$PScedF, 1, 2) == 2, 1, NA)),
    psych_7 = ifelse(num(d$PScedG, 1, 2) == 1, 1, ifelse(num(d$PScedG, 1, 2) == 2, 0, NA)),
    psych_8 = ifelse(num(d$PScedH, 1, 2) == 1, 1, ifelse(num(d$PScedH, 1, 2) == 2, 0, NA)),
    shlt = num(d$shlt, 1, 5), bmi = num(d$mbmi, 0, 200),
    hypertension = bin01(d$hibpe), diabetes = bin01(d$diabe),
    heart_disease = bin01(d$hearte), stroke = bin01(d$stroke),
    cancer = bin01(d$cancre), arthritis = bin01(d$arthre)
  )
  fi <- fi_score(out$shlt, out$bmi,
                 out[c("hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis")],
                 "poor_is_low")
  out$fi_primary <- fi$primary; out$fi_complete8 <- fi$complete8
  out$fi_without_srh <- fi$without_srh; out$fi_observed_n <- fi$observed_n
  out
}

prepare_charls <- function() {
  path <- file.path(ROOT, "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta")
  cols <- c("inw3", "r3imrc", "r3dlrc", "r3orient", "r3ser7", "r3walk100a", "r3walk1kma",
            "r3lgrip1", "r3lgrip2", "r3rgrip1", "r3rgrip2", "r3depresl", "r3effortl",
            "r3sleeprl", "r3whappyl", "r3flonel", "r3botherl", "r3goingl", "r3fhopel",
            "r3mindtsl", "r3fearll", "r3shlt", "r3mbmi", "r3hibpe", "r3diabe", "r3hearte",
            "r3stroke", "r3cancre", "r3arthre")
  d <- read_dta(path, col_select = cols)
  d <- d[d$inw3 == 1, , drop = FALSE]
  out <- data.frame(
    imrc = num(d$r3imrc, 0, 10), dlrc = num(d$r3dlrc, 0, 10), orient = num(d$r3orient, 0, 4), verbf = num(d$r3ser7, 0, 5),
    loc_1 = num(d$r3walk100a, 0, 1), loc_2 = num(d$r3walk1kma, 0, 1),
    grip_1 = num(d$r3lgrip1, 0, 100), grip_2 = num(d$r3lgrip2, 0, 100),
    grip_3 = num(d$r3rgrip1, 0, 100), grip_4 = num(d$r3rgrip2, 0, 100),
    psych_1 = num(d$r3depresl, 1, 4), psych_2 = num(d$r3effortl, 1, 4), psych_3 = num(d$r3sleeprl, 1, 4),
    psych_4 = 5 - num(d$r3whappyl, 1, 4), psych_5 = num(d$r3flonel, 1, 4), psych_6 = num(d$r3botherl, 1, 4),
    psych_7 = num(d$r3goingl, 1, 4), psych_8 = 5 - num(d$r3fhopel, 1, 4), psych_9 = num(d$r3mindtsl, 1, 4), psych_10 = num(d$r3fearll, 1, 4),
    shlt = num(d$r3shlt, 1, 5), bmi = num(d$r3mbmi, 0, 200),
    hypertension = bin01(d$r3hibpe), diabetes = bin01(d$r3diabe), heart_disease = bin01(d$r3hearte),
    stroke = bin01(d$r3stroke), cancer = bin01(d$r3cancre), arthritis = bin01(d$r3arthre)
  )
  fi <- fi_score(out$shlt, out$bmi,
                 out[c("hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis")],
                 "poor_is_high")
  out$fi_primary <- fi$primary; out$fi_complete8 <- fi$complete8
  out$fi_without_srh <- fi$without_srh; out$fi_observed_n <- fi$observed_n
  out
}

prepare_hrs <- function() {
  path <- file.path(ROOT, "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta")
  cols <- c("inw10", "r10proxy", "r10imrc", "r10dlrc", "r10ser7", "r10bwc20", "r10walkra", "r10walk1a", "r10walksa",
            "r10grpl", "r10grpr", "r10depres", "r10effort", "r10sleepr", "r10whappy", "r10flone", "r10fsad", "r10going", "r10enlife",
            "r10shlt", "r10bmi", "r10hibpe", "r10diabe", "r10hearte", "r10stroke", "r10cancre", "r10arthre")
  d <- read_dta(path, col_select = cols)
  d <- d[!is.na(d$inw10) & d$inw10 == 1 & !is.na(d$r10proxy) & d$r10proxy == 0, , drop = FALSE]
  out <- data.frame(
    cog_imrc = num(d$r10imrc, 0, 10), cog_dlrc = num(d$r10dlrc, 0, 10), cog_ser7 = num(d$r10ser7, 0, 5), cog_bwc20 = num(d$r10bwc20, 0, 2),
    loc_1 = num(d$r10walkra, 0, 2), loc_2 = num(d$r10walk1a, 0, 2), loc_3 = num(d$r10walksa, 0, 2),
    grip_1 = num(d$r10grpl, 0, 100), grip_2 = num(d$r10grpr, 0, 100),
    psych_1 = num(d$r10depres, 0, 1), psych_2 = num(d$r10effort, 0, 1), psych_3 = num(d$r10sleepr, 0, 1),
    psych_4 = 1 - num(d$r10whappy, 0, 1), psych_5 = num(d$r10flone, 0, 1), psych_6 = num(d$r10fsad, 0, 1),
    psych_7 = num(d$r10going, 0, 1), psych_8 = 1 - num(d$r10enlife, 0, 1),
    shlt = num(d$r10shlt, 1, 5), bmi = num(d$r10bmi, 0, 200),
    hypertension = bin01(d$r10hibpe), diabetes = bin01(d$r10diabe), heart_disease = bin01(d$r10hearte),
    stroke = bin01(d$r10stroke), cancer = bin01(d$r10cancre), arthritis = bin01(d$r10arthre)
  )
  fi <- fi_score(out$shlt, out$bmi,
                 out[c("hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis")],
                 "poor_is_high")
  out$fi_primary <- fi$primary; out$fi_complete8 <- fi$complete8
  out$fi_without_srh <- fi$without_srh; out$fi_observed_n <- fi$observed_n
  out
}

fit_one <- function(cohort, dat, ordered, cognition_names, locomotion_names, psych_names) {
  syntax <- paste(
    paste("cognition =~", paste(cognition_names, collapse = " + ")),
    paste("locomotion =~", paste(locomotion_names, collapse = " + ")),
    paste("grip_vitality =~", paste(if (cohort == "HRS") c("grip_1", "grip_2") else c("grip_1", "grip_2", "grip_3", "grip_4"), collapse = " + ")),
    paste("psychological =~", paste(psych_names, collapse = " + ")),
    "cognition ~~ locomotion + grip_vitality + psychological",
    "locomotion ~~ grip_vitality + psychological",
    "grip_vitality ~~ psychological", sep = "\n")
  needed <- c(cognition_names, locomotion_names,
              if (cohort == "HRS") c("grip_1", "grip_2") else c("grip_1", "grip_2", "grip_3", "grip_4"),
              psych_names)
  fit <- lavaan::cfa(syntax, data = dat, ordered = ordered, estimator = "WLSMV",
                     std.lv = TRUE, missing = "pairwise")
  fm <- lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))
  fit_row <- data.frame(dataset = cohort, n = nrow(dat), n_complete_all_indicators = sum(complete.cases(dat[, needed, drop = FALSE])),
                        converged = lavaan::lavInspect(fit, "converged"), post_check = lavaan::lavInspect(fit, "post.check"),
                        t(as.data.frame(fm)), check.names = FALSE)
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE)
  load <- pe[pe$op == "=~", c("lhs", "rhs", "est", "se", "pvalue", "std.all")]
  load$dataset <- cohort
  list(fit = fit, fit_row = fit_row, load = load, syntax = syntax, needed = needed)
}

as_named_metric <- function(x) {
  if (is.null(x)) return(numeric())
  if (is.data.frame(x)) {
    if (ncol(x) == 1) x <- x[[1]] else x <- unlist(x, use.names = TRUE)
  }
  nm <- names(x)
  if (is.null(nm)) nm <- character(length(x))
  x <- as.numeric(x)
  names(x) <- nm
  x
}

domain_metrics <- function(cohort, obj, dat, domain_items) {
  fit <- obj$fit
  domains <- names(domain_items)
  out <- list()
  omega <- tryCatch(as_named_metric(semTools::compRelSEM(fit, simplify = TRUE)), error = function(e) numeric())
  ave <- tryCatch(as_named_metric(semTools::AVE(fit)), error = function(e) numeric())
  corlv <- tryCatch(lavaan::lavInspect(fit, "cor.lv"), error = function(e) NULL)
  fs0 <- lavaan::lavPredict(fit, type = "lv", method = "EBM", rel = TRUE)
  fs_rel <- attr(fs0, "rel")
  fs <- as.matrix(fs0)
  if (is.list(fs_rel)) fs_rel <- fs_rel[[1]]
  fs_rel <- as_named_metric(fs_rel)
  dir_sign <- c(cognition = 1, locomotion = -1, grip_vitality = 1, psychological = -1)
  fs_cap <- sweep(fs, 2, dir_sign[colnames(fs)], "*")
  # lavaan can return a numerical placeholder for rows with no observed model
  # indicators. Such rows have no identifiable factor score and are reset to
  # missing before aggregate coverage and correlation summaries.
  empty_case <- rowSums(!is.na(dat[, obj$needed, drop = FALSE])) == 0
  fs_cap[empty_case, ] <- NA_real_
  latent_var <- tryCatch(lavaan::lavInspect(fit, "cov.lv"), error = function(e) NULL)
  for (g in domains) {
    others <- setdiff(domains, g)
    maxcorr <- if (!is.null(corlv) && length(others)) max(abs(corlv[g, others]), na.rm = TRUE) else NA_real_
    av <- if (g %in% names(ave)) ave[[g]] else NA_real_
    om <- if (g %in% names(omega)) omega[[g]] else NA_real_
    rel <- if (g %in% names(fs_rel)) fs_rel[[g]] else NA_real_
    nscore <- sum(!is.na(fs_cap[, g]))
    all_ind_complete <- complete.cases(dat[, domain_items[[g]], drop = FALSE])
    score_complete <- all_ind_complete & !is.na(fs_cap[, g])
    # lavaan does not attach `rel` for WLSMV factor scores. For this ordinal
    # pilot, report a clearly labeled empirical determinacy approximation on
    # the complete-indicator subset: sqrt(var(score) / model-implied var(latent)).
    # It is a diagnostic, not a substitute for a planned latent-score model.
    det_approx <- NA_real_
    if (sum(score_complete) > 2 && !is.null(latent_var) && g %in% rownames(latent_var)) {
      lv <- latent_var[g, g]
      sv <- stats::var(fs_cap[score_complete, g], na.rm = TRUE)
      if (is.finite(lv) && lv > 0 && is.finite(sv)) det_approx <- sqrt(max(0, min(1, sv / lv)))
    }
    out[[length(out) + 1L]] <- data.frame(
      dataset = cohort, domain = g, omega = om, AVE = av,
      sqrt_AVE = ifelse(is.na(av) || av < 0, NA_real_, sqrt(av)),
      max_abs_latent_r_other_domains = maxcorr,
      FL_margin = ifelse(is.na(av) || is.na(maxcorr), NA_real_, sqrt(av) - maxcorr),
      factor_score_reliability = rel, factor_score_determinacy_approx = det_approx,
      n_factor_score = nscore,
      pct_factor_score = 100 * nscore / nrow(dat),
      n_all_indicator_complete = sum(all_ind_complete),
      pct_all_indicator_complete = 100 * mean(all_ind_complete),
      stringsAsFactors = FALSE)
  }
  list(metrics = do.call(rbind, out), factor_scores_capacity = fs_cap, latent_cor = corlv)
}

pairwise_rho <- function(a, b) {
  ok <- !is.na(a) & !is.na(b)
  if (sum(ok) < 3 || length(unique(a[ok])) < 2 || length(unique(b[ok])) < 2) return(c(n = sum(ok), rho = NA_real_))
  c(n = sum(ok), rho = suppressWarnings(cor(a[ok], b[ok], method = "spearman")))
}

association_table <- function(cohort, dat, fs_cap, domain_items) {
  domains <- colnames(fs_cap)
  rows <- list(); missing_rows <- list()
  for (g in domains) {
    for (fi_name in c("fi_primary", "fi_complete8", "fi_without_srh")) {
      z <- pairwise_rho(fs_cap[, g], dat[[fi_name]])
      rows[[length(rows) + 1L]] <- data.frame(dataset = cohort, domain = g,
        score_orientation = "higher_capacity", fi_variant = fi_name, sample = "factor_score_pairwise",
        n_pairwise = z[["n"]], spearman = z[["rho"]], stringsAsFactors = FALSE)
      # Complete indicators within the same domain: a missingness sensitivity,
      # not a substitute for the primary pairwise scoring rule.
      complete_domain <- complete.cases(dat[, domain_items[[g]], drop = FALSE])
      z2 <- pairwise_rho(fs_cap[complete_domain, g], dat[[fi_name]][complete_domain])
      rows[[length(rows) + 1L]] <- data.frame(dataset = cohort, domain = g,
        score_orientation = "higher_capacity", fi_variant = fi_name, sample = "domain_indicator_complete",
        n_pairwise = z2[["n"]], spearman = z2[["rho"]], stringsAsFactors = FALSE)
      if (fi_name == "fi_primary") {
        missing_rows[[length(missing_rows) + 1L]] <- data.frame(dataset = cohort, domain = g,
          n_active = nrow(dat), n_domain_score = sum(!is.na(fs_cap[, g])),
          pct_domain_score = 100 * mean(!is.na(fs_cap[, g])), n_domain_indicator_complete = sum(complete_domain),
          pct_domain_indicator_complete = 100 * mean(complete_domain), n_fi_primary = sum(!is.na(dat$fi_primary)),
          pct_fi_primary = 100 * mean(!is.na(dat$fi_primary)), stringsAsFactors = FALSE)
      }
    }
  }
  list(assoc = do.call(rbind, rows), missing = do.call(rbind, missing_rows))
}

run_one <- function(cohort, dat, ordered, cognition, locomotion, psych, domain_items) {
  obj <- fit_one(cohort, dat, ordered, cognition, locomotion, psych)
  dm <- domain_metrics(cohort, obj, dat, domain_items)
  a <- association_table(cohort, dat, dm$factor_scores_capacity, domain_items)
  latent <- as.data.frame(as.table(dm$latent_cor), stringsAsFactors = FALSE)
  names(latent) <- c("domain_a", "domain_b", "latent_r")
  latent$dataset <- cohort
  list(fit = obj, fit_row = obj$fit_row, metrics = dm$metrics, assoc = a$assoc, missing = a$missing,
       latent = latent, load = obj$load, scores = dm$factor_scores_capacity)
}

elsa <- prepare_elsa()
charls <- prepare_charls()
hrs <- prepare_hrs()

results <- list(
  ELSA = run_one("ELSA", elsa, c("loc_1", "loc_2", paste0("psych_", 1:8)),
                 c("imrc", "dlrc", "orient"), c("loc_1", "loc_2"), paste0("psych_", 1:8),
                 list(cognition = c("imrc", "dlrc", "orient"), locomotion = c("loc_1", "loc_2"),
                      grip_vitality = paste0("grip_", 1:4), psychological = paste0("psych_", 1:8))),
  CHARLS = run_one("CHARLS", charls, c("loc_1", "loc_2", paste0("psych_", 1:10)),
                   c("imrc", "dlrc", "orient", "verbf"), c("loc_1", "loc_2"), paste0("psych_", 1:10),
                   list(cognition = c("imrc", "dlrc", "orient", "verbf"), locomotion = c("loc_1", "loc_2"),
                        grip_vitality = paste0("grip_", 1:4), psychological = paste0("psych_", 1:10))),
  HRS = run_one("HRS", hrs, c("loc_1", "loc_2", "loc_3", paste0("psych_", 1:8)),
                c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20"), c("loc_1", "loc_2", "loc_3"), paste0("psych_", 1:8),
                list(cognition = c("cog_imrc", "cog_dlrc", "cog_ser7", "cog_bwc20"), locomotion = paste0("loc_", 1:3),
                     grip_vitality = c("grip_1", "grip_2"), psychological = paste0("psych_", 1:8)))
)

fits <- do.call(rbind, lapply(results, `[[`, "fit_row"))
metrics <- do.call(rbind, lapply(results, `[[`, "metrics"))
assoc <- do.call(rbind, lapply(results, `[[`, "assoc"))
missing <- do.call(rbind, lapply(results, `[[`, "missing"))
latent <- do.call(rbind, lapply(results, `[[`, "latent"))
loads <- do.call(rbind, lapply(results, `[[`, "load"))

write.csv(fits, file.path(TAB, "stage8_domain_cfa_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(loads, file.path(TAB, "stage8_domain_cfa_loadings.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(metrics, file.path(TAB, "stage8_domain_reliability_discriminant.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(latent, file.path(TAB, "stage8_domain_latent_correlations.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(assoc, file.path(TAB, "stage8_ic_fi_factor_score_spearman.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(missing, file.path(TAB, "stage8_domain_missing_sensitivity.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(unlist(lapply(results, function(x) c(x$fit$syntax, ""))), file.path(OUT, "stage8_domain_reliability_syntax.txt"))

safe_num <- function(x) ifelse(length(x) == 0 || is.na(x), NA_character_, format(round(x, 3), nsmall = 3))
memo_lines <- c(
  "# Stage 8 domain reliability and IC--FI discriminant-validity audit", "",
  "This is an exploratory within-cohort audit for the primary ELSA/CHARLS/HRS anchor-wave comparison. It does not establish cross-cohort measurement invariance or latent-mean comparability.", "",
  "The four-domain CFA was refit with WLSMV, ordered locomotion/psychological indicators, pairwise available records, and the previously frozen HRS non-proxy sample (`inw10 == 1 & r10proxy == 0`). The outcome-disjoint FI uses six chronic-disease indicators, label-aware self-rated health, BMI <18.5 or >=30, a minimum of 6/8 observed components, and the actual observed denominator.", "",
  "Reliability is summarized with semTools `compRelSEM()` (model-based omega) and `AVE()`. Because lavaan 0.7-2 does not attach a factor-score reliability attribute for WLSMV, the table also reports a clearly labeled empirical factor-score determinacy approximation on the domain-complete subset (`sqrt[var(score)/var(latent)]`); this is a diagnostic and is not a substitute for a planned latent-score model. The factor-score association table uses capacity-oriented signs: higher cognition and grip scores indicate better capacity, and locomotion/psychological factor scores are sign-reversed so that higher values also indicate better capacity. Spearman correlations are unweighted and exploratory.", "",
  "Use the aggregate tables for the gate decision. A domain with low omega/AVE, a negative Fornell--Larcker margin, a weak factor-score reliability, or a large change between pairwise and domain-complete IC--FI associations needs sensitivity modeling before cross-cohort alignment. These are diagnostics, not automatic item-deletion rules.", "",
  "Outputs: `tables/stage8_domain_reliability_discriminant.csv`, `tables/stage8_domain_latent_correlations.csv`, `tables/stage8_ic_fi_factor_score_spearman.csv`, `tables/stage8_domain_missing_sensitivity.csv`, and `tables/stage8_domain_cfa_loadings.csv`. Individual factor scores are not written to disk."
)
writeLines(memo_lines, file.path(OUT, "stage8_domain_reliability_memo.md"))

run_info <- c(
  paste0("timestamp=", format(Sys.time(), tz = "UTC")),
  paste0("R=", R.version.string), paste0("lavaan=", as.character(packageVersion("lavaan"))),
  paste0("semTools=", as.character(packageVersion("semTools"))),
  "design=exploratory unweighted aggregate audit; raw source files read-only",
  "factor_scores=memory_only; no individual IDs or scores exported",
  "hrs_ic_sample=inw10==1 & r10proxy==0",
  "fi_rule=8 components; minimum 6 observed; observed denominator; BMI<18.5 or >=30"
)
writeLines(run_info, file.path(OUT, "stage8_domain_reliability_run_info.txt"))
cat("Wrote Stage 8 aggregate reliability, discriminant-validity, and IC-FI audit outputs.\n")
