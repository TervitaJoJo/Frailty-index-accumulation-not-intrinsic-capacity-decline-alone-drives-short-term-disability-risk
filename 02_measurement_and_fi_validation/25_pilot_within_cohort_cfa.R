#!/usr/bin/env Rscript
# Pilot within-cohort CFA for scheme A. Aggregate outputs only.

suppressPackageStartupMessages({
  library(haven)
  library(lavaan)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
TAB <- file.path(OUT, "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

num <- function(x, low = 0, high = Inf) {
  x <- as.numeric(x)
  x[x < low | x > high] <- NA_real_
  x
}

fit_one <- function(cohort, dat, ordered, cognition_names, psych_names) {
  syntax <- paste(
    paste("cognition =~", paste(cognition_names, collapse = " + ")),
    "locomotion =~ loc_1 + loc_2",
    "grip_vitality =~ grip_1 + grip_2 + grip_3 + grip_4",
    paste("psychological =~", paste(psych_names, collapse = " + ")),
    "cognition ~~ locomotion + grip_vitality + psychological",
    "locomotion ~~ grip_vitality + psychological",
    "grip_vitality ~~ psychological",
    sep = "\n"
  )
  needed <- c(cognition_names, "loc_1", "loc_2", "grip_1", "grip_2", "grip_3", "grip_4", psych_names)
  stopifnot(all(needed %in% names(dat)))
  fit <- lavaan::cfa(syntax, data = dat, ordered = ordered,
                     estimator = "WLSMV", std.lv = TRUE,
                     missing = "pairwise")
  fm <- lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))
  fit_row <- data.frame(dataset = cohort, n = nrow(dat), n_complete_any = sum(complete.cases(dat[, needed])),
                        converged = lavaan::lavInspect(fit, "converged"),
                        post_check = lavaan::lavInspect(fit, "post.check"),
                        t(as.data.frame(fm)), check.names = FALSE)
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE)
  load <- pe[pe$op == "=~", c("lhs", "rhs", "est", "se", "pvalue", "std.all")]
  load$dataset <- cohort
  list(fit = fit_row, load = load, syntax = syntax)
}

prepare_elsa <- function() {
  work_path <- file.path(ROOT, "ELSA/Working_data/elsa.dta")
  raw_path <- file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_data_v2.dta")
  nurse_path <- file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_nurse_data_v2.dta")
  w <- read_dta(work_path)
  w <- w[w$wave == 6, , drop = FALSE]
  r <- read_dta(raw_path, col_select = c("idauniq", paste0("PSced", LETTERS[1:8])))
  n <- read_dta(nurse_path, col_select = c("idauniq", "mmgsd1", "mmgsd2", "mmgsd3", "mmgsn1", "mmgsn2", "mmgsn3"))
  w$.__key <- as.character(w$idauniqc)
  r$.__key <- as.character(r$idauniq)
  n$.__key <- as.character(n$idauniq)
  d <- merge(w[, c(".__key", "imrc", "dlrc", "orient", "verbf", "walkra", "walk100a")], r[, c(".__key", paste0("PSced", LETTERS[1:8]))], by = ".__key", all.x = TRUE, sort = FALSE)
  d <- merge(d, n[, c(".__key", "mmgsd1", "mmgsd2", "mmgsd3", "mmgsn1", "mmgsn2", "mmgsn3")], by = ".__key", all.x = TRUE, sort = FALSE)
  out <- data.frame(
    imrc = num(d$imrc, 0, 10), dlrc = num(d$dlrc, 0, 10), orient = num(d$orient, 0, 4),
    loc_1 = num(d$walkra, 0, 1), loc_2 = num(d$walk100a, 0, 1),
    grip_1 = num(d$mmgsd1), grip_2 = num(d$mmgsd2), grip_3 = num(d$mmgsn1), grip_4 = num(d$mmgsn2),
    psych_1 = ifelse(num(d$PScedA, 1, 2) == 1, 1, ifelse(num(d$PScedA, 1, 2) == 2, 0, NA)),
    psych_2 = ifelse(num(d$PScedB, 1, 2) == 1, 1, ifelse(num(d$PScedB, 1, 2) == 2, 0, NA)),
    psych_3 = ifelse(num(d$PScedC, 1, 2) == 1, 1, ifelse(num(d$PScedC, 1, 2) == 2, 0, NA)),
    psych_4 = ifelse(num(d$PScedD, 1, 2) == 1, 0, ifelse(num(d$PScedD, 1, 2) == 2, 1, NA)),
    psych_5 = ifelse(num(d$PScedE, 1, 2) == 1, 1, ifelse(num(d$PScedE, 1, 2) == 2, 0, NA)),
    psych_6 = ifelse(num(d$PScedF, 1, 2) == 1, 0, ifelse(num(d$PScedF, 1, 2) == 2, 1, NA)),
    psych_7 = ifelse(num(d$PScedG, 1, 2) == 1, 1, ifelse(num(d$PScedG, 1, 2) == 2, 0, NA)),
    psych_8 = ifelse(num(d$PScedH, 1, 2) == 1, 1, ifelse(num(d$PScedH, 1, 2) == 2, 0, NA))
  )
  out
}

prepare_charls <- function() {
  path <- file.path(ROOT, "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta")
  d <- read_dta(path)
  d <- d[d$inw3 == 1, , drop = FALSE]
  out <- data.frame(
    imrc = num(d$r3imrc, 0, 10), dlrc = num(d$r3dlrc, 0, 10), orient = num(d$r3orient, 0, 4), verbf = num(d$r3ser7, 0, 5),
    loc_1 = num(d$r3walk100a, 0, 1), loc_2 = num(d$r3walk1kma, 0, 1),
    grip_1 = num(d$r3lgrip1), grip_2 = num(d$r3lgrip2), grip_3 = num(d$r3rgrip1), grip_4 = num(d$r3rgrip2),
    psych_1 = num(d$r3depresl, 1, 4), psych_2 = num(d$r3effortl, 1, 4), psych_3 = num(d$r3sleeprl, 1, 4),
    psych_4 = 5 - num(d$r3whappyl, 1, 4), psych_5 = num(d$r3flonel, 1, 4), psych_6 = num(d$r3botherl, 1, 4),
    psych_7 = num(d$r3goingl, 1, 4), psych_8 = 5 - num(d$r3fhopel, 1, 4),
    psych_9 = num(d$r3mindtsl, 1, 4), psych_10 = num(d$r3fearll, 1, 4)
  )
  out
}

results <- list()
for (spec in list(
  ELSA = list(data = prepare_elsa(), ordered = c("loc_1", "loc_2", paste0("psych_", 1:8))),
  CHARLS = list(data = prepare_charls(), ordered = c("loc_1", "loc_2", paste0("psych_", 1:8)))
)) {
  # Named list iteration is handled below; this branch is intentionally explicit.
}
elsa <- prepare_elsa()
charls <- prepare_charls()
e <- fit_one("ELSA", elsa, c("loc_1", "loc_2", paste0("psych_", 1:8)), c("imrc", "dlrc", "orient"), paste0("psych_", 1:8))
c <- fit_one("CHARLS", charls, c("loc_1", "loc_2", paste0("psych_", 1:10)), c("imrc", "dlrc", "orient", "verbf"), paste0("psych_", 1:10))
fits <- rbind(e$fit, c$fit)
loads <- rbind(e$load, c$load)
write.csv(fits, file.path(TAB, "pilot_domain_cfa_fit.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(loads, file.path(TAB, "pilot_domain_cfa_loadings.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(c(e$syntax, "\n# CHARLS", c$syntax), file.path(OUT, "pilot_domain_cfa_syntax.txt"))
cat("Wrote pilot domain CFA aggregate outputs.\n")
