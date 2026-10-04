#!/usr/bin/env Rscript

# Ordinal-compatible missing-data sensitivity for the frozen longitudinal
# observed-proxy prediction track. The imputation model operates on raw
# ordinal/binary indicators and FI components; domains and FI are recomputed
# passively after each completed data set. No person-level output is written.

suppressPackageStartupMessages({
  library(haven)
  library(mice)
  library(sandwich)
})

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
TAB <- file.path(OUT, "tables")
dir.create(TAB, recursive=TRUE, showWarnings=FALSE)

M <- 5L
MAXIT <- 3L
SEED <- 20260925L

num <- function(x, lo=-Inf, hi=Inf) {
  z <- as.numeric(x)
  z[is.na(z) | z < lo | z > hi] <- NA_real_
  z
}

valid01 <- function(x) {
  z <- num(x)
  z[!(z %in% c(0, 1))] <- NA_real_
  z
}

sex01 <- function(x) {
  z <- num(x)
  u <- sort(unique(z[is.finite(z)]))
  if (length(u) == 2 && all(u %in% c(1, 2))) return(ifelse(z == 2, 1, ifelse(z == 1, 0, NA_real_)))
  valid01(z)
}

fi_from_components <- function(d, suffix="") {
  nms <- paste0(c("hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis"), suffix)
  z <- as.data.frame(lapply(nms, function(n) valid01(d[[n]])))
  names(z) <- c("hypertension", "diabetes", "heart_disease", "stroke", "cancer", "arthritis")
  sh <- num(d[[paste0("shlt", suffix)]], 1, 5)
  z$self_rated_health <- (5 - sh) / 4
  bmi <- num(d[[paste0("bmi", suffix)]], 0, 200)
  z$bmi <- ifelse(is.na(bmi), NA_real_, as.numeric(bmi < 18.5 | bmi >= 30))
  n <- rowSums(!is.na(z))
  z$fi <- rowSums(z, na.rm=TRUE) / n
  z$fi[n < 6] <- NA_real_
  z
}

mean_min <- function(d, cols, minimum=1, scale=1, reverse=FALSE) {
  z <- as.data.frame(lapply(cols, function(n) num(d[[n]]) / scale))
  if (reverse) z <- 1 - z
  out <- rowMeans(z, na.rm=TRUE)
  out[rowSums(!is.na(z)) < minimum] <- NA_real_
  out[is.nan(out)] <- NA_real_
  out
}

max_valid <- function(d, cols, scale=1) {
  z <- as.data.frame(lapply(cols, function(n) num(d[[n]]) / scale))
  out <- apply(z, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm=TRUE))
  as.numeric(out)
}

psych_mean <- function(d, cols, reverse=FALSE, max_value=1) {
  z <- as.data.frame(lapply(cols, function(n) num(d[[n]], 0, max_value)))
  if (reverse) z <- -z
  out <- rowMeans(z, na.rm=TRUE)
  out[is.nan(out)] <- NA_real_
  out
}

make_elsa <- function() {
  work <- read_dta(file.path(ROOT, "ELSA/Working_data/elsa.dta"))
  raw <- read_dta(file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_data_v2.dta"))
  nur <- read_dta(file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_nurse_data_v2.dta"))
  work$key <- as.character(work$idauniqc); raw$key <- as.character(raw$idauniq); nur$key <- as.character(nur$idauniq)
  b <- merge(work[work$wave == 6, ], raw[, c("key", paste0("PSced", LETTERS[1:8]))], by="key", all.x=TRUE, sort=FALSE)
  b <- merge(b, nur[, c("key", "mmgsd1", "mmgsn1", "mmgsd2", "mmgsn2")], by="key", all.x=TRUE, sort=FALSE)
  f <- work[work$wave == 7, ]
  out <- data.frame(
    age=num(b$agey), sex=sex01(b$ragender),
    imrc_b=num(b$imrc,0,10), dlrc_b=num(b$dlrc,0,10), orient_b=num(b$orient,0,4),
    loc1_b=num(b$walkra,0,1), loc2_b=num(b$walk100a,0,1),
    grip1_b=num(b$mmgsd1,0,100), grip2_b=num(b$mmgsd2,0,100), grip3_b=num(b$mmgsn1,0,100), grip4_b=num(b$mmgsn2,0,100),
    psych1_b=ifelse(num(b$PScedA,1,2)==1,1,ifelse(num(b$PScedA,1,2)==2,0,NA)),
    psych2_b=ifelse(num(b$PScedB,1,2)==1,1,ifelse(num(b$PScedB,1,2)==2,0,NA)),
    psych3_b=ifelse(num(b$PScedC,1,2)==1,1,ifelse(num(b$PScedC,1,2)==2,0,NA)),
    psych4_b=ifelse(num(b$PScedD,1,2)==1,0,ifelse(num(b$PScedD,1,2)==2,1,NA)),
    hypertension_b=valid01(b$hibpe), diabetes_b=valid01(b$diabe), heart_disease_b=valid01(b$hearte),
    stroke_b=valid01(b$stroke), cancer_b=valid01(b$cancre), arthritis_b=valid01(b$arthre),
    shlt_b=num(b$shlt,1,5), bmi_b=num(b$mbmi,0,200),
    hypertension_f=valid01(f$hibpe[match(b$idauniqc, f$idauniqc)]), diabetes_f=valid01(f$diabe[match(b$idauniqc, f$idauniqc)]),
    heart_disease_f=valid01(f$hearte[match(b$idauniqc, f$idauniqc)]), stroke_f=valid01(f$stroke[match(b$idauniqc, f$idauniqc)]),
    cancer_f=valid01(f$cancre[match(b$idauniqc, f$idauniqc)]), arthritis_f=valid01(f$arthre[match(b$idauniqc, f$idauniqc)]),
    shlt_f=num(f$shlt[match(b$idauniqc, f$idauniqc)],1,5), bmi_f=num(f$mbmi[match(b$idauniqc, f$idauniqc)],0,200)
  )
  out
}

make_charls <- function() {
  d <- read_dta(file.path(ROOT, "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"))
  active <- num(d$inw3) == 1 & num(d$inw4) == 1
  p <- function(w, x) {
    nm <- paste0("r", w, x)
    if (nm %in% names(d)) d[[nm]] else rep(NA_real_, nrow(d))
  }
  data.frame(
    age=num(p(3,"agey")), sex=sex01(d$ragender),
    imrc_b=num(p(3,"imrc"),0,10), dlrc_b=num(p(3,"dlrc"),0,10), orient_b=num(p(3,"orient"),0,4),
    loc1_b=num(p(3,"walk100a"),0,1), loc2_b=num(p(3,"walk1kma"),0,1),
    grip1_b=num(p(3,"lgrip1"),0,100), grip2_b=num(p(3,"lgrip2"),0,100), grip3_b=num(p(3,"rgrip1"),0,100), grip4_b=num(p(3,"rgrip2"),0,100),
    psych1_b=num(p(3,"depresl"),1,4), psych2_b=num(p(3,"effortl"),1,4), psych3_b=num(p(3,"sleeprl"),1,4), psych4_b=num(p(3,"whappyl"),1,4),
    hypertension_b=valid01(p(3,"hibpe")), diabetes_b=valid01(p(3,"diabe")), heart_disease_b=valid01(p(3,"hearte")),
    stroke_b=valid01(p(3,"stroke")), cancer_b=valid01(p(3,"cancre")), arthritis_b=valid01(p(3,"arthre")),
    shlt_b=num(p(3,"shlt"),1,5), bmi_b=num(p(3,"mbmi"),0,200),
    hypertension_f=valid01(p(4,"hibpe")), diabetes_f=valid01(p(4,"diabe")), heart_disease_f=valid01(p(4,"hearte")),
    stroke_f=valid01(p(4,"stroke")), cancer_f=valid01(p(4,"cancre")), arthritis_f=valid01(p(4,"arthre")),
    shlt_f=num(p(4,"shlt"),1,5), bmi_f=num(p(4,"mbmi"),0,200)
  )[active, , drop=FALSE]
}

make_hrs <- function() {
  d <- read_dta(file.path(ROOT, "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"))
  p <- function(w, x) {
    nm <- paste0("r", w, x)
    if (nm %in% names(d)) d[[nm]] else rep(NA_real_, nrow(d))
  }
  active <- num(d$inw10) == 1 & num(d$inw11) == 1 & num(p(10,"proxy")) == 0 & num(p(11,"proxy")) == 0
  data.frame(
    age=num(p(10,"agey_m")), sex=sex01(d$ragender),
    imrc_b=num(p(10,"imrc"),0,10), dlrc_b=num(p(10,"dlrc"),0,10), ser7_b=num(p(10,"ser7"),0,5), bwc20_b=num(p(10,"bwc20"),0,2),
    loc1_b=num(p(10,"walkra"),0,2), loc2_b=num(p(10,"walk1a"),0,2), loc3_b=num(p(10,"walksa"),0,2),
    grip1_b=num(p(10,"grpl"),0,100), grip2_b=num(p(10,"grpr"),0,100),
    psych1_b=num(p(10,"depres"),0,1), psych2_b=num(p(10,"effort"),0,1), psych3_b=num(p(10,"sleepr"),0,1), psych4_b=num(p(10,"whappy"),0,1),
    hypertension_b=valid01(p(10,"hibpe")), diabetes_b=valid01(p(10,"diabe")), heart_disease_b=valid01(p(10,"hearte")),
    stroke_b=valid01(p(10,"stroke")), cancer_b=valid01(p(10,"cancre")), arthritis_b=valid01(p(10,"arthre")),
    shlt_b=num(p(10,"shlt"),1,5), bmi_b=num(p(10,"bmi"),0,200),
    hypertension_f=valid01(p(11,"hibpe")), diabetes_f=valid01(p(11,"diabe")), heart_disease_f=valid01(p(11,"hearte")),
    stroke_f=valid01(p(11,"stroke")), cancer_f=valid01(p(11,"cancre")), arthritis_f=valid01(p(11,"arthre")),
    shlt_f=num(p(11,"shlt"),1,5), bmi_f=num(p(11,"bmi"),0,200)
  )[active, , drop=FALSE]
}

configure_mice <- function(dat, cohort) {
  # Convert ordinal and binary fields before assigning imputation methods.
  # Two-level indicators are explicitly treated as binary.  This avoids
  # silently asking polr to fit a two-category outcome and then falling back
  # to multinom.  logreg.boot is used for binary imputation to reduce
  # separation/non-convergence in sparse disease indicators.
  ord <- c("imrc_b","dlrc_b","orient_b","loc1_b","loc2_b","loc3_b","ser7_b","bwc20_b","shlt_b","shlt_f",
           "psych1_b","psych2_b","psych3_b","psych4_b")
  ord <- intersect(ord, names(dat))
  ord_binary <- character(0)
  for (v in ord) {
    lv <- sort(unique(dat[[v]][is.finite(dat[[v]])]))
    if (length(lv) <= 2) {
      dat[[v]] <- factor(dat[[v]], levels=lv)
      ord_binary <- c(ord_binary, v)
    } else {
      dat[[v]] <- ordered(dat[[v]], levels=lv)
    }
  }
  bin <- grep("^(sex|hypertension|diabetes|heart_disease|stroke|cancer|arthritis)_", names(dat), value=TRUE)
  # ELSA psychological indicators are already binary; CHARLS/HRS psychological
  # indicators are ordinal/binary and are handled above where appropriate.
  for (v in bin) dat[[v]] <- factor(dat[[v]], levels=c(0,1))
  ini <- mice(dat, m=1, maxit=0, printFlag=FALSE)
  meth <- ini$method
  pred <- ini$predictorMatrix
  pred[,] <- 1
  diag(pred) <- 0
  if ("id" %in% names(dat)) {
    meth["id"] <- ""
    pred[, "id"] <- 0
  }
  for (v in setdiff(ord, ord_binary)) meth[v] <- "polr"
  for (v in c(bin, ord_binary)) meth[v] <- "logreg.boot"
  cont <- setdiff(names(dat), c("id", ord, bin))
  cont <- setdiff(cont, "")
  for (v in cont) meth[v] <- "pmm"
  # Do not impute fully observed variables; this also avoids accidental
  # perturbation of cohort inclusion rules.
  for (v in names(dat)) if (all(!is.na(dat[[v]])) || sum(!is.na(dat[[v]])) < 2) meth[v] <- ""
  list(data=dat, method=meth, predictor=pred, ord_binary=ord_binary)
}

numeric_value <- function(x) {
  if (is.factor(x)) return(as.numeric(as.character(x)))
  as.numeric(x)
}

derive_proxy <- function(d, cohort) {
  for (v in names(d)) if (is.factor(d[[v]])) d[[v]] <- numeric_value(d[[v]])
  d$baseline_fi <- fi_from_components(d, "_b")$fi
  d$future_fi <- fi_from_components(d, "_f")$fi
  if (cohort == "ELSA") {
    # Orientation has a different range from recall; use native scales.
    z <- cbind(d$imrc_b/10, d$dlrc_b/10, d$orient_b/4)
    d$cognition <- rowMeans(z, na.rm=TRUE); d$cognition[rowSums(!is.na(z))<2] <- NA_real_
    d$locomotion <- mean_min(d, c("loc1_b","loc2_b"), minimum=1, scale=1, reverse=TRUE)
    d$grip_vitality <- max_valid(d, c("grip1_b","grip2_b","grip3_b","grip4_b"), scale=100)
    d$psychological <- rowMeans(cbind(d$psych1_b,d$psych2_b,d$psych3_b,d$psych4_b), na.rm=TRUE)
  } else if (cohort == "CHARLS") {
    z <- cbind(d$imrc_b/10, d$dlrc_b/10, d$orient_b/4)
    d$cognition <- rowMeans(z, na.rm=TRUE); d$cognition[rowSums(!is.na(z))<2] <- NA_real_
    d$locomotion <- mean_min(d, c("loc1_b","loc2_b"), minimum=1, reverse=TRUE)
    d$grip_vitality <- max_valid(d, c("grip1_b","grip2_b","grip3_b","grip4_b"), scale=100)
    d$psychological <- rowMeans(cbind((4-d$psych1_b)/3,(4-d$psych2_b)/3,(4-d$psych3_b)/3,(d$psych4_b-1)/3), na.rm=TRUE)
  } else {
    z <- cbind(d$imrc_b/10, d$dlrc_b/10, d$ser7_b/5, d$bwc20_b/2)
    d$cognition <- rowMeans(z, na.rm=TRUE); d$cognition[rowSums(!is.na(z))<2] <- NA_real_
    d$locomotion <- rowMeans(cbind(1-(d$loc1_b>0),1-(d$loc2_b>0),1-(d$loc3_b>0)), na.rm=TRUE)
    d$grip_vitality <- max_valid(d, c("grip1_b","grip2_b"), scale=100)
    d$psychological <- -rowMeans(cbind(d$psych1_b,d$psych2_b,d$psych3_b,d$psych4_b), na.rm=TRUE)
  }
  d$psychological[is.nan(d$psychological)] <- NA_real_
  d
}

zscore <- function(x) as.numeric(scale(as.numeric(x)))

fit_pair <- function(d, cohort, imp_no) {
  vars <- c("future_fi","baseline_fi","age","sex","cognition","locomotion","grip_vitality","psychological")
  dd <- d[complete.cases(d[, vars]), vars]
  if (nrow(dd) < 100) return(data.frame(cohort=cohort, imputation=imp_no, n=nrow(dd), delta_r2=NA_real_, model_error="too_few_complete"))
  xbase <- data.frame(baseline_fi=zscore(dd$baseline_fi), age=zscore(dd$age), sex=as.numeric(dd$sex))
  xfull <- cbind(xbase, cognition=zscore(dd$cognition), locomotion=zscore(dd$locomotion), grip_vitality=zscore(dd$grip_vitality), psychological=zscore(dd$psychological))
  y <- dd$future_fi
  mb <- lm(y ~ ., data=xbase)
  mf <- lm(y ~ ., data=xfull)
  vb <- sandwich::vcovHC(mb, type="HC3"); vf <- sandwich::vcovHC(mf, type="HC3")
  co <- coef(mf); se <- sqrt(diag(vf))
  terms <- names(co)
  out <- data.frame(cohort=cohort, imputation=imp_no, n=nrow(dd), delta_r2=summary(mf)$r.squared-summary(mb)$r.squared, term=terms, estimate=as.numeric(co), se_hc3=as.numeric(se), stringsAsFactors=FALSE)
  out$ci_low <- out$estimate - 1.96*out$se_hc3; out$ci_high <- out$estimate + 1.96*out$se_hc3
  out$model_error <- ""
  out
}

pool_results <- function(rows) {
  # Rubin pooling for coefficients; delta R2 is summarised across imputations
  # because it is a derived, non-linear contrast.
  coef_rows <- rows[!is.na(rows$estimate) & rows$term != "", ]
  pooled <- do.call(rbind, lapply(split(coef_rows, list(coef_rows$cohort, coef_rows$term), drop=TRUE), function(x) {
    m <- nrow(x); q <- mean(x$estimate); u <- mean(x$se_hc3^2); b <- if (m > 1) var(x$estimate) else 0
    total <- u + (1 + 1/m)*b; se <- sqrt(total)
    data.frame(cohort=x$cohort[1], term=x$term[1], m=m, n_mean=mean(x$n), pooled_estimate=q, pooled_se=se, pooled_ci_low=q-1.96*se, pooled_ci_high=q+1.96*se, between_imputation_var=b, stringsAsFactors=FALSE)
  }))
  by_cohort <- split(rows$delta_r2, rows$cohort)
  dr2_out <- do.call(rbind, lapply(names(by_cohort), function(nm) {
    x <- by_cohort[[nm]]
    data.frame(cohort=nm, delta_r2_mean=mean(x, na.rm=TRUE), delta_r2_min=min(x, na.rm=TRUE), delta_r2_max=max(x, na.rm=TRUE), stringsAsFactors=FALSE)
  }))
  list(coef=pooled, dr2=dr2_out)
}

run_one <- function(dat, cohort) {
  cfg <- configure_mice(dat, cohort)
  set.seed(SEED + match(cohort, c("ELSA","CHARLS","HRS")))
  imp <- mice(cfg$data, m=M, maxit=MAXIT, method=cfg$method, predictorMatrix=cfg$predictor, seed=SEED, printFlag=FALSE, polr.to.loggedEvents=TRUE)
  rows <- do.call(rbind, lapply(seq_len(M), function(i) fit_pair(derive_proxy(complete(imp, i), cohort), cohort, i)))
  miss <- data.frame(cohort=cohort, variable=names(cfg$data), missing_fraction=sapply(cfg$data, function(x) mean(is.na(x))), method=unname(cfg$method), stringsAsFactors=FALSE)
  pool <- pool_results(rows)
  list(rows=rows, pool=pool, miss=miss, logged=imp$loggedEvents, m=M, maxit=MAXIT)
}

datasets <- list(ELSA=make_elsa(), CHARLS=make_charls(), HRS=make_hrs())
res <- lapply(names(datasets), function(nm) run_one(datasets[[nm]], nm)); names(res) <- names(datasets)

all_coef <- do.call(rbind, lapply(res, function(x) x$pool$coef))
all_dr2 <- do.call(rbind, lapply(res, function(x) x$pool$dr2))
all_imp <- do.call(rbind, lapply(res, function(x) x$rows))
all_miss <- do.call(rbind, lapply(res, function(x) x$miss))
all_coef$analysis <- "ordinal_compatible_MI_longitudinal_proxy"
all_dr2$analysis <- "ordinal_compatible_MI_longitudinal_proxy"
all_imp$analysis <- "ordinal_compatible_MI_longitudinal_proxy"
all_miss$analysis <- "ordinal_compatible_MI_longitudinal_proxy"
write.csv(all_coef, file.path(TAB,"ordinal_mi_longitudinal_coefficients_pooled.csv"), row.names=FALSE, fileEncoding="UTF-8")
write.csv(all_dr2, file.path(TAB,"ordinal_mi_longitudinal_delta_r2.csv"), row.names=FALSE, fileEncoding="UTF-8")
write.csv(all_imp, file.path(TAB,"ordinal_mi_longitudinal_imputation_replicates.csv"), row.names=FALSE, fileEncoding="UTF-8")
write.csv(all_miss, file.path(TAB,"ordinal_mi_longitudinal_missingness_methods.csv"), row.names=FALSE, fileEncoding="UTF-8")
logs <- do.call(rbind, lapply(names(res), function(nm) {
  z <- res[[nm]]$logged
  if (is.null(z) || nrow(z) == 0) {
    return(data.frame(cohort=character(0), it=integer(0), im=integer(0),
                      dep=character(0), meth=character(0), out=character(0),
                      stringsAsFactors=FALSE))
  }
  cbind(cohort=nm, z)
}))
write.csv(logs, file.path(TAB,"ordinal_mi_longitudinal_mice_log.csv"), row.names=FALSE, fileEncoding="UTF-8")

memo <- c(
  "# Ordinal-compatible multiple-imputation sensitivity for longitudinal proxy prediction", "",
  paste0("This analysis was a bounded sensitivity of the frozen observed-proxy longitudinal specification. It used ", M, " imputations and ", MAXIT, " MICE iterations per cohort (seed ", SEED, "). Multi-level ordinal indicators were imputed with proportional-odds regression (`polr`), two-level indicators and disease/FI components with bootstrap logistic regression (`logreg.boot`), and continuous age, grip and BMI variables with predictive mean matching (`pmm`). Domains and the outcome-disjoint FI were recomputed after each completed data set."), "",
  "The base model contained baseline FI, age and sex. The full model added baseline cognition, locomotion, grip/vitality and psychological proxies. Predictors were standardised within completed cohort-window samples, matching the frozen proxy specification. Coefficients were pooled with Rubin's rules using HC3 within-imputation covariance; delta R-squared is summarised as the mean and range across imputations because it is a derived contrast.", "",
  "This is an ordinal-compatible missing-data sensitivity, not a replacement for the primary partial-metric WLSMV analysis and not evidence that structurally absent modules are recoverable. It is reported as a robustness diagnostic for the conditional incremental predictive-validity track. No individual identifiers, imputed rows or individual predictions are written.", ""
)
writeLines(memo, file.path(OUT,"ordinal_mi_longitudinal_sensitivity_memo.md"), useBytes=TRUE)
run_info <- c(
  paste0("timestamp=", format(Sys.time(), tz="UTC")), paste0("R=", R.version.string), paste0("mice=", as.character(packageVersion("mice"))), paste0("lavaan=", as.character(packageVersion("lavaan"))),
  paste0("m=", M, "; maxit=", MAXIT, "; seed=", SEED), "methods=multilevel ordinal polr; two-level/binary logreg.boot; continuous pmm", "estimand=future FI adjusted baseline FI age sex plus four observed IC proxies", "cohorts=ELSA 6->7; CHARLS 3->4; HRS 10->11", "FI=6 chronic diseases + self-rated health + BMI; minimum 6/8 observed before imputation and recomputed after imputation", "individual_level_output=false"
)
writeLines(run_info, file.path(OUT,"ordinal_mi_longitudinal_run_info.txt"), useBytes=TRUE)
cat("Ordinal-compatible MI sensitivity completed.\n")
print(all_dr2)
print(all_coef[all_coef$term %in% c("cognition","locomotion","grip_vitality","psychological"), ])
