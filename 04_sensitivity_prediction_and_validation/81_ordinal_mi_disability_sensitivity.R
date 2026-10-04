#!/usr/bin/env Rscript

# Ordinal-compatible MI sensitivity for the primary incident-disability
# prediction endpoint. The analysis frame is restricted to participants with
# observed baseline ADL/IADL summaries equal to zero, matching the primary
# estimand. Future ADL/IADL summary scores and baseline predictors are imputed
# from raw ordinal/binary variables; disability, FI and IC proxy domains are
# recomputed after each completed data set. No identifiers or row-level output
# are written.

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

make_elsa <- function() {
  work <- read_dta(file.path(ROOT, "ELSA/Working_data/elsa.dta"))
  raw <- read_dta(file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_data_v2.dta"))
  nur <- read_dta(file.path(ROOT, "ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_nurse_data_v2.dta"))
  work$key <- as.character(work$idauniqc); raw$key <- as.character(raw$idauniq); nur$key <- as.character(nur$idauniq)
  b <- merge(work[work$wave == 6, ], raw[, c("key", paste0("PSced", LETTERS[1:8]))], by="key", all.x=TRUE, sort=FALSE)
  b <- merge(b, nur[, c("key", "mmgsd1", "mmgsn1", "mmgsd2", "mmgsn2")], by="key", all.x=TRUE, sort=FALSE)
  b <- b[!duplicated(b$idauniqc), , drop=FALSE]
  f <- work[work$wave == 7, ]
  f <- f[!duplicated(f$idauniqc), , drop=FALSE]
  mid <- match(b$idauniqc, f$idauniqc)
  out <- data.frame(
    age=num(b$agey), sex=sex01(b$ragender), education=num(b$raeducl),
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
    adl_b=num(b$adltot6,0,6), iadl_b=num(b$iadltot2_e,0,30),
    adl_f=num(f$adltot6[mid],0,6), iadl_f=num(f$iadltot2_e[mid],0,30)
  )
  out[!is.na(out$adl_b) & !is.na(out$iadl_b) & out$adl_b == 0 & out$iadl_b == 0, , drop=FALSE]
}

make_charls <- function() {
  d <- read_dta(file.path(ROOT, "CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta"))
  active <- num(d$inw3) == 1 & num(d$inw4) == 1
  p <- function(w, x) { nm <- paste0("r", w, x); if (nm %in% names(d)) d[[nm]] else rep(NA_real_, nrow(d)) }
  out <- data.frame(
    age=num(p(3,"agey")), sex=sex01(d$ragender), education=num(d$raeduc_c),
    imrc_b=num(p(3,"imrc"),0,10), dlrc_b=num(p(3,"dlrc"),0,10), orient_b=num(p(3,"orient"),0,4),
    loc1_b=num(p(3,"walk100a"),0,1), loc2_b=num(p(3,"walk1kma"),0,1),
    grip1_b=num(p(3,"lgrip1"),0,100), grip2_b=num(p(3,"lgrip2"),0,100), grip3_b=num(p(3,"rgrip1"),0,100), grip4_b=num(p(3,"rgrip2"),0,100),
    psych1_b=num(p(3,"depresl"),1,4), psych2_b=num(p(3,"effortl"),1,4), psych3_b=num(p(3,"sleeprl"),1,4), psych4_b=num(p(3,"whappyl"),1,4),
    hypertension_b=valid01(p(3,"hibpe")), diabetes_b=valid01(p(3,"diabe")), heart_disease_b=valid01(p(3,"hearte")),
    stroke_b=valid01(p(3,"stroke")), cancer_b=valid01(p(3,"cancre")), arthritis_b=valid01(p(3,"arthre")),
    shlt_b=num(p(3,"shlt"),1,5), bmi_b=num(p(3,"mbmi"),0,200),
    adl_b=num(p(3,"adlwa"),0,3), iadl_b=num(p(3,"iadla"),0,3),
    adl_f=num(p(4,"adlwa"),0,3), iadl_f=num(p(4,"iadla"),0,3)
  )
  out[active & !is.na(out$adl_b) & !is.na(out$iadl_b) & out$adl_b == 0 & out$iadl_b == 0, , drop=FALSE]
}

make_hrs <- function() {
  d <- read_dta(file.path(ROOT, "HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta"))
  d <- d[!duplicated(d$hhidpn), , drop=FALSE]
  p <- function(w, x) { nm <- paste0("r", w, x); if (nm %in% names(d)) d[[nm]] else rep(NA_real_, nrow(d)) }
  active <- num(d$inw10) == 1 & num(d$inw11) == 1 & num(p(10,"proxy")) == 0 & num(p(11,"proxy")) == 0
  out <- data.frame(
    age=num(p(10,"agey_m")), sex=sex01(d$ragender), education=num(d$raeduc),
    imrc_b=num(p(10,"imrc"),0,10), dlrc_b=num(p(10,"dlrc"),0,10), ser7_b=num(p(10,"ser7"),0,5), bwc20_b=num(p(10,"bwc20"),0,2),
    loc1_b=num(p(10,"walkra"),0,2), loc2_b=num(p(10,"walk1a"),0,2), loc3_b=num(p(10,"walksa"),0,2),
    grip1_b=num(p(10,"grpl"),0,100), grip2_b=num(p(10,"grpr"),0,100),
    psych1_b=num(p(10,"depres"),0,1), psych2_b=num(p(10,"effort"),0,1), psych3_b=num(p(10,"sleepr"),0,1), psych4_b=num(p(10,"whappy"),0,1),
    hypertension_b=valid01(p(10,"hibpe")), diabetes_b=valid01(p(10,"diabe")), heart_disease_b=valid01(p(10,"hearte")),
    stroke_b=valid01(p(10,"stroke")), cancer_b=valid01(p(10,"cancre")), arthritis_b=valid01(p(10,"arthre")),
    shlt_b=num(p(10,"shlt"),1,5), bmi_b=num(p(10,"bmi"),0,200),
    adl_b=num(p(10,"adl5a"),0,5), iadl_b=num(p(10,"iadl5a"),0,5),
    adl_f=num(p(11,"adl5a"),0,5), iadl_f=num(p(11,"iadl5a"),0,5)
  )
  out[active & !is.na(out$adl_b) & !is.na(out$iadl_b) & out$adl_b == 0 & out$iadl_b == 0, , drop=FALSE]
}

configure_mice <- function(dat) {
  ord <- c("imrc_b","dlrc_b","orient_b","loc1_b","loc2_b","loc3_b","ser7_b","bwc20_b","shlt_b","shlt_f",
           "psych1_b","psych2_b","psych3_b","psych4_b","adl_b","iadl_b","adl_f","iadl_f")
  ord <- intersect(ord, names(dat)); ord_binary <- character(0)
  for (v in ord) {
    lv <- sort(unique(dat[[v]][is.finite(dat[[v]])]))
    if (length(lv) <= 2) { dat[[v]] <- factor(dat[[v]], levels=lv); ord_binary <- c(ord_binary, v) }
    else dat[[v]] <- ordered(dat[[v]], levels=lv)
  }
  bin <- grep("^(sex|hypertension|diabetes|heart_disease|stroke|cancer|arthritis|psych)", names(dat), value=TRUE)
  for (v in bin) dat[[v]] <- factor(dat[[v]], levels=c(0,1))
  ini <- mice(dat, m=1, maxit=0, printFlag=FALSE)
  meth <- ini$method; pred <- ini$predictorMatrix; pred[,] <- 1; diag(pred) <- 0
  for (v in setdiff(ord, ord_binary)) meth[v] <- "polr"
  for (v in c(bin, ord_binary)) meth[v] <- "logreg.boot"
  cont <- setdiff(names(dat), c(ord, bin)); for (v in cont) meth[v] <- "pmm"
  for (v in names(dat)) if (all(!is.na(dat[[v]])) || sum(!is.na(dat[[v]])) < 2) meth[v] <- ""
  list(data=dat, method=meth, predictor=pred)
}

numeric_value <- function(x) if (is.factor(x)) as.numeric(as.character(x)) else as.numeric(x)

derive <- function(d, cohort) {
  for (v in names(d)) if (is.factor(d[[v]])) d[[v]] <- numeric_value(d[[v]])
  d$baseline_fi <- fi_from_components(d, "_b")$fi
  d$future_disability <- as.numeric(d$adl_f > 0 | d$iadl_f > 0)
  if (cohort == "ELSA") {
    z <- cbind(d$imrc_b/10, d$dlrc_b/10, d$orient_b/4); d$cognition <- rowMeans(z, na.rm=TRUE); d$cognition[rowSums(!is.na(z)) < 2] <- NA_real_
    d$locomotion <- mean_min(d, c("loc1_b","loc2_b"), minimum=1, reverse=TRUE)
    d$grip_vitality <- max_valid(d, c("grip1_b","grip2_b","grip3_b","grip4_b"), scale=100)
    d$psychological <- rowMeans(cbind(d$psych1_b,d$psych2_b,d$psych3_b,d$psych4_b), na.rm=TRUE)
  } else if (cohort == "CHARLS") {
    z <- cbind(d$imrc_b/10, d$dlrc_b/10, d$orient_b/4); d$cognition <- rowMeans(z, na.rm=TRUE); d$cognition[rowSums(!is.na(z)) < 2] <- NA_real_
    d$locomotion <- mean_min(d, c("loc1_b","loc2_b"), minimum=1, reverse=TRUE)
    d$grip_vitality <- max_valid(d, c("grip1_b","grip2_b","grip3_b","grip4_b"), scale=100)
    d$psychological <- rowMeans(cbind((4-d$psych1_b)/3,(4-d$psych2_b)/3,(4-d$psych3_b)/3,(d$psych4_b-1)/3), na.rm=TRUE)
  } else {
    z <- cbind(d$imrc_b/10, d$dlrc_b/10, d$ser7_b/5, d$bwc20_b/2); d$cognition <- rowMeans(z, na.rm=TRUE); d$cognition[rowSums(!is.na(z)) < 2] <- NA_real_
    d$locomotion <- rowMeans(cbind(1-(d$loc1_b>0),1-(d$loc2_b>0),1-(d$loc3_b>0)), na.rm=TRUE)
    d$grip_vitality <- max_valid(d, c("grip1_b","grip2_b"), scale=100)
    d$psychological <- -rowMeans(cbind(d$psych1_b,d$psych2_b,d$psych3_b,d$psych4_b), na.rm=TRUE)
  }
  d$psychological[is.nan(d$psychological)] <- NA_real_; d
}

auc_rank <- function(y, p) {
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  if (length(unique(y)) < 2) return(NA_real_)
  r <- rank(p, ties.method="average"); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1*(n1+1)/2) / (n1*n0)
}

fit_one <- function(d, cohort, imp_no) {
  vars <- c("future_disability","baseline_fi","age","sex","education","cognition","locomotion","grip_vitality","psychological")
  dd <- d[complete.cases(d[, vars]), vars]
  if (nrow(dd) < 100 || length(unique(dd$future_disability)) < 2) return(data.frame(cohort=cohort, imputation=imp_no, n=nrow(dd), events=sum(dd$future_disability, na.rm=TRUE), delta_auc=NA_real_, delta_brier=NA_real_, model_error="too_few_or_single_class"))
  zscore <- function(x) as.numeric(scale(as.numeric(x)))
  base <- data.frame(age=zscore(dd$age), sex=as.numeric(dd$sex), education=zscore(dd$education), baseline_fi=zscore(dd$baseline_fi))
  full <- cbind(base, cognition=zscore(dd$cognition), locomotion=zscore(dd$locomotion), grip_vitality=zscore(dd$grip_vitality), psychological=zscore(dd$psychological))
  mb <- glm(dd$future_disability ~ ., data=base, family=binomial())
  mf <- glm(dd$future_disability ~ ., data=full, family=binomial())
  pb <- predict(mb, type="response"); pf <- predict(mf, type="response"); y <- dd$future_disability
  vb <- sandwich::vcovHC(mb, type="HC3"); vf <- sandwich::vcovHC(mf, type="HC3")
  co <- coef(mf); se <- sqrt(diag(vf)); terms <- names(co)
  out <- data.frame(cohort=cohort, imputation=imp_no, n=nrow(dd), events=sum(y), delta_auc=auc_rank(y,pf)-auc_rank(y,pb), delta_brier=mean((y-pf)^2)-mean((y-pb)^2), term=terms, estimate=as.numeric(co), se_hc3=as.numeric(se), stringsAsFactors=FALSE)
  out$ci_low <- out$estimate - 1.96*out$se_hc3; out$ci_high <- out$estimate + 1.96*out$se_hc3; out$model_error <- ""; out
}

pool_results <- function(rows) {
  coef_rows <- rows[!is.na(rows$estimate) & rows$term != "", ]
  pooled <- do.call(rbind, lapply(split(coef_rows, list(coef_rows$cohort, coef_rows$term), drop=TRUE), function(x) {
    m <- nrow(x); q <- mean(x$estimate); u <- mean(x$se_hc3^2); b <- if (m > 1) var(x$estimate) else 0; se <- sqrt(u + (1 + 1/m)*b)
    data.frame(cohort=x$cohort[1], term=x$term[1], m=m, n_mean=mean(x$n), events_mean=mean(x$events), pooled_estimate=q, pooled_se=se, pooled_ci_low=q-1.96*se, pooled_ci_high=q+1.96*se, between_imputation_var=b, stringsAsFactors=FALSE)
  }))
  metric_rows <- rows[!duplicated(rows[, c("cohort", "imputation")]), , drop=FALSE]
  met <- do.call(rbind, lapply(split(metric_rows, metric_rows$cohort), function(x) data.frame(cohort=x$cohort[1], m=nrow(x), n_mean=mean(x$n), events_mean=mean(x$events), delta_auc_mean=mean(x$delta_auc,na.rm=TRUE), delta_auc_min=min(x$delta_auc,na.rm=TRUE), delta_auc_max=max(x$delta_auc,na.rm=TRUE), delta_brier_mean=mean(x$delta_brier,na.rm=TRUE), delta_brier_min=min(x$delta_brier,na.rm=TRUE), delta_brier_max=max(x$delta_brier,na.rm=TRUE), stringsAsFactors=FALSE)))
  list(coef=pooled, metrics=met)
}

run_one <- function(dat, cohort) {
  cfg <- configure_mice(dat); set.seed(SEED + match(cohort, c("ELSA","CHARLS","HRS")))
  imp <- mice(cfg$data, m=M, maxit=MAXIT, method=cfg$method, predictorMatrix=cfg$predictor, seed=SEED, printFlag=FALSE, polr.to.loggedEvents=TRUE)
  rows <- do.call(rbind, lapply(seq_len(M), function(i) fit_one(derive(complete(imp, i), cohort), cohort, i)))
  miss <- data.frame(cohort=cohort, variable=names(cfg$data), missing_fraction=sapply(cfg$data, function(x) mean(is.na(x))), method=unname(cfg$method), stringsAsFactors=FALSE)
  list(rows=rows, pool=pool_results(rows), miss=miss, logged=imp$loggedEvents)
}

datasets <- list(ELSA=make_elsa(), CHARLS=make_charls(), HRS=make_hrs())
res <- lapply(names(datasets), function(nm) run_one(datasets[[nm]], nm)); names(res) <- names(datasets)
all_coef <- do.call(rbind, lapply(res, function(x) x$pool$coef)); all_met <- do.call(rbind, lapply(res, function(x) x$pool$metrics)); all_rep <- do.call(rbind, lapply(res, function(x) x$rows)); all_miss <- do.call(rbind, lapply(res, function(x) x$miss))
for (x in c("all_coef","all_met","all_rep","all_miss")) assign(x, within(get(x), analysis <- "ordinal_compatible_MI_incident_disability"))
write.csv(all_coef, file.path(TAB,"ordinal_mi_disability_coefficients_pooled.csv"), row.names=FALSE, fileEncoding="UTF-8")
write.csv(all_met, file.path(TAB,"ordinal_mi_disability_delta_metrics.csv"), row.names=FALSE, fileEncoding="UTF-8")
write.csv(all_rep, file.path(TAB,"ordinal_mi_disability_metrics_by_imputation.csv"), row.names=FALSE, fileEncoding="UTF-8")
write.csv(all_miss, file.path(TAB,"ordinal_mi_disability_missingness_methods.csv"), row.names=FALSE, fileEncoding="UTF-8")
logs <- do.call(rbind, lapply(names(res), function(nm) { z <- res[[nm]]$logged; if (is.null(z) || nrow(z) == 0) return(data.frame(cohort=character(0), it=integer(0), im=integer(0), dep=character(0), meth=character(0), out=character(0), stringsAsFactors=FALSE)); cbind(cohort=nm, z) }))
write.csv(logs, file.path(TAB,"ordinal_mi_disability_mice_log.csv"), row.names=FALSE, fileEncoding="UTF-8")

memo <- c("# Ordinal-compatible MI sensitivity for incident disability", "", paste0("The analysis used ", M, " imputations and ", MAXIT, " MICE iterations per cohort (seed ", SEED, "). The analysis frame was restricted to participants with observed baseline ADL/IADL summary scores equal to zero, matching the primary incident-disability estimand. Future ADL/IADL summary scores and baseline predictors were imputed before recomputing incident disability, the outcome-disjoint FI and the four observed IC proxies."), "", "Multi-level ordinal variables used proportional-odds imputation (polr), two-level/binary variables used bootstrap logistic imputation (logreg.boot), and continuous variables used predictive mean matching (pmm). The base model contained age, sex, education and baseline FI; the full model added cognition, locomotion, grip/vitality and psychological proxies. Continuous predictors were standardised within completed cohort-window samples. Coefficients were pooled with Rubin's rules and HC3 covariance; delta AUC and delta Brier are summarised across imputations because they are nonlinear contrasts.", "", "This is a missing-data sensitivity for the primary incident-disability endpoint, not a replacement for the complete-case primary analysis or the fixed-coefficient SHARE external validation. Structural missingness and HRS proxy restrictions remain design boundaries. No identifiers, imputed rows or individual predictions were written.", "")
writeLines(memo, file.path(OUT,"ordinal_mi_disability_sensitivity_memo.md"), useBytes=TRUE)
run_info <- c(paste0("timestamp=",format(Sys.time(),tz="UTC")), paste0("R=",R.version.string), paste0("mice=",as.character(packageVersion("mice"))), paste0("m=",M,"; maxit=",MAXIT,"; seed=",SEED), "methods=multilevel ordinal polr; two-level/binary logreg.boot; continuous pmm", "estimand=incident any ADL/IADL limitation after observed baseline ADL/IADL-free frame", "cohorts=ELSA 6->7; CHARLS 3->4; HRS 10->11", "HRS proxy respondents excluded at baseline and follow-up", "individual_level_output=false")
writeLines(run_info, file.path(OUT,"ordinal_mi_disability_run_info.txt"), useBytes=TRUE)
cat("Ordinal-compatible incident-disability MI sensitivity completed.\n"); print(all_met)
