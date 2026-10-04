#!/usr/bin/env Rscript
# Fixed-prediction paired bootstrap intervals for the latent primary analysis.
# This sources the frozen Stage-67 script, keeps all scores in memory, and
# exports aggregate rows only.
suppressPackageStartupMessages({ library(haven); library(lavaan) })
ROOT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
GATE <- file.path(ROOT, "robustness_gate"); dir.create(GATE, recursive=TRUE, showWarnings=FALSE)
set.seed(20260925)
B <- 400
source(file.path(CODE, "04_sensitivity_prediction_and_validation", "67_partial_metric_factor_disability.R"), local=FALSE)

auc_fun <- function(y,p) {
  n1 <- sum(y==1); n0 <- sum(y==0); if(n1==0 || n0==0) return(NA_real_)
  r <- rank(p); (sum(r[y==1])-n1*(n1+1)/2)/(n1*n0)
}
boot_row <- function(d, label, window) {
  d <- d[d$baseline_none & d$observed,]
  d <- d[complete.cases(d[,c("event",full_terms)]),]
  mb <- glm(as.formula(paste("event~",paste(base_terms,collapse="+"))), data=d, family=binomial)
  mf <- glm(as.formula(paste("event~",paste(full_terms,collapse="+"))), data=d, family=binomial)
  p0 <- predict(mb, type="response"); p1 <- predict(mf, type="response"); y <- d$event
  da <- db <- numeric(0)
  for (b in seq_len(B)) {
    ii <- sample.int(nrow(d), nrow(d), replace=TRUE); yy <- y[ii]
    if(length(unique(yy)) < 2) next
    da <- c(da, auc_fun(yy,p1[ii])-auc_fun(yy,p0[ii]))
    db <- c(db, mean((yy-p1[ii])^2)-mean((yy-p0[ii])^2))
  }
  data.frame(track="latent", sample=label, window=window, n=nrow(d), events=sum(y),
             delta_auc=auc_fun(y,p1)-auc_fun(y,p0), delta_auc_low=quantile(da,.025),
             delta_auc_high=quantile(da,.975), delta_brier=mean((y-p1)^2)-mean((y-p0)^2),
             delta_brier_low=quantile(db,.025), delta_brier_high=quantile(db,.975),
             bootstrap_replicates=length(da), prediction_type="fixed_prediction_internal",
             interval_type="paired_apparent_bootstrap", stringsAsFactors=FALSE)
}

rows <- list()
for (g in levels(dat$cohort)) {
  rows[[length(rows)+1]] <- boot_row(dat[dat$cohort==g,], g, unique(dat$window[dat$cohort==g]))
}
rows[[length(rows)+1]] <- boot_row(dat, "ELSA+CHARLS+HRS", "anchor_windows")
out <- do.call(rbind, rows)
write.csv(out, file.path(GATE,"latent_primary_delta_bootstrap.csv"), row.names=FALSE, fileEncoding="UTF-8")
print(out)
