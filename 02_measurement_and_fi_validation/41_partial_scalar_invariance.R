#!/usr/bin/env Rscript
# Stage 15b: focused partial-scalar follow-up. Fits only the two models needed
# for the formal ordered-indicator threshold/intercept gate, which keeps peak
# memory lower than retaining all Stage 15 candidate fits simultaneously.
suppressPackageStartupMessages({ library(haven); library(lavaan) })
ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables"); dir.create(TAB, recursive = TRUE, showWarnings = FALSE)
source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local = .GlobalEnv)

max_na <- function(...) { z <- cbind(...); as.numeric(apply(z, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm=TRUE))) }
common_frame <- function(dat, cohort) {
  if (cohort == "ELSA") {
    cog_imrc <- dat$imrc; cog_dlrc <- dat$dlrc; cog_exec <- dat$orient
    loc_room <- ifelse(is.na(dat$loc_1), NA_real_, 1-dat$loc_1); loc_100m <- ifelse(is.na(dat$loc_2), NA_real_, 1-dat$loc_2)
    grip_left <- max_na(dat$grip_1,dat$grip_2); grip_right <- max_na(dat$grip_3,dat$grip_4)
  } else if (cohort == "CHARLS") {
    cog_imrc <- dat$imrc; cog_dlrc <- dat$dlrc; cog_exec <- dat$orient
    loc_room <- ifelse(is.na(dat$loc_1), NA_real_, 1-dat$loc_1); loc_100m <- ifelse(is.na(dat$loc_2), NA_real_, 1-dat$loc_2)
    grip_left <- max_na(dat$grip_1,dat$grip_2); grip_right <- max_na(dat$grip_3,dat$grip_4)
  } else {
    cog_imrc <- dat$cog_imrc; cog_dlrc <- dat$cog_dlrc; cog_exec <- pmin(dat$cog_ser7,4)
    loc_room <- ifelse(is.na(dat$loc_1), NA_real_, 1-as.numeric(dat$loc_1>0)); loc_100m <- ifelse(is.na(dat$loc_2), NA_real_, 1-as.numeric(dat$loc_2>0))
    grip_left <- dat$grip_1; grip_right <- dat$grip_2
  }
  psych_binary <- function(x) { x <- as.numeric(x); if (cohort == "CHARLS") ifelse(is.na(x),NA_real_,as.numeric(x>=3)) else ifelse(is.na(x),NA_real_,as.numeric(x>=1)) }
  out <- data.frame(cog_imrc=as.numeric(cog_imrc),cog_dlrc=as.numeric(cog_dlrc),cog_exec=as.numeric(cog_exec),
    loc_room=as.numeric(loc_room),loc_100m=as.numeric(loc_100m),grip_left=as.numeric(grip_left),grip_right=as.numeric(grip_right),
    psych_dep=psych_binary(dat$psych_1),psych_effort=psych_binary(dat$psych_2),psych_sleep=psych_binary(dat$psych_3),psych_happy=psych_binary(dat$psych_4),cohort=cohort)
  out
}
harm <- do.call(rbind, list(common_frame(elsa,"ELSA"),common_frame(charls,"CHARLS"),common_frame(hrs,"HRS")))
harm$cohort <- factor(harm$cohort, levels=c("ELSA","CHARLS","HRS"))
items <- c("cog_imrc","cog_dlrc","cog_exec","loc_room","loc_100m","grip_left","grip_right","psych_dep","psych_effort","psych_sleep","psych_happy")
ordered_items <- c("loc_room","loc_100m","psych_dep","psych_effort","psych_sleep","psych_happy")
for (v in ordered_items) harm[[v]] <- as.ordered(harm[[v]])
syntax <- paste("cognition =~ cog_imrc + cog_dlrc + cog_exec","locomotion =~ loc_room + loc_100m","grip_vitality =~ grip_left + grip_right","psychological =~ psych_dep + psych_effort + psych_sleep + psych_happy","cognition ~~ locomotion + grip_vitality + psychological","locomotion ~~ grip_vitality + psychological","grip_vitality ~~ psychological",sep="\n")
partial_domain <- c("cognition=~cog_exec","grip_vitality=~grip_right","locomotion=~loc_100m","psychological=~psych_happy")
fit_safe <- function(equal) tryCatch(lavaan::cfa(syntax,data=harm,group="cohort",ordered=ordered_items,estimator="WLSMV",parameterization="theta",std.lv=TRUE,meanstructure=TRUE,missing="pairwise",group.equal=equal,group.partial=partial_domain), error=function(e) structure(list(error=conditionMessage(e)),class="fit_error"))
fits <- list(metric_partial_domain=fit_safe("loadings"),scalar_partial_domain=fit_safe(c("loadings","thresholds","intercepts")))
fit_row <- function(label,f) { if (inherits(f,"fit_error")) return(data.frame(model=label,converged=FALSE,error=f$error)); fm <- fitMeasures(f,c("chisq","df","cfi","tli","rmsea","srmr")); data.frame(model=label,converged=lavInspect(f,"converged"),post_check=tryCatch(lavInspect(f,"post.check"),error=function(e)NA),t(as.data.frame(fm)),error=NA_character_,check.names=FALSE) }
fit_tab <- do.call(rbind,Map(fit_row,names(fits),fits))
param_tab <- do.call(rbind,lapply(names(fits),function(label){ f<-fits[[label]]; if(inherits(f,"fit_error")) return(NULL); pt<-parameterEstimates(f,standardized=TRUE); keep <- (pt$op=="=~" & pt$lhs %in% c("cognition","locomotion","grip_vitality","psychological")) | (pt$op %in% c("|","~1") & pt$lhs %in% ordered_items); z<-pt[keep,c("lhs","op","rhs","group","est","se","pvalue","std.all")]; z$model<-label; z }))
constraint_tab <- do.call(rbind,lapply(names(fits),function(label){f<-fits[[label]];if(inherits(f,"fit_error"))return(NULL);pt<-parTable(f);th<-pt[pt$op=="|" & pt$lhs %in% ordered_items,c("lhs","group","free","label","plabel")]; data.frame(model=label,n_threshold_rows=nrow(th),n_unique_nonempty_labels=length(unique(th$label[nchar(th$label)>0])),threshold_labels_shared=all(tapply(th$label,th$lhs,function(z)length(unique(z[nchar(z)>0]))<=1)),df=fitMeasures(f,"df"),npar=fitMeasures(f,"npar"))}))
write.csv(fit_tab,file.path(TAB,"stage15b_partial_scalar_fit.csv"),row.names=FALSE,fileEncoding="UTF-8")
write.csv(param_tab,file.path(TAB,"stage15b_partial_scalar_parameters.csv"),row.names=FALSE,fileEncoding="UTF-8")
write.csv(constraint_tab,file.path(TAB,"stage15b_partial_scalar_constraint_audit.csv"),row.names=FALSE,fileEncoding="UTF-8")
writeLines(c("# Stage 15b focused partial-scalar follow-up","","The formal scalar gate combines thresholds for ordered indicators and intercepts for continuous indicators. The four prespecified loading exceptions are retained.","","A threshold-only model is retained in Stage 15 for label auditing; under mixed continuous/ordered mean-structure identification it has zero df change versus partial metric and is not used as an invariance LRT.","","Outputs: tables/stage15b_partial_scalar_fit.csv, tables/stage15b_partial_scalar_parameters.csv, tables/stage15b_partial_scalar_constraint_audit.csv."),file.path(OUT,"stage15b_partial_scalar_memo.md"))
writeLines(c(paste0("timestamp=",format(Sys.time(),tz="UTC")),paste0("R=",R.version.string),paste0("lavaan=",as.character(packageVersion("lavaan"))),"design=focused partial metric vs partial scalar; ELSA6 CHARLS3 HRS10 non-proxy"),file.path(OUT,"stage15b_partial_scalar_run_info.txt"))
cat("Wrote Stage 15b focused partial-scalar outputs.\n")
