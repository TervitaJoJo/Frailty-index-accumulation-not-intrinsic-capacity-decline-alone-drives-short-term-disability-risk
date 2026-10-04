#!/usr/bin/env Rscript
# Stage 16: cross-cohort partial-metric IC--FI structural comparison.
# The harmonized indicators are used only to put the four IC domains on a
# common loading scale. FI remains an observed outcome; no latent means are
# estimated or compared.
suppressPackageStartupMessages({ library(haven); library(lavaan) })
ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
CODE <- Sys.getenv("IC_FRAILTY_CODE_ROOT", unset = "PATH_TO_IC_FRAILTY_CODE")
TAB <- file.path(OUT, "tables"); dir.create(TAB, recursive=TRUE, showWarnings=FALSE)
source(file.path(CODE, "02_measurement_and_fi_validation", "29_domain_reliability_audit.R"), local=.GlobalEnv)

max_na <- function(...) { z<-cbind(...); as.numeric(apply(z,1,function(x) if(all(is.na(x))) NA_real_ else max(x,na.rm=TRUE))) }
common_frame <- function(dat, cohort) {
  if (cohort=="ELSA") {
    ci<-dat$imrc; cd<-dat$dlrc; ce<-dat$orient
    lr<-ifelse(is.na(dat$loc_1),NA_real_,1-dat$loc_1); lm<-ifelse(is.na(dat$loc_2),NA_real_,1-dat$loc_2)
    gl<-max_na(dat$grip_1,dat$grip_2); gr<-max_na(dat$grip_3,dat$grip_4)
  } else if (cohort=="CHARLS") {
    ci<-dat$imrc; cd<-dat$dlrc; ce<-dat$orient
    lr<-ifelse(is.na(dat$loc_1),NA_real_,1-dat$loc_1); lm<-ifelse(is.na(dat$loc_2),NA_real_,1-dat$loc_2)
    gl<-max_na(dat$grip_1,dat$grip_2); gr<-max_na(dat$grip_3,dat$grip_4)
  } else {
    ci<-dat$cog_imrc; cd<-dat$cog_dlrc; ce<-pmin(dat$cog_ser7,4)
    lr<-ifelse(is.na(dat$loc_1),NA_real_,1-as.numeric(dat$loc_1>0)); lm<-ifelse(is.na(dat$loc_2),NA_real_,1-as.numeric(dat$loc_2>0))
    gl<-dat$grip_1; gr<-dat$grip_2
  }
  pb <- function(x) { x<-as.numeric(x); if(cohort=="CHARLS") ifelse(is.na(x),NA_real_,as.numeric(x>=3)) else ifelse(is.na(x),NA_real_,as.numeric(x>=1)) }
  # Put continuous indicators on comparable numerical scales for WLSMV
  # optimization; the transformations are monotone and do not alter the
  # construct direction. FI is expressed as percentage points in the SEM.
  data.frame(cog_imrc=as.numeric(ci)/10,cog_dlrc=as.numeric(cd)/10,cog_exec=as.numeric(ce)/4,loc_room=as.numeric(lr),loc_100m=as.numeric(lm),grip_left=as.numeric(gl)/100,grip_right=as.numeric(gr)/100,psych_dep=pb(dat$psych_1),psych_effort=pb(dat$psych_2),psych_sleep=pb(dat$psych_3),psych_happy=pb(dat$psych_4),fi_primary=100*as.numeric(dat$fi_primary),fi_complete8=100*as.numeric(dat$fi_complete8),fi_without_srh=100*as.numeric(dat$fi_without_srh),cohort=cohort)
}
harm <- do.call(rbind,list(common_frame(elsa,"ELSA"),common_frame(charls,"CHARLS"),common_frame(hrs,"HRS")))
harm$cohort <- factor(harm$cohort,levels=c("ELSA","CHARLS","HRS"))
ordered_items <- c("loc_room","loc_100m","psych_dep","psych_effort","psych_sleep","psych_happy")
for(v in ordered_items) harm[[v]] <- as.ordered(harm[[v]])

partial_loadings <- c("cognition=~cog_exec","grip_vitality=~grip_right","locomotion=~loc_100m","psychological=~psych_happy")
all_structural <- c("fi_primary~cognition","fi_primary~locomotion","fi_primary~grip_vitality","fi_primary~psychological")
other_structural <- c("fi_primary~cognition","fi_primary~grip_vitality","fi_primary~psychological")
base_syntax <- paste("cognition =~ cog_imrc + cog_dlrc + cog_exec","locomotion =~ loc_room + loc_100m","grip_vitality =~ grip_left + grip_right","psychological =~ psych_dep + psych_effort + psych_sleep + psych_happy","cognition ~~ locomotion + grip_vitality + psychological","locomotion ~~ grip_vitality + psychological","grip_vitality ~~ psychological","fi_primary ~ cognition + locomotion + grip_vitality + psychological",sep="\n")
fit_safe <- function(equal=character(0), partial=character(0)) tryCatch(lavaan::sem(base_syntax,data=harm,group="cohort",ordered=ordered_items,estimator="WLSMV",parameterization="theta",std.lv=TRUE,meanstructure=TRUE,missing="pairwise",group.equal=equal,group.partial=partial), error=function(e) structure(list(error=conditionMessage(e)),class="fit_error"))
fits <- list(
  configural_free = fit_safe(),
  partial_metric_free = fit_safe("loadings",partial_loadings),
  partial_metric_equal_all_slopes = fit_safe(c("loadings","regressions"),c(partial_loadings)),
  partial_metric_equal_locomotion = fit_safe(c("loadings","regressions"),c(partial_loadings,other_structural))
)
fit_row <- function(label,f) {
  if(inherits(f,"fit_error")) return(data.frame(model=label,converged=FALSE,post_check=FALSE,n=NA,chisq=NA,df=NA,cfi=NA,tli=NA,rmsea=NA,srmr=NA,error=f$error,check.names=FALSE))
  conv <- tryCatch(lavInspect(f,"converged"),error=function(e)FALSE)
  fm <- if (isTRUE(conv)) tryCatch(fitMeasures(f,c("chisq","df","cfi","tli","rmsea","srmr")),error=function(e)setNames(rep(NA_real_,6),c("chisq","df","cfi","tli","rmsea","srmr"))) else setNames(rep(NA_real_,6),c("chisq","df","cfi","tli","rmsea","srmr"))
  data.frame(model=label,converged=conv,post_check=tryCatch(lavInspect(f,"post.check"),error=function(e)NA),n=tryCatch(sum(lavInspect(f,"nobs")),error=function(e)NA),t(as.data.frame(fm)),error=NA_character_,check.names=FALSE)
}
fit_tab <- do.call(rbind,Map(fit_row,names(fits),fits))
coef_tab <- do.call(rbind,lapply(names(fits),function(label){ f<-fits[[label]]; if(inherits(f,"fit_error") || !isTRUE(tryCatch(lavInspect(f,"converged"),error=function(e)FALSE)))return(NULL); pe<-parameterEstimates(f,standardized=TRUE); z<-pe[pe$op=="~" & pe$lhs=="fi_primary",c("lhs","rhs","group","est","se","pvalue","std.all")]; z$model<-label; # locomotion indicators are already reverse-coded to higher capacity; psychological remains higher-worse
  z$capacity_oriented_est<-ifelse(z$rhs %in% c("psychological"),-z$est,z$est); z$capacity_oriented_std_all<-ifelse(z$rhs %in% c("psychological"),-z$std.all,z$std.all); z }))
safe_lrt <- function(a,b,label){
  cols <- c("Df","AIC","BIC","Chisq","Chisq diff","Df diff","Pr(>Chisq)","model","comparison","error")
  empty <- function(msg) { z <- as.data.frame(matrix(NA,nrow=1,ncol=length(cols))); names(z)<-cols; z$comparison<-label; z$error<-msg; z }
  if(inherits(a,"fit_error")||inherits(b,"fit_error")) return(empty("one model failed"))
  z<-tryCatch(lavTestLRT(a,b),error=function(e)NULL); if(is.null(z)) return(empty("lavTestLRT failed"))
  zz<-as.data.frame(z);zz$model<-rownames(zz);zz$comparison<-label;zz$error<-NA_character_;rownames(zz)<-NULL
  for (nm in setdiff(cols,names(zz))) zz[[nm]] <- NA
  zz[,cols,drop=FALSE]
}
lrt_tab <- rbind(safe_lrt(fits$configural_free,fits$partial_metric_free,"configural_vs_partial_metric"),safe_lrt(fits$partial_metric_free,fits$partial_metric_equal_all_slopes,"free_vs_equal_all_structural_slopes"),safe_lrt(fits$partial_metric_free,fits$partial_metric_equal_locomotion,"free_vs_equal_locomotion_slope"))
write.csv(fit_tab,file.path(TAB,"stage16_cross_cohort_partial_metric_sem_fit.csv"),row.names=FALSE,fileEncoding="UTF-8")
write.csv(coef_tab,file.path(TAB,"stage16_cross_cohort_partial_metric_sem_coefficients.csv"),row.names=FALSE,fileEncoding="UTF-8")
write.csv(lrt_tab,file.path(TAB,"stage16_cross_cohort_partial_metric_sem_lrt.csv"),row.names=FALSE,fileEncoding="UTF-8")
writeLines(c("# Stage 16 cross-cohort partial-metric IC--FI SEM","","This exploratory multi-group SEM uses the same cross-cohort analogue indicators and the prespecified partial-metric loading exceptions from Stage 15. FI is observed and uses the primary 6/8 observed-component denominator with clinical BMI threshold. HRS remains restricted to non-proxy respondents.","","The free-slope model estimates cohort-specific structural slopes after partial metric alignment. Two nested sensitivity constraints test (i) all four slopes equal and (ii) only the locomotion slope equal while cognition, grip/vitality and psychological slopes remain cohort-specific. These are structural association comparisons, not causal effects; no latent means are compared.","","The harmonized locomotion indicators were reverse-coded before modeling, so locomotion coefficients are already capacity-oriented (higher locomotion = better capacity). Psychological indicators retain higher-worse symptom direction; the coefficient table reverses only that domain in its capacity-oriented columns.","","Threshold/scalar invariance is not used for the main structural comparison because the scalar candidate had post-check failure and a HRS grip-related standardized loading above one. The result is therefore a partial-metric structural comparison with explicit measurement limitations.","","Outputs: `tables/stage16_cross_cohort_partial_metric_sem_fit.csv`, `tables/stage16_cross_cohort_partial_metric_sem_coefficients.csv`, and `tables/stage16_cross_cohort_partial_metric_sem_lrt.csv`."),file.path(OUT,"stage16_cross_cohort_partial_metric_sem_memo.md"))
writeLines(c(paste0("timestamp=",format(Sys.time(),tz="UTC")),paste0("R=",R.version.string),paste0("lavaan=",as.character(packageVersion("lavaan"))),"estimator=WLSMV; partial metric; observed FI; HRS non-proxy"),file.path(OUT,"stage16_cross_cohort_partial_metric_sem_run_info.txt"))
cat("Wrote Stage 16 cross-cohort partial-metric structural outputs.\n")
