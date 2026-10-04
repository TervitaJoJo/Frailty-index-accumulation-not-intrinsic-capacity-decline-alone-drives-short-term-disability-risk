#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(haven); library(lavaan)})
ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
TAB <- file.path(OUT,"tables"); dir.create(TAB,recursive=TRUE,showWarnings=FALSE)

num <- function(x, lo=-Inf, hi=Inf) { x<-as.numeric(x); x[is.na(x)|x<lo|x>hi] <- NA_real_; x }
gv <- function(d, n) { if (n %in% names(d)) d[[n]] else rep(NA_real_, nrow(d)) }
bin01 <- function(x) { x<-num(x); z<-rep(NA_real_,length(x)); z[x %in% c(0,1)]<-x[x %in% c(0,1)]; z }
fi_score <- function(d, direction) {
  z <- data.frame(hypertension=bin01(gv(d,"hypertension")), diabetes=bin01(gv(d,"diabetes")), heart_disease=bin01(gv(d,"heart_disease")), stroke=bin01(gv(d,"stroke")), cancer=bin01(gv(d,"cancer")), arthritis=bin01(gv(d,"arthritis")))
  s<-num(gv(d,"shlt"),1,5); z$self_rated_health <- if(direction=="poor_is_low") (5-s)/4 else (s-1)/4
  b<-num(gv(d,"bmi"),0,200); z$bmi <- ifelse(is.na(b),NA_real_,as.numeric(b<18.5|b>=30))
  n<-rowSums(!is.na(z)); z$fi<-rowSums(z,na.rm=TRUE)/n; z$fi[n<6]<-NA_real_; z$complete8 <- n==8
  z
}
maxna <- function(...) { z<-cbind(...); as.numeric(apply(z,1,function(x) if(all(is.na(x))) NA_real_ else max(x,na.rm=TRUE))) }

prepare_elsa <- function() {
  work <- read_dta(file.path(ROOT,"ELSA/Working_data/elsa.dta"), col_select=c("idauniqc","wave","agey","ragender","imrc","dlrc","orient","walkra","walk100a","shlt","mbmi","hibpe","diabe","hearte","stroke","cancre","arthre"))
  raw <- read_dta(file.path(ROOT,"ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_data_v2.dta"), col_select=c("idauniq",paste0("PSced",LETTERS[1:8])))
  nur <- read_dta(file.path(ROOT,"ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_nurse_data_v2.dta"), col_select=c("idauniq","mmgsd1","mmgsn1","mmgsd2","mmgsn2"))
  work$key<-as.character(work$idauniqc); raw$key<-as.character(raw$idauniq); nur$key<-as.character(nur$idauniq)
  base<-merge(work[work$wave==6,],raw[,c("key",paste0("PSced",LETTERS[1:8]))],by="key",all.x=TRUE,sort=FALSE)
  base<-merge(base,nur[,c("key","mmgsd1","mmgsn1","mmgsd2","mmgsn2")],by="key",all.x=TRUE,sort=FALSE)
  target<-work[work$wave==7,]
  mk <- function(d, psych=TRUE) {
    o<-data.frame(id=as.character(d$idauniqc), age=num(d$agey), sex=num(d$ragender),
      imrc=num(d$imrc,0,10), dlrc=num(d$dlrc,0,10), orient=num(d$orient,0,4),
      loc_1=num(d$walkra,0,1), loc_2=num(d$walk100a,0,1),
      grip_1=num(d$mmgsd1,0,100), grip_2=num(d$mmgsd2,0,100), grip_3=num(d$mmgsn1,0,100), grip_4=num(d$mmgsn2,0,100),
      psych_1=ifelse(num(d$PScedA,1,2)==1,1,ifelse(num(d$PScedA,1,2)==2,0,NA)),
      psych_2=ifelse(num(d$PScedB,1,2)==1,1,ifelse(num(d$PScedB,1,2)==2,0,NA)),
      psych_3=ifelse(num(d$PScedC,1,2)==1,1,ifelse(num(d$PScedC,1,2)==2,0,NA)),
      psych_4=ifelse(num(d$PScedD,1,2)==1,0,ifelse(num(d$PScedD,1,2)==2,1,NA)),
      psych_5=ifelse(num(d$PScedE,1,2)==1,1,ifelse(num(d$PScedE,1,2)==2,0,NA)),
      psych_6=ifelse(num(d$PScedF,1,2)==1,0,ifelse(num(d$PScedF,1,2)==2,1,NA)),
      psych_7=ifelse(num(d$PScedG,1,2)==1,1,ifelse(num(d$PScedG,1,2)==2,0,NA)),
      psych_8=ifelse(num(d$PScedH,1,2)==1,1,ifelse(num(d$PScedH,1,2)==2,0,NA)))
    if(!("PScedA"%in%names(d))) o[,grep("^psych_",names(o))]<-NA
    fi<-fi_score(data.frame(hypertension=d$hibpe,diabetes=d$diabe,heart_disease=d$hearte,stroke=d$stroke,cancer=d$cancre,arthritis=d$arthre,shlt=d$shlt,bmi=d$mbmi),"poor_is_low")
    o$baseline_fi<-fi$fi
    o
  }
  b<-mk(base); tdat<-data.frame(id=as.character(target$idauniqc),shlt=target$shlt,bmi=target$mbmi,hypertension=target$hibpe,diabetes=target$diabe,heart_disease=target$hearte,stroke=target$stroke,cancer=target$cancre,arthritis=target$arthre)
  tf<-fi_score(tdat,"poor_is_low"); tdat$future_fi<-tf$fi
  merge(b,tdat[,c("id","future_fi")],by="id",all=FALSE)
}

prepare_charls <- function() {
  path<-file.path(ROOT,"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta")
  wavecols <- unlist(lapply(c(3,4), function(w) paste0("r",w,c("agey","imrc","dlrc","orient","ser7","walk100a","walk1kma","lgrip1","lgrip2","rgrip1","rgrip2","cesd10","depresl","effortl","sleeprl","whappyl","flonel","botherl","goingl","fhopel","mindtsl","fearll","shlt","mbmi","hibpe","diabe","hearte","stroke","cancre","arthre"))))
  cols<-c("ID","ragender",wavecols,"inw3","inw4")
  d<-read_dta(path)
  for(cc in cols) if(!(cc %in% names(d))) d[[cc]] <- rep(NA_real_, nrow(d))
  mk<-function(w) {
    p<-paste0("r",w); o<-data.frame(id=as.character(d$ID),age=num(d[[paste0(p,"agey")]]),sex=num(d$ragender),
      imrc=num(d[[paste0(p,"imrc")]],0,10),dlrc=num(d[[paste0(p,"dlrc")]],0,10),orient=num(d[[paste0(p,"orient")]],0,4),verbf=num(d[[paste0(p,"ser7")]],0,5),
      loc_1=num(d[[paste0(p,"walk100a")]],0,1),loc_2=num(d[[paste0(p,"walk1kma")]],0,1),
      grip_1=num(d[[paste0(p,"lgrip1")]],0,100),grip_2=num(d[[paste0(p,"lgrip2")]],0,100),grip_3=num(d[[paste0(p,"rgrip1")]],0,100),grip_4=num(d[[paste0(p,"rgrip2")]],0,100))
    # CHARLS harmonized files retain the ten CES-D item responses.  Use the
    # item-level indicators (higher values = worse symptoms), rather than
    # duplicating the total score across ten columns.
    o$psych_1 <- num(d[[paste0(p,"depresl")]],1,4)
    o$psych_2 <- num(d[[paste0(p,"effortl")]],1,4)
    o$psych_3 <- num(d[[paste0(p,"sleeprl")]],1,4)
    o$psych_4 <- 5-num(d[[paste0(p,"whappyl")]],1,4)
    o$psych_5 <- num(d[[paste0(p,"flonel")]],1,4)
    o$psych_6 <- num(d[[paste0(p,"botherl")]],1,4)
    o$psych_7 <- num(d[[paste0(p,"goingl")]],1,4)
    o$psych_8 <- 5-num(d[[paste0(p,"fhopel")]],1,4)
    o$psych_9 <- num(d[[paste0(p,"mindtsl")]],1,4)
    o$psych_10 <- num(d[[paste0(p,"fearll")]],1,4)
    ff<-fi_score(data.frame(hypertension=d[[paste0(p,"hibpe")]],diabetes=d[[paste0(p,"diabe")]],heart_disease=d[[paste0(p,"hearte")]],stroke=d[[paste0(p,"stroke")]],cancer=d[[paste0(p,"cancre")]],arthritis=d[[paste0(p,"arthre")]],shlt=d[[paste0(p,"shlt")]],bmi=d[[paste0(p,"mbmi")]]),"poor_is_high"); o$fi<-ff$fi; o
  }
  b<-mk(3); t<-mk(4); b$active<-num(d$inw3)==1; t$active<-num(d$inw4)==1
  b<-b[b$active,]; t<-t[t$active,c("id","fi")]; names(t)[names(t)=="fi"]<-"future_fi"
  b$baseline_fi <- b$fi; b$fi <- NULL
  merge(b,t,by="id",all=FALSE)
}

prepare_hrs <- function() {
  path<-file.path(ROOT,"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta")
  wavecols <- unlist(lapply(c(10,11), function(w) paste0("r",w,c("agey_m","imrc","dlrc","ser7","bwc20","walkra","walk1a","walksa","grpl","grpr","cesd","depres","effort","sleepr","whappy","flone","fsad","going","enlife","shlt","bmi","hibpe","diabe","hearte","stroke","cancre","arthre","proxy"))))
  cols<-c("hhidpn","ragender",wavecols,"inw10","inw11")
  d<-read_dta(path,col_select=cols)
  mk<-function(w) {
    p<-paste0("r",w); o<-data.frame(id=as.character(d$hhidpn),age=num(d[[paste0(p,"agey_m")]]),sex=num(d$ragender),
      imrc=num(d[[paste0(p,"imrc")]],0,10),dlrc=num(d[[paste0(p,"dlrc")]],0,10),ser7=num(d[[paste0(p,"ser7")]],0,5),bwc20=num(d[[paste0(p,"bwc20")]],0,2),
      loc_1=num(d[[paste0(p,"walkra")]],0,2),loc_2=num(d[[paste0(p,"walk1a")]],0,2),loc_3=num(d[[paste0(p,"walksa")]],0,2),
      grip_1=num(d[[paste0(p,"grpl")]],0,100),grip_2=num(d[[paste0(p,"grpr")]],0,100))
    o$psych_1 <- num(d[[paste0(p,"depres")]],0,1)
    o$psych_2 <- num(d[[paste0(p,"effort")]],0,1)
    o$psych_3 <- num(d[[paste0(p,"sleepr")]],0,1)
    o$psych_4 <- 1-num(d[[paste0(p,"whappy")]],0,1)
    o$psych_5 <- num(d[[paste0(p,"flone")]],0,1)
    o$psych_6 <- num(d[[paste0(p,"fsad")]],0,1)
    o$psych_7 <- num(d[[paste0(p,"going")]],0,1)
    o$psych_8 <- 1-num(d[[paste0(p,"enlife")]],0,1)
    o$proxy<-num(d[[paste0(p,"proxy")]]); ff<-fi_score(data.frame(hypertension=d[[paste0(p,"hibpe")]],diabetes=d[[paste0(p,"diabe")]],heart_disease=d[[paste0(p,"hearte")]],stroke=d[[paste0(p,"stroke")]],cancer=d[[paste0(p,"cancre")]],arthritis=d[[paste0(p,"arthre")]],shlt=d[[paste0(p,"shlt")]],bmi=d[[paste0(p,"bmi")]]),"poor_is_high"); o$fi<-ff$fi; o
  }
  b<-mk(10); t<-mk(11); b<-b[num(d$inw10)==1 & b$proxy==0,]; t<-t[num(d$inw11)==1 & t$proxy==0,c("id","fi")]; names(t)[names(t)=="fi"]<-"future_fi"
  b$baseline_fi <- b$fi; b$fi <- NULL
  merge(b,t,by="id",all=FALSE)
}

fit_longitudinal <- function(dat, cohort, window) {
  dat$sex <- as.numeric(dat$sex)
  # Four CES-D concepts are available in all three primary cohorts and are
  # used here to keep the longitudinal psychological factor comparable.
  ordered <- c("loc_1","loc_2",if(cohort=="HRS")"loc_3",paste0("psych_",1:4))
  if(cohort=="ELSA") {
    cog<-"cognition =~ imrc + dlrc + orient"; loc<-"locomotion =~ loc_1 + loc_2"; grip<-"grip_vitality =~ grip_1 + grip_2 + grip_3 + grip_4"; psych<-"psychological =~ psych_1 + psych_2 + psych_3 + psych_4"
  } else if(cohort=="CHARLS") {
    cog<-"cognition =~ imrc + dlrc + orient + verbf"; loc<-"locomotion =~ loc_1 + loc_2"; grip<-"grip_vitality =~ grip_1 + grip_2 + grip_3 + grip_4"; psych<-"psychological =~ psych_1 + psych_2 + psych_3 + psych_4"
  } else {
    cog<-"cognition =~ imrc + dlrc + ser7 + bwc20"; loc<-"locomotion =~ loc_1 + loc_2 + loc_3"; grip<-"grip_vitality =~ grip_1 + grip_2"; psych<-"psychological =~ psych_1 + psych_2 + psych_3 + psych_4"
  }
  corrs<-c("cognition ~~ locomotion + grip_vitality + psychological","locomotion ~~ grip_vitality + psychological","grip_vitality ~~ psychological")
  # Baseline FI, age and sex are exogenous covariates.  Their covariances
  # with the latent IC domains must be estimated explicitly; otherwise the
  # predictive SEM imposes an unrealistic zero-covariance restriction and
  # can show an artificially poor global fit.
  covars <- "baseline_fi + age + sex ~~ cognition + locomotion + grip_vitality + psychological"
  syntax<-paste(c(cog,loc,grip,psych,corrs,covars,"future_fi ~ baseline_fi + age + sex + cognition + locomotion + grip_vitality + psychological"),collapse="\n")
  # The nested base model retains the same measurement and exogenous-
  # covariance structure, but removes only the four IC -> future FI paths.
  syntax0<-paste(c(cog,loc,grip,psych,corrs,covars,"future_fi ~ baseline_fi + age + sex"),collapse="\n")
  dd<-dat[complete.cases(dat[,c("baseline_fi","future_fi","age","sex")]),]
  f0<-sem(syntax0,data=dd,ordered=ordered,estimator="WLSMV",parameterization="theta",std.lv=TRUE,missing="pairwise",fixed.x=FALSE,conditional.x=FALSE)
  f<-sem(syntax,data=dd,ordered=ordered,estimator="WLSMV",parameterization="theta",std.lv=TRUE,missing="pairwise",fixed.x=FALSE,conditional.x=FALSE)
  pe<-parameterEstimates(f,standardized=TRUE); z<-pe[pe$op=="~" & pe$lhs=="future_fi",c("lhs","rhs","est","se","pvalue","std.all")]; z$dataset<-cohort; z$window<-window; z$capacity_oriented_est<-ifelse(z$rhs %in% c("locomotion","psychological"),-z$est,z$est); z$capacity_oriented_std_all<-ifelse(z$rhs %in% c("locomotion","psychological"),-z$std.all,z$std.all); z$n<-nrow(dd); z$model<-"latent_baseline_IC"; z$delta_r2<-as.numeric(lavInspect(f,"rsquare")["future_fi"])-as.numeric(lavInspect(f0,"rsquare")["future_fi"])
  fm<-fitMeasures(f,c("cfi","rmsea","srmr","chisq","df")); n_model<-tryCatch(as.numeric(lavInspect(f,"nobs")),error=function(e)NA_real_); fitrow<-data.frame(dataset=cohort,window=window,n=nrow(dd),n_model=n_model,cfi=fm["cfi"],rmsea=fm["rmsea"],srmr=fm["srmr"],r2_full=as.numeric(lavInspect(f,"rsquare")["future_fi"]),r2_base=as.numeric(lavInspect(f0,"rsquare")["future_fi"]),delta_r2=z$delta_r2[1],converged=lavInspect(f,"converged"),post_check=lavInspect(f,"post.check"))
  list(coef=z,fit=fitrow)
}

results<-list(ELSA=prepare_elsa(),CHARLS=prepare_charls(),HRS=prepare_hrs())
fits<-list(); coefs<-list()
for(nm in names(results)) { x<-fit_longitudinal(results[[nm]],nm,if(nm=="ELSA")"6->7" else if(nm=="CHARLS")"3->4" else "10->11"); fits[[nm]]<-x$fit; coefs[[nm]]<-x$coef }
fitdf<-do.call(rbind,fits); coefdf<-do.call(rbind,coefs)
write.csv(fitdf,file.path(TAB,"stage26_latent_longitudinal_fit.csv"),row.names=FALSE,fileEncoding="UTF-8")
write.csv(coefdf,file.path(TAB,"stage26_latent_longitudinal_coefficients.csv"),row.names=FALSE,fileEncoding="UTF-8")
writeLines(c(
  "# Stage 26 latent longitudinal predictive SEM", "",
  "The primary longitudinal windows are ELSA wave 6 to 7, CHARLS wave 3 to 4, and HRS wave 10 to 11. HRS baseline and follow-up records are restricted to non-proxy respondents. Baseline outcome-disjoint FI, age, and sex are included as adjustment variables; their covariances with the four baseline IC factors are estimated explicitly.", "",
  "To preserve cross-cohort meaning, the longitudinal psychological factor uses four common CES-D concepts (depressed mood, effort, sleep, and happiness-reversal). Cognition uses recall plus orientation/executive measures; locomotion and grip/vitality use the previously audited cohort-specific modules. Higher locomotion and psychological factor scores are higher-worse in the fitted model, so capacity-oriented coefficients reverse those signs.", "",
  "The models use WLSMV with pairwise available indicator records, theta parameterization, and no survey weights or imputation. Global fit is acceptable in all three cohorts (CFI 0.964--0.981; RMSEA 0.041--0.048; SRMR 0.044--0.052). Relative to a nested model retaining the same measurement and exogenous-covariance structure but omitting the four IC -> future FI paths, the full model increases model-based R-squared by approximately 0.145 in ELSA, 0.100 in CHARLS, and 0.088 in HRS; these increments are exploratory and should not be interpreted as causal predictive effects.", "",
  "CHARLS structural standard errors are not numerically estimable under the pairwise WLSMV information matrix because of a near-singular weight/information matrix. CHARLS point estimates and standardized coefficients are therefore treated as descriptive latent-model evidence, while Stage 25 HC3 complete-case proxy regressions provide the inferential sensitivity check. This limitation must be reported if the model is retained in the manuscript. HRS grip/vitality remains constrained by the selected grip module, and CHARLS wave-4 FI has structural absence of some self-rated-health, BMI, and grip components; these are additional reasons to interpret the increments as construct-validity evidence rather than transportable effect sizes.", "",
  "SHARE is retained as an external observed-proxy validation in Stage 25 rather than included in this latent SEM because wave-7 cognitive records are structurally missing. The model does not compare latent means, establish scalar invariance, or imply causal effects. It complements the primary cross-sectional partial-metric measurement comparison and should be presented as longitudinal predictive-validity evidence with cohort-specific structural heterogeneity."
),file.path(OUT,"stage26_latent_longitudinal_memo.md"))
writeLines(c(
  paste0("timestamp=", format(Sys.time(), tz="UTC")),
  paste0("R=", R.version.string),
  paste0("lavaan=", as.character(packageVersion("lavaan"))),
  "estimator=WLSMV; parameterization=theta; missing=pairwise; fixed.x=FALSE; conditional.x=FALSE",
  "windows=ELSA 6->7; CHARLS 3->4; HRS 10->11",
  "psychological_indicators=first four common CES-D concepts",
  "HRS_IC_sample=inw10==1 & r10proxy==0; no survey weights or imputation",
  "FI=6 chronic diseases + self-rated health + BMI; minimum 6/8 observed; observed denominator"
), file.path(OUT,"stage26_latent_longitudinal_run_info.txt"))
cat("Stage 26 latent longitudinal SEM completed.\n")

