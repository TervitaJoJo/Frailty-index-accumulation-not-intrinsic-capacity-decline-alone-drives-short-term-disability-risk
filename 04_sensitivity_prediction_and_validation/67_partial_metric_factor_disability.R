#!/usr/bin/env Rscript
# Main latent-score criterion-validity analysis.
# Factor scores are created in memory from the frozen partial-metric CFA and
# are never written with identifiers. Aggregate metrics only are exported.
suppressPackageStartupMessages({ library(haven); library(lavaan) })

ROOT <- Sys.getenv("IC_FRAILTY_STATISTICAL_MODELING", unset = "PATH_TO_STATISTICAL_MODELING")
OUT <- Sys.getenv("IC_FRAILTY_ANALYSIS_ROOT", unset = "PATH_TO_IC_FRAILTY")
TAB <- file.path(OUT, "tables"); dir.create(TAB, recursive=TRUE, showWarnings=FALSE)

num <- function(x, lo=-Inf, hi=Inf) { x <- as.numeric(x); x[is.na(x) | x < lo | x > hi] <- NA_real_; x }
bin01 <- function(x) { x <- as.numeric(x); out <- rep(NA_real_, length(x)); out[x %in% c(0,1)] <- x[x %in% c(0,1)]; out }
max_na <- function(...) { z <- cbind(...); apply(z,1,function(x) if(all(is.na(x))) NA_real_ else max(x,na.rm=TRUE)) }
mean_min <- function(...) { z <- cbind(...); apply(z,1,function(x) if(sum(!is.na(x)) < 1) NA_real_ else mean(x,na.rm=TRUE)) }

fi8 <- function(shlt,bmi,disease) {
  z <- as.data.frame(lapply(disease,bin01)); z$srh <- (num(shlt,1,5)-1)/4
  b <- num(bmi,0,100); z$bmi <- ifelse(is.na(b),NA_real_,as.numeric(b < 18.5 | b >= 30))
  n <- rowSums(!is.na(z)); out <- rowSums(z,na.rm=TRUE)/n; out[n<6] <- NA_real_; out
}

psych_bin <- function(x, charls=FALSE) {
  x <- as.numeric(x)
  if(charls) return(ifelse(x %in% 1:4, as.numeric(x>=3), NA_real_))
  if(all(na.omit(x) %in% c(0,1))) return(ifelse(x %in% c(0,1),x,NA_real_))
  ifelse(x %in% c(1,2), as.numeric(x==1), NA_real_)
}

make_elsa <- function() {
  wp <- file.path(ROOT,"ELSA/Working_data/elsa.dta")
  rp <- file.path(ROOT,"ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_data_v2.dta")
  np <- file.path(ROOT,"ELSA/Raw_data/UKDA-5050-stata/stata/stata13_se/wave6/wave_6_elsa_nurse_data_v2.dta")
  w <- read_dta(wp,col_select=c("idauniqc","wave","agey","ragender","raeducl","imrc","dlrc","orient","walkra","walk100a","shlt","mbmi","hibpe","diabe","hearte","stroke","cancre","arthre","adltot6","iadltot2_e"))
  w <- w[w$wave %in% c(6,7),]; r <- read_dta(rp,col_select=c("idauniq",paste0("PSced",LETTERS[1:8]))); n <- read_dta(np,col_select=c("idauniq","mmgsd1","mmgsd2","mmgsn1","mmgsn2"))
  w$key <- as.character(w$idauniqc); r$key <- as.character(r$idauniq); n$key <- as.character(n$idauniq)
  d <- merge(w,r[,c("key",paste0("PSced",LETTERS[1:8]))],by="key",all.x=TRUE,sort=FALSE)
  d <- merge(d,n[,c("key","mmgsd1","mmgsd2","mmgsn1","mmgsn2")],by="key",all.x=TRUE,sort=FALSE)
  base <- d[d$wave==6,]; fut <- d[d$wave==7,]
  # The first four PSced items are the audited psychological indicators.
  frame <- function(x) data.frame(id=as.character(x$idauniqc), age=num(x$agey), sex=num(x$ragender), education=num(x$raeducl),
    cog_imrc=num(x$imrc,0,10)/10, cog_dlrc=num(x$dlrc,0,10)/10, cog_exec=num(x$orient,0,4)/4,
    loc_room=1-num(x$walkra,0,1), loc_100m=1-num(x$walk100a,0,1),
    grip_left=max_na(num(x$mmgsd1,0,100),num(x$mmgsd2,0,100))/100, grip_right=max_na(num(x$mmgsn1,0,100),num(x$mmgsn2,0,100))/100,
    psych_dep=psych_bin(x$PScedA), psych_effort=psych_bin(x$PScedB), psych_sleep=psych_bin(x$PScedC),
    psych_happy=psych_bin(x$PScedD), fi=fi8(x$shlt,x$mbmi,x[c("hibpe","diabe","hearte","stroke","cancre","arthre")]),
    adl=num(x$adltot6,0,30), iadl=num(x$iadltot2_e,0,30))
  b <- frame(base); f <- frame(fut); ff <- merge(b,f[,c("id","adl","iadl")],by="id",suffixes=c("","_f")); ff$event <- as.numeric((ff$adl_f+ff$iadl_f)>0); ff$baseline_none <- (ff$adl+ff$iadl)==0; ff$observed <- !is.na(ff$adl_f)&!is.na(ff$iadl_f)
  ff$cohort <- "ELSA"; ff$window <- "6->7"; ff
}

make_charls <- function() {
  p <- file.path(ROOT,"CHARLS/Harmonized_CHARLS/H_CHARLS_D_Data.dta")
  cs <- c("ID","inw3","inw4","r3agey","ragender","raeduc_c","r3imrc","r3dlrc","r3orient","r3walk100a","r3walk1kma","r3lgrip1","r3lgrip2","r3rgrip1","r3rgrip2","r3depresl","r3effortl","r3sleeprl","r3whappyl","r3shlt","r3mbmi","r3hibpe","r3diabe","r3hearte","r3stroke","r3cancre","r3arthre","r3adlwa","r3iadla","r4adlwa","r4iadla")
  d <- read_dta(p,col_select=cs); d <- d[d$inw3==1 & d$inw4==1,]
  out <- data.frame(id=as.character(d$ID), age=num(d$r3agey), sex=as.numeric(d$ragender==2), education=num(d$raeduc_c),
    cog_imrc=num(d$r3imrc,0,10)/10,cog_dlrc=num(d$r3dlrc,0,10)/10,cog_exec=num(d$r3orient,0,4)/4,
    loc_room=1-num(d$r3walk100a,0,1),loc_100m=1-num(d$r3walk1kma,0,1),
    grip_left=max_na(num(d$r3lgrip1,0,100),num(d$r3lgrip2,0,100))/100,grip_right=max_na(num(d$r3rgrip1,0,100),num(d$r3rgrip2,0,100))/100,
    psych_dep=psych_bin(d$r3depresl,TRUE),psych_effort=psych_bin(d$r3effortl,TRUE),psych_sleep=psych_bin(d$r3sleeprl,TRUE),psych_happy=psych_bin(d$r3whappyl,TRUE),
    fi=fi8(d$r3shlt,d$r3mbmi,d[c("r3hibpe","r3diabe","r3hearte","r3stroke","r3cancre","r3arthre")]),adl=num(d$r3adlwa,0,30),iadl=num(d$r3iadla,0,30),adl_f=num(d$r4adlwa,0,30),iadl_f=num(d$r4iadla,0,30))
  out$event <- as.numeric((out$adl_f+out$iadl_f)>0); out$baseline_none <- (out$adl+out$iadl)==0; out$observed <- !is.na(out$adl_f)&!is.na(out$iadl_f); out$cohort <- "CHARLS"; out$window <- "3->4"; out
}

make_hrs <- function() {
  p <- file.path(ROOT,"HRS/RAND HRS Data/Longitudinal and Cross-Wave Data Products/randhrs1992_2022v1.dta")
  cs <- c("hhidpn","inw10","inw11","r10proxy","r11proxy","r10agey_m","ragender","raeduc","r10imrc","r10dlrc","r10ser7","r10walkra","r10walk1a","r10walksa","r10grpl","r10grpr","r10depres","r10effort","r10sleepr","r10whappy","r10flone","r10fsad","r10going","r10enlife","r10shlt","r10bmi","r10hibpe","r10diabe","r10hearte","r10stroke","r10cancre","r10arthre","r10adl5a","r10iadl5a","r11adl5a","r11iadl5a")
  d <- read_dta(p,col_select=cs); d <- d[d$inw10==1&d$inw11==1&d$r10proxy==0&d$r11proxy==0,]
  out <- data.frame(id=as.character(d$hhidpn),age=num(d$r10agey_m),sex=as.numeric(d$ragender==2),education=num(d$raeduc),
    cog_imrc=num(d$r10imrc,0,10)/10,cog_dlrc=num(d$r10dlrc,0,10)/10,cog_exec=pmin(num(d$r10ser7,0,5),4)/4,
    loc_room=1-as.numeric(num(d$r10walkra,0,2)>0),loc_100m=1-as.numeric(num(d$r10walk1a,0,2)>0),
    grip_left=num(d$r10grpl,0,100)/100,grip_right=num(d$r10grpr,0,100)/100,
    psych_dep=psych_bin(d$r10depres),psych_effort=psych_bin(d$r10effort),psych_sleep=psych_bin(d$r10sleepr),psych_happy=psych_bin(d$r10whappy),
    fi=fi8(d$r10shlt,d$r10bmi,d[c("r10hibpe","r10diabe","r10hearte","r10stroke","r10cancre","r10arthre")]),adl=num(d$r10adl5a,0,30),iadl=num(d$r10iadl5a,0,30),adl_f=num(d$r11adl5a,0,30),iadl_f=num(d$r11iadl5a,0,30))
  out$event <- as.numeric((out$adl_f+out$iadl_f)>0); out$baseline_none <- (out$adl+out$iadl)==0; out$observed <- !is.na(out$adl_f)&!is.na(out$iadl_f); out$cohort <- "HRS"; out$window <- "10->11"; out
}

elsa <- make_elsa(); charls <- make_charls(); hrs <- make_hrs(); dat <- rbind(elsa,charls,hrs); dat$cohort <- factor(dat$cohort,levels=c("ELSA","CHARLS","HRS"))
items <- c("loc_room","loc_100m","psych_dep","psych_effort","psych_sleep","psych_happy")
for(v in items) dat[[v]] <- ordered(dat[[v]])
syntax <- paste("cognition =~ cog_imrc + cog_dlrc + cog_exec","locomotion =~ loc_room + loc_100m","grip_vitality =~ grip_left + grip_right","psychological =~ psych_dep + psych_effort + psych_sleep + psych_happy","cognition ~~ locomotion + grip_vitality + psychological","locomotion ~~ grip_vitality + psychological","grip_vitality ~~ psychological",sep="\n")
partial <- c("cognition=~cog_exec","grip_vitality=~grip_right","locomotion=~loc_100m","psychological=~psych_happy")
fit <- cfa(syntax,data=dat,group="cohort",ordered=items,estimator="WLSMV",parameterization="theta",std.lv=TRUE,meanstructure=TRUE,missing="pairwise",group.equal="loadings",group.partial=partial)
raw_fs <- lavPredict(fit,type="lv",method="Bartlett")
indicator_all <- c("cog_imrc","cog_dlrc","cog_exec","loc_room","loc_100m","grip_left","grip_right","psych_dep","psych_effort","psych_sleep","psych_happy")
fs <- matrix(NA_real_,nrow=nrow(dat),ncol=4); colnames(fs) <- c("cognition","locomotion","grip_vitality","psychological")
if(is.list(raw_fs)) {
  for(i in seq_along(levels(dat$cohort))) {
    g <- levels(dat$cohort)[i]; idx <- which(dat$cohort==g & rowSums(!is.na(dat[,indicator_all]))>0); z <- as.matrix(raw_fs[[i]]); fs[idx[seq_len(min(length(idx),nrow(z)))],] <- z[seq_len(min(length(idx),nrow(z))),,drop=FALSE]
  }
} else {
  idx <- which(rowSums(!is.na(dat[,indicator_all]))>0); fs[idx[seq_len(min(length(idx),nrow(raw_fs)))],] <- as.matrix(raw_fs)[seq_len(min(length(idx),nrow(raw_fs))),,drop=FALSE]
}
fs <- as.data.frame(fs); fs$psychological <- -fs$psychological; dat <- cbind(dat,fs)
zwithin <- function(x,g) ave(x,g,FUN=function(y) as.numeric((y-mean(y,na.rm=TRUE))/sd(y,na.rm=TRUE)))
for(v in c("age","education","fi","cognition","locomotion","grip_vitality","psychological")) dat[[paste0(v,"_z")]] <- zwithin(dat[[v]],dat$cohort)
base_terms <- c("age_z","sex","education_z","fi_z"); full_terms <- c(base_terms,"cognition_z","locomotion_z","grip_vitality_z","psychological_z")
auc_fun <- function(y,p) { n1<-sum(y==1); n0<-sum(y==0); if(n1==0||n0==0) return(NA_real_); r<-rank(p); (sum(r[y==1])-n1*(n1+1)/2)/(n1*n0) }
metric <- function(y,p,cohort,window,model,sample) { p<-pmin(pmax(p,1e-6),1-1e-6); cal<-glm(y~qlogis(p),family=binomial); data.frame(sample=sample,cohort=cohort,window=window,model=model,n=length(y),events=sum(y),event_rate=mean(y),auc_horizon_cindex=auc_fun(y,p),brier=mean((y-p)^2),calibration_intercept=coef(cal)[1],calibration_slope=coef(cal)[2],mean_predicted_risk=mean(p),observed_risk=mean(y),observed_expected_ratio=mean(y)/mean(p)) }
rows <- list(); coefs <- list()
for(g in levels(dat$cohort)) { d<-dat[dat$cohort==g & dat$baseline_none & dat$observed,]; d<-d[complete.cases(d[,c("event",full_terms)]),]; if(nrow(d)<100) next; mb<-glm(as.formula(paste("event~",paste(base_terms,collapse="+"))),data=d,family=binomial); mf<-glm(as.formula(paste("event~",paste(full_terms,collapse="+"))),data=d,family=binomial); rows[[length(rows)+1]]<-metric(d$event,predict(mb,type="response"),g,unique(d$window),"base","latent_internal"); rows[[length(rows)+1]]<-metric(d$event,predict(mf,type="response"),g,unique(d$window),"base_plus_partial_metric_IC","latent_internal"); coefs[[length(coefs)+1]]<-data.frame(cohort=g,model="base_plus_partial_metric_IC",term=names(coef(mf)),estimate=coef(mf),odds_ratio=exp(coef(mf))) }
d <- dat[dat$baseline_none & dat$observed,]; d <- d[complete.cases(d[,c("event",full_terms)]),]; mb<-glm(as.formula(paste("event~",paste(base_terms,collapse="+"))),data=d,family=binomial); mf<-glm(as.formula(paste("event~",paste(full_terms,collapse="+"))),data=d,family=binomial); rows[[length(rows)+1]]<-metric(d$event,predict(mb,type="response"),"ELSA+CHARLS+HRS","anchor_windows","base","latent_development_pooled"); rows[[length(rows)+1]]<-metric(d$event,predict(mf,type="response"),"ELSA+CHARLS+HRS","anchor_windows","base_plus_partial_metric_IC","latent_development_pooled"); coefs[[length(coefs)+1]]<-data.frame(cohort="ELSA+CHARLS+HRS",model="base_plus_partial_metric_IC",term=names(coef(mf)),estimate=coef(mf),odds_ratio=exp(coef(mf)))
write.csv(do.call(rbind,rows),file.path(TAB,"incident_disability_latent_factor_metrics.csv"),row.names=FALSE,fileEncoding="UTF-8")
write.csv(do.call(rbind,coefs),file.path(TAB,"incident_disability_latent_factor_coefficients.csv"),row.names=FALSE,fileEncoding="UTF-8")
summ <- do.call(rbind,lapply(levels(dat$cohort),function(g) { d<-dat[dat$cohort==g & dat$baseline_none & dat$observed,]; data.frame(cohort=g,eligible=nrow(d),events=sum(d$event,na.rm=TRUE),complete_latent=sum(complete.cases(d[,c("event",full_terms)]))) }))
write.csv(summ,file.path(TAB,"incident_disability_latent_factor_sample_summary.csv"),row.names=FALSE,fileEncoding="UTF-8")
writeLines(c("# Partial-metric latent factor-score disability analysis","","The primary-cohort CFA uses the audited four-domain indicator set and the prespecified partial-metric loading exceptions. Individual factor scores are estimated in memory with lavaan and are not exported. The scores are sign-aligned so higher values indicate better capacity. Base and IC-augmented logistic models include age, sex, education and baseline outcome-disjoint FI. Predictors are standardised within cohort. The fixed-window AUC is the C-index equivalent because onset dates are not observed. This is the latent-score primary analysis; the Python proxy analysis supplies SHARE external validation and measurement-sensitive sensitivity results."),file.path(OUT,"incident_disability_latent_factor_memo.md"))
writeLines(c(paste0("timestamp=",format(Sys.time(),tz="UTC")),paste0("R=",R.version.string),paste0("lavaan=",as.character(packageVersion("lavaan"))),"factor_scores=partial_metric_CFA; IDs_and_scores_not_exported","primary_window=ELSA6-7; CHARLS3-4; HRS10-11"),file.path(OUT,"incident_disability_latent_factor_run_info.txt"))
print(do.call(rbind,rows))
