# Base R only. Recreate the bundled numeric references without installing packages.
# Usage: Rscript verify_loess.R INPUT_CSV OUTPUT_DIRECTORY
# INPUT_CSV is data/r_reference/r_input.csv prepared from the hash-pinned source.
# The source's original method is geom_smooth(method='loess') on SampleDepth.max.
# This runs stats::loess itself; it does not emulate the entire publication figure.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Expected INPUT_CSV OUTPUT_DIRECTORY')
d <- read.csv(args[1], stringsAsFactors=FALSE, fileEncoding='UTF-8')
out <- args[2]; dir.create(out, recursive=TRUE, showWarnings=FALSE)
stopifnot(nrow(d)==973L, !anyDuplicated(d$source_row))
d$study_id <- as.character(d$study_id)
d$permutable <- as.logical(d$permutable)
scenarios <- list(all_rows_upper=d,
  no_arguello_upper=d[d$lead_author!='David Arguello',],
  conservative_omit_17_27=d[!d$study_id %in% c('17','27'),],
  audited_farm_composites_20_31=d[d$study_id %in% c('20','31'),])
predictions <- list(); warnings_out <- list(); counts <- list(); at <- 1L
for (scenario in names(scenarios)) {
  dat <- scenarios[[scenario]]
  dat <- dat[order(dat$sample_id),] # original merge(..., by="Sample.ID") default sort
  values <- list(observed=dat$total_cd_mg_kg,
    flattened=ave(dat$total_cd_mg_kg,dat$site_depth_series,FUN=mean))
  if (scenario=='audited_farm_composites_20_31') {
    stopifnot(all(dat$permutable))
    values$forced_nonincreasing <- dat$total_cd_mg_kg
    values$forced_nondecreasing <- dat$total_cd_mg_kg
    for (sid in unique(dat$site_depth_series)) {
      j <- which(dat$site_depth_series==sid)
      j <- j[order(dat$depth_max_cm[j])]
      values$forced_nonincreasing[j] <- sort(dat$total_cd_mg_kg[j],decreasing=TRUE)
      values$forced_nondecreasing[j] <- sort(dat$total_cd_mg_kg[j])
    }
  }
  grid <- sort(unique(c(seq(min(dat$depth_max_cm),max(dat$depth_max_cm),length.out=80),20,30,50,80)))
  for (response in names(values)) {
    fits <- list(); ws <- list()
    for (surface in c('interpolate','direct')) for (sgn in c(1,-1)) {
      tmp <- data.frame(x=sgn*dat$depth_max_cm,y=values[[response]])
      fit <- withCallingHandlers(stats::loess(y~x,data=tmp,span=.75,degree=2,
        family='gaussian',control=stats::loess.control(surface=surface)),
        warning=function(w){ws[[length(ws)+1L]] <<- data.frame(surface=surface,x_sign=sgn,stage='fit',warning=conditionMessage(w));invokeRestart('muffleWarning')})
      pred <- withCallingHandlers(stats::predict(fit,newdata=data.frame(x=sgn*grid)),
        warning=function(w){ws[[length(ws)+1L]] <<- data.frame(surface=surface,x_sign=sgn,stage='predict',warning=conditionMessage(w));invokeRestart('muffleWarning')})
      fits[[paste(surface,sgn)]] <- as.numeric(pred)
    }
    predictions[[at]] <- data.frame(scenario=scenario,response=response,
      coordinate_cm=grid,r_positive_x=fits[['interpolate 1']],r_reversed_x=fits[['interpolate -1']],
      reversal_difference=fits[['interpolate -1']]-fits[['interpolate 1']],
      r_direct_positive_x=fits[['direct 1']],r_direct_reversed_x=fits[['direct -1']],n_rows=nrow(dat))
    counts[[at]] <- data.frame(scenario=scenario,response=response,n_rows=nrow(dat),
      n_series=length(unique(dat$site_depth_series)),n_studies=length(unique(dat$study_id)),
      max_abs_reversal_difference=max(abs(fits[['interpolate -1']]-fits[['interpolate 1']]),na.rm=TRUE),
      n_nonfinite_predictions=sum(!is.finite(fits[['interpolate -1']])) )
    if (length(ws)) warnings_out[[length(warnings_out)+1L]] <- data.frame(scenario=scenario,response=response,unique(do.call(rbind,ws)))
    at <- at+1L
  }
}
write.csv(do.call(rbind,predictions),file.path(out,'r_loess_reference.csv'),row.names=FALSE,na='')
write.csv(do.call(rbind,counts),file.path(out,'r_fit_counts.csv'),row.names=FALSE,na='')
warnings_df <- if(length(warnings_out)) do.call(rbind,warnings_out) else data.frame(scenario=character(),response=character(),surface=character(),x_sign=integer(),stage=character(),warning=character())
write.csv(warnings_df,file.path(out,'r_warnings.csv'),row.names=FALSE,na='')
capture.output(sessionInfo(),file=file.path(out,'R_SESSION_INFO.txt'))
cat('R_REFERENCE_COMPLETE\n')
cat(R.version.string,'\n')
cat('rows=',nrow(do.call(rbind,predictions)),'\n')
