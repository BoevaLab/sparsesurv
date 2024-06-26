log <- file(snakemake@log[[1]], open = "wt")
sink(log)

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(readr)
  library(vroom)
  library(dplyr)
  library(microbenchmark)
  library(splitTools)
  library(glmnetUtils)
})

# Prevent early stoping.
glmnet::glmnet.control(
  fdev = 0,
  devmax = 1.0
)

config <- rjson::fromJSON(
  file = snakemake@params[["config_path"]]
)

set.seed(config$seed)

# https://stackoverflow.com/questions/7196450/create-a-dataframe-of-unequal-lengths
na.pad <- function(x, len) {
  x[1:len]
}

makePaddedDataFrame <- function(l, ...) {
  maxlen <- max(sapply(l, length))
  data.frame(lapply(l, na.pad, len = maxlen), ...)
}

timing <- list()

for (tune_l1_ratio in c(TRUE, FALSE)) {
  for (cancer in c(config$datasets)) {
    timing[[cancer]] <- c()
    data <- data.frame(vroom::vroom(
      paste("results/preprocess_data/", paste0(cancer, ".csv"), sep = "/")
    )[, -1], check.names = FALSE)
    x <- as.matrix(data[, -(1:2)])
    y <- Surv(data$OS_days, data$OS)
    fold_ids <- rep(0, length(y))

    fold_helper <- create_folds(
      y = data$OS,
      k = config$n_inner_cv,
      type = c("stratified"),
      invert = TRUE,
      seed = config$seed
    )
    for (i in 1:length(fold_helper)) {
      fold_ids[fold_helper[[i]]] <- i
    }
    if (tune_l1_ratio) {
      tim <- microbenchmark(
        cva.glmnet(
          x = x,
          y = y,
          family = "cox",
          alpha = config$l1_ratio_tuned,
          lambda.min.ratio = config$eps,
          standardize = TRUE,
          nlambda = config$n_alphas,
          foldid = fold_ids,
          grouped = TRUE
        ),
        times = config$timing_reps
      )
    } else {
      tim <- microbenchmark(
        cv.glmnet(
          x = x,
          y = y,
          family = "cox",
          alpha = config$l1_ratio,
          lambda.min.ratio = config$eps,
          standardize = TRUE,
          nlambda = config$n_alphas,
          foldid = fold_ids,
          grouped = TRUE
        ),
        times = config$timing_reps
      )
    }

    timing[[cancer]] <- tim$time * 1e-9
  }
  if (tune_l1_ratio) {
    data.frame(timing) %>% write_csv(paste("results", "non_kd", "breslow", "timing_tuned_l1_ratio.csv", sep = "/"))
  } else {
    data.frame(timing) %>% write_csv(paste("results", "non_kd", "breslow", "timing.csv", sep = "/"))
  }
}
