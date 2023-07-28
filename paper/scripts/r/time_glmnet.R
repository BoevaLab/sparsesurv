library(rjson)
library(glmnet)
library(penAFT)
library(survival)
library(coefplot)
library(pec)
library(readr)
library(vroom)
library(dplyr)
library(microbenchmark)

glmnet::glmnet.control(
    fdev = 0,
    devmax = 1.0
)

config <- rjson::fromJSON(
  file = here::here(
    "config.json"
  )
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

for (cancer in c(config$datasets)) {
  timing[[cancer]] <- c()
  data <- data.frame(vroom::vroom(
    here::here(
      "data", "processed",  "TCGA",
      paste0(cancer, "_data_preprocessed.csv")
    )
  )[, -1], check.names = FALSE)
  x <- as.matrix(data[, -(1:2)])
  y <- Surv(data$OS_days, data$OS)
  tim <- microbenchmark(
    cv.glmnet(
        x = x,
        y = y,
        family = "cox",
        alpha = config$l1_ratio,
        lambda.min.ratio = config$eps,
        standardize = TRUE,
        nlambda = 100,
        nfolds = config$n_inner_cv,
        grouped = TRUE
    ),
    times = 5L
  )
  timing[[cancer]] <- tim$time * 1e-9
}

data.frame(timing) %>% write_csv(here::here("results", "non_kd", "breslow", "timing.csv"))
