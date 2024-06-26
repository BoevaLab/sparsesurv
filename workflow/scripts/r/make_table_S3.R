log <- file(snakemake@log[[1]], open = "wt")
sink(log)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(vroom)
  library(cowplot)
  library(ggsignif)
  library(readr)
  library(tidyr)
  options(warn = 1)
})

config <- rjson::fromJSON(
  file = snakemake@params[["config_path"]]
)

tmp_glmnet <- (unname(unlist(
  vroom::vroom(paste(
    "results", "non_kd", "breslow", "failures_vvh_lambda.min.csv",
    sep = "/"
  ), delim = ",")[1, ]
)))

failures <- data.frame(t(data.frame(
  glmnet = c(tmp_glmnet[1], diff(tmp_glmnet)),
  glmnet_tuned = sapply(c(
    "BLCA",
    "BRCA",
    "HNSC",
    "KIRC",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "OV",
    "STAD"
  ), function(cancer) {
    as.numeric((vroom::vroom(paste(
      "results", "non_kd", "breslow", cancer, "failures_tuned_l1_ratio_vvh_lambda.min.csv",
      sep = "/"
    ), delim = ",")[1, 1]))
  }),
  kd_cox_nnet = unlist(unname(
    vroom::vroom(paste(
      "results", "kd", "cox_nnet", "failures_linear_predictor_min.csv",
      sep = "/"
    ), delim = ",")[, 1]
  )),
  kd_breslow = unlist(unname(
    vroom::vroom(paste(
      "results", "kd", "breslow", "failures_linear_predictor_min.csv",
      sep = "/"
    ), delim = ",")[, 1]
  )),
  kd_breslow_pcvl = unlist(unname(
    vroom::vroom(paste(
      "results", "kd", "breslow", "failures_linear_predictor_pcvl.csv",
      sep = "/"
    ), delim = ",")[, 1]
  ))
)))

colnames(failures) <- c(
  "BLCA",
  "BRCA",
  "HNSC",
  "KIRC",
  "LGG",
  "LIHC",
  "LUAD",
  "LUSC",
  "OV",
  "STAD"
)
failures$model <- c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)")


failures %>%
  select(model, BLCA, BRCA, HNSC, KIRC, LGG, LIHC, LUAD, LUSC, OV, STAD) %>%
  write_csv(paste("results", "tables", "table_S3.csv", sep = "/"))
