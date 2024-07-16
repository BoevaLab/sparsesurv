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

tmp_glmnet_1se <- (unname(unlist(
  vroom::vroom(paste(
    "results", "non_kd", "breslow", "failures_vvh_lambda.1se.csv",
    sep = "/"
  ), delim = ",")[1, ]
)))

failures <- data.frame(t(data.frame(
  glmnet = c(tmp_glmnet[1], diff(tmp_glmnet)),
  glmnet_1se = c(tmp_glmnet_1se[1], diff(tmp_glmnet_1se)),
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
  glmnet_tuned_1se = sapply(c(
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
      "results", "non_kd", "breslow", cancer, "failures_tuned_l1_ratio_vvh_lambda.1se.csv",
      sep = "/"
    ), delim = ",")[1, 1]))
  }),
  apply(vroom::vroom(
    paste(
      "results", "kd", "cox_nnet", "sparsity_linear_predictor_min.csv",
      sep = "/"
    )
  ), 2, function(x) sum(x == 0)),
  apply(vroom::vroom(
    paste(
      "results", "kd", "breslow", "sparsity_linear_predictor_min.csv",
      sep = "/"
    )
  )[1:25, ], 2, function(x) sum(x == 0)),
  apply(vroom::vroom(
    paste(
      "results", "kd", "breslow", "sparsity_linear_predictor_pcvl.csv",
      sep = "/"
    )
  ), 2, function(x) sum(x == 0)),
  apply(vroom::vroom(
    paste(
      "results", "kd", "breslow", "sparsity_vvh_1se.csv",
      sep = "/"
    )
  ), 2, function(x) sum(x == 0))
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
failures$model <- c(
  "glmnet (Breslow)",
  "glmnet (Breslow - 1se)",
  "glmnet tuned (Breslow)",
  "glmnet tuned (Breslow - 1se)",
  "KD Cox-Nnet (min)",
  "KD Breslow (min)", 
  "KD Breslow (pcvl)",
  "KD Breslow (1se)"
  
)


failures %>%
  select(model, BLCA, BRCA, HNSC, KIRC, LGG, LIHC, LUAD, LUSC, OV, STAD) %>%
  write_csv(paste("results", "tables", "table_S3.csv", sep = "/"))
