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

timing <- data.frame(
  time = c(
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "non_kd", "breslow", "timing.csv",
        sep = "/"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "non_kd", "breslow", "timing_tuned_l1_ratio.csv",
        sep = "/"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "kd", "breslow", "timing_tuned_teacher.csv",
        sep = "/"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "kd", "cox_nnet", "timing.csv",
        sep = "/"
      )
    )))
  ),
  cancer = rep(rep(config$datasets, each = 5), 4),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow", "KD Cox-Nnet"), each = 50)
)

timing$cancer <- factor(timing$cancer, levels = timing %>% group_by(cancer) %>% summarise(mean = mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer))
timing$model <- factor(timing$model, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow", "KD Cox-Nnet"))

timing %>%
  group_by(model, cancer) %>%
  summarise(value = paste0(round(mean(time), 2), " (", round(sd(time), 2), ")")) %>%
  pivot_wider(names_from = cancer, values_from = value) %>%
  write_csv(paste("results", "tables", "table_S4.csv", sep = "/"))
