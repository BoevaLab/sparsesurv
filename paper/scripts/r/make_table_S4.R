library(ggplot2)
library(dplyr)
library(vroom)
library(cowplot)
library(ggpubfigs)
library(ggsignif)
library(readr)
library(tidyr)
options(warn = 1)


config <- rjson::fromJSON(
  file = here::here(
    "config.json"
  )
)

timing <- data.frame(
  time = c(
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "non_kd", "breslow", "timing.csv"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "non_kd", "breslow", "timing_tuned_l1_ratio.csv"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "kd", "breslow", "timing_tuned_teacher.csv"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "kd", "cox_nnet", "timing.csv"
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
  write_csv(here::here("tables", "table_S4.csv"))
