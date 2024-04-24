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

metrics <- vroom::vroom(here::here("results", "metrics", "metrics_overall.csv"))


regular_metrics <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl")) %>%
  filter(metric %in% c("IBS"))
regular_metrics$model_type <- ifelse(regular_metrics$model == "breslow" & regular_metrics$lambda == "pcvl", "KD Breslow (pcvl)",
                                     ifelse(regular_metrics$model == "breslow" & regular_metrics$kd,
                                            "KD Breslow (min)",
                                            ifelse(
                                              regular_metrics$model == "cox_nnet",
                                              "KD Cox-Nnet (min)",
                                              ifelse(
                                                regular_metrics$tuned, "glmnet tuned (Breslow)",
                                                "glmnet (Breslow)"
                                              )
                                            )
                                     )
)
ibs_metrics <- metrics %>%
  filter(model == "breslow" & !kd & lambda == 0) %>%
  filter(metric %in% c("IBS"))

regular_metrics %>%
  left_join(ibs_metrics, by = c("cancer" = "cancer", "split" = "split")) %>%
  mutate(model = model_type, cancer = cancer, split = split, is_kd = value.x == value.y) %>%
  group_by(model, cancer) %>%
  summarise(sum = sum(is_kd)) %>%
  pivot_wider(names_from = cancer, values_from = sum) %>%
  write_csv(here::here("tables", "table_S3.csv"))
