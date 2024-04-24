library(ggplot2)
library(dplyr)
library(vroom)
library(cowplot)
library(ggpubfigs)
library(ggsignif)
options(warn = 1)


config <- rjson::fromJSON(
  file = here::here(
    "config.json"
  )
)


sparsity <- data.frame(
  sparsity = c(
    c(
      unlist(as.vector(vroom::vroom(
        here::here(
          "results", "non_kd", "breslow", "sparsity_vvh_lambda.min.csv"
        )
      )[1:25, ]))
    ),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "non_kd", "breslow", "sparsity_tuned_l1_ratio_vvh_lambda.min.csv"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "kd", "breslow", "sparsity_linear_predictor_min.csv"
      )
    )[1:25, ])),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "kd", "breslow", "sparsity_linear_predictor_pcvl.csv"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "results", "kd", "cox_nnet", "sparsity_linear_predictor_min.csv"
      )
    )))
  ),
  cancer = rep(rep(config$datasets, each = 25), 9),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"), each = 225)
)


sparsity %>%
  group_by(model, cancer) %>%
  summarise(value = paste0(round(mean(sparsity), 2), " (", round(sd(sparsity), 2), ")")) %>%
  pivot_wider(names_from = cancer, values_from = value) %>%
  write_csv(here::here("tables", "table_S2.csv"))
