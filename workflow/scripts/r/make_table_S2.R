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

sparsity <- data.frame(
  sparsity = c(
    c(
      unlist(as.vector(vroom::vroom(
        paste(
          "results", "non_kd", "breslow", "sparsity_vvh_lambda.min.csv",
          sep = "/"
        )
      )[1:25, ]))
    ),
    c(
      unlist(as.vector(vroom::vroom(
        paste(
          "results", "non_kd", "breslow", "sparsity_vvh_lambda.1se.csv",
          sep = "/"
        )
      )))
    ),
    sapply(c(
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
      unname(unlist(vroom::vroom(paste(
        "results", "non_kd", "breslow", cancer, "sparsity_tuned_l1_ratio_vvh_lambda.min.csv",
        sep = "/"
      ), delim = ",")[, 1]))[1:25]
    }),
    sapply(c(
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
      unname(unlist(vroom::vroom(paste(
        "results", "non_kd", "breslow", cancer, "sparsity_tuned_l1_ratio_vvh_lambda.1se.csv",
        sep = "/"
      ), delim = ",")[, 1]))[1:25]
    }),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "kd", "breslow", "sparsity_linear_predictor_min.csv",
        sep = "/"
      )
    )[1:25, ])),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "kd", "breslow", "sparsity_linear_predictor_pcvl.csv",
        sep = "/"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "kd", "breslow", "sparsity_vvh_1se.csv",
        sep = "/"
      )
    ))),
    unlist(as.vector(vroom::vroom(
      paste(
        "results", "kd", "cox_nnet", "sparsity_linear_predictor_min.csv",
        sep = "/"
      )
    )))
  ),
  cancer = rep(rep(config$datasets, each = 25), 8),
  model = rep(c(
    "glmnet (Breslow)",
    "glmnet (Breslow - 1se)",
    "glmnet tuned (Breslow)",
    "glmnet tuned (Breslow - 1se)",
    "KD Breslow (min)", "KD Breslow (pcvl)",
    "KD Breslow (1se)",
    "KD Cox-Nnet (min)"
  ), each = 250)
)


sparsity %>%
  group_by(model, cancer) %>%
  summarise(value = paste0(round(mean(sparsity), 2), " (", round(sd(sparsity), 2), ")")) %>%
  pivot_wider(names_from = cancer, values_from = value) %>%
  write_csv(paste("results", "tables", "table_S2.csv", sep = "/"))
