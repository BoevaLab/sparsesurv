library(ggplot2)
library(dplyr)
library(vroom)
library(cowplot)
library(ggpubfigs)
library(ggsignif)
options(warn = 1)


config <- rjson::fromJSON(
  file = here::here(
    "/", "Volumes", "Backup", "cr", "sparsesurv", "config.json"
  )
)

dataset_overview <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "dataset_overview.csv"))

knitr::kable(dataset_overview, "latex", booktabs = TRUE, digits = 3)
