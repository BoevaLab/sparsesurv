log <- file(snakemake@log[[1]], open = "wt")
sink(log)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(vroom)
  library(cowplot)
  library(ggsignif)
  options(warn = 1)
})

config <- rjson::fromJSON(
  file = snakemake@params[["config_path"]]
)

theme_big_simple <- function() {
  theme_bw(base_size = 16, base_family = "") %+replace%
    theme(
      plot.background = element_rect(fill = "transparent", colour = NA),
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.key = element_rect(fill = "transparent", colour = NA),
      legend.title = element_text(size = 24),
      legend.text = element_text(size = 20),
      axis.line = element_line(color = "black", size = 1, linetype = "solid"),
      axis.ticks = element_line(colour = "black", size = 1),
      panel.background = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_blank(),
      panel.border = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(size = 24, hjust = 0.0, vjust = 1.75),
      axis.text.x = element_text(color = "black", size = 20, margin = margin(t = 4, r = 0, b = 0, l = 0)),
      axis.text.y = element_text(color = "black", size = 20, margin = margin(t = 0, r = 4, b = 0, l = 0)),
      axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0), angle = 90, size = 24),
      axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0), angle = 0, size = 24),
      axis.ticks.length = unit(0.20, "cm"),
      strip.background = element_rect(color = "black", size = 1, linetype = "solid"),
      strip.text.x = element_text(size = 20, color = "black"),
      strip.text.y = element_text(size = 20, color = "black")
    )
}

friendly_pals <- list(
  bright_seven = c("#4477AA", "#228833", "#AA3377", "#BBBBBB", "#66CCEE", "#CCBB44", "#EE6677"),
  contrast_three = c("#004488", "#BB5566", "#DDAA33"),
  vibrant_seven = c("#0077BB", "#EE7733", "#33BBEE", "#CC3311", "#009988", "#EE3377", "#BBBBBB"),
  muted_nine = c("#332288", "#117733", "#CC6677", "#88CCEE", "#999933", "#882255", "#44AA99", "#DDCC77", "#AA4499"),
  nickel_five = c("#648FFF", "#FE6100", "#785EF0", "#FFB000", "#DC267F"),
  ito_seven = c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#F0E442"),
  ibm_five = c("#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"),
  wong_eight = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"),
  tol_eight = c("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"),
  zesty_four = c("#F5793A", "#A95AA1", "#85C0F9", "#0F2080"),
  retro_four = c("#601A4A", "#EE442F", "#63ACBE", "#F9F4EC")
)

metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep = "/"))
metrics_125 <- vroom::vroom(paste("results", "metrics", "metrics_overall_125_full.csv", sep = "/"))
metrics_stratified <- vroom::vroom(paste("results", "metrics", "metrics_overall_cved.csv", sep = "/"))

metrics <- rbind(
  cbind(metrics %>% filter(model == "breslow" & lambda %in% c("lambda.min", "min")),
    calc_type = "5-fold CV 5 reps (per split)"
  ),
  cbind(metrics_125, calc_type = "5-fold CV 25 reps (per split)"),
  cbind(metrics_stratified, calc_type = "5-fold CV 25 reps (per CV)")
)

metrics$model_type <- ifelse(
  metrics$kd, "KD Breslow (min)",
  ifelse(
    metrics$tuned, "glmnet tuned (Breslow)",
    "glmnet (Breslow)"
  )
)

a <- metrics %>%
  filter(metric == "Harrell's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Harrell's C", fill = "") +
  facet_wrap(~ interaction(calc_type)) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])

metrics$model_type <- ifelse(
  metrics$kd, "KD Breslow (min)",
  ifelse(
    metrics$tuned, "glmnet tuned (Breslow)",
    "glmnet (Breslow)"
  )
)

b <- metrics %>%
  filter(metric == "Uno's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Harrell's C", fill = "") +
  facet_wrap(~ interaction(calc_type)) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])


cv_fig <- cowplot::plot_grid(a, b,
  labels = c("A", "B"),
  nrow = 2,
  label_size = 24
)

ggsave(paste("results", "figures", "fig-S3_finalized.pdf", sep = "/"), plot = cv_fig, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S3_finalized.svg", sep = "/"), plot = cv_fig, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
