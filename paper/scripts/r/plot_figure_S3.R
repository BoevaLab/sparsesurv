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

metrics <- vroom::vroom(here::here("results", "metrics", "metrics_overall.csv"))
metrics_125 <- vroom::vroom(here::here("results", "metrics", "metrics_overall_125_full.csv"))
metrics_stratified <- vroom::vroom(here::here("results", "metrics", "metrics_overall_cved.csv"))

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
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])

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
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])


cv_fig <- cowplot::plot_grid(a, b,
  labels = c("A", "B"),
  nrow = 2,
  label_size = 24
)

ggsave(here::here("figures", "fig-S3_finalized.pdf"), plot = cv_fig, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
ggsave(here::here("figures", "fig-S3_finalized.svg"), plot = cv_fig, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
