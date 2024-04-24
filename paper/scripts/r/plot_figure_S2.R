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

metrics %>%
  filter(score %in% c("path")) %>%
  filter((metric == "Antolini's C" & model == "breslow" & !kd) | (kd & model == "cox_nnet" & metric == "Antolini's C")) -> path_data
teacher_line <- metrics %>%
  filter(score %in% c("teacher")) %>%
  filter(metric == "Antolini's C") %>%
  filter(model == "cox_nnet") %>%
  group_by(cancer) %>%
  summarise(mean = mean(value))
path_data$cancer <- factor(path_data$cancer, levels = cancer_ordering)


path_data$model_type <- ifelse(path_data$kd, "KD Cox-Nnet",
  "glmnet (Breslow)"
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Cox-Nnet"))
path_data$cancer <- factor(path_data$cancer, as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>%
  group_by(cancer, model_type, lambda) %>%
  summarise(mean = mean(value), sd = sd(value) / sqrt(n()))

path_data_summarised$cancer <- factor(path_data_summarised$cancer, levels = cancer_ordering)
g <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) +
  geom_line(aes(y = mean, color = model_type), linewidth = 1) +
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values = ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) +
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Cox-Nnet teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha = 0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() +
  labs(x = "Regularization index (from sparse to dense)", y = "Antolini's C", fill = "", color = "")



p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()
teacher_legend <- p + geom_hline(aes(lty = "Cox-Nnet teacher teacher", yintercept = 20), linewidth = 1, color = "red", show_guide = TRUE) + scale_linetype_manual(name = "", values = 2) + theme_big_simple() + guides(color = guide_legend(override.aes = list(linetype = c("dashed")))) + theme(legend.key.width = unit(2, "cm"))


metrics <- vroom::vroom(here::here("results", "metrics", "metrics_overall_fixed.csv"))

metrics %>%
  filter(score %in% c("path")) %>%
  filter((metric == "IBS" & model == "breslow" & !kd) | (kd & model == "cox_nnet" & metric == "IBS")) -> path_data
teacher_line <- metrics %>%
  filter(score %in% c("teacher")) %>%
  filter(metric == "IBS") %>%
  filter(model == "cox_nnet") %>%
  group_by(cancer) %>%
  summarise(mean = mean(value))


path_data$model_type <- ifelse(path_data$kd, "KD Cox-Nnet",
  "glmnet (Breslow)"
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Cox-Nnet"))
path_data$cancer <- factor(path_data$cancer, levels = as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>%
  group_by(cancer, model_type, lambda) %>%
  summarise(mean = mean(value), sd = sd(value) / sqrt(n()))


h <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) +
  geom_line(aes(y = mean, color = model_type), linewidth = 1) +
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values = ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) +
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Cox-Nnet teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha = 0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() +
  labs(x = "Regularization index (from sparse to dense)", y = "Integrated Brier Score", fill = "", color = "")


line_legend <- get_legend(
  g + theme(legend.box.margin = margin(0, 0, 0, 0))
)

teacher_legend <- get_legend(
  teacher_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)

both_legends <- plot_grid(
  line_legend, teacher_legend
)

reg_path <- plot_grid(
  cowplot::plot_grid(g + theme(legend.position = "none"), both_legends, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1),
  cowplot::plot_grid(h + theme(legend.position = "none"), both_legends, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1),
  labels = c("A", "B"),
  nrow = 2,
  label_size = 24
)

ggsave(here::here("figures", "fig-S2_finalized.pdf"), plot = reg_path, dpi = 300, height = 20 / 1.75, width = 15, units = "in")
ggsave(here::here("figures", "fig-S2_finalized.svg"), plot = reg_path, dpi = 300, height = 20 / 1.75, width = 15, units = "in")
