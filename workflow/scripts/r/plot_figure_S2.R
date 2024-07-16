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
timing_summarised <- timing %>%
  group_by(cancer, model) %>%
  summarise(mean = mean(time), sd = sd(time) / sqrt(n()))

cancer_ordering <- timing %>%
  group_by(cancer) %>%
  summarise(mean = mean(time)) %>%
  arrange(desc(`mean`)) %>%
  pull(cancer)

metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep="/"))

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
  scale_color_manual(values = friendly_pals$ito_seven[c(1, 6)]) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 6)]) +
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Cox-Nnet teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha = 0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() +
  labs(x = "Regularization index (from sparse to dense)", y = "Antolini's C", fill = "", color = "")



p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()
teacher_legend <- p + geom_hline(aes(lty = "Cox-Nnet teacher teacher", yintercept = 20), linewidth = 1, color = "red", show_guide = TRUE) + scale_linetype_manual(name = "", values = 2) + theme_big_simple() + guides(color = guide_legend(override.aes = list(linetype = c("dashed")))) + theme(legend.key.width = unit(2, "cm"))


metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep="/"))

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
  scale_color_manual(values = friendly_pals$ito_seven[c(1, 6)]) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 6)]) +
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

ggsave(paste("results", "figures", "fig-S2_finalized.pdf", sep="/"), plot = reg_path, dpi = 300, height = 20 / 1.75, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S2_finalized.tiff", sep="/"), plot = reg_path, dpi = 300, height = 20 / 1.75, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S2_finalized.eps", sep="/"), plot = reg_path, dpi = 300, height = 20 / 1.75, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S2_finalized.svg", sep="/"), plot = reg_path, dpi = 300, height = 20 / 1.75, width = 15, units = "in")
