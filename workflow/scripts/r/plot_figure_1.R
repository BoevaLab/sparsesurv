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

fig_1_ab <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl")) %>%
  filter(metric %in% c("Harrell's C", "Uno's C"))

fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_ab$model == "breslow" & fig_1_ab$kd,
    "KD Breslow (min)",
    ifelse(
      fig_1_ab$model == "cox_nnet",
      "KD Cox-Nnet (min)",
      ifelse(
        fig_1_ab$tuned, "glmnet tuned (Breslow)",
        "glmnet (Breslow)"
      )
    )
  )
)

fig_1_ab$model_type <- factor(fig_1_ab$model_type, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"))

fig_1_cd <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl")) %>%
  filter(metric %in% c("Antolini's C", "IBS"))


fig_1_cd$model_type <- ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_cd$model == "breslow" & fig_1_cd$kd,
    "KD Breslow (min)",
    ifelse(
      fig_1_cd$model == "cox_nnet",
      "KD Cox-Nnet (min)",
      ifelse(
        fig_1_cd$tuned, "glmnet tuned (Breslow)",
        "glmnet (Breslow)"
      )
    )
  )
)


fig_1_cd$model_type <- factor(fig_1_cd$model_type, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"))

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 4),
  end = c("KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "glmnet tuned (Breslow)"),
  start = rep("glmnet (Breslow)", 4)
)





a <- fig_1_ab %>%
  filter(metric == "Harrell's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Harrell's C", fill = "") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.0, 1.05, 1.1, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 4),
  end = c("KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "glmnet tuned (Breslow)"),
  start = rep("glmnet (Breslow)", 4)
)


b <- fig_1_ab %>%
  filter(metric == "Uno's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Uno's C", fill = "") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.1, 1.175, 1.25, 1.025)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 4),
  end = c("KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "glmnet tuned (Breslow)"),
  start = rep("glmnet (Breslow)", 4)
)

c <- fig_1_cd %>%
  filter(metric == "Antolini's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Antolini's C", fill = "") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.0, 1.05, 1.1, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
  )

c_legend <- fig_1_cd %>%
  filter(metric == "Antolini's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Antolini's C", fill = "") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.0, 1.05, 1.1, 0.95)),
    textsize = 3 * 1.5 * 2, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 4),
  end = c("KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "glmnet tuned (Breslow)"),
  start = rep("glmnet (Breslow)", 4)
)

d <- fig_1_cd %>%
  filter(metric == "IBS") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "Integrated Brier Score", fill = "") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.375, 0.4, 0.425, 0.35)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
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



f <- ggplot(timing_summarised, aes(x = cancer, group = model)) +
  geom_line(aes(y = mean, color = model), linewidth = 1) +
  scale_color_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 6)]) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 6)]) +
  theme_big_simple() +
  labs(x = "", y = "Time (s)", fill = "", color = "") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))


f_legend <- ggplot(timing_summarised, aes(x = cancer, group = model)) +
  geom_line(aes(y = mean, color = model), linewidth = 1) +
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model), alpha = .1) +
  scale_color_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) +
  theme_big_simple() +
  labs(x = "", y = "Time (s)", fill = "", color = "")

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
    ), function(cancer)  {
      unname(unlist(vroom::vroom(paste(
            "results", "non_kd", "breslow", cancer, "sparsity_tuned_l1_ratio_vvh_lambda.min.csv",
            sep = "/"
          ), delim = ",")[, 1]))[1:25]

    }

    ),
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
        "results", "kd", "cox_nnet", "sparsity_linear_predictor_min.csv",
        sep = "/"
      )
    )))
  ),
  cancer = rep(rep(config$datasets, each = 25), 5),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"), each = 250)
)

sparsity$cancer <- factor(sparsity$cancer, levels = sparsity %>% group_by(cancer) %>% summarise(mean = mean(sparsity)) %>% arrange(desc(`mean`)) %>% pull(cancer))
sparsity$model <- factor(sparsity$model, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"))
sparsity_summarised <- sparsity %>%
  group_by(cancer, model) %>%
  summarise(mean = mean(sparsity), sd = sd(sparsity) / sqrt(n()))


e <- sparsity %>% ggplot(aes(x = model, y = sparsity, fill = model)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "# non-zero covariates", fill = "") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])


metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep = "/"))

metrics %>%
  filter(score %in% c("path")) %>%
  filter(metric == "Antolini's C" & model == "breslow") -> path_data
teacher_line <- metrics %>%
  filter(score %in% c("teacher")) %>%
  filter(metric == "Antolini's C") %>%
  filter(model == "breslow") %>%
  group_by(cancer) %>%
  summarise(mean = mean(value))
teacher_line_coxnnet <- metrics %>%
  filter(score %in% c("teacher")) %>%
  filter(metric == "Antolini's C") %>%
  filter(model == "cox_nnet") %>%
  group_by(cancer) %>%
  summarise(mean = mean(value))
path_data$cancer <- factor(path_data$cancer, levels = cancer_ordering)
path_data


path_data$model_type <- ifelse(path_data$model == "breslow" & path_data$kd, "KD Breslow",
  "glmnet (Breslow)"
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Breslow"))
path_data$cancer <- factor(path_data$cancer, as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>%
  group_by(cancer, model_type, lambda) %>%
  summarise(mean = mean(value), sd = sd(value) / sqrt(n()))

path_data_summarised$cancer <- factor(path_data_summarised$cancer, levels = cancer_ordering)
g <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) +
  geom_line(aes(y = mean, color = model_type), linewidth = 1) +
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values = friendly_pals$ito_seven[c(1, 4)]) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 4)]) +
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Breslow teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha = 0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() +
  labs(x = "Regularization index (from sparse to dense)", y = "Antolini's C", fill = "", color = "")



p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()
teacher_legend <- p + geom_hline(aes(lty = "Breslow teacher", yintercept = 20), linewidth = 1, color = "red", show_guide = TRUE) + scale_linetype_manual(name = "", values = 2) + theme_big_simple() + guides(color = guide_legend(override.aes = list(linetype = c("dashed")))) + theme(legend.key.width = unit(2, "cm"))




metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep = "/"))

metrics %>%
  filter(score %in% c("path")) %>%
  filter(metric == "IBS" & model == "breslow") -> path_data
teacher_line <- metrics %>%
  filter(score %in% c("teacher")) %>%
  filter(metric == "IBS") %>%
  filter(model == "breslow") %>%
  group_by(cancer) %>%
  summarise(mean = mean(value))


path_data$model_type <- ifelse(path_data$model == "breslow" & path_data$kd, "KD Breslow",
  "glmnet (Breslow)"
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Breslow"))
path_data$cancer <- factor(path_data$cancer, levels = as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>%
  group_by(cancer, model_type, lambda) %>%
  summarise(mean = mean(value), sd = sd(value) / sqrt(n()))


h <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) +
  geom_line(aes(y = mean, color = model_type), linewidth = 1) +
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values = friendly_pals$ito_seven[c(1, 4)]) +
  scale_fill_manual(values = friendly_pals$ito_seven[c(1, 4)]) +
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Breslow teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha = 0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() +
  labs(x = "Regularization index (from sparse to dense)", y = "Integrated Brier Score", fill = "", color = "")


first_row <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), labels = "AUTO", nrow = 1, ncol = 3, label_size = 24)
boxplot_legend <- get_legend(
  c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
first_row_with_legend <- cowplot::plot_grid(first_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

second_row <- cowplot::plot_grid(d + theme(legend.position = "none"), e + theme(legend.position = "none"), f + theme(legend.position = c(0.7, 0.8)), labels = c("D", "E", "F"), nrow = 1, rel_widths = c(0.25, 0.25, 0.5), label_size = 24)

second_row_with_legend <- cowplot::plot_grid(second_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

line_legend <- get_legend(
  g + theme(legend.box.margin = margin(0, 0, 0, 0))
)

teacher_legend <- get_legend(
  teacher_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)

both_legends <- plot_grid(
  line_legend, teacher_legend
)

panels <- plot_grid(
  first_row_with_legend,
  second_row_with_legend,
  cowplot::plot_grid(g + theme(legend.position = "none"), both_legends, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1),
  cowplot::plot_grid(h + theme(legend.position = "none"), both_legends, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1),
  labels = c("", "", "G", "H"),
  nrow = 4,
  label_size = 24
)

ggsave(paste("results", "figures", "fig-1_finalized.png", sep = "/"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-1_finalized.pdf", sep = "/"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-1_finalized.tiff", sep = "/"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-1_finalized.eps", sep = "/"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-1_finalized.svg", sep = "/"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")
