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

metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep="/")) %>% filter(score != "teacher")
metrics_teacher <- vroom::vroom(paste("results", "metrics", "metrics_overall_teachers.csv", sep="/"))
metrics <- rbind(metrics, metrics_teacher)

fig_1_ab <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>%
  filter(metric %in% c("Harrell's C", "Uno's C")) %>%
  filter(!(lambda == "teacher" & model == "cox_nnet"))

fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_ab$model == "breslow" & fig_1_ab$kd & !fig_1_ab$lambda == "teacher",
    "KD Breslow (min)",
    ifelse(
      fig_1_ab$model == "cox_nnet" & !fig_1_ab$lambda == "teacher",
      "KD Cox-Nnet (min)",
      ifelse(
        fig_1_ab$tuned, "glmnet tuned (Breslow)",
        ifelse(!fig_1_ab$tuned & fig_1_ab$model == "breslow" & !fig_1_ab$kd,
          "glmnet (Breslow)",
          ifelse(
            fig_1_ab$model == "cox_nnet",
            "Teacher Cox-Nnet",
            "Teacher Breslow"
          )
        )
      )
    )
  )
)

fig_1_ab$model_type <- factor(fig_1_ab$model_type, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "Teacher Breslow"))

fig_1_cd <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>%
  filter(metric %in% c("Antolini's C", "IBS")) %>%
  filter(!(lambda == "teacher" & model == "cox_nnet"))


fig_1_cd$model_type <- ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_cd$model == "breslow" & fig_1_cd$kd & !fig_1_cd$lambda == "teacher",
    "KD Breslow (min)",
    ifelse(
      fig_1_cd$model == "cox_nnet" & !fig_1_cd$lambda == "teacher",
      "KD Cox-Nnet (min)",
      ifelse(
        fig_1_cd$tuned, "glmnet tuned (Breslow)",
        ifelse(!fig_1_cd$tuned & fig_1_cd$model == "breslow" & !fig_1_cd$kd,
          "glmnet (Breslow)",
          ifelse(
            fig_1_cd$model == "cox_nnet",
            "Teacher Cox-Nnet",
            "Teacher Breslow"
          )
        )
      )
    )
  )
)


fig_1_cd$model_type <- factor(fig_1_cd$model_type, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "Teacher Breslow"))

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Breslow", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Breslow", 5)
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Breslow", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Breslow", 5)
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.25, 1.175, 1.1, 1.025, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Breslow", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Breslow", 5)
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7, 3)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.25, 1.175, 1.1, 1.025, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.15, 1.125, 1.075, 1.025, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Breslow", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Breslow", 5)
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = (rev(c(0.35, 0.385, 0.41, 0.435, 0.46))) + 0.025),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
  )





metrics <- vroom::vroom(paste("results", "metrics", "metrics_overall.csv", sep="/")) %>% filter(score != "teacher")
metrics_teacher <- vroom::vroom(paste("results", "metrics", "metrics_overall_teachers.csv", sep="/"))
metrics <- rbind(metrics, metrics_teacher)


fig_1_ab <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>%
  filter(metric %in% c("Harrell's C", "Uno's C")) %>%
  filter(!(lambda == "teacher" & model == "breslow"))


fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_ab$model == "breslow" & fig_1_ab$kd & !fig_1_ab$lambda == "teacher",
    "KD Breslow (min)",
    ifelse(
      fig_1_ab$model == "cox_nnet" & !fig_1_ab$lambda == "teacher",
      "KD Cox-Nnet (min)",
      ifelse(
        fig_1_ab$tuned, "glmnet tuned (Breslow)",
        ifelse(!fig_1_ab$tuned & fig_1_ab$model == "breslow" & !fig_1_ab$kd,
          "glmnet (Breslow)",
          ifelse(
            fig_1_ab$model == "cox_nnet",
            "Teacher Cox-Nnet",
            "Teacher Breslow"
          )
        )
      )
    )
  )
)

fig_1_ab$model_type <- factor(fig_1_ab$model_type, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "Teacher Cox-Nnet"))

fig_1_cd <- metrics %>%
  filter(model %in% c("breslow", "cox_nnet")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>%
  filter(metric %in% c("Antolini's C", "IBS")) %>%
  filter(!(lambda == "teacher" & model == "breslow"))


fig_1_cd$model_type <- ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_cd$model == "breslow" & fig_1_cd$kd & !fig_1_cd$lambda == "teacher",
    "KD Breslow (min)",
    ifelse(
      fig_1_cd$model == "cox_nnet" & !fig_1_cd$lambda == "teacher",
      "KD Cox-Nnet (min)",
      ifelse(
        fig_1_cd$tuned, "glmnet tuned (Breslow)",
        ifelse(!fig_1_cd$tuned & fig_1_cd$model == "breslow" & !fig_1_cd$kd,
          "glmnet (Breslow)",
          ifelse(
            fig_1_cd$model == "cox_nnet",
            "Teacher Cox-Nnet",
            "Teacher Breslow"
          )
        )
      )
    )
  )
)


fig_1_cd$model_type <- factor(fig_1_cd$model_type, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)", "Teacher Cox-Nnet"))

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Cox-Nnet", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Cox-Nnet", 5)
)





a_bottom <- fig_1_ab %>%
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Cox-Nnet", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Cox-Nnet", 5)
)


b_bottom <- fig_1_ab %>%
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = (c(1.275, 1.2, 1.125, 1.05, 0.95)) + 0.1),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "less"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Cox-Nnet", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Cox-Nnet", 5)
)

c_bottom <- fig_1_cd %>%
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

c_bottom_legend <- fig_1_cd %>%
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Cox-Nnet (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("Teacher Cox-Nnet", 5),
  end = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"),
  start = rep("Teacher Cox-Nnet", 5)
)

d_bottom <- fig_1_cd %>%
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
  scale_fill_manual(values = c(friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) +
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = (rev(c(0.35, 0.385, 0.41, 0.435, 0.46))) + 0.025),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
  )

first_row <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), d + theme(legend.position = "none"), labels = "AUTO", nrow = 1, ncol = 4, label_size = 24)
boxplot_legend <- get_legend(
  c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
first_row_with_legend <- cowplot::plot_grid(first_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

second_row <- cowplot::plot_grid(a_bottom + theme(legend.position = "none"), b_bottom + theme(legend.position = "none"), c_bottom + theme(legend.position = "none"), d_bottom + theme(legend.position = "none"), labels = c("E", "F", "G", "H"), nrow = 1, ncol = 4, label_size = 24)
boxplot_legend <- get_legend(
  c_bottom_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
second_row_with_legend <- cowplot::plot_grid(second_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

s1 <- cowplot::plot_grid(first_row_with_legend, second_row_with_legend, nrow = 2)

ggsave(paste("results", "figures", "fig-S1_finalized.pdf", sep="/"), plot = s1, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S1_finalized.tiff", sep="/"), plot = s1, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S1_finalized.eps", sep="/"), plot = s1, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S1_finalized.svg", sep="/"), plot = s1, dpi = 300, height = 20 / 1.5, width = 15, units = "in")
