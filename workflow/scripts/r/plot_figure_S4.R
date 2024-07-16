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
  filter(model %in% c("breslow")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl", "lambda.1se", "1se")) %>%
  filter(metric %in% c("Harrell's C", "Uno's C"))

fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_ab$model == "breslow" & fig_1_ab$kd & fig_1_ab$lambda == "1se",
    "KD Breslow (1se)",
    ifelse(fig_1_ab$model == "breslow" & fig_1_ab$kd,
      "KD Breslow (min)",
      ifelse(
        fig_1_ab$tuned & fig_1_ab$lambda == "lambda.min",
        "glmnet tuned (Breslow)",
        ifelse(
          fig_1_ab$tuned & fig_1_ab$lambda == "lambda.1se",
          "glmnet tuned (Breslow - 1se)",
          ifelse(
            fig_1_ab$lambda == "lambda.min",
            "glmnet (Breslow)",
            "glmnet (Breslow - 1se)"
          )
        )
      )
    )
  )
)

fig_1_ab$model_type <- factor(fig_1_ab$model_type, levels = c(
  "glmnet (Breslow)",
  "glmnet (Breslow - 1se)",
  "glmnet tuned (Breslow)",
  "glmnet tuned (Breslow - 1se)",
  "KD Breslow (min)", "KD Breslow (pcvl)", "KD Breslow (1se)"
))

fig_1_cd <- metrics %>%
  filter(model %in% c("breslow")) %>%
  filter(lambda %in% c("min", "lambda.min", "pcvl", "lambda.1se", "1se")) %>%
  filter(metric %in% c("Antolini's C", "IBS"))


fig_1_cd$model_type <- ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl", "KD Breslow (pcvl)",
  ifelse(fig_1_cd$model == "breslow" & fig_1_cd$kd & fig_1_cd$lambda == "1se",
    "KD Breslow (1se)",
    ifelse(fig_1_cd$model == "breslow" & fig_1_cd$kd,
      "KD Breslow (min)",
      ifelse(
        fig_1_cd$tuned & fig_1_cd$lambda == "lambda.min",
        "glmnet tuned (Breslow)",
        ifelse(
          fig_1_cd$tuned & fig_1_cd$lambda == "lambda.1se",
          "glmnet tuned (Breslow - 1se)",
          ifelse(
            fig_1_cd$lambda == "lambda.min",
            "glmnet (Breslow)",
            "glmnet (Breslow - 1se)"
          )
        )
      )
    )
  )
)


fig_1_cd$model_type <- factor(fig_1_cd$model_type, levels = c(
  "glmnet (Breslow)",
  "glmnet (Breslow - 1se)",
  "glmnet tuned (Breslow)",
  "glmnet tuned (Breslow - 1se)",
  "KD Breslow (min)", "KD Breslow (pcvl)", "KD Breslow (1se)"
))


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
  scale_fill_manual(values = c(
    friendly_pals$ito_seven[1],
    friendly_pals$contrast_three[1],
    friendly_pals$ito_seven[2],
    friendly_pals$contrast_three[2],
    friendly_pals$ito_seven[4],
    friendly_pals$ito_seven[5],
    friendly_pals$contrast_three[2]
  ))

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
  scale_fill_manual(values = c(
    friendly_pals$ito_seven[1],
    friendly_pals$contrast_three[1],
    friendly_pals$ito_seven[2],
    friendly_pals$contrast_three[2],
    friendly_pals$ito_seven[4],
    friendly_pals$ito_seven[5],
    friendly_pals$contrast_three[2]
  ))


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
  scale_fill_manual(values = c(
    friendly_pals$ito_seven[1],
    friendly_pals$contrast_three[1],
    friendly_pals$ito_seven[2],
    friendly_pals$contrast_three[2],
    friendly_pals$ito_seven[4],
    friendly_pals$ito_seven[5],
    friendly_pals$contrast_three[2]
  ))

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
  scale_fill_manual(values = c(
    friendly_pals$ito_seven[1],
    friendly_pals$contrast_three[1],
    friendly_pals$ito_seven[2],
    friendly_pals$contrast_three[2],
    friendly_pals$ito_seven[4],
    friendly_pals$ito_seven[5],
    friendly_pals$contrast_three[2]
  ))

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
  scale_fill_manual(values = c(
    friendly_pals$ito_seven[1],
    friendly_pals$contrast_three[1],
    friendly_pals$ito_seven[2],
    friendly_pals$contrast_three[2],
    friendly_pals$ito_seven[4],
    friendly_pals$ito_seven[5],
    friendly_pals$contrast_three[2]
  ))

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
    )))
  ),
  cancer = rep(rep(config$datasets, each = 25), 7),
  model = rep(c("glmnet (Breslow)", 
                "glmnet (Breslow - 1se)",
                "glmnet tuned (Breslow)", 
                "glmnet tuned (Breslow - 1se)", 
                "KD Breslow (min)", "KD Breslow (pcvl)", "KD Breslow (1se)"), each = 250)
)

sparsity$cancer <- factor(sparsity$cancer, levels = sparsity %>% group_by(cancer) %>% summarise(mean = mean(sparsity)) %>% arrange(desc(`mean`)) %>% pull(cancer))
sparsity$model <- factor(sparsity$model, levels = c(
  "glmnet (Breslow)",
  "glmnet (Breslow - 1se)",
  "glmnet tuned (Breslow)",
  "glmnet tuned (Breslow - 1se)",
  "KD Breslow (min)", "KD Breslow (pcvl)", "KD Breslow (1se)"
))
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
  scale_fill_manual(values = c(
    friendly_pals$ito_seven[1],
    friendly_pals$contrast_three[1],
    friendly_pals$ito_seven[2],
    friendly_pals$contrast_three[2],
    friendly_pals$ito_seven[4],
    friendly_pals$ito_seven[5],
    friendly_pals$contrast_three[2]
  ))

first_row <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), labels = "AUTO", nrow = 1, ncol = 3, label_size = 24)
boxplot_legend <- get_legend(
  c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
first_row_with_legend <- cowplot::plot_grid(first_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

second_row <- cowplot::plot_grid(d + theme(legend.position = "none"), e + theme(legend.position = "none"), NULL, labels = c("D", "E", ""), nrow = 1, rel_widths = c(0.25, 0.25, 0.25), label_size = 24)

second_row_with_legend <- cowplot::plot_grid(second_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)


panels <- plot_grid(
  first_row_with_legend,
  second_row_with_legend,
  nrow = 2,
  label_size = 24
)

ggsave(paste("results", "figures", "fig-S4_finalized.png", sep = "/"), plot = panels, dpi = 300, height = 14, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S4_finalized.pdf", sep = "/"), plot = panels, dpi = 300, height = 14, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S4_finalized.tiff", sep = "/"), plot = panels, dpi = 300, height = 14, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S4_finalized.eps", sep = "/"), plot = panels, dpi = 300, height = 14, width = 15, units = "in")
ggsave(paste("results", "figures", "fig-S4_finalized.svg", sep = "/"), plot = panels, dpi = 300, height = 14, width = 15, units = "in")
