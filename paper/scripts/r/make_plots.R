library(ggplot2)
library(dplyr)
library(vroom)
library(cowplot)
library(ggpubfigs)
library(ggsignif)


config <- rjson::fromJSON(
  file = here::here(
    "config.json"
  )
)

metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "metrics", "metrics_overall.csv"))
#metrics_old <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "metrics_overall.csv"))


fig_1_ab <- metrics %>% filter(model %in% c("efron", "breslow")) %>% filter(lambda %in% c("min", "lambda.min")) %>% filter(metric %in% c("Harrell's C", "Uno's C"))
                  

fig_1_ab$model_type <- ifelse(fig_1_ab$model == "efron","KD Efron", 
                                     ifelse(fig_1_ab$model == "breslow" & fig_1_ab$pc,
                                            "KD Breslow",
                                            "glmnet (Breslow)"
                                            
                                       
                                     )
                                    
                              )

fig_1_ab$model_type <- factor(fig_1_ab$model_type, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))

fig_1_cd <- metrics %>% filter(model %in% c("efron", "breslow")) %>% filter(lambda %in% c("min", "lambda.min")) %>% filter(metric %in% c("Antolini's C", "IBS"))


fig_1_cd$model_type <- ifelse(fig_1_cd$model == "efron","KD Efron", 
                                     ifelse(fig_1_cd$model == "breslow" & fig_1_cd$pc,
                                            "KD Breslow",
                                            "glmnet (Breslow)"
                                            
                                            
                                     )
                              )

fig_1_cd$model_type <- factor(fig_1_cd$model_type, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Efron") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 2),
  end = c("KD Breslow", "KD Efron"),
  start = rep("glmnet (Breslow)", 2)
)





a <- fig_1_ab %>% filter(metric == "Harrell's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Harrell's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Efron") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 2),
  end = c("KD Breslow", "KD Efron"),
  start = rep("glmnet (Breslow)", 2)
)

b  <- fig_1_ab %>% filter(metric == "Uno's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Uno's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Efron") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  end = c("KD Breslow", "KD Efron"),
  start = rep("glmnet (Breslow)", 2)
)

c  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

c_legend  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Efron") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  end = c("KD Breslow", "KD Efron"),
  start = rep("glmnet (Breslow)", 2)
)

d  <- fig_1_cd %>% filter(metric == "IBS") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Integrated Brier Score", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.35, 0.375)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )


timing <- data.frame(
  time = c(
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "non_pc", "breslow", "timing.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "timing.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "timing.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 5), 3),
  model = rep(c("glmnet (Breslow)", "KD Breslow", "KD Efron"), each = 50)
  
)

timing$cancer <- factor(timing$cancer, levels = timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer))
timing$model <- factor(timing$model, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))
timing_summarised <- timing %>% group_by(cancer, model) %>% summarise(mean=mean(time), sd = sd(time) / sqrt(n()))

cancer_ordering <- timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer)

#timing %>% ggplot(aes(x = cancer, y = time, color = model)) + geom_path(group=1)  + scale_y_log10()



f <- ggplot(timing_summarised, aes(x = cancer, group = model)) + 
  geom_line(aes(y = mean, color = model), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model), alpha = .2) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  theme_simple() + labs(x = "", y = "Time (s)", fill = "", color = "")


f_legend <- ggplot(timing_summarised, aes(x = cancer, group = model)) + 
  geom_line(aes(y = mean, color = model), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model), alpha = .2) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  theme_big_simple() + labs(x = "", y = "Time (s)", fill = "", color = "")


sparsity <- data.frame(
  sparsity = c(
    c(
      unlist(as.vector(vroom::vroom(
        here::here(
          "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "non_pc", "breslow", "sparsity_vvh_lambda.min.csv"  
        )
      )))
      
    ),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "sparsity_linear_predictor_min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "sparsity_linear_predictor_min.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 25), 3),
  model = rep(c("glmnet (Breslow)", "KD Breslow", "KD Efron"), each = 250)
  
)

sparsity$cancer <- factor(sparsity$cancer, levels = sparsity %>% group_by(cancer) %>% summarise(mean=mean(sparsity)) %>% arrange(desc(`mean`)) %>% pull(cancer))
sparsity$model <- factor(sparsity$model, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))
sparsity_summarised <- sparsity %>% group_by(cancer, model) %>% summarise(mean=mean(sparsity), sd = sd(sparsity) / sqrt(n()))


e <- sparsity %>% ggplot(aes(x = model, y = sparsity, fill = model)) + geom_boxplot() + theme_simple() + labs(x = "", y = "# non-zero covariates", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3])


metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "metrics", "metrics_overall.csv"))

metrics %>% filter(score %in% c("path")) %>% filter(metric == "Antolini's C") -> path_data
teacher_line <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "Antolini's C") %>% filter(model == "efron") %>% group_by(cancer) %>% summarise(mean=mean(value))
path_data$cancer <- factor(path_data$cancer, levels = cancer_ordering)
path_data


path_data$model_type <- ifelse(path_data$model == "efron","KD Efron", 
                                     ifelse(path_data$model == "breslow" & path_data$pc,
                                            "KD Breslow",
                                            "glmnet (Breslow)"
                                            
                                            
                                     )
                      
)
path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))
path_data$cancer <- factor(path_data$cancer, as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>% group_by(cancer, model_type, lambda) %>% summarise(mean=mean(value), sd = sd(value) / sqrt(n()))

g <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) + 
  geom_line(aes(y = mean, color = model_type), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .2) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Efron teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha=0.75) +
  #geom_segment(data = teacher_line, aes(x=0,xend=100,y=mean,yend=mean)) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_simple() + 
  labs(x = "Lambda index", y = "Antolini's C", fill = "", color = "") 

 #%>% ggplot(aes(x = lambda, y = value)) +geom_boxplot() + facet_wrap(~cancer)


p <- ggplot(mtcars, aes(x = wt, y=mpg)) + geom_point()
teacher_legend <- p + geom_hline(aes(lty="Efron teacher",yintercept=20), linewidth = 1, color = "red", show_guide=TRUE) + scale_linetype_manual(name="",values=2) + theme_big_simple() + guides(color = guide_legend(override.aes = list(linetype = c("dashed")))) + theme(legend.key.width = unit(2,"cm"))

#+
#+ 
  #scale_linewidth_manual(values = 10) + scale_color_manual(values = "red") + theme_simple()



metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "metrics", "metrics_overall.csv"))
metrics %>% filter(score == "path") %>% filter(metric == "IBS") -> path_data
teacher_line <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "IBS") %>% filter(model == "efron") %>% group_by(cancer) %>% summarise(mean=mean(value))

path_data$model_type <- ifelse(path_data$model == "efron","KD Efron", 
                              
                                      ifelse(path_data$model == "breslow" & path_data$pc,
                                             "KD Breslow",
                                             "glmnet (Breslow)"
                                             
                                             
                                      )
                              
)
path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))
path_data$cancer <- factor(path_data$cancer, levels = as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>% group_by(cancer, model_type, lambda) %>% summarise(mean=mean(value), sd = sd(value) / sqrt(n()))


h  <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) + 
  geom_line(aes(y = mean, color = model_type), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .2) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Efron teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha=0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_simple() + labs(x = "Lambda index", y = "Integrated Brier Score", fill = "", color = "")
#%>% ggplot(aes(x = lambda, y = value)) +geom_boxplot() + facet_wrap(~cancer)


# A, B, C in one plot with shared legend

# D, E in one plot with shared legend
# F with own legend

# G with own legend

first_row <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), labels = "AUTO", nrow = 1, ncol = 3)
boxplot_legend <- get_legend(
  # create some space to the left of the legend
  c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
#top_row_full <- plot_grid(a_plots, legend, nrow = 2, rel_heights = c(0.95, 0.05))

second_row <- cowplot::plot_grid(d + theme(legend.position = "none"), e + theme(legend.position = "none"), f + theme(legend.position = "none"), labels = c("D", "E", "F"), nrow = 1, rel_widths = c(0.25, 0.25, 0.5))

line_legend <- get_legend(
  # create some space to the left of the legend
  f_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)

teacher_legend <- get_legend(
  # create some space to the left of the legend
  teacher_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)

panels <- plot_grid(
  first_row,
  second_row,
  g + theme(legend.position = "none"),
  h + theme(legend.position = "none"),
  labels = c("", "", "G", "H"),
  nrow = 4
)

both_legends <- plot_grid(
  line_legend, teacher_legend
)
final_legend <- plot_grid(boxplot_legend, both_legends, nrow = 2, ncol = 1)

final_panels <- plot_grid(
panels, final_legend
, nrow = 2, rel_heights = c(0.95, 0.05)
)

ggsave(here::here("~", "Downloads", "sparsesurv_final", "plots", "fig-1.png"), plot = last_plot(), dpi = 300, height = 20, width = 15, units = "in")


dataset_overview <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "dataset_overview.csv"))

knitr::kable(dataset_overview, "latex", booktabs = TRUE, digits = 3)

#knitr::kableEx


### Supplementary



metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "metrics", "metrics_overall.csv"))
#metrics_old <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "metrics_overall.csv"))


fig_1_ab <- metrics %>% filter(model %in% c("efron", "breslow")) %>% filter(lambda %in% c("pcvl", "lambda.min")) %>% filter(metric %in% c("Harrell's C", "Uno's C"))


fig_1_ab$model_type <- ifelse(fig_1_ab$model == "efron","KD Efron (pcvl)", 
                              ifelse(fig_1_ab$model == "breslow" & fig_1_ab$pc,
                                     "KD Breslow (pcvl)",
                                     "glmnet (Breslow)"
                                     
                                     
                              )
                              
)

fig_1_ab$model_type <- factor(fig_1_ab$model_type, levels = c("glmnet (Breslow)", "KD Breslow (pcvl)", "KD Efron (pcvl)"))

fig_1_cd <- metrics %>% filter(model %in% c("efron", "breslow")) %>% filter(lambda %in% c("min", "lambda.min")) %>% filter(metric %in% c("Antolini's C", "IBS"))


fig_1_cd$model_type <- ifelse(fig_1_cd$model == "efron","KD Efron (pcvl)", 
                              ifelse(fig_1_cd$model == "breslow" & fig_1_cd$pc,
                                     "KD Breslow (pcvl)",
                                     "glmnet (Breslow)"
                                     
                                     
                              )
)

fig_1_cd$model_type <- factor(fig_1_cd$model_type, levels = c("glmnet (Breslow)", "KD Breslow (pcvl)", "KD Efron (pcvl)"))

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Efron (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 2),
  end = c("KD Breslow (pcvl)", "KD Efron (pcvl)"),
  start = rep("glmnet (Breslow)", 2)
)





a <- fig_1_ab %>% filter(metric == "Harrell's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Harrell's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "KD Efron (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_ab %>% filter(metric == "Uno's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  model_type = rep("glmnet (Breslow)", 2),
  end = c("KD Breslow (pcvl)", "KD Efron (pcvl)"),
  start = rep("glmnet (Breslow)", 2)
)

b  <- fig_1_ab %>% filter(metric == "Uno's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Uno's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "KD Efron (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "Antolini's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  end = c("KD Breslow (pcvl)", "KD Efron (pcvl)"),
  start = rep("glmnet (Breslow)", 2)
)

c  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

c_legend  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.95, 1.0)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

signif_frame <- data.frame(
  pval = c(
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Breslow (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3),
    round(wilcox.test(
      x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "KD Efron (pcvl)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      paired = TRUE,
      alternative = "greater"
    )$p.val, 3)
  ),
  end = c("KD Breslow (pcvl)", "KD Efron (pcvl)"),
  start = rep("glmnet (Breslow)", 2)
)

d  <- fig_1_cd %>% filter(metric == "IBS") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_simple() + labs(x = "", y = "Integrated Brier Score", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[1:3]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(0.35, 0.375)),
    textsize = 3, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

s1_panels <- plot_grid(
  a + theme(legend.position = "none"),
  b + theme(legend.position = "none"),
  c + theme(legend.position = "none"),
  d + theme(legend.position = "none"),
  labels = c("A", "B", "C", "D"),
  ncol = 4
)

s1_complete <- plot_grid(
  s1_panels,
  get_legend(
    # create some space to the left of the legend
    c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
  ),
  nrow = 2,
  rel_heights = c(0.95, 0.05)
)



ggsave(here::here("~", "Downloads", "sparsesurv_final", "plots", "fig-s1.pdf"), plot = last_plot(), dpi = 300, height = 20 / 3, width = 15, units = "in")


timing <- data.frame(
  time = c(
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "non_pc", "breslow", "timing.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "timing.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "timing.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 5), 3),
  model = rep(c("glmnet (Breslow)", "KD Breslow", "KD Efron"), each = 50)
  
)
library(tidyr)
timing$cancer <- factor(timing$cancer, levels = timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer))
timing$model <- factor(timing$model, levels = c("glmnet (Breslow)", "KD Breslow", "KD Efron"))
timing_summarised <- timing %>% group_by(cancer, model) %>% summarise(measure=paste0(round(mean(time), 3), " (", sd = round(sd(time) / sqrt(n()),  3), ")"))
timing_summarised %>% pivot_wider(id_cols = model, names_from = cancer, values_from = measure) %>% knitr::kable("latex", booktabs = TRUE)



failures <- data.frame(
  failures = c(
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "non_pc", "breslow", "failures_vvh_lambda.min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "failures_linear_predictor_min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "failures_linear_predictor_min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "failures_linear_predictor_pcvl.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "failures_linear_predictor_pcvl.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 1), 5),
  model = rep(c("glmnet (Breslow)", "KD Efron (min)", "KD Efron (pcvl)", "KD Breslow (min)", "KD Breslow (pcvl)"), each = 10)
  
)

failures$cancer <- factor(failures$cancer, levels = as.character(timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer)))
failures[failures$model == "glmnet (Breslow)", "failures"] <- diff(c(0, failures[failures$model == "glmnet (Breslow)", "failures"]))
failures %>% group_by(model, cancer) %>% summarise(failures = mean(failures)) %>% pivot_wider(id_cols = model, names_from = cancer, values_from = failures) %>% knitr::kable("latex", booktabs = TRUE)


sparsity <- data.frame(
  sparsity = c(
    c(
      unlist(as.vector(vroom::vroom(
        here::here(
          "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "non_pc", "breslow", "sparsity_vvh_lambda.min.csv"  
        )
      )))
      
    ),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "sparsity_linear_predictor_min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "breslow", "sparsity_linear_predictor_pcvl.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "sparsity_linear_predictor_min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "/", "Volumes", "Backup", "cr",  "sparsesurv", "results", "pc", "efron", "sparsity_linear_predictor_pcvl.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 25), 5),
  model = rep(c("glmnet (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Efron (min)", "KD Efron (pcvl)"), each = 250)
  
)

sparsity$cancer <- factor(sparsity$cancer, levels = sparsity %>% group_by(cancer) %>% summarise(mean=mean(sparsity)) %>% arrange(desc(`mean`)) %>% pull(cancer))
sparsity$model <- factor(sparsity$model, levels = c("glmnet (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Efron (min)", "KD Efron (pcvl)"))
sparsity_summarised <- sparsity %>% group_by(cancer, model) %>% summarise(measure=paste0(round(mean(sparsity), 3), " (", sd = round(sd(sparsity) / sqrt(n()),  3), ")"))

sparsity_summarised %>% pivot_wider(id_cols = model, names_from = cancer, values_from = measure) %>% knitr::kable("latex", booktabs = TRUE)
