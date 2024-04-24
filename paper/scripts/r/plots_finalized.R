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

metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))
#metrics_old <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall.csv"))
#metrics_old <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "metrics_overall.csv"))


fig_1_ab <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl")) %>% filter(metric %in% c("Harrell's C", "Uno's C"))
                  

fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl","KD Breslow (pcvl)", 
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

fig_1_cd <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl")) %>% filter(metric %in% c("Antolini's C", "IBS"))


fig_1_cd$model_type <-ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl","KD Breslow (pcvl)", 
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
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
  #    paired = TRUE,
  #    alternative = "greater"
  #  )$p.val, 3),
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





a <- fig_1_ab %>% filter(metric == "Harrell's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Harrell's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.0, 1.05, 1.1, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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


b  <- fig_1_ab %>% filter(metric == "Uno's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Uno's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.1, 1.175, 1.25, 1.025)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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

c  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.0, 1.05, 1.1, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

c_legend  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.0, 1.05, 1.1, 0.95)),
    textsize = 3 * 1.5 * 2, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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

d  <- fig_1_cd %>% filter(metric == "IBS") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Integrated Brier Score", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
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
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "non_kd", "breslow", "timing.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv","results",  "non_kd", "breslow", "timing_tuned_l1_ratio.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "breslow", "timing_tuned_teacher.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "cox_nnet", "timing.csv"  
      )
    )))
    
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 5), 4),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow", "KD Cox-Nnet"), each = 50)
  
)

timing$cancer <- factor(timing$cancer, levels = timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer))
timing$model <- factor(timing$model, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow", "KD Cox-Nnet"))
timing_summarised <- timing %>% group_by(cancer, model) %>% summarise(mean=mean(time), sd = sd(time) / sqrt(n()))

cancer_ordering <- timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer)

#timing %>% ggplot(aes(x = cancer, y = time, color = model)) + geom_path(group=1)  + scale_y_log10()



f <- ggplot(timing_summarised, aes(x = cancer, group = model)) + 
  geom_line(aes(y = mean, color = model), linewidth = 1) + 
  #geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model), alpha = .1) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 6)]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 6)]) + 
  theme_big_simple() + labs(x = "", y = "Time (s)", fill = "", color = "") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


f_legend <- ggplot(timing_summarised, aes(x = cancer, group = model)) + 
  geom_line(aes(y = mean, color = model), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model), alpha = .1) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) + 
  theme_big_simple() + labs(x = "", y = "Time (s)", fill = "", color = "")


sparsity <- data.frame(
  sparsity = c(
    c(
      unlist(as.vector(vroom::vroom(
        here::here(
          "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "non_kd", "breslow", "sparsity_vvh_lambda.min.csv"  
        )
      )[1:25, ]))
      
    ),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "non_kd", "breslow", "sparsity_tuned_l1_ratio_vvh_lambda.min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "breslow", "sparsity_linear_predictor_min.csv"  
      )
    )[1:25, ])),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "breslow", "sparsity_linear_predictor_pcvl.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "cox_nnet", "sparsity_linear_predictor_min.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 25), 9),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"), each = 225)
  
)

sparsity$cancer <- factor(sparsity$cancer, levels = sparsity %>% group_by(cancer) %>% summarise(mean=mean(sparsity)) %>% arrange(desc(`mean`)) %>% pull(cancer))
sparsity$model <- factor(sparsity$model, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"))
sparsity_summarised <- sparsity %>% group_by(cancer, model) %>% summarise(mean=mean(sparsity), sd = sd(sparsity) / sqrt(n()))


e <- sparsity %>% ggplot(aes(x = model, y = sparsity, fill = model)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "# non-zero covariates", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])


metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))

metrics %>% filter(score %in% c("path")) %>% filter(metric == "Antolini's C" & model == "breslow") -> path_data
teacher_line <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "Antolini's C") %>% filter(model == "breslow") %>% group_by(cancer) %>% summarise(mean=mean(value))
teacher_line_coxnnet <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "Antolini's C") %>% filter(model == "cox_nnet") %>% group_by(cancer) %>% summarise(mean=mean(value))
path_data$cancer <- factor(path_data$cancer, levels = cancer_ordering)
path_data


path_data$model_type <- ifelse(path_data$model == "breslow" & path_data$kd,"KD Breslow", 
                                "glmnet (Breslow)"
                                            
                                            
                                     )
                      

path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Breslow"))
path_data$cancer <- factor(path_data$cancer, as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>% group_by(cancer, model_type, lambda) %>% summarise(mean=mean(value), sd = sd(value) / sqrt(n()))

path_data_summarised$cancer <- factor(path_data_summarised$cancer, levels=cancer_ordering)
g <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) + 
  geom_line(aes(y = mean, color = model_type), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 4)]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 4)]) + 
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Breslow teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha=0.75) +
  #geom_segment(data = teacher_line, aes(x=0,xend=100,y=mean,yend=mean)) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() + 
  labs(x = "Regularization index (from sparse to dense)", y = "Antolini's C", fill = "", color = "") 

 #%>% ggplot(aes(x = lambda, y = value)) +geom_boxplot() + facet_wrap(~cancer)


p <- ggplot(mtcars, aes(x = wt, y=mpg)) + geom_point()
teacher_legend <- p + geom_hline(aes(lty="Breslow teacher",yintercept=20), linewidth = 1, color = "red", show_guide=TRUE) + scale_linetype_manual(name="",values=2) + theme_big_simple() + guides(color = guide_legend(override.aes = list(linetype = c("dashed")))) + theme(legend.key.width = unit(2,"cm"))

#+
#+ 
  #scale_linewidth_manual(values = 10) + scale_color_manual(values = "red") + theme_big_simple()



metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))

metrics %>% filter(score %in% c("path")) %>% filter(metric == "IBS" & model == "breslow") -> path_data
teacher_line <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "IBS") %>% filter(model == "breslow") %>% group_by(cancer) %>% summarise(mean=mean(value))


path_data$model_type <- ifelse(path_data$model == "breslow" & path_data$kd,"KD Breslow", 
                               "glmnet (Breslow)"
                               
                               
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Breslow"))
path_data$cancer <- factor(path_data$cancer, levels = as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>% group_by(cancer, model_type, lambda) %>% summarise(mean=mean(value), sd = sd(value) / sqrt(n()))


h  <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) + 
  geom_line(aes(y = mean, color = model_type), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 4)]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 4)]) + 
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Breslow teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha=0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() + labs(x = "Regularization index (from sparse to dense)", y = "Integrated Brier Score", fill = "", color = "")
#%>% ggplot(aes(x = lambda, y = value)) +geom_boxplot() + facet_wrap(~cancer)


# A, B, C in one plot with shared legend

# D, E in one plot with shared legend
# F with own legend

# G with own legend

first_row <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), labels = "AUTO", nrow = 1, ncol = 3, label_size = 24)
boxplot_legend <- get_legend(
  # create some space to the left of the legend
  c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
first_row_with_legend <- cowplot::plot_grid(first_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)
#top_row_full <- plot_grid(a_plots, legend, nrow = 2, rel_heights = c(0.95, 0.1))

second_row <- cowplot::plot_grid(d + theme(legend.position = "none"), e + theme(legend.position = "none"), f + theme(legend.position = c(0.7, 0.8)), labels = c("D", "E", "F"), nrow = 1, rel_widths = c(0.25, 0.25, 0.5), label_size = 24)

second_row_with_legend <- cowplot::plot_grid(second_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

line_legend <- get_legend(
  # create some space to the left of the legend
  g + theme(legend.box.margin = margin(0, 0, 0, 0))
)

teacher_legend <- get_legend(
  # create some space to the left of the legend
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


#final_legend <- plot_grid(boxplot_legend, both_legends, nrow = 2, ncol = 1)

#final_panels <- plot_grid(
#panels, final_legend
#, nrow = 2, rel_heights = c(0.95, 0.1)
#)

ggsave(here::here("~", "Downloads", "fig-1_finalized.png"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")
ggsave(here::here("~", "Downloads", "fig-1_finalized.svg"), plot = panels, dpi = 300, height = 20, width = 15, units = "in")





library(ggplot2)
library(dplyr)
library(vroom)
library(cowplot)
library(ggpubfigs)
library(ggsignif)



library(ggplot2)
library(dplyr)
library(vroom)
library(cowplot)
library(ggpubfigs)
library(ggsignif)


config <- rjson::fromJSON(
  file = here::here(
    "/", "Volumes", "Backup", "cr", "sparsesurv", "config.json"
  )
)

metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv")) %>% filter(score != "teacher")
metrics_teacher <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed_teachers.csv"))
#metrics_old <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall.csv"))
metrics <- rbind(metrics, metrics_teacher)
#metrics_old <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "metrics_overall.csv"))


fig_1_ab <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>% filter(metric %in% c("Harrell's C", "Uno's C")) %>% filter(!(lambda == "teacher" & model == "cox_nnet"))


fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl","KD Breslow (pcvl)", 
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

fig_1_cd <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>% filter(metric %in% c("Antolini's C", "IBS")) %>% filter(!(lambda == "teacher" & model == "cox_nnet"))


fig_1_cd$model_type <-ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl","KD Breslow (pcvl)", 
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
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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





a <- fig_1_ab %>% filter(metric == "Harrell's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Harrell's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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


b  <- fig_1_ab %>% filter(metric == "Uno's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Uno's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.25, 1.175, 1.1, 1.025, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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

c  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7, 3)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.25, 1.175, 1.1, 1.025, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

c_legend  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.15, 1.125, 1.075, 1.025, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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

d  <- fig_1_cd %>% filter(metric == "IBS") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Integrated Brier Score", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)])) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = (rev(c(0.35, 0.385, 0.41, 0.435, 0.46)))+0.025),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )





metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv")) %>% filter(score != "teacher")
metrics_teacher <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed_teachers.csv"))
#metrics_old <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall.csv"))
metrics <- rbind(metrics, metrics_teacher)
#metrics_old <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "metrics_overall.csv"))


fig_1_ab <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>% filter(metric %in% c("Harrell's C", "Uno's C")) %>% filter(!(lambda == "teacher" & model == "breslow"))


fig_1_ab$model_type <- ifelse(fig_1_ab$model == "breslow" & fig_1_ab$lambda == "pcvl","KD Breslow (pcvl)", 
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

fig_1_cd <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl", "teacher")) %>% filter(metric %in% c("Antolini's C", "IBS")) %>% filter(!(lambda == "teacher" & model == "breslow"))


fig_1_cd$model_type <-ifelse(fig_1_cd$model == "breslow" & fig_1_cd$lambda == "pcvl","KD Breslow (pcvl)", 
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
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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





a_bottom <- fig_1_ab %>% filter(metric == "Harrell's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Harrell's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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


b_bottom  <- fig_1_ab %>% filter(metric == "Uno's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Uno's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = (c(1.275, 1.2, 1.125, 1.05, 0.95))+0.1),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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

c_bottom  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

c_bottom_legend  <- fig_1_cd %>% filter(metric == "Antolini's C") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Antolini's C", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) + 
  geom_signif(
    data = signif_frame,
    # 0.95 1 1.05 1.1 1.15
    aes(xmin = start, xmax = end, annotations = pval, y_position = c(1.16, 1.11, 1.06, 1.01, 0.95)),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    
  )

signif_frame <- data.frame(
  pval = c(
    #round(wilcox.test(
    #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
    #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
    #    paired = TRUE,
    #    alternative = "greater"
    #  )$p.val, 3),
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

d_bottom  <- fig_1_cd %>% filter(metric == "IBS") %>% ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Integrated Brier Score", fill = "") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=c(ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 3)])) + 
  geom_signif(
    data = signif_frame,
    aes(xmin = start, xmax = end, annotations = pval, y_position = (rev(c(0.35, 0.385, 0.41, 0.435, 0.46)))+0.025),
    textsize = 3 * 1.5, vjust = -0.2,
    manual = TRUE,
    inherit.aes = FALSE
    
  )

first_row <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), d+ theme(legend.position = "none"), labels = "AUTO", nrow = 1, ncol = 4, label_size = 24)
boxplot_legend <- get_legend(
  # create some space to the left of the legend
  c_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
first_row_with_legend <- cowplot::plot_grid(first_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

second_row <- cowplot::plot_grid(a_bottom + theme(legend.position = "none"), b_bottom + theme(legend.position = "none"), c_bottom + theme(legend.position = "none"), d_bottom+ theme(legend.position = "none"), labels = c("E", "F", "G", "H"), nrow = 1, ncol = 4, label_size = 24)
boxplot_legend <- get_legend(
  # create some space to the left of the legend
  c_bottom_legend + theme(legend.box.margin = margin(0, 0, 0, 0))
)
second_row_with_legend <- cowplot::plot_grid(second_row, boxplot_legend, rel_heights = c(0.95, 0.1), nrow = 2, ncol = 1)

s1 <- cowplot::plot_grid(first_row_with_legend, second_row_with_legend, nrow=2)

ggsave(here::here("~", "Downloads", "fig-S1_finalized.pdf"), plot = s1, dpi = 300, height = 20/1.5, width = 15, units = "in")
ggsave(here::here("~", "Downloads", "fig-S1_finalized.svg"), plot = s1, dpi = 300, height = 20/1.5, width = 15, units = "in")




metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))

metrics %>% filter(score %in% c("path")) %>% filter((metric == "Antolini's C" & model == "breslow" & !kd)|(kd & model == "cox_nnet" & metric == "Antolini's C")) -> path_data
teacher_line <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "Antolini's C") %>% filter(model == "cox_nnet") %>% group_by(cancer) %>% summarise(mean=mean(value))
path_data$cancer <- factor(path_data$cancer, levels = cancer_ordering)


path_data$model_type <- ifelse(path_data$kd,"KD Cox-Nnet", 
                               "glmnet (Breslow)"
                               
                               
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Cox-Nnet"))
path_data$cancer <- factor(path_data$cancer, as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>% group_by(cancer, model_type, lambda) %>% summarise(mean=mean(value), sd = sd(value) / sqrt(n()))

path_data_summarised$cancer <- factor(path_data_summarised$cancer, levels=cancer_ordering)
g <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) + 
  geom_line(aes(y = mean, color = model_type), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) + 
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Cox-Nnet teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha=0.75) +
  #geom_segment(data = teacher_line, aes(x=0,xend=100,y=mean,yend=mean)) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() + 
  labs(x = "Regularization index (from sparse to dense)", y = "Antolini's C", fill = "", color = "") 

#%>% ggplot(aes(x = lambda, y = value)) +geom_boxplot() + facet_wrap(~cancer)


p <- ggplot(mtcars, aes(x = wt, y=mpg)) + geom_point()
teacher_legend <- p + geom_hline(aes(lty="Cox-Nnet teacher teacher",yintercept=20), linewidth = 1, color = "red", show_guide=TRUE) + scale_linetype_manual(name="",values=2) + theme_big_simple() + guides(color = guide_legend(override.aes = list(linetype = c("dashed")))) + theme(legend.key.width = unit(2,"cm"))

#+
#+ 
#scale_linewidth_manual(values = 10) + scale_color_manual(values = "red") + theme_big_simple()



metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))

metrics %>% filter(score %in% c("path")) %>% filter((metric == "IBS" & model == "breslow" & !kd)|(kd & model == "cox_nnet" & metric == "IBS")) -> path_data
teacher_line <- metrics %>% filter(score %in% c("teacher")) %>% filter(metric == "IBS") %>% filter(model == "cox_nnet") %>% group_by(cancer) %>% summarise(mean=mean(value))


path_data$model_type <- ifelse(path_data$kd,"KD Cox-Nnet", 
                               "glmnet (Breslow)"
                               
                               
)


path_data$model_type <- factor(path_data$model_type, levels = c("glmnet (Breslow)", "KD Cox-Nnet"))
path_data$cancer <- factor(path_data$cancer, levels = as.character(cancer_ordering))
teacher_line$cancer <- factor(teacher_line$cancer, as.character(cancer_ordering))
path_data_summarised <- path_data %>% group_by(cancer, model_type, lambda) %>% summarise(mean=mean(value), sd = sd(value) / sqrt(n()))


h  <- ggplot(path_data_summarised, aes(x = as.numeric(lambda), group = model_type)) + 
  geom_line(aes(y = mean, color = model_type), linewidth = 1) + 
  geom_ribbon(aes(y = mean, ymin = mean - sd, ymax = mean + sd, fill = model_type), alpha = .1) +
  scale_color_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) + 
  scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 6)]) + 
  geom_hline(data = teacher_line, aes(yintercept = mean, linetype = "Cox-Nnet teacher"), color = "red", lwd = 0.5, linetype = 2, show.legend = FALSE, alpha=0.75) +
  facet_wrap(~cancer, scales = "free_y", nrow = 2) +
  theme_big_simple() + labs(x = "Regularization index (from sparse to dense)", y = "Integrated Brier Score", fill = "", color = "")


line_legend <- get_legend(
  # create some space to the left of the legend
  g + theme(legend.box.margin = margin(0, 0, 0, 0))
)

teacher_legend <- get_legend(
  # create some space to the left of the legend
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

ggsave(here::here("~", "Downloads", "fig-S2_finalized.pdf"), plot = reg_path, dpi = 300, height = 20/1.75, width = 15, units = "in")
ggsave(here::here("~", "Downloads", "fig-S2_finalized.svg"), plot = reg_path, dpi = 300, height = 20/1.75, width = 15, units = "in")

config <- rjson::fromJSON(
  file = here::here(
    "/", "Volumes", "Backup", "cr", "sparsesurv", "config.json"
  )
)

metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))
metrics_125 <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed_125_full.csv"))
metrics_stratified <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed_cved.csv"))

metrics <- rbind(cbind(metrics %>% filter(model == "breslow"  & lambda %in% c("lambda.min", "min")), 
            calc_type = "5-fold CV 5 reps (per split)"),
      cbind(metrics_125, calc_type="5-fold CV 25 reps (per split)"),
      cbind(metrics_stratified, calc_type="5-fold CV 25 reps (per CV)"))

metrics$model_type <- ifelse(
  metrics$kd, "KD Breslow (min)",
  ifelse(
    metrics$tuned, "glmnet tuned (Breslow)",
    "glmnet (Breslow)"
  )
)

a <- metrics %>% 
  filter(metric == "Harrell's C") %>%
  ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Harrell's C", fill = "") +
  facet_wrap(~interaction(calc_type))+ 
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
    scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) 
  
metrics$model_type <- ifelse(
    metrics$kd, "KD Breslow (min)",
    ifelse(
      metrics$tuned, "glmnet tuned (Breslow)",
      "glmnet (Breslow)"
    )
  )
  
b <- metrics %>% 
    filter(metric == "Uno's C") %>%
    ggplot(aes(x = model_type, y = value, fill = model_type)) + geom_boxplot() + theme_big_simple() + labs(x = "", y = "Harrell's C", fill = "") +
    facet_wrap(~interaction(calc_type))+ 
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) +
    scale_fill_manual(values=ggpubfigs::friendly_pals$ito_seven[c(1, 2, 4, 5, 6, 7)]) 

  
cv_fig <- cowplot::plot_grid(a, b,   labels = c("A", "B"),
                   nrow = 2,
                   label_size = 24)

ggsave(here::here("~", "Downloads", "fig-S3_finalized.pdf"), plot = cv_fig, dpi = 300, height = 20/1.5, width = 15, units = "in")
ggsave(here::here("~", "Downloads", "fig-S3_finalized.svg"), plot = cv_fig, dpi = 300, height = 20/1.5, width = 15, units = "in")

signif_frame <- data.frame(
    pval = c(
      #round(wilcox.test(
      #  x = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "KD Breslow") %>% group_by(cancer) %>% summarise(mean(val#e)) %>% pull(`mean(value)`),
      #  y = fig_1_ab %>% filter(metric == "Harrell's C") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
      #    paired = TRUE,
      #    alternative = "greater"
      #  )$p.val, 3),
      round(wilcox.test(
        x = metrics %>% filter(metric == "Harrell's C" & calc_type == "5-fold CV 25 reps (per CV)") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
        y = metrics %>% filter(metric == "Harrell's C" & calc_type == "5-fold CV 25 reps (per CV)") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
        paired = TRUE,
        alternative = "less"
      )$p.val, 3),
      round(wilcox.test(
        x = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "glmnet tuned (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
        y = fig_1_cd %>% filter(metric == "IBS") %>% filter(model_type == "Teacher Cox-Nnet") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
        paired = TRUE,
        alternative = "greater"
      )$p.val, 3)
    ),
    model_type = rep("KD Breslow (min)", 2),
    end = c("KD Breslow (min)", "KD Breslow (min)"),
    start = c("glmnet (Breslow)", "glmnet tuned (Breslow)")
  )


round(wilcox.test(
  x = metrics %>% filter(metric == "Uno's C" & calc_type == "5-fold CV 25 reps (per CV)") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
  y = metrics %>% filter(metric == "Uno's C" & calc_type == "5-fold CV 25 reps (per CV)") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
  paired = TRUE,
  alternative = "less"
)$p.val, 3)

round(wilcox.test(
  x = metrics %>% filter(metric == "Harrell's C" & calc_type == "5-fold CV 5 reps (per split)") %>% filter(model_type == "glmnet (Breslow)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
  y = metrics %>% filter(metric == "Harrell's C" & calc_type == "5-fold CV 5 reps (per split)") %>% filter(model_type == "KD Breslow (min)") %>% group_by(cancer) %>% summarise(mean(value)) %>% pull(`mean(value)`),
  paired = TRUE,
  alternative = "less"
)$p.val, 3)



sparsity <- data.frame(
  sparsity = c(
    c(
      unlist(as.vector(vroom::vroom(
        here::here(
          "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "non_kd", "breslow", "sparsity_vvh_lambda.min.csv"  
        )
      )[1:25, ]))
      
    ),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "non_kd", "breslow", "sparsity_tuned_l1_ratio_vvh_lambda.min.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "breslow", "sparsity_linear_predictor_min.csv"  
      )
    )[1:25, ])),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "breslow", "sparsity_linear_predictor_pcvl.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "cox_nnet", "sparsity_linear_predictor_min.csv"  
      )
    )))
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 25), 9),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow (min)", "KD Breslow (pcvl)", "KD Cox-Nnet (min)"), each = 225)
  
)


sparsity %>% group_by(model, cancer) %>% summarise(value=paste0(round(mean(sparsity), 2), " (", round(sd(sparsity), 2), ")")) %>% pivot_wider(names_from = cancer, values_from = value) %>% knitr::kable(booktabs = TRUE, format = "latex")
                                                   


config <- rjson::fromJSON(
  file = here::here(
    "/", "Volumes", "Backup", "cr", "sparsesurv", "config.json"
  )
)

metrics <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall_fixed.csv"))
#metrics_old <- vroom::vroom(here::here("/", "Volumes", "Backup", "cr", "sparsesurv", "results", "metrics", "metrics_overall.csv"))
#metrics_old <- vroom::vroom(here::here("~", "Downloads", "sparsesurv_final", "plots", "metrics_overall.csv"))


regular_metrics <- metrics %>% filter(model %in% c("breslow", "cox_nnet")) %>% filter(lambda %in% c("min", "lambda.min", "pcvl")) %>% filter(metric %in% c("IBS"))
regular_metrics$model_type <- ifelse(regular_metrics$model == "breslow" & regular_metrics$lambda == "pcvl","KD Breslow (pcvl)", 
                              ifelse(regular_metrics$model == "breslow" & regular_metrics$kd,
                                     "KD Breslow (min)",
                                     ifelse(
                                       regular_metrics$model == "cox_nnet",
                                       "KD Cox-Nnet (min)",
                                       ifelse(
                                         regular_metrics$tuned, "glmnet tuned (Breslow)",
                                         "glmnet (Breslow)"
                                       )
                                       
                                     )
                                     
                                     
                                     
                              )
                              
)
ibs_metrics <- metrics %>% filter(model == "breslow" & !kd & lambda == 0) %>% filter(metric %in% c("IBS"))

regular_metrics %>% left_join(ibs_metrics, by = c("cancer" = "cancer", "split" = "split")) %>%
  mutate(model = model_type, cancer = cancer, split = split, is_kd = value.x == value.y) %>%
  group_by(model, cancer) %>% summarise(sum=sum(is_kd)) %>%
  pivot_wider(names_from=cancer, values_from = sum) %>%
  knitr::kable(booktabs=TRUE, format="latex")
  #
  #group_by()




timing <- data.frame(
  time = c(
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "non_kd", "breslow", "timing.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv","results",  "non_kd", "breslow", "timing_tuned_l1_ratio.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "breslow", "timing_tuned_teacher.csv"  
      )
    ))),
    unlist(as.vector(vroom::vroom(
      here::here(
        "//", "Volumes", "Backup", "cr", "sparsesurv", "results", "kd", "cox_nnet", "timing.csv"  
      )
    )))
    
    
    
    
    
  ),
  cancer = rep(rep(config$datasets, each = 5), 4),
  model = rep(c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow", "KD Cox-Nnet"), each = 50)
  
)

timing$cancer <- factor(timing$cancer, levels = timing %>% group_by(cancer) %>% summarise(mean=mean(time)) %>% arrange(desc(`mean`)) %>% pull(cancer))
timing$model <- factor(timing$model, levels = c("glmnet (Breslow)", "glmnet tuned (Breslow)", "KD Breslow", "KD Cox-Nnet"))

timing  %>% group_by(model, cancer) %>% 
  summarise(value=paste0(round(mean(time), 2), " (", round(sd(time), 2), ")")) %>% pivot_wider(names_from = cancer, values_from = value) %>% knitr::kable(booktabs = TRUE, format = "latex")
