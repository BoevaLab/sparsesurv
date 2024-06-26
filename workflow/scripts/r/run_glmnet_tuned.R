log <- file(snakemake@log[[1]], open = "wt")
sink(log)

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(coefplot)
  library(pec)
  library(readr)
  library(vroom)
  library(dplyr)
  library(splitTools)
  library(glmnetUtils)
  library(parallel)
  library(doParallel)
})

glmnet::glmnet.control(
  fdev = 0,
  devmax = 1.0
)

config <- rjson::fromJSON(
  file = snakemake@params[["config_path"]]
)

# Get alpha.
get_alpha <- function(fit) {
  alpha <- fit$alpha
  error <- sapply(fit$modlist, function(mod) {
    min(mod$cvm)
  })
  alpha[which.min(error)]
}

set.seed(config$seed)

# https://stackoverflow.com/questions/7196450/create-a-dataframe-of-unequal-lengths
na.pad <- function(x, len) {
  x[1:len]
}

makePaddedDataFrame <- function(l, ...) {
  maxlen <- max(sapply(l, length))
  data.frame(lapply(l, na.pad, len = maxlen), ...)
}

for (tune_l1_ratio in c(TRUE)) {
  for (score in c("vvh")) {
    for (cv_score in c("lambda.min")) {
      n_failures <- 0
      failures <- list()
      sparsity <- list()

      for (cancer in snakemake@params[["cancer"]]) {
        result_sparsity <- c()
        lp_df <- list()
        data <- data.frame(vroom::vroom(
          paste(
            "results", "preprocess_data",
            paste0(cancer, ".csv"),
            sep = "/"
          )
        )[, -1], check.names = FALSE)
        train_splits <- data.frame(vroom::vroom(
          paste(
            "results", "make_splits",
            paste0(cancer, "_train_splits.csv"),
            sep = "/"
          )
        ), check.names = FALSE)
        test_splits <- data.frame(vroom::vroom(
          paste(
            "results", "make_splits",
            paste0(cancer, "_test_splits.csv"),
            sep = "/"
          )
        ), check.names = FALSE)


        for (split in 1:125) {
          train_ix <- as.numeric(unname(train_splits[split, ]))
          train_ix <- train_ix[!is.na(train_ix)] + 1

          test_ix <- as.numeric(unname(test_splits[split, ]))
          test_ix <- test_ix[!is.na(test_ix)] + 1

          X_train <- data[train_ix, -(1:2)]
          X_test <- data[test_ix, -(1:2)]
          y_train <- Surv(data$OS_days[train_ix], data$OS[train_ix])
          y_test <- Surv(data$OS_days[test_ix], data$OS[test_ix])
          fold_ids <- rep(0, length(y_train))

          fold_helper <- create_folds(
            y = data$OS[train_ix],
            k = config$n_inner_cv,
            type = c("stratified"),
            invert = TRUE,
            seed = config$seed
          )
          for (i in 1:length(fold_helper)) {
            fold_ids[fold_helper[[i]]] <- i
          }
          result <- tryCatch(
            {
              if (tune_l1_ratio) {
                fit <- cva.glmnet(
                  x = as.matrix(X_train),
                  y = y_train,
                  family = "cox",
                  alpha = config$l1_ratio_tuned,
                  lambda.min.ratio = config$eps,
                  standardize = TRUE,
                  nlambda = config$n_alphas,
                  nfolds = config$n_inner_cv,
                  grouped = score == "vvh",
                  foldid = fold_ids # ,
                  # outerParallel = {inner_cl <- parallel::makeForkCluster(7); parallel::clusterSetRNGStream(inner_cl, config$seed); inner_cl}
                )

                fit <- fit$modlist[[which((fit$alpha == get_alpha(fit)))]]
              } else {
                fit <- cv.glmnet(
                  x = as.matrix(X_train),
                  y = y_train,
                  family = "cox",
                  alpha = config$l1_ratio,
                  lambda.min.ratio = config$eps,
                  standardize = TRUE,
                  nlambda = config$n_alphas,
                  nfolds = config$n_inner_cv,
                  grouped = score == "vvh",
                  foldid = fold_ids
                )
              }

              if (cv_score %in% c("lambda.min", "lambda.1se")) {
                n_sparsity <- nrow(extract.coef(fit, cv_score))
                if (n_sparsity == 0) {
                  stop()
                }
                linear_predictor <- as.vector(predict(fit, as.matrix(X_test), s = fit$lambda.min))
                X_train_survival <- cbind(data$OS_days[train_ix], data$OS[train_ix], X_train[, sapply(rownames(extract.coef(fit, cv_score)), function(x) grep(x, colnames(X_test), fixed = TRUE)), drop = FALSE])
                X_test_survival <- X_test[, sapply(rownames(extract.coef(fit, cv_score)), function(x) grep(x, colnames(X_test), fixed = TRUE)), drop = FALSE]
              } else {
                if (which(fit$lambda.min == fit$lambda) == 1) {
                  stop()
                }
                transformed_error_space <- fit$cvm - (((fit$lambda[which(fit$lambda == fit$lambda.min)] - fit$cvm[1]) / (fit$nzero[which(fit$lambda == fit$lambda.min)])) * fit$nzero)
                lambda_ix <- which.min(transformed_error_space[1:which(fit$lambda == fit$lambda.min)])
                coefs <- fit$glmnet.fit$beta[, lambda_ix]
                coefs <- coefs[coefs != 0.0]
                n_sparsity <- length(coefs)
                if (n_sparsity == 0) {
                  stop()
                }

                linear_predictor <- as.vector(as.matrix(X_test) %*% as.matrix(fit$glmnet.fit$beta[, lambda_ix]))
                X_train_survival <- cbind(data$OS_days[train_ix], data$OS[train_ix], X_train[, sapply(names(coefs), function(x) grep(x, colnames(X_test), fixed = TRUE)), drop = FALSE])
                X_test_survival <- X_test[, sapply(names(coefs), function(x) grep(x, colnames(X_test), fixed = TRUE)), drop = FALSE]
              }

              colnames(X_train_survival)[1:2] <- c("time", "event")

              cox_helper <- coxph(Surv(time, event) ~ ., data = X_train_survival, ties = "breslow", init = extract.coef(fit, cv_score)[, 1], iter.max = 0, x = TRUE)
              surv <- pec::predictSurvProb(cox_helper, X_test_survival, unique(sort(y_test[, 1])))
              if (length(which(is.na(surv[1, ]))) > 1) {
                surv[, which(is.na(surv[1, ]))] <- matrix(rep(surv[, max(which(!is.na(surv[1, ])))], length(which(is.na(surv[1, ])))), ncol = length(which(is.na(surv[1, ]))))
              } else {
                surv[, which(is.na(surv[1, ]))] <- surv[, max(which(!is.na(surv[1, ])))]
              }
              colnames(surv) <- unique(sort(y_test[, 1]))
              list(sparsity = n_sparsity, linear_predictor = linear_predictor, surv = surv, failures = 0)
            },
            error = function(cond) {
              times <- sort(unique(y_test[, 1]))
              km <- exp(-survfit(y_test ~ 1)$cumhaz)
              km_surv <- matrix(rep(km, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
              colnames(km_surv) <- times
              return(list(sparsity = 0, linear_predictor = rep(0, nrow(X_test)), surv = km_surv, failures = 1))
            }
          )
          if (tune_l1_ratio) {
            data.frame(result$surv, check.names = FALSE) %>% readr::write_csv(
              paste(
                "results", "non_kd", "breslow", cancer, paste0("survival_function_tuned_l1_ratio_", score, "_", cv_score, "_", split, ".csv"),
                sep = "/"
              )
            )
          } else {
            data.frame(result$surv, check.names = FALSE) %>% readr::write_csv(
              paste(
                "results", "non_kd", "breslow", cancer, paste0("survival_function_", score, "_", cv_score, "_", split, ".csv"),
                sep = "/"
              )
            )
          }

          n_failures <- n_failures + result$failures
          result_sparsity <- c(result_sparsity, result$sparsity)
          lp_df[[split]] <- result$linear_predictor
        }
        failures[[cancer]] <- n_failures
        sparsity[[cancer]] <- result_sparsity
        lp_df <- makePaddedDataFrame(lp_df)
        colnames(lp_df) <- 1:ncol(lp_df)
        if (tune_l1_ratio) {
          lp_df %>%
            write.csv(
              paste(
                "results", "non_kd", "breslow", cancer, paste0("eta_tuned_l1_ratio_", score, "_", cv_score, ".csv"),
                sep = "/"
              )
            )
        } else {
          lp_df %>%
            write.csv(
              paste(
                "results", "non_kd", "breslow", cancer, paste0("eta_", score, "_", cv_score, ".csv"),
                sep = "/"
              )
            )
        }
      }

      if (tune_l1_ratio) {
        data.frame(failures) %>% write_csv(
          paste(
            "results", "non_kd", "breslow", cancer, paste0("failures_tuned_l1_ratio_", score, "_", cv_score, ".csv"),
            sep = "/"
          )
        )

        data.frame(sparsity) %>% write_csv(
          paste(
            "results", "non_kd", "breslow", cancer, paste0("sparsity_tuned_l1_ratio_", score, "_", cv_score, ".csv"),
            sep = "/"
          )
        )
      } else {
        data.frame(failures) %>% write_csv(
          paste(
            "results", "non_kd", "breslow", paste0("failures_", score, "_", cv_score, ".csv"),
            sep = "/"
          )
        )

        data.frame(sparsity) %>% write_csv(
          paste(
            "results", "non_kd", "breslow", paste0("sparsity_", score, "_", cv_score, ".csv"),
            sep = "/"
          )
        )
      }
    }
  }
}
