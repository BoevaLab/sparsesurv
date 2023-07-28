library(rjson)
library(glmnet)
library(survival)
library(coefplot)
library(pec)
library(readr)
library(vroom)
library(dplyr)

glmnet::glmnet.control(
    fdev = 0,
    devmax = 1.0
)


config <- rjson::fromJSON(
  file = here::here(
    "config.json"
  )
)

set.seed(config$seed)


# https://stackoverflow.com/questions/7196450/create-a-dataframe-of-unequal-lengths
na.pad <- function(x, len) {
  x[1:len]
}

makePaddedDataFrame <- function(l, ...) {
  maxlen <- max(sapply(l, length))
  data.frame(lapply(l, na.pad, len = maxlen), ...)
}

for (score in c("vvh")) {
  for (cv_score in c("lambda.pcvl")) {
    n_failures <- 0
    failures <- list()
    sparsity <- list()

    for (cancer in config$dataset) {
      result_sparsity <- c()
      lp_df <- list()
      data <- data.frame(vroom::vroom(
        here::here(
          "data", "processed", "TCGA",
          paste0(cancer, "_data_preprocessed.csv")
        )
      )[, -1], check.names = FALSE)
      train_splits <- data.frame(vroom::vroom(
        here::here(
          "data", "splits", "TCGA",
          paste0(cancer, "_train_splits.csv")
        )
      ), check.names = FALSE)
      test_splits <- data.frame(vroom::vroom(
        here::here(
          "data", "splits", "TCGA",
          paste0(cancer, "_test_splits.csv")
        )
      ), check.names = FALSE)


      for (split in 1:25) {
        train_ix <- as.numeric(unname(train_splits[split, ]))
        train_ix <- train_ix[!is.na(train_ix)] + 1

        test_ix <- as.numeric(unname(test_splits[split, ]))
        test_ix <- test_ix[!is.na(test_ix)] + 1

        X_train <- data[train_ix, -(1:2)]
        X_test <- data[test_ix, -(1:2)]
        y_train <- Surv(data$OS_days[train_ix], data$OS[train_ix])
        y_test <- Surv(data$OS_days[test_ix], data$OS[test_ix])
        result <- tryCatch(
          {
            fit <- cv.glmnet(
              x = as.matrix(X_train),
              y = y_train,
              family = "cox",
              alpha = config$l1_ratio,
              lambda.min.ratio = config$eps,
              standardize = TRUE,
              nlambda = 100,
              nfolds = config$n_inner_cv,
              grouped = score == "vvh"
            )
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
            print(cond)
            times <- sort(unique(y_test[, 1]))
            km <- exp(-survfit(y_test ~ 1)$cumhaz)
            km_surv <- matrix(rep(km, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
            colnames(km_surv) <- times
            return(list(sparsity = 0, linear_predictor = rep(0, nrow(X_test)), surv = km_surv, failures = 1))
          }
        )

        data.frame(result$surv, check.names = FALSE) %>% readr::write_csv(
          here::here(
            "results", "non_kd", "breslow", cancer, paste0("survival_function_", score, "_", cv_score, "_", split, ".csv")
          )
        )
        n_failures <- n_failures + result$failures
        result_sparsity <- c(result_sparsity, result$sparsity)
        lp_df[[split]] <- result$linear_predictor
      }
      failures[[cancer]] <- n_failures
      sparsity[[cancer]] <- result_sparsity
      lp_df <- makePaddedDataFrame(lp_df)
      colnames(lp_df) <- 1:ncol(lp_df)
      lp_df %>%
        write.csv(
          here::here(
            "results", "non_kd", "breslow", cancer, paste0("eta_", score, "_", cv_score, ".csv")
          )
        )
    }

    data.frame(failures) %>% write_csv(
      here::here(
        "results", "non_kd", "breslow", paste0("failures_", score, "_", cv_score, ".csv")
      )
    )

    data.frame(sparsity) %>% write_csv(
      here::here(
        "results", "non_kd", "breslow", paste0("sparsity_", score, "_", cv_score, ".csv")
      )
    )
  }
}
