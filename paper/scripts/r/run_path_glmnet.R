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



failures <- list()

for (cancer in config$dataset) {
  n_failures <- 0
  result_sparsity <- list()
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
    result_sparsity[[split]] <- vector()
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
        fit <- glmnet(
          x = as.matrix(X_train),
          y = y_train,
          family = "cox",
          alpha = config$l1_ratio,
          lambda.min.ratio = config$eps,
          standardize = TRUE,
          nlambda = 100
        )
        pred <- predict(fit, as.matrix(X_test))
        path_coefs <- coef(fit)
        if (ncol(path_coefs) < 100) {
          stop()
        }
        for (z in 1:100) {
          if (z > ncol(path_coefs)) {
            result_sparsity[[split]] <- c(result_sparsity[[split]], 0)
            times <- sort(unique(y_test[, 1]))
            km <- exp(-survfit(y_test ~ 1)$cumhaz)
            km_surv <- matrix(rep(km, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
            colnames(km_surv) <- times
            data.frame(km_surv, check.names = FALSE) %>% readr::write_csv(
              here::here(
                "results", "non_kd", "breslow", cancer, "path", paste0("survival_function_", z, "_alpha_", split, ".csv")
              )
            )
          }
          else {
          current_coef <- path_coefs[, z]
          current_coef <- current_coef[current_coef != 0.0]
          if (length(current_coef) == 0) {
            result_sparsity[[split]] <- c(result_sparsity[[split]], 0)
            times <- sort(unique(y_test[, 1]))
            km <- exp(-survfit(y_test ~ 1)$cumhaz)
            km_surv <- matrix(rep(km, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
            colnames(km_surv) <- times
            data.frame(km_surv, check.names = FALSE) %>% readr::write_csv(
              here::here(
                "results", "non_kd", "breslow", cancer, "path", paste0("survival_function_", z, "_alpha_", split, ".csv")
              )
            )
          } else {
            X_train_survival <- cbind(data$OS_days[train_ix], data$OS[train_ix], X_train[, sapply(names(current_coef), function(x) grep(x, colnames(X_test), fixed = TRUE)), drop = FALSE])
            X_test_survival <- X_test[, sapply(names(current_coef), function(x) grep(x, colnames(X_test), fixed = TRUE)), drop = FALSE]
            colnames(X_train_survival)[1:2] <- c("time", "event")
            cox_helper <- coxph(Surv(time, event) ~ ., data = X_train_survival, ties = "breslow", init = current_coef, iter.max = 0, x = TRUE)
            surv <- pec::predictSurvProb(cox_helper, X_test_survival, unique(sort(y_test[, 1])))
            if (length(which(is.na(surv[1, ]))) > 1) {
              surv[, which(is.na(surv[1, ]))] <- matrix(rep(surv[, max(which(!is.na(surv[1, ])))], length(which(is.na(surv[1, ])))), ncol = length(which(is.na(surv[1, ]))))
            } else {
              surv[, which(is.na(surv[1, ]))] <- surv[, max(which(!is.na(surv[1, ])))]
            }
            colnames(surv) <- unique(sort(y_test[, 1]))
            result_sparsity[[split]] <- c(result_sparsity[[split]], length(current_coef))
            data.frame(surv, check.names = FALSE) %>% readr::write_csv(
              here::here(
                "results", "non_kd", "breslow", cancer, "path", paste0("survival_function_", z, "_alpha_", split, ".csv")
              )
            )
          }
          }

        }
      },
      error = function(cond) {
        print(cond)
        n_failures <- n_failures + 1
        result_sparsity[[split]] <- vector()
        times <- sort(unique(y_test[, 1]))
        km <- exp(-survfit(y_test ~ 1)$cumhaz)
        km_surv <- matrix(rep(km, nrow(X_test)), nrow = nrow(X_test), byrow = TRUE)
        colnames(km_surv) <- times
        for (z in 1:100) {
          result_sparsity[[split]] <- c(result_sparsity[[split]], 0)
          data.frame(km_surv, check.names = FALSE) %>% readr::write_csv(
            here::here(
              "results", "non_kd", "breslow", cancer, "path", paste0("survival_function_", z, "_alpha_", split, ".csv")
            )
          )
        }
      }
    )
  }
  data.frame(result_sparsity) %>% write_csv(
    here::here(
      "results", "non_kd", "breslow", cancer, "path", "sparsity.csv"
    )
  )
  failures[[cancer]] <- n_failures
}

data.frame(failures) %>% write_csv(
  here::here(
    "results", "non_kd", "breslow", cancer, "path", "failures.csv"
  )
)
