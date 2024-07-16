import sys

with open(snakemake.log[0], "w") as f:
    sys.stderr = sys.stdout = f
    import json

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from skorch.callbacks import EarlyStopping
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.linear_model.coxph import BreslowEstimator
    from sparsesurv.loss import breslow_negative_likelihood
    from sparsesurv.neuralsurv.python.model.model import SKORCH_MODULE_FACTORY
    from sparsesurv.neuralsurv.python.model.skorch_infra import FixSeed
    from sparsesurv.neuralsurv.python.utils.factories import (
        CRITERION_FACTORY,
        SKORCH_NET_FACTORY,
    )
    from sparsesurv.neuralsurv.python.utils.misc_utils import (
        StratifiedSkorchSurvivalSplit,
        StratifiedSurvivalKFold,
    )
    from sparsesurv.utils import inverse_transform_survival, transform_survival

    def breslow_score_wrapper(y_true, y_pred):
        time, event = inverse_transform_survival(y_true)
        return np.negative(
            breslow_negative_likelihood(
                linear_predictor=np.squeeze(y_pred), time=time, event=event
            )
        )

    with open(snakemake.params["config_path"]) as f:
        config = json.load(f)

    np.random.seed(config["random_state"])
    g = np.random.default_rng(config.get("random_state"))

    results = {}
    for tie_correction in ["breslow"]:
        teacher = GridSearchCV(
            estimator=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                PCA(
                    n_components=config["pc_n_components"],
                    random_state=config["random_state"],
                ),
                CoxPHSurvivalAnalysis(ties=tie_correction),
            ),
            param_grid={"pca__n_components": config["pc_n_components_tuned"]},
            n_jobs=config["n_jobs"],
            scoring=make_scorer(breslow_score_wrapper),
            cv=StratifiedSurvivalKFold(
                n_splits=config["n_inner_cv"],
                shuffle=config["shuffle_cv"],
                random_state=config["random_state"],
            ),
        )

        for cancer in config["datasets"]:
            results[cancer] = {}
            train_splits = pd.read_csv(f"results/make_splits/{cancer}_train_splits.csv")
            test_splits = pd.read_csv(f"results/make_splits/{cancer}_test_splits.csv")
            data = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[:, 1:]
            X_ = data.iloc[:, 3:]
            y_ = transform_survival(
                time=data["OS_days"].values, event=data["OS"].values
            )
            for split in range(25):
                train_ix = train_splits.iloc[split, :].dropna().to_numpy().astype(int)
                test_ix = test_splits.iloc[split, :].dropna().to_numpy().astype(int)
                X_train = X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
                y_train = y_[train_ix].copy()
                y_test = y_[test_ix].copy()
                X_test = X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()

                teacher.fit(X_train, y_train)
                results[cancer][split] = teacher.predict(X_test)
                (
                    cumulative_baseline_hazards_times,
                    cumulative_baseline_hazards,
                ) = (
                    teacher.best_estimator_[3].cum_baseline_hazard_.x,
                    teacher.best_estimator_[3].cum_baseline_hazard_.y,
                )
                cumulative_baseline_hazards = np.concatenate(
                    [np.array([0.0]), cumulative_baseline_hazards]
                )
                cumulative_baseline_hazards_times: np.array = np.concatenate(
                    [np.array([0.0]), cumulative_baseline_hazards_times]
                )
                cumulative_baseline_hazards: np.array = np.tile(
                    A=cumulative_baseline_hazards[
                        np.digitize(
                            x=np.unique(y_test["time"]),
                            bins=cumulative_baseline_hazards_times,
                            right=False,
                        )
                        - 1
                    ],
                    reps=X_test.shape[0],
                ).reshape((X_test.shape[0], np.unique(y_test["time"]).shape[0]))
                log_hazards: np.array = (
                    np.tile(
                        A=teacher.predict(X_test),
                        reps=np.unique(y_test["time"]).shape[0],
                    )
                    .reshape((np.unique(y_test["time"]).shape[0], X_test.shape[0]))
                    .T
                )
                surv: pd.DataFrame = np.exp(
                    -pd.DataFrame(
                        cumulative_baseline_hazards * np.exp(log_hazards),
                        columns=np.unique(y_test["time"]),
                    )
                )
                surv.to_csv(
                    f"results/kd/{tie_correction}/{cancer}/survival_function_teacher_{split+1}.csv",
                    index=False,
                )
            pd.concat(
                [pd.DataFrame(results[cancer][i]) for i in range(25)],
                axis=1,
            ).to_csv(
                f"results/kd/{tie_correction}/{cancer}/eta_teacher.csv",
                index=False,
            )

    results = {}
    for cancer in config["datasets"]:
        results[cancer] = {}
        train_splits = pd.read_csv(f"results/make_splits/{cancer}_train_splits.csv")
        test_splits = pd.read_csv(f"results/make_splits/{cancer}_test_splits.csv")
        data = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[:, 1:]
        X_ = data.iloc[:, 3:]
        y_ = transform_survival(time=data["OS_days"].values, event=data["OS"].values)
        teacher_cox_nnet = RandomizedSearchCV(
            estimator=make_pipeline(
                StandardScaler(),
                SKORCH_NET_FACTORY["cox"](
                    module=SKORCH_MODULE_FACTORY["cox"],
                    criterion=CRITERION_FACTORY["cox"],
                    module__fusion_method="early",
                    module__blocks=[[i for i in range(X_.shape[1])]],
                    iterator_train__shuffle=True,
                    optimizer=torch.optim.AdamW,
                    max_epochs=config["max_epochs"],
                    verbose=False,
                    train_split=StratifiedSkorchSurvivalSplit(
                        config["validation_set_neural"],
                        stratified=config["stratify_cv"],
                        random_state=config.get("random_state"),
                    ),
                    callbacks=[
                        (
                            "es",
                            EarlyStopping(
                                monitor="valid_loss",
                                patience=config["early_stopping_patience"],
                                load_best=True,
                            ),
                        ),
                        ("seed", FixSeed(generator=g)),
                    ],
                    module__activation=torch.nn.ReLU,
                ),
            ),
            param_distributions={
                "coxphneuralnet__lr": config["tune_lr"],
                "coxphneuralnet__optimizer__weight_decay": config["tune_weight_decay"],
                "coxphneuralnet__module__modality_hidden_layer_size": config[
                    "tune_modality_hidden_layer_size"
                ],
                "coxphneuralnet__module__modality_hidden_layers": config[
                    "tune_modality_hidden_layers"
                ],
                "coxphneuralnet__module__p_dropout": config["tune_p_dropout"],
                "coxphneuralnet__batch_size": config["tune_batch_size"],
            },
            n_jobs=config["n_jobs"],
            random_state=config["random_state"],
            scoring=make_scorer(breslow_score_wrapper),
            cv=StratifiedSurvivalKFold(
                n_splits=config["n_inner_cv"],
                shuffle=config["shuffle_cv"],
                random_state=config["random_state"],
            ),
            error_score=config["error_score"],
            verbose=False,
            n_iter=config["random_search_n_iter"],
        )
        for split in range(25):
            train_ix = train_splits.iloc[split, :].dropna().to_numpy().astype(int)
            test_ix = test_splits.iloc[split, :].dropna().to_numpy().astype(int)
            X_train = (
                X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy(np.float32)
            )
            y_train = y_[train_ix].copy()
            y_test = y_[test_ix].copy()
            X_test = (
                X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy(np.float32)
            )

            teacher_cox_nnet.fit(X_train, y_train)
            results[cancer][split] = teacher_cox_nnet.predict(X_test)
            breslow = BreslowEstimator()
            breslow.fit(
                linear_predictor=teacher_cox_nnet.predict(X_train),
                time=data["OS_days"][train_ix].values,
                event=data["OS"][train_ix].values,
            )
            (
                cumulative_baseline_hazards_times,
                cumulative_baseline_hazards,
            ) = (
                breslow.cum_baseline_hazard_.x,
                breslow.cum_baseline_hazard_.y,
            )
            cumulative_baseline_hazards = np.concatenate(
                [np.array([0.0]), cumulative_baseline_hazards]
            )
            cumulative_baseline_hazards_times: np.array = np.concatenate(
                [np.array([0.0]), cumulative_baseline_hazards_times]
            )
            cumulative_baseline_hazards: np.array = np.tile(
                A=cumulative_baseline_hazards[
                    np.digitize(
                        x=np.unique(y_test["time"]),
                        bins=cumulative_baseline_hazards_times,
                        right=False,
                    )
                    - 1
                ],
                reps=X_test.shape[0],
            ).reshape((X_test.shape[0], np.unique(y_test["time"]).shape[0]))
            log_hazards: np.array = (
                np.tile(
                    A=teacher_cox_nnet.predict(X_test).squeeze(),
                    reps=np.unique(y_test["time"]).shape[0],
                )
                .reshape((np.unique(y_test["time"]).shape[0], X_test.shape[0]))
                .T
            )
            surv: pd.DataFrame = np.exp(
                -pd.DataFrame(
                    cumulative_baseline_hazards * np.exp(log_hazards),
                    columns=np.unique(y_test["time"]),
                )
            )
            surv.to_csv(
                f"results/kd/cox_nnet/{cancer}/survival_function_teacher_{split+1}.csv",
                index=False,
            )

        pd.concat(
            [pd.DataFrame(results[cancer][i]) for i in range(25)],
            axis=1,
        ).to_csv(
            f"results/kd/cox_nnet/{cancer}/eta_teacher.csv",
            index=False,
        )
