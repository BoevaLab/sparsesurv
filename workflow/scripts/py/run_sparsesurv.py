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

    from sparsesurv._base import KDSurv
    from sparsesurv.cv import KDPHElasticNetCV
    from sparsesurv.loss import breslow_negative_likelihood, efron_negative_likelihood
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

    with open(snakemake.params["config_path"]) as f:
        config = json.load(f)

    def breslow_score_wrapper(y_true, y_pred):
        time, event = inverse_transform_survival(y_true)
        return np.negative(
            breslow_negative_likelihood(
                linear_predictor=np.squeeze(y_pred), time=time, event=event
            )
        )

    def efron_score_wrapper(y_true, y_pred):
        time, event = inverse_transform_survival(y_true)
        return np.negative(
            efron_negative_likelihood(
                linear_predictor=np.squeeze(y_pred), time=time, event=event
            )
        )

    np.random.seed(config["random_state"])
    g = np.random.default_rng(config.get("random_state"))

    for tune_l1_ratio in [False]:
        for tie_correction in ["breslow"]:
            for score_type in ["min", "pcvl"]:
                for score in ["linear_predictor"]:
                    results = {}
                    failures = {}
                    sparsity = {}
                    pipe = KDSurv(
                        teacher=GridSearchCV(
                            estimator=make_pipeline(
                                VarianceThreshold(),
                                StandardScaler(),
                                PCA(n_components=config["pc_n_components_tuned"], random_state=config["random_state"]),
                                CoxPHSurvivalAnalysis(ties=tie_correction),
                            ),
                            param_grid={
                                "pca__n_components": config["pc_n_components_tuned"]
                            },
                            n_jobs=config["n_jobs"],
                            scoring=make_scorer(efron_score_wrapper),
                            cv=StratifiedSurvivalKFold(
                                n_splits=config["n_inner_cv"],
                                shuffle=config["shuffle_cv"],
                                random_state=config["random_state"],
                            ),
                        ),
                        student=make_pipeline(
                            VarianceThreshold(),
                            StandardScaler(),
                            KDPHElasticNetCV(
                                tie_correction=tie_correction,
                                l1_ratio=config[
                                    f"l1_ratio{'_tuned' if tune_l1_ratio else ''}"
                                ],
                                eps=config["eps"],
                                n_alphas=config["n_alphas"],
                                cv=config["n_inner_cv"],
                                stratify_cv=config["stratify_cv"],
                                seed=np.random.RandomState(config["random_state"]),
                                shuffle_cv=config["shuffle_cv"],
                                n_jobs=config["n_jobs"],
                                cv_score_method=score,
                                alpha_type=score_type,
                            ),
                        ),
                    )

                    for cancer in config["datasets"]:

                        train_splits = pd.read_csv(
                            f"results/make_splits/{cancer}_train_splits.csv"
                        )
                        test_splits = pd.read_csv(
                            f"results/make_splits/{cancer}_test_splits.csv"
                        )
                        data = pd.read_csv(
                            f"results/preprocess_data/{cancer}.csv"
                        ).iloc[:, 1:]
                        X_ = data.iloc[:, 3:]
                        y_ = transform_survival(
                            time=data["OS_days"].values, event=data["OS"].values
                        )
                        for split in range(125 if score_type == "min" else 25):
                            train_ix = (
                                train_splits.iloc[split, :]
                                .dropna()
                                .to_numpy()
                                .astype(int)
                            )
                            test_ix = (
                                test_splits.iloc[split, :]
                                .dropna()
                                .to_numpy()
                                .astype(int)
                            )
                            X_train = (
                                X_.iloc[train_ix, :]
                                .copy()
                                .reset_index(drop=True)
                                .to_numpy()
                            )
                            y_train = y_[train_ix].copy()
                            y_test = y_[test_ix].copy()
                            X_test = (
                                X_.iloc[test_ix, :]
                                .copy()
                                .reset_index(drop=True)
                                .to_numpy()
                            )
                            if split == 0:
                                results[cancer] = {}
                                sparsity[cancer] = {}
                                failures[cancer] = [0]
                            try:
                                pipe.fit(X_train, y_train)
                                sparsity[cancer][split] = np.sum(
                                    pipe.student[-1].coef_ != 0
                                )
                                results[cancer][split] = pipe.predict(X_test)
                                surv = pipe.predict_survival_function(
                                    X_test, np.unique(y_test["time"])
                                )
                                surv.to_csv(
                                    f"results/kd/{tie_correction}/{cancer}/survival_function{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}_{split+1}.csv",
                                    index=False,
                                )
                            except ValueError as e:
                                failures[cancer][0] += 1
                                results[cancer][split] = np.zeros(test_ix.shape[0])
                                sparsity[cancer][split] = 0

                        pd.concat(
                            [
                                pd.DataFrame(results[cancer][i])
                                for i in range(125 if score_type == "min" else 25)
                            ],
                            axis=1,
                        ).to_csv(
                            f"results/kd/{tie_correction}/{cancer}/eta{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}.csv",
                            index=False,
                        )

                    pd.DataFrame(sparsity).to_csv(
                        f"results/kd/{tie_correction}/sparsity{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}.csv",
                        index=False,
                    )
                    pd.DataFrame(failures).to_csv(
                        f"results/kd/{tie_correction}/failures{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}.csv",
                        index=False,
                    )

    for tune_l1_ratio in [False]:
        for score_type in ["min"]:
            results = {}
            failures = {}
            sparsity = {}
            for cancer in config["datasets"]:
                for score in ["linear_predictor"]:

                    train_splits = pd.read_csv(
                        f"results/make_splits/{cancer}_train_splits.csv"
                    )
                    test_splits = pd.read_csv(
                        f"results/make_splits/{cancer}_test_splits.csv"
                    )
                    data = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[
                        :, 1:
                    ]
                    X_ = data.iloc[:, 3:]
                    y_ = transform_survival(
                        time=data["OS_days"].values, event=data["OS"].values
                    )
                    pipe = KDSurv(
                        teacher=RandomizedSearchCV(
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
                                                patience=config[
                                                    "early_stopping_patience"
                                                ],
                                                load_best=True,
                                            ),
                                        ),
                                        ("seed", FixSeed(generator=g)),
                                    ],
                                    module__activation=torch.nn.ReLU,
                                ),
                            ),
                            param_distributions=[
                                {
                                    "coxphneuralnet__lr": config["tune_lr"],
                                    "coxphneuralnet__optimizer__weight_decay": config[
                                        "tune_weight_decay"
                                    ],
                                    "coxphneuralnet__module__modality_hidden_layer_size": config[
                                        "tune_modality_hidden_layer_size"
                                    ],
                                    "coxphneuralnet__module__modality_hidden_layers": config[
                                        "tune_modality_hidden_layers"
                                    ],
                                    "coxphneuralnet__module__p_dropout": config[
                                        "tune_p_dropout"
                                    ],
                                    "coxphneuralnet__batch_size": config[
                                        "tune_batch_size"
                                    ],
                                }
                            ],
                            n_jobs=config["n_jobs"],
                            scoring=make_scorer(breslow_score_wrapper),
                            cv=StratifiedSurvivalKFold(
                                n_splits=config["n_inner_cv"],
                                shuffle=config["shuffle_cv"],
                                random_state=config["random_state"],
                            ),
                            error_score=config["error_score"],
                            verbose=False,
                            n_iter=config["random_search_n_iter"],
                            random_state=config["random_state"],
                        ),
                        student=make_pipeline(
                            VarianceThreshold(),
                            StandardScaler(),
                            KDPHElasticNetCV(
                                tie_correction="breslow",
                                l1_ratio=config[
                                    f"l1_ratio{'_tuned' if tune_l1_ratio else ''}"
                                ],
                                eps=config["eps"],
                                n_alphas=config["n_alphas"],
                                cv=config["n_inner_cv"],
                                stratify_cv=config["stratify_cv"],
                                seed=np.random.RandomState(config["random_state"]),
                                shuffle_cv=config["shuffle_cv"],
                                cv_score_method=score,
                                n_jobs=config["n_jobs"],
                                alpha_type=score_type,
                            ),
                        ),
                    )
                    for split in range(25):
                        train_ix = (
                            train_splits.iloc[split, :].dropna().to_numpy().astype(int)
                        )
                        test_ix = (
                            test_splits.iloc[split, :].dropna().to_numpy().astype(int)
                        )
                        X_train = (
                            X_.iloc[train_ix, :]
                            .copy()
                            .reset_index(drop=True)
                            .to_numpy(np.float32)
                        )
                        y_train = y_[train_ix].copy()
                        y_test = y_[test_ix].copy()
                        X_test = (
                            X_.iloc[test_ix, :]
                            .copy()
                            .reset_index(drop=True)
                            .to_numpy(np.float32)
                        )
                        if split == 0:
                            results[cancer] = {}
                            sparsity[cancer] = {}
                            failures[cancer] = [0]
                        try:
                            pipe.fit(X_train, y_train)
                            sparsity[cancer][split] = np.sum(
                                pipe.student[-1].coef_ != 0
                            )
                            results[cancer][split] = pipe.predict(X_test)

                            surv = pipe.predict_survival_function(
                                X_test.astype(np.float32), np.unique(y_test["time"])
                            )

                            surv.to_csv(
                                f"results/kd/cox_nnet/{cancer}/survival_function{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}_{split+1}.csv",
                                index=False,
                            )
                        except ValueError as e:
                            raise e
                    pd.concat(
                        [pd.DataFrame(results[cancer][i]) for i in range(25)],
                        axis=1,
                    ).to_csv(
                        f"results/kd/cox_nnet/{cancer}/eta{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}.csv",
                        index=False,
                    )
            pd.DataFrame(sparsity).to_csv(
                f"results/kd/cox_nnet/sparsity{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}.csv",
                index=False,
            )
            pd.DataFrame(failures).to_csv(
                f"results/kd/cox_nnet/failures{'_tuned_l1' if tune_l1_ratio else ''}_{score}_{score_type}.csv",
                index=False,
            )
