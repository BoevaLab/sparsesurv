import sys

with open(snakemake.log[0], "w") as f:
    sys.stderr = sys.stdout = f
    import json
    from timeit import default_timer as timer

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
                linear_predictor=np.squeeze(y_pred.astype(np.float64)),
                time=time,
                event=event,
            )
        )

    def efron_score_wrapper(y_true, y_pred):
        time, event = inverse_transform_survival(y_true)
        return np.negative(
            efron_negative_likelihood(linear_predictor=y_pred, time=time, event=event)
        )

    SCORE_FACTORY = {"breslow": breslow_score_wrapper, "efron": efron_score_wrapper}

    g = np.random.default_rng(config.get("random_state"))
    np.random.seed(config["random_state"])

    for tune_teacher in [True]:
        for tune_l1_ratio in [False]:
            for tie_correction in ["breslow"]:
                timing = {}
                for cancer in config["datasets"]:
                    timing[cancer] = []
                    print(f"Starting: {cancer}")
                    train_splits = pd.read_csv(
                        f"results/make_splits/{cancer}_train_splits.csv"
                    )
                    test_splits = pd.read_csv(
                        f"results/make_splits/{cancer}_test_splits.csv"
                    )
                    data = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[
                        :, 1:
                    ]
                    X_ = data.iloc[:, 3:].to_numpy()
                    y_ = transform_survival(
                        time=data["OS_days"].values, event=data["OS"].values
                    )
                    for rep in range(config["timing_reps"]):
                        pipe = KDSurv(
                            teacher=GridSearchCV(
                                estimator=make_pipeline(
                                    StandardScaler(),
                                    PCA(
                                        n_components=config["pc_n_components"],
                                        random_state=config["random_state"],
                                    ),
                                    CoxPHSurvivalAnalysis(ties=tie_correction),
                                ),
                                param_grid={
                                    "pca__n_components": config[
                                        f"pc_n_components{'_tuned' if tune_teacher else ''}"
                                    ]
                                },
                                n_jobs=1,
                                verbose=0,
                                scoring=make_scorer(SCORE_FACTORY[tie_correction]),
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
                                    cv_score_method="linear_predictor",
                                    n_jobs=1,
                                ),
                            ),
                        )
                        start = timer()
                        pipe.fit(X_, y_)
                        end = timer()
                        timing[cancer].append(end - start)
                if tune_l1_ratio:
                    pd.DataFrame(timing).to_csv(
                        f"results/kd/{tie_correction}/timing_tuned_l1_ratio{'_tuned_teacher' if tune_teacher else '' }.csv",
                        index=False,
                    )
                else:
                    pd.DataFrame(timing).to_csv(
                        f"results/kd/{tie_correction}/timing{'_tuned_teacher' if tune_teacher else '' }.csv",
                        index=False,
                    )

    for tune_l1_ratio in [False]:
        timing = {}
        for cancer in config["datasets"]:
            timing[cancer] = []
            print(f"Starting: {cancer}")
            train_splits = pd.read_csv(f"results/make_splits/{cancer}_train_splits.csv")
            test_splits = pd.read_csv(f"results/make_splits/{cancer}_test_splits.csv")
            data = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[:, 1:]
            X_ = data.iloc[:, 3:].to_numpy().astype(np.float32)
            y_ = transform_survival(
                time=data["OS_days"].values, event=data["OS"].values
            )
            for rep in range(config["timing_reps"]):
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
                            "coxphneuralnet__batch_size": config["tune_batch_size"],
                        },
                        n_jobs=1,
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
                            cv_score_method="linear_predictor",
                            n_jobs=1,
                        ),
                    ),
                )
                start = timer()
                pipe.fit(X_, y_)
                end = timer()
                timing[cancer].append(end - start)

        if tune_l1_ratio:
            pd.DataFrame(timing).to_csv(
                f"results/kd/cox_nnet/timing_tuned_l1_ratio.csv",
                index=False,
            )

        else:
            pd.DataFrame(timing).to_csv(
                f"results/kd/cox_nnet/timing.csv",
                index=False,
            )
