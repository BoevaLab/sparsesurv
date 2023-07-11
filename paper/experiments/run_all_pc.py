import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from survhive._base import PCSurv
from survhive.aft import AFT
from survhive.cv import PCAFTElasticNetCV, PCEHElasticNetCV, PCPHElasticNetCV
from survhive.eh import EH
from survhive.utils import transform_survival

with open(f"../config.json") as f:
    config = json.load(f)

np.random.seed(config["random_state"])

for score in ["linear_predictor", "mse", "basic", "vvh"]:
    results = {}
    failures = {}
    sparsity = {}
    pipe = PCSurv(
        pc_pipe=make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCA(n_components=config["pc_n_components"]),
            CoxPHSurvivalAnalysis(ties="efron"),
        ),
        model_pipe=make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCPHElasticNetCV(
                tie_correction="efron",
                l1_ratio=config["l1_ratio"],
                eps=config["eps"],
                n_alphas=config["n_alphas"],
                cv=config["n_inner_cv"],
                stratify_cv=config["stratify_cv"],
                seed=config["seed"],
                shuffle_cv=config["shuffle_cv"],
                cv_score_method=score,
                n_jobs=5,
            ),
        ),
    )

    for cancer in config["datasets"]:
        print(f"Starting: {cancer}")
        train_splits = pd.read_csv(
            f"../data/splits/TCGA/{cancer}_train_splits.csv"
        )
        test_splits = pd.read_csv(
            f"../data/splits/TCGA/{cancer}_test_splits.csv"
        )
        data = pd.read_csv(
            f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
        ).iloc[:, 1:]
        X_ = data.iloc[:, 3:]
        y_ = transform_survival(
            time=data["OS_days"].values, event=data["OS"].values
        )
        for split in range(25):
            print(f"Starting split: {split+1} / 25")
            train_ix = (
                train_splits.iloc[split, :].dropna().to_numpy().astype(int)
            )
            test_ix = (
                test_splits.iloc[split, :].dropna().to_numpy().astype(int)
            )
            X_train = (
                X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
            )
            y_train = y_[train_ix].copy()
            y_test = y_[test_ix].copy()
            X_test = (
                X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()
            )
            if split == 0:
                results[cancer] = {}
                sparsity[cancer] = {}
                failures[cancer] = [0]
            try:
                pipe.fit(X_train, y_train)
                sparsity[cancer][split] = np.sum(pipe.model_pipe[2].coef_ != 0)
                results[cancer][split] = pipe.predict(X_test)
                surv = pipe.predict_survival_function(
                    X_test, np.unique(y_test["time"])
                )
                surv.to_csv(
                    f"../results/pc/efron/{cancer}/survival_function_{score}_{split+1}.csv",
                    index=False,
                )
            except ValueError as e:
                failures[cancer][0] += 1
                results[cancer][split] = np.zeros(test_ix.shape[0])
                sparsity[cancer][split] = 0

        pd.concat(
            [pd.DataFrame(results[cancer][i]) for i in range(25)], axis=1
        ).to_csv(f"../results/pc/efron/{cancer}/eta_{score}.csv", index=False)

    pd.DataFrame(sparsity).to_csv(
        f"../results/pc/efron/sparsity_{score}.csv", index=False
    )
    pd.DataFrame(failures).to_csv(
        f"../results/pc/efron/failures_{score}.csv", index=False
    )


# for score in ["linear_predictor", "mse", "basic", "vvh"]:
#     results = {}
#     failures = {}
#     sparsity = {}
#     pipe = PCSurv(
#         pc_pipe=make_pipeline(
#             VarianceThreshold(),
#             StandardScaler(),
#             PCA(n_components=config["pc_n_components"]),
#             CoxPHSurvivalAnalysis(ties="breslow"),
#         ),
#         model_pipe=make_pipeline(
#             VarianceThreshold(),
#             StandardScaler(),
#             PCPHElasticNetCV(
#                 tie_correction="breslow",
#                 l1_ratio=config["l1_ratio"],
#                 eps=config["eps"],
#                 n_alphas=config["n_alphas"],
#                 cv=config["n_inner_cv"],
#                 stratify_cv=config["stratify_cv"],
#                 seed=config["seed"],
#                 shuffle_cv=config["shuffle_cv"],
#                 cv_score_method=score,
#                 n_jobs=5,
#             ),
#         ),
#     )

#     for cancer in config["datasets"]:
#         print(f"Starting: {cancer}")
#         train_splits = pd.read_csv(
#             f"../data/splits/TCGA/{cancer}_train_splits.csv"
#         )
#         test_splits = pd.read_csv(
#             f"../data/splits/TCGA/{cancer}_test_splits.csv"
#         )
#         data = pd.read_csv(
#             f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
#         ).iloc[:, 1:]
#         X_ = data.iloc[:, 3:]
#         y_ = transform_survival(
#             time=data["OS_days"].values, event=data["OS"].values
#         )
#         for split in range(25):
#             print(f"Starting split: {split+1} / 25")
#             train_ix = (
#                 train_splits.iloc[split, :].dropna().to_numpy().astype(int)
#             )
#             test_ix = (
#                 test_splits.iloc[split, :].dropna().to_numpy().astype(int)
#             )
#             X_train = (
#                 X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
#             )
#             y_train = y_[train_ix].copy()
#             y_test = y_[test_ix].copy()
#             X_test = (
#                 X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()
#             )
#             if split == 0:
#                 results[cancer] = {}
#                 sparsity[cancer] = {}
#                 failures[cancer] = [0]
#             try:
#                 pipe.fit(X_train, y_train)
#                 sparsity[cancer][split] = np.sum(pipe.model_pipe[2].coef_ != 0)
#                 results[cancer][split] = pipe.predict(X_test)
#                 surv = pipe.predict_survival_function(
#                     X_test, np.unique(y_test["time"])
#                 )
#                 surv.to_csv(
#                     f"../results/pc/breslow/{cancer}/survival_function_{score}_{split+1}.csv",
#                     index=False,
#                 )
#             except ValueError as e:
#                 failures[cancer][0] += 1
#                 results[cancer][split] = np.zeros(test_ix.shape[0])
#                 sparsity[cancer][split] = 0

#         pd.concat(
#             [pd.DataFrame(results[cancer][i]) for i in range(25)], axis=1
#         ).to_csv(
#             f"../results/pc/breslow/{cancer}/eta_{score}.csv", index=False
#         )

#     pd.DataFrame(sparsity).to_csv(
#         f"../results/pc/breslow/sparsity_{score}.csv", index=False
#     )
#     pd.DataFrame(failures).to_csv(
#         f"../results/pc/breslow/failures_{score}.csv", index=False
#     )


for score in ["mse", "basic", "vvh"]:
    np.random.seed(config["random_state"])
    results = {}
    failures = {}
    sparsity = {}
    pipe = PCSurv(
        pc_pipe=make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCA(n_components=config["pc_n_components"]),
            AFT(),
        ),
        model_pipe=make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCAFTElasticNetCV(
                bandwidth=None,
                l1_ratio=config["l1_ratio"],
                eps=config["eps"],
                n_alphas=config["n_alphas"],
                cv=config["n_inner_cv"],
                stratify_cv=config["stratify_cv"],
                seed=config["seed"],
                shuffle_cv=config["shuffle_cv"],
                cv_score_method=score,
                n_jobs=5,
            ),
        ),
    )

    for cancer in config["datasets"]:
        print(f"Starting: {cancer}")
        train_splits = pd.read_csv(
            f"../data/splits/TCGA/{cancer}_train_splits.csv"
        )
        test_splits = pd.read_csv(
            f"../data/splits/TCGA/{cancer}_test_splits.csv"
        )
        data = pd.read_csv(
            f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
        ).iloc[:, 1:]
        X_ = data.iloc[:, 3:]
        y_ = transform_survival(
            time=data["OS_days"].values, event=data["OS"].values
        )
        for split in range(25):
            print(f"Starting split: {split+1} / 25")
            train_ix = (
                train_splits.iloc[split, :].dropna().to_numpy().astype(int)
            )
            test_ix = (
                test_splits.iloc[split, :].dropna().to_numpy().astype(int)
            )
            X_train = (
                X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
            )
            y_train = y_[train_ix].copy()
            y_test = y_[test_ix].copy()
            X_test = (
                X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()
            )
            if split == 0:
                results[cancer] = {}
                sparsity[cancer] = {}
                failures[cancer] = [0]
            try:
                pipe.fit(X_train, y_train)
                sparsity[cancer][split] = np.sum(pipe.model_pipe[2].coef_ != 0)
                results[cancer][split] = pipe.predict(X_test)
                surv = pipe.predict_survival_function(
                    X_test, np.unique(y_test["time"])
                )
                surv.to_csv(
                    f"../results/pc/aft/{cancer}/survival_function_{score}_{split+1}.csv",
                    index=False,
                )
            except ValueError as e:
                failures[cancer][0] += 1
                results[cancer][split] = np.zeros(test_ix.shape[0])
                sparsity[cancer][split] = 0

        pd.concat(
            [pd.DataFrame(results[cancer][i]) for i in range(25)], axis=1
        ).to_csv(f"../results/pc/aft/{cancer}/eta_{score}.csv", index=False)

    pd.DataFrame(sparsity).to_csv(
        f"../results/pc/aft/sparsity_{score}.csv", index=False
    )
    pd.DataFrame(failures).to_csv(
        f"../results/pc/aft/failures_{score}.csv", index=False
    )


for score in ["mse", "basic", "vvh"]:
    np.random.seed(config["random_state"])
    results = {}
    failures = {}
    sparsity = {}
    pipe = PCSurv(
        pc_pipe=make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCA(n_components=config["pc_n_components"]),
            EH(),
        ),
        model_pipe=make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCEHElasticNetCV(
                l1_ratio=config["l1_ratio"],
                eps=config["eps"],
                n_alphas=config["n_alphas"],
                cv=config["n_inner_cv"],
                stratify_cv=config["stratify_cv"],
                seed=config["seed"],
                shuffle_cv=config["shuffle_cv"],
                cv_score_method=score,
                n_jobs=5,
            ),
        ),
    )

    for cancer in config["datasets"]:
        print(f"Starting: {cancer}")
        train_splits = pd.read_csv(
            f"../data/splits/TCGA/{cancer}_train_splits.csv"
        )
        test_splits = pd.read_csv(
            f"../data/splits/TCGA/{cancer}_test_splits.csv"
        )
        data = pd.read_csv(
            f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
        ).iloc[:, 1:]
        X_ = data.iloc[:, 3:]
        y_ = transform_survival(
            time=data["OS_days"].values, event=data["OS"].values
        )
        for split in range(25):
            print(f"Starting split: {split+1} / 25")
            train_ix = (
                train_splits.iloc[split, :].dropna().to_numpy().astype(int)
            )
            test_ix = (
                test_splits.iloc[split, :].dropna().to_numpy().astype(int)
            )
            X_train = (
                X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
            )
            y_train = y_[train_ix].copy()
            y_test = y_[test_ix].copy()
            X_test = (
                X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()
            )
            if split == 0:
                results[cancer] = {}
                sparsity[cancer] = {}
                failures[cancer] = [0]
            try:
                pipe.fit(X_train, y_train)
                sparsity[cancer][split] = int(
                    np.sum(pipe.model_pipe[2].coef_ != 0) / 2
                )
                results[cancer][split] = pipe.predict(X_test)
                surv = pipe.predict_survival_function(
                    X_test, np.unique(y_test["time"])
                )
                surv.to_csv(
                    f"../results/pc/eh/{cancer}/survival_function_{score}_{split+1}.csv",
                    index=False,
                )
            except Exception as e:
                failures[cancer][0] += 1
                results[cancer][split] = np.zeros(test_ix.shape[0])
                sparsity[cancer][split] = 0
                time_km, survival_km = kaplan_meier_estimator(
                    y_train["event"].astype(bool), y_train["time"]
                )
                sf_df = pd.DataFrame(
                    [survival_km for i in range(X_test.shape[0])]
                )
                sf_df.columns = time_km
                sf_df.to_csv(
                    f"../results/pc/eh/{cancer}/survival_function_{score}_{split+1}.csv",
                    index=False,
                )

        pd.concat(
            [pd.DataFrame(results[cancer][i]) for i in range(25)], axis=1
        ).to_csv(f"../results/pc/eh/{cancer}/eta_{score}.csv", index=False)

    pd.DataFrame(sparsity).to_csv(
        f"../results/pc/eh/sparsity_{score}.csv", index=False
    )
    pd.DataFrame(failures).to_csv(
        f"../results/pc/eh/failures_{score}.csv", index=False
    )
