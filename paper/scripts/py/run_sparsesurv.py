import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sparsesurv._base import KDSurv
from sparsesurv.cv import KDPHElasticNetCV
from sparsesurv.utils import transform_survival

with open("./config.json") as f:
    config = json.load(f)

np.random.seed(config["random_state"])

for score_type in ["min", "pcvl"]:
    for score in ["linear_predictor"]:
        results = {}
        failures = {}
        sparsity = {}
        pipe = KDSurv(
            teacher=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                PCA(n_components=config["pc_n_components"]),
                CoxPHSurvivalAnalysis(ties="efron"),
            ),
            student=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                KDPHElasticNetCV(
                    tie_correction="efron",
                    l1_ratio=config["l1_ratio"],
                    eps=config["eps"],
                    n_alphas=config["n_alphas"],
                    cv=config["n_inner_cv"],
                    stratify_cv=config["stratify_cv"],
                    seed=np.random.RandomState(config["random_state"]),
                    shuffle_cv=config["shuffle_cv"],
                    cv_score_method=score,
                    n_jobs=5,
                    alpha_type=score_type,
                ),
            ),
        )

        for cancer in config["datasets"]:
            print(f"Starting: {cancer}")
            train_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_train_splits.csv")
            test_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_test_splits.csv")
            data = pd.read_csv(
                f"./data/processed/TCGA/{cancer}_data_preprocessed.csv"
            ).iloc[:, 1:]
            X_ = data.iloc[:, 3:]
            y_ = transform_survival(
                time=data["OS_days"].values, event=data["OS"].values
            )
            for split in range(25):
                print(f"Starting split: {split+1} / 25")
                train_ix = train_splits.iloc[split, :].dropna().to_numpy().astype(int)
                test_ix = test_splits.iloc[split, :].dropna().to_numpy().astype(int)
                X_train = X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
                y_train = y_[train_ix].copy()
                y_test = y_[test_ix].copy()
                X_test = X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()
                if split == 0:
                    results[cancer] = {}
                    sparsity[cancer] = {}
                    failures[cancer] = [0]
                try:
                    pipe.fit(X_train, y_train)
                    print(pipe.student)
                    sparsity[cancer][split] = np.sum(pipe.student[-1].coef_ != 0)
                    results[cancer][split] = pipe.predict(X_test)
                    surv = pipe.predict_survival_function(
                        X_test, np.unique(y_test["time"])
                    )
                    surv.to_csv(
                        f"./results/kd/efron/{cancer}/survival_function_{score}_{score_type}_{split+1}.csv",
                        index=False,
                    )
                except ValueError as e:
                    failures[cancer][0] += 1
                    results[cancer][split] = np.zeros(test_ix.shape[0])
                    sparsity[cancer][split] = 0

            pd.concat(
                [pd.DataFrame(results[cancer][i]) for i in range(25)], axis=1
            ).to_csv(
                f"./results/kd/efron/{cancer}/eta_{score}_{score_type}.csv",
                index=False,
            )

        pd.DataFrame(sparsity).to_csv(
            f"./results/kd/efron/sparsity_{score}_{score_type}.csv",
            index=False,
        )
        pd.DataFrame(failures).to_csv(
            f"./results/kd/efron/failures_{score}_{score_type}.csv",
            index=False,
        )

    for score in ["linear_predictor"]:
        results = {}
        failures = {}
        sparsity = {}
        pipe = KDSurv(
            teacher=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                PCA(n_components=config["pc_n_components"]),
                CoxPHSurvivalAnalysis(ties="breslow"),
            ),
            student=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                KDPHElasticNetCV(
                    tie_correction="breslow",
                    l1_ratio=config["l1_ratio"],
                    eps=config["eps"],
                    n_alphas=config["n_alphas"],
                    cv=config["n_inner_cv"],
                    stratify_cv=config["stratify_cv"],
                    seed=np.random.RandomState(config["random_state"]),
                    shuffle_cv=config["shuffle_cv"],
                    cv_score_method=score,
                    n_jobs=5,
                    alpha_type=score_type,
                ),
            ),
        )

        for cancer in config["datasets"]:
            print(f"Starting: {cancer}")
            train_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_train_splits.csv")
            test_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_test_splits.csv")
            data = pd.read_csv(
                f"./data/processed/TCGA/{cancer}_data_preprocessed.csv"
            ).iloc[:, 1:]
            X_ = data.iloc[:, 3:]
            y_ = transform_survival(
                time=data["OS_days"].values, event=data["OS"].values
            )
            for split in range(25):
                print(f"Starting split: {split+1} / 25")
                train_ix = train_splits.iloc[split, :].dropna().to_numpy().astype(int)
                test_ix = test_splits.iloc[split, :].dropna().to_numpy().astype(int)
                X_train = X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
                y_train = y_[train_ix].copy()
                y_test = y_[test_ix].copy()
                X_test = X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()
                if split == 0:
                    results[cancer] = {}
                    sparsity[cancer] = {}
                    failures[cancer] = [0]
                try:
                    pipe.fit(X_train, y_train)
                    sparsity[cancer][split] = np.sum(pipe.student[-1].coef_ != 0)
                    results[cancer][split] = pipe.predict(X_test)
                    surv = pipe.predict_survival_function(
                        X_test, np.unique(y_test["time"])
                    )
                    surv.to_csv(
                        f"./results/kd/breslow/{cancer}/survival_function_{score}_{score_type}_{split+1}.csv",
                        index=False,
                    )
                except ValueError as e:
                    failures[cancer][0] += 1
                    results[cancer][split] = np.zeros(test_ix.shape[0])
                    sparsity[cancer][split] = 0

            pd.concat(
                [pd.DataFrame(results[cancer][i]) for i in range(25)], axis=1
            ).to_csv(
                f"./results/kd/breslow/{cancer}/eta_{score}_{score_type}.csv",
                index=False,
            )

        pd.DataFrame(sparsity).to_csv(
            f"./results/kd/breslow/sparsity_{score}_{score_type}.csv",
            index=False,
        )
        pd.DataFrame(failures).to_csv(
            f"./results/kd/breslow/failures_{score}_{score_type}.csv",
            index=False,
        )
