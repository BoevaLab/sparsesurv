import json

import celer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sparsesurv.cv import KDPHElasticNetCV
from sparsesurv.utils import transform_survival

with open("./config.json") as f:
    config = json.load(f)

np.random.seed(config["random_state"])


model_pipe = make_pipeline(
    VarianceThreshold(),
    StandardScaler(),
)
en = celer.ElasticNet(
    l1_ratio=config["l1_ratio"],
    fit_intercept=False,
)

pc_pipe = make_pipeline(
    VarianceThreshold(),
    StandardScaler(),
    PCA(n_components=config["pc_n_components"]),
    CoxPHSurvivalAnalysis(ties="efron"),
)

for cancer in config["datasets"]:
    sparsity = {}

    print(f"Starting: {cancer}")
    train_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_train_splits.csv")
    test_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_test_splits.csv")
    data = pd.read_csv(f"./data/processed/TCGA/{cancer}_data_preprocessed.csv").iloc[
        :, 1:
    ]
    X_ = data.iloc[:, 3:]
    y_ = transform_survival(time=data["OS_days"].values, event=data["OS"].values)
    for split in range(25):
        print(f"Starting split: {split+1} / 25")
        train_ix = train_splits.iloc[split, :].dropna().to_numpy().astype(int)
        test_ix = test_splits.iloc[split, :].dropna().to_numpy().astype(int)
        X_train = X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
        y_train = y_[train_ix].copy()
        y_test = y_[test_ix].copy()
        X_test = X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()

        pc_pipe.fit(X_train, y_train)
        path_results = en.path(
            X=model_pipe.fit_transform(X_train),
            y=pc_pipe.predict(X_train),
            l1_ratio=config["l1_ratio"],
            eps=config["eps"],
            n_alphas=100,
            alphas=None,
        )

        for z in range(100):
            path_coef = path_results[1][:, z]
            if z == 0:
                sparsity[split] = []
            sparsity[split].append(np.sum(path_coef != 0.0))
            helper = KDPHElasticNetCV(
                tie_correction="efron",
                seed=np.random.RandomState(config["random_state"]),
            )
            helper.coef_ = path_coef
            ix_sort = np.argsort(y_train["time"])
            helper.train_time_ = y_train["time"][ix_sort]
            helper.train_event_ = y_train["event"][ix_sort]
            helper.train_eta_ = helper.predict(model_pipe.transform(X_train))[ix_sort]
            surv = helper.predict_survival_function(
                model_pipe.transform(X_test), np.unique(y_test["time"])
            )
            surv.to_csv(
                f"./results/kd/efron/{cancer}/path/survival_function_{z+1}_alpha_{split+1}.csv",
                index=False,
            )

    pd.DataFrame(sparsity).to_csv(
        f"./results/kd/efron/{cancer}/path/sparsity.csv",
        index=False,
    )


pc_pipe = make_pipeline(
    VarianceThreshold(),
    StandardScaler(),
    PCA(n_components=config["pc_n_components"]),
    CoxPHSurvivalAnalysis(ties="breslow"),
)
for cancer in config["datasets"]:
    sparsity = {}

    print(f"Starting: {cancer}")
    train_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_train_splits.csv")
    test_splits = pd.read_csv(f"./data/splits/TCGA/{cancer}_test_splits.csv")
    data = pd.read_csv(f"./data/processed/TCGA/{cancer}_data_preprocessed.csv").iloc[
        :, 1:
    ]
    X_ = data.iloc[:, 3:]
    y_ = transform_survival(time=data["OS_days"].values, event=data["OS"].values)
    for split in range(25):
        print(f"Starting split: {split+1} / 25")
        train_ix = train_splits.iloc[split, :].dropna().to_numpy().astype(int)
        test_ix = test_splits.iloc[split, :].dropna().to_numpy().astype(int)
        X_train = X_.iloc[train_ix, :].copy().reset_index(drop=True).to_numpy()
        y_train = y_[train_ix].copy()
        y_test = y_[test_ix].copy()
        X_test = X_.iloc[test_ix, :].copy().reset_index(drop=True).to_numpy()

        pc_pipe.fit(X_train, y_train)
        path_results = en.path(
            X=model_pipe.fit_transform(X_train),
            y=pc_pipe.predict(X_train),
            l1_ratio=config["l1_ratio"],
            eps=config["eps"],
            n_alphas=100,
            alphas=None,
        )

        for z in range(100):
            path_coef = path_results[1][:, z]
            if z == 0:
                sparsity[split] = []
            sparsity[split].append(np.sum(path_coef != 0.0))
            helper = KDPHElasticNetCV(
                tie_correction="breslow",
                seed=np.random.RandomState(config["random_state"]),
            )
            helper.coef_ = path_coef
            ix_sort = np.argsort(y_train["time"])
            helper.train_time_ = y_train["time"][ix_sort]
            helper.train_event_ = y_train["event"][ix_sort]
            helper.train_eta_ = helper.predict(model_pipe.transform(X_train))[ix_sort]

            surv = helper.predict_survival_function(
                model_pipe.transform(X_test), np.unique(y_test["time"])
            )
            surv.to_csv(
                f"./results/kd/breslow/{cancer}/path/survival_function_{z+1}_alpha_{split+1}.csv",
                index=False,
            )

    pd.DataFrame(sparsity).to_csv(
        f"./results/kd/breslow/{cancer}/path/sparsity.csv",
        index=False,
    )
