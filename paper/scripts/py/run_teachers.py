import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sparsesurv.utils import transform_survival

with open(f"./config.json") as f:
    config = json.load(f)

np.random.seed(config["random_state"])


teacher_efron = make_pipeline(
    VarianceThreshold(),
    StandardScaler(),
    PCA(n_components=config["pc_n_components"]),
    CoxPHSurvivalAnalysis(ties="efron"),
)

for cancer in config["datasets"]:
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

        teacher_efron.fit(X_train, y_train)
        (
            cumulative_baseline_hazards_times,
            cumulative_baseline_hazards,
        ) = (
            teacher_efron[3].cum_baseline_hazard_.x,
            teacher_efron[3].cum_baseline_hazard_.y,
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
                A=teacher_efron.predict(X_test),
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
            f"./results/kd/efron/{cancer}/survival_function_teacher_{split+1}.csv",
            index=False,
        )


teacher_breslow = make_pipeline(
    VarianceThreshold(),
    StandardScaler(),
    PCA(n_components=config["pc_n_components"]),
    CoxPHSurvivalAnalysis(ties="efron"),
)

for cancer in config["datasets"]:
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

        teacher_breslow.fit(X_train, y_train)
        (
            cumulative_baseline_hazards_times,
            cumulative_baseline_hazards,
        ) = (
            teacher_breslow[3].cum_baseline_hazard_.x,
            teacher_breslow[3].cum_baseline_hazard_.y,
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
                A=teacher_breslow.predict(X_test),
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
            f"./results/kd/breslow/{cancer}/survival_function_teacher_{split+1}.csv",
            index=False,
        )
