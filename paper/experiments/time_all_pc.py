import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from survhive._base import PCSurv
from survhive.aft import AFT
from survhive.cv import PCAFTElasticNetCV, PCEHElasticNetCV, PCPHElasticNetCV
from survhive.eh import EH
from survhive.utils import transform_survival

with open(f"../config.json") as f:
    config = json.load(f)

np.random.seed(config["random_state"])


efron_timing = {}
breslow_timing = {}
aft_timing = {}
eh_timing = {}

#for cancer in config["datasets"]:
for cancer in ["KIRC"]:
    efron_timing[cancer] = []
    print(f"Starting: {cancer}")
    train_splits = pd.read_csv(
        f"../data/splits/TCGA/{cancer}_train_splits.csv"
    )
    test_splits = pd.read_csv(f"../data/splits/TCGA/{cancer}_test_splits.csv")
    data = pd.read_csv(
        f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
    ).iloc[:, 1:]
    X_ = data.iloc[:, 3:].to_numpy()
    y_ = transform_survival(
        time=data["OS_days"].values, event=data["OS"].values
    )
    for rep in range(5):
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
                    seed=config["seed"] + rep,
                    shuffle_cv=config["shuffle_cv"],
                    cv_score_method="linear_predictor",
                    n_jobs=1,
                ),
            ),
        )
        start = timer()
        pipe.fit(X_, y_)
        end = timer()
        efron_timing[cancer].append(end - start)


#for cancer in config["datasets"]:
for cancer in ["KIRC"]:
    breslow_timing[cancer] = []
    print(f"Starting: {cancer}")
    train_splits = pd.read_csv(
        f"../data/splits/TCGA/{cancer}_train_splits.csv"
    )
    test_splits = pd.read_csv(f"../data/splits/TCGA/{cancer}_test_splits.csv")
    data = pd.read_csv(
        f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
    ).iloc[:, 1:]
    X_ = data.iloc[:, 3:].to_numpy()
    y_ = transform_survival(
        time=data["OS_days"].values, event=data["OS"].values
    )
    for rep in range(5):
        pipe = PCSurv(
            pc_pipe=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                PCA(n_components=config["pc_n_components"]),
                CoxPHSurvivalAnalysis(ties="breslow"),
            ),
            model_pipe=make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                PCPHElasticNetCV(
                    tie_correction="breslow",
                    l1_ratio=config["l1_ratio"],
                    eps=config["eps"],
                    n_alphas=config["n_alphas"],
                    cv=config["n_inner_cv"],
                    stratify_cv=config["stratify_cv"],
                    seed=config["seed"] + rep,
                    shuffle_cv=config["shuffle_cv"],
                    cv_score_method="linear_predictor",
                    n_jobs=1,
                ),
            ),
        )
        start = timer()
        pipe.fit(X_, y_)
        end = timer()
        breslow_timing[cancer].append(end - start)


#for cancer in config["datasets"]:
for cancer in ["KIRC"]:
    aft_timing[cancer] = []
    print(f"Starting: {cancer}")
    train_splits = pd.read_csv(
        f"../data/splits/TCGA/{cancer}_train_splits.csv"
    )
    test_splits = pd.read_csv(f"../data/splits/TCGA/{cancer}_test_splits.csv")
    data = pd.read_csv(
        f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
    ).iloc[:, 1:]
    X_ = data.iloc[:, 3:].to_numpy()
    y_ = transform_survival(
        time=data["OS_days"].values, event=data["OS"].values
    )
    for rep in range(5):
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
                    seed=config["seed"] + rep,
                    shuffle_cv=config["shuffle_cv"],
                    cv_score_method="linear_predictor",
                    n_jobs=1,
                ),
            ),
        )
        start = timer()
        pipe.fit(X_, y_)
        end = timer()
        aft_timing[cancer].append(end - start)

#for cancer in config["datasets"]:
for cancer in ["KIRC"]:
    eh_timing[cancer] = []
    print(f"Starting: {cancer}")
    train_splits = pd.read_csv(
        f"../data/splits/TCGA/{cancer}_train_splits.csv"
    )
    test_splits = pd.read_csv(f"../data/splits/TCGA/{cancer}_test_splits.csv")
    data = pd.read_csv(
        f"../data/processed/TCGA/{cancer}_data_preprocessed.csv"
    ).iloc[:, 1:]
    X_ = data.iloc[:, 3:].to_numpy()
    y_ = transform_survival(
        time=data["OS_days"].values, event=data["OS"].values
    )
    for rep in range(5):
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
                    bandwidth=None,
                    l1_ratio=config["l1_ratio"],
                    eps=config["eps"],
                    n_alphas=config["n_alphas"],
                    cv=config["n_inner_cv"],
                    stratify_cv=config["stratify_cv"],
                    seed=config["seed"] + rep,
                    shuffle_cv=config["shuffle_cv"],
                    cv_score_method="linear_predictor",
                    n_jobs=1,
                ),
            ),
        )
        start = timer()
        pipe.fit(X_, y_)
        end = timer()
        eh_timing[cancer].append(end - start)

pd.DataFrame(efron_timing).to_csv(
    "../results/pc/efron/timing.csv", index=False
)
pd.DataFrame(breslow_timing).to_csv(
    "../results/pc/breslow/timing.csv", index=False
)
pd.DataFrame(aft_timing).to_csv("../results/pc/aft/timing.csv", index=False)
pd.DataFrame(eh_timing).to_csv("../results/pc/eh/timing.csv", index=False)
