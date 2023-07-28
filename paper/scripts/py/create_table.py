import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from survhive._base import PCSurvCV
from survhive.aft import AFT
from survhive.cv import (
    PCAFTElasticNetCV,
    PCEHMultiTaskLassoCV,
    PCPHElasticNetCV,
)
from survhive.eh import EH
from survhive.utils import transform_survival

with open(f"./config.json") as f:
    config = json.load(f)

np.random.seed(config["random_state"])

cancer_type = config["datasets"]
tissue = [
    "Bladder",
    "Breast",
    "Head and neck",
    "Kidney",
    "Brain",
    "Liver",
    "Lung",
    "Lung",
    "Ovaries",
    "Stomach",
]
full_name = [
    "Bladder Urothelial Carcinoma",
    "Breast invasive carcinoma",
    "Head and neck squamous cell carcinoma",
    "Kidney renal clear cell carcinoma",
    "Brain lower grade glioma",
    "Liver hepatocellular carcinoma",
    "Lung adenocarcinoma",
    "Lung squamous cell carcinoma",
    "Ovarian serous cystadenocarcinoma",
    "Stomach adenocarcinoma",
]
p = []
n = []
event_ratio = []
min_event_time = []
max_event_time = []
median_event_time = []


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
    p.append(X_.shape[1])
    n.append(X_.shape[0])
    event_ratio.append(np.mean(data["OS"].values))
    min_event_time.append(np.min(data["OS_days"].values))
    max_event_time.append(np.max(data["OS_days"].values))
    median_event_time.append(np.median(data["OS_days"].values))

pd.DataFrame(
    {
        "type": cancer_type,
        "tissue": tissue,
        "full_name": full_name,
        "p": p,
        "n": n,
        "event_ratio": event_ratio,
        "min_event_time": min_event_time,
        "max_event_time": max_event_time,
        "median_event_time": median_event_time,
    }
).to_csv("./results/metrics/dataset_overview.csv", index=False)
