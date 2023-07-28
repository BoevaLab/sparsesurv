import json
import os

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

# from survhive.utils import transform_survival
from sksurv.util import Surv


def main() -> int:
    with open("./config.json") as f:
        config = json.load(f)
    np.random.seed(config["random_state"])
    sksurv_converter = Surv()
    transform_survival = sksurv_converter.from_arrays

    splits_path = os.path.join(".", "data", "splits", "TCGA")
    data_path = os.path.join(".", "data", "processed", "TCGA")
    os.makedirs(splits_path, exist_ok=True)
    model = []
    pc = []
    score = []
    metric = []
    value = []
    split = []
    cancer_val = []
    lambda_val = []

    # LP eval PC
    for cancer in config["datasets"]:
        data_path = f"./data/processed/TCGA/{cancer}_data_preprocessed.csv"
        df = pd.read_csv(data_path)
        time = df["OS_days"].values
        event = df["OS"].values
        test_splits = pd.read_csv(
            f"./data/splits/TCGA/{cancer}_test_splits.csv"
        )
        train_splits = pd.read_csv(
            f"./data/splits/TCGA/{cancer}_train_splits.csv"
        )
        for lambda_type in ["min", "pcvl"]:
            for score_function in ["linear_predictor"]:
                for model_type in ["efron", "breslow", "aft"]:
                    lp = pd.read_csv(
                        f"./results/pc/{model_type}/{cancer}/eta_{score_function}_{lambda_type}.csv"
                    )
                    for i in range(25):
                        lp_split = lp.iloc[:, i].dropna().values
                        test_split = (
                            test_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        train_split = (
                            train_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        value.append(
                            concordance_index_censored(
                                event[test_split].astype(bool),
                                time[test_split],
                                lp_split,
                                1e-8,
                            )[0]
                        )
                        value.append(
                            concordance_index_ipcw(
                                transform_survival(
                                    event[train_split].astype(bool),
                                    time[train_split],
                                ),
                                transform_survival(
                                    event[test_split].astype(bool),
                                    time[test_split],
                                ),
                                lp_split,
                                np.max(time[train_split][event[train_split]])
                                - 1e-8,
                                1e-8,
                            )[0]
                        )
                        model = model + [model_type for q in range(2)]
                        pc = pc + [True for q in range(2)]
                        metric = metric + ["Harrell's C", "Uno's C"]
                        score = score + [score_function for q in range(2)]
                        split = split + [i for q in range(2)]
                        cancer_val = cancer_val + [cancer for q in range(2)]
                        lambda_val = lambda_val + [
                            lambda_type for q in range(2)
                        ]
        for lambda_type in ["lambda.min", "lambda.pcvl"]:
            for score_function in ["vvh"]:
                for model_type in ["breslow"]:
                    lp = pd.read_csv(
                        f"./results/non_pc/{model_type}/{cancer}/eta_{score_function}_{lambda_type}.csv"
                    ).iloc[:, 1:]
                    for i in range(25):
                        lp_split = lp.iloc[:, i].dropna().values
                        test_split = (
                            test_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        train_split = (
                            train_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        value.append(
                            concordance_index_censored(
                                event[test_split].astype(bool),
                                time[test_split],
                                lp_split,
                                1e-8,
                            )[0]
                        )
                        value.append(
                            concordance_index_ipcw(
                                transform_survival(
                                    event[train_split].astype(bool),
                                    time[train_split],
                                ),
                                transform_survival(
                                    event[test_split].astype(bool),
                                    time[test_split],
                                ),
                                lp_split,
                                np.max(time[train_split][event[train_split]])
                                - 1e-8,
                                1e-8,
                            )[0]
                        )
                        model = model + [model_type for q in range(2)]
                        pc = pc + [False for q in range(2)]
                        metric = metric + ["Harrell's C", "Uno's C"]
                        score = score + [score_function for q in range(2)]
                        split = split + [i for q in range(2)]
                        cancer_val = cancer_val + [cancer for q in range(2)]
                        lambda_val = lambda_val + [
                            lambda_type for q in range(2)
                        ]
        for lambda_type in ["min", "pcvl"]:
            for score_function in ["linear_predictor"]:
                for model_type in ["efron", "breslow"]:
                    for i in range(25):

                        surv = pd.read_csv(
                            f"./results/pc/{model_type}/{cancer}/survival_function_{score_function}_{lambda_type}_{i+1}.csv"
                        ).T
                        surv.index = surv.index.astype(float)
                        test_split = (
                            test_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        train_split = (
                            train_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        ev = EvalSurv(
                            surv,
                            time[test_split],
                            event[test_split],
                            censor_surv="km",
                        )
                        value.append(ev.concordance_td())
                        time_grid = np.linspace(
                            time[test_split].min(), time[test_split].max(), 100
                        )
                        value.append(ev.integrated_brier_score(time_grid))

                        model = model + [model_type for q in range(2)]
                        pc = pc + [True for q in range(2)]
                        metric = metric + ["Antolini's C", "IBS"]
                        score = score + [score_function for q in range(2)]
                        split = split + [i for q in range(2)]
                        cancer_val = cancer_val + [cancer for q in range(2)]
                        lambda_val = lambda_val + [
                            lambda_type for q in range(2)
                        ]

        for model_type in ["efron", "breslow"]:
            for i in range(25):

                surv = pd.read_csv(
                    f"./results/pc/{model_type}/{cancer}/survival_function_teacher_{i+1}.csv"
                ).T
                surv.index = surv.index.astype(float)
                test_split = test_splits.iloc[i, :].dropna().values.astype(int)
                train_split = (
                    train_splits.iloc[i, :].dropna().values.astype(int)
                )
                ev = EvalSurv(
                    surv,
                    time[test_split],
                    event[test_split],
                    censor_surv="km",
                )
                value.append(ev.concordance_td())
                time_grid = np.linspace(
                    time[test_split].min(), time[test_split].max(), 100
                )
                value.append(ev.integrated_brier_score(time_grid))

                model = model + [model_type for q in range(2)]
                pc = pc + [False for q in range(2)]
                metric = metric + ["Antolini's C", "IBS"]
                score = score + ["teacher" for q in range(2)]
                split = split + [i for q in range(2)]
                cancer_val = cancer_val + [cancer for q in range(2)]
                lambda_val = lambda_val + ["teacher" for q in range(2)]
        for lambda_type in ["lambda.min", "lambda.pcvl"]:
            for score_function in ["vvh"]:
                for model_type in ["breslow"]:
                    for i in range(25):

                        surv = pd.read_csv(
                            f"./results/non_pc/{model_type}/{cancer}/survival_function_{score_function}_{lambda_type}_{i+1}.csv"
                        ).T
                        surv.index = surv.index.astype(float)
                        test_split = (
                            test_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        train_split = (
                            train_splits.iloc[i, :].dropna().values.astype(int)
                        )
                        ev = EvalSurv(
                            surv,
                            time[test_split],
                            event[test_split],
                            censor_surv="km",
                        )
                        value.append(ev.concordance_td())
                        time_grid = np.linspace(
                            time[test_split].min(), time[test_split].max(), 100
                        )
                        value.append(ev.integrated_brier_score(time_grid))

                        model = model + [model_type for q in range(2)]
                        pc = pc + [False for q in range(2)]
                        metric = metric + ["Antolini's C", "IBS"]
                        score = score + [score_function for q in range(2)]
                        split = split + [i for q in range(2)]
                        cancer_val = cancer_val + [cancer for q in range(2)]
                        lambda_val = lambda_val + [
                            lambda_type for q in range(2)
                        ]

        for path_num in range(100):
            for model_type in ["efron", "breslow"]:
                for i in range(25):
                    surv = pd.read_csv(
                        f"./results/pc/{model_type}/{cancer}/path/survival_function_{path_num+1}_alpha_{i+1}.csv"
                    ).T
                    surv.index = surv.index.astype(float)
                    test_split = (
                        test_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    train_split = (
                        train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    ev = EvalSurv(
                        surv,
                        time[test_split],
                        event[test_split],
                        censor_surv="km",
                    )
                    value.append(ev.concordance_td())
                    time_grid = np.linspace(
                        time[test_split].min(), time[test_split].max(), 100
                    )
                    value.append(ev.integrated_brier_score(time_grid))

                    model = model + [model_type for q in range(2)]
                    pc = pc + [True for q in range(2)]
                    metric = metric + ["Antolini's C", "IBS"]
                    score = score + ["path" for q in range(2)]
                    split = split + [i for q in range(2)]
                    cancer_val = cancer_val + [cancer for q in range(2)]
                    lambda_val = lambda_val + [path_num for q in range(2)]
        for path_num in range(100):
            for model_type in ["breslow"]:
                for i in range(25):

                    surv = pd.read_csv(
                        f"./results/non_pc/{model_type}/{cancer}/path/survival_function_{path_num+1}_alpha_{i+1}.csv"
                    ).T
                    surv.index = surv.index.astype(float)
                    test_split = (
                        test_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    train_split = (
                        train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    ev = EvalSurv(
                        surv,
                        time[test_split],
                        event[test_split],
                        censor_surv="km",
                    )
                    value.append(ev.concordance_td())
                    time_grid = np.linspace(
                        time[test_split].min(), time[test_split].max(), 100
                    )
                    value.append(ev.integrated_brier_score(time_grid))

                    model = model + [model_type for q in range(2)]
                    pc = pc + [False for q in range(2)]
                    metric = metric + ["Antolini's C", "IBS"]
                    score = score + ["path" for q in range(2)]
                    split = split + [i for q in range(2)]
                    cancer_val = cancer_val + [cancer for q in range(2)]
                    lambda_val = lambda_val + [path_num for q in range(2)]
    pd.DataFrame(
        {
            "value": value,
            "model": model,
            "pc": pc,
            "metric": metric,
            "score": score,
            "split": split,
            "cancer": cancer_val,
            "lambda": lambda_val,
        }
    ).to_csv("./results/metrics/metrics_overall.csv", index=False)
    return 0


if __name__ == "__main__":
    main()
