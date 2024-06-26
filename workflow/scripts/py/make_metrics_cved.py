import sys

with open(snakemake.log[0], "w") as f:
    sys.stderr = sys.stdout = f

    import json

    import numpy as np
    import pandas as pd
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
    from sksurv.util import Surv

    with open(snakemake.params["config_path"]) as f:
        config = json.load(f)
    np.random.seed(config["random_state"])
    sksurv_converter = Surv()
    transform_survival = sksurv_converter.from_arrays

    model = []
    pc = []
    score = []
    metric = []
    value = []
    split = []
    cancer_val = []
    lambda_val = []
    variables = []
    tuned = []

    for cancer in config["datasets"]:
        print(f"Starting: {cancer}")
        df = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[:, 1:]
        time = df["OS_days"].values
        event = df["OS"].values
        test_splits = pd.read_csv(f"results/make_splits/{cancer}_test_splits.csv")
        train_splits = pd.read_csv(f"results/make_splits/{cancer}_train_splits.csv")
        for n_variables in [""]:
            if n_variables == "":
                n_variables_string = 0
            elif n_variables == "_10":
                n_variables_string = 10
            else:
                n_variables_string = 50

            if n_variables_string in [10, 50]:
                model_list = ["cox_nnet", "breslow", "boosting"]
            else:
                model_list = ["cox_nnet", "breslow"]
            for lambda_type in ["min"]:
                if lambda_type == "min":
                    model_list = ["breslow"]
                else:
                    model_list = ["breslow"]
                for score_function in ["linear_predictor"]:
                    for model_type in model_list:
                        lp = pd.read_csv(
                            f"results/kd/{model_type}/{cancer}/eta_{score_function}_{lambda_type+n_variables}.csv"
                        )
                        for i in range(125):
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
                            try:
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
                                        np.partition(
                                            time[train_split][event[train_split]], -3
                                        )[-3]
                                        - 1e-8,
                                        1e-8,
                                    )[0]
                                )
                            except ValueError:
                                value.append(0.5)
                            model = model + [model_type for q in range(2)]
                            pc = pc + [True for q in range(2)]
                            metric = metric + ["Harrell's C", "Uno's C"]
                            score = score + [score_function for q in range(2)]
                            split = split + [i for q in range(2)]
                            cancer_val = cancer_val + [cancer for q in range(2)]
                            lambda_val = lambda_val + [lambda_type for q in range(2)]
                            variables = variables + [
                                n_variables_string for q in range(2)
                            ]
                            tuned = tuned + [False for q in range(2)]

            for lambda_type in ["lambda.min"]:
                for score_function in ["vvh"]:
                    for model_type in ["breslow"]:
                        lp = pd.read_csv(
                            f"results/non_kd/{model_type}/{cancer}/eta_tuned_l1_ratio_{score_function}_{lambda_type+n_variables}.csv"
                        ).iloc[:, 1:]
                        for i in range(125):
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
                            try:
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
                                        np.partition(
                                            time[train_split][event[train_split]],
                                            -3,
                                        )[-3]
                                        - 1e-8,
                                        1e-8,
                                    )[0]
                                )
                            except:
                                value.append(0.5)
                            model = model + [model_type for q in range(2)]
                            pc = pc + [False for q in range(2)]
                            metric = metric + ["Harrell's C", "Uno's C"]
                            score = score + [score_function for q in range(2)]
                            split = split + [i for q in range(2)]
                            cancer_val = cancer_val + [cancer for q in range(2)]
                            lambda_val = lambda_val + [lambda_type for q in range(2)]
                            variables = variables + [
                                n_variables_string for q in range(2)
                            ]
                            tuned = tuned + [True for q in range(2)]

            for lambda_type in ["lambda.min"]:
                for score_function in ["vvh"]:
                    for model_type in ["breslow"]:
                        lp = pd.read_csv(
                            f"results/non_kd/{model_type}/{cancer}/eta_{score_function}_{lambda_type+n_variables}.csv"
                        ).iloc[:, 1:]
                        for i in range(125):
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
                            try:
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
                                        np.partition(
                                            time[train_split][event[train_split]],
                                            -3,
                                        )[-3]
                                        - 1e-8,
                                        1e-8,
                                    )[0]
                                )
                            except:
                                value.append(0.5)
                            model = model + [model_type for q in range(2)]
                            pc = pc + [False for q in range(2)]
                            metric = metric + ["Harrell's C", "Uno's C"]
                            score = score + [score_function for q in range(2)]
                            split = split + [i for q in range(2)]
                            cancer_val = cancer_val + [cancer for q in range(2)]
                            lambda_val = lambda_val + [lambda_type for q in range(2)]
                            variables = variables + [
                                n_variables_string for q in range(2)
                            ]
                            tuned = tuned + [False for q in range(2)]

    pd.DataFrame(
        {
            "value": value,
            "model": model,
            "kd": pc,
            "metric": metric,
            "score": score,
            "split": split,
            "cancer": cancer_val,
            "lambda": lambda_val,
            "n_variables": variables,
            "tuned": tuned,
        }
    ).to_csv("results/metrics/metrics_overall_125_full.csv", index=False)
    model = []
    pc = []
    score = []
    metric = []
    value = []
    split = []
    cancer_val = []
    lambda_val = []
    variables = []
    tuned = []

    for cancer in config["datasets"]:
        print(f"Starting: {cancer}")
        df = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[:, 1:]
        time = df["OS_days"].values
        event = df["OS"].values
        test_splits = pd.read_csv(f"results/make_splits/{cancer}_test_splits.csv")
        train_splits = pd.read_csv(f"results/make_splits/{cancer}_train_splits.csv")
        for n_variables in [""]:
            if n_variables == "":
                n_variables_string = 0
            elif n_variables == "_10":
                n_variables_string = 10
            else:
                n_variables_string = 50

            if n_variables_string in [10, 50]:
                model_list = ["cox_nnet", "breslow", "boosting"]
            else:
                model_list = ["cox_nnet", "breslow"]
            for lambda_type in ["min"]:
                if lambda_type == "min":
                    model_list = ["breslow"]
                else:
                    model_list = ["breslow"]
                for score_function in ["linear_predictor"]:
                    for model_type in model_list:
                        lp = pd.read_csv(
                            f"results/kd/{model_type}/{cancer}/eta_{score_function}_{lambda_type+n_variables}.csv"
                        )
                        for i in range(0, 125, 5):
                            lp_split = np.concatenate(
                                [
                                    (
                                        lp.iloc[:, q].dropna().values
                                        - np.mean(lp.iloc[:, q].dropna().values)
                                    )
                                    for q in range(i, i + 5)
                                ]
                            )
                            test_split = np.concatenate(
                                [
                                    test_splits.iloc[q, :].dropna().values.astype(int)
                                    for q in range(i, i + 5)
                                ]
                            )

                            train_split = np.concatenate(
                                [
                                    train_splits.iloc[q, :].dropna().values.astype(int)
                                    for q in range(i, i + 5)
                                ]
                            )
                            value.append(
                                concordance_index_censored(
                                    event[test_split].astype(bool),
                                    time[test_split],
                                    lp_split,
                                    1e-8,
                                )[0]
                            )
                            try:
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
                                        np.partition(
                                            time[train_split][event[train_split]], -3
                                        )[-3]
                                        - 1e-8,
                                        1e-8,
                                    )[0]
                                )
                            except ValueError:
                                value.append(0.5)
                            model = model + [model_type for q in range(2)]
                            pc = pc + [True for q in range(2)]
                            metric = metric + ["Harrell's C", "Uno's C"]
                            score = score + [score_function for q in range(2)]
                            split = split + [i for q in range(2)]
                            cancer_val = cancer_val + [cancer for q in range(2)]
                            lambda_val = lambda_val + [lambda_type for q in range(2)]
                            variables = variables + [
                                n_variables_string for q in range(2)
                            ]
                            tuned = tuned + [False for q in range(2)]

            for lambda_type in ["lambda.min"]:
                for score_function in ["vvh"]:
                    for model_type in ["breslow"]:
                        lp = pd.read_csv(
                            f"results/non_kd/{model_type}/{cancer}/eta_tuned_l1_ratio_{score_function}_{lambda_type+n_variables}.csv"
                        ).iloc[:, 1:]
                        for i in range(0, 125, 5):
                            lp_split = np.concatenate(
                                [
                                    (
                                        lp.iloc[:, q].dropna().values
                                        - np.mean(lp.iloc[:, q].dropna().values)
                                    )
                                    / (
                                        1
                                        if np.std(lp.iloc[:, q].dropna().values) == 0
                                        else np.std(lp.iloc[:, q].dropna().values)
                                    )
                                    for q in range(i, i + 5)
                                ]
                            )
                            test_split = np.concatenate(
                                [
                                    test_splits.iloc[q, :].dropna().values.astype(int)
                                    for q in range(i, i + 5)
                                ]
                            )
                            train_split = np.concatenate(
                                [
                                    train_splits.iloc[q, :].dropna().values.astype(int)
                                    for q in range(i, i + 5)
                                ]
                            )
                            value.append(
                                concordance_index_censored(
                                    event[test_split].astype(bool),
                                    time[test_split],
                                    lp_split,
                                    1e-8,
                                )[0]
                            )
                            try:
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
                                        np.partition(
                                            time[train_split][event[train_split]],
                                            -3,
                                        )[-3]
                                        - 1e-8,
                                        1e-8,
                                    )[0]
                                )
                            except:
                                value.append(0.5)
                            model = model + [model_type for q in range(2)]
                            pc = pc + [False for q in range(2)]
                            metric = metric + ["Harrell's C", "Uno's C"]
                            score = score + [score_function for q in range(2)]
                            split = split + [i for q in range(2)]
                            cancer_val = cancer_val + [cancer for q in range(2)]
                            lambda_val = lambda_val + [lambda_type for q in range(2)]
                            variables = variables + [
                                n_variables_string for q in range(2)
                            ]
                            tuned = tuned + [True for q in range(2)]

            for lambda_type in ["lambda.min"]:
                for score_function in ["vvh"]:
                    for model_type in ["breslow"]:
                        lp = pd.read_csv(
                            f"results/non_kd/{model_type}/{cancer}/eta_{score_function}_{lambda_type+n_variables}.csv"
                        ).iloc[:, 1:]
                        for i in range(0, 125, 5):
                            lp_split = np.concatenate(
                                [
                                    (
                                        lp.iloc[:, q].dropna().values
                                        - np.mean(lp.iloc[:, q].dropna().values)
                                    )
                                    / (
                                        1
                                        if np.std(lp.iloc[:, q].dropna().values) == 0
                                        else np.std(lp.iloc[:, q].dropna().values)
                                    )
                                    for q in range(i, i + 5)
                                ]
                            )
                            test_split = np.concatenate(
                                [
                                    test_splits.iloc[q, :].dropna().values.astype(int)
                                    for q in range(i, i + 5)
                                ]
                            )
                            train_split = np.concatenate(
                                [
                                    train_splits.iloc[q, :].dropna().values.astype(int)
                                    for q in range(i, i + 5)
                                ]
                            )
                            value.append(
                                concordance_index_censored(
                                    event[test_split].astype(bool),
                                    time[test_split],
                                    lp_split,
                                    1e-8,
                                )[0]
                            )
                            try:
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
                                        np.partition(
                                            time[train_split][event[train_split]],
                                            -3,
                                        )[-3]
                                        - 1e-8,
                                        1e-8,
                                    )[0]
                                )
                            except:
                                value.append(0.5)
                            model = model + [model_type for q in range(2)]
                            pc = pc + [False for q in range(2)]
                            metric = metric + ["Harrell's C", "Uno's C"]
                            score = score + [score_function for q in range(2)]
                            split = split + [i for q in range(2)]
                            cancer_val = cancer_val + [cancer for q in range(2)]
                            lambda_val = lambda_val + [lambda_type for q in range(2)]
                            variables = variables + [
                                n_variables_string for q in range(2)
                            ]
                            tuned = tuned + [False for q in range(2)]

    pd.DataFrame(
        {
            "value": value,
            "model": model,
            "kd": pc,
            "metric": metric,
            "score": score,
            "split": split,
            "cancer": cancer_val,
            "lambda": lambda_val,
            "n_variables": variables,
            "tuned": tuned,
        }
    ).to_csv("results/metrics/metrics_overall_cved.csv", index=False)
