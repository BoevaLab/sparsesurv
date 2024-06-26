import sys

with open(snakemake.log[0], "w") as f:
    sys.stderr = sys.stdout = f

    import json

    import numpy as np
    import pandas as pd

    with open(snakemake.params["config_path"]) as f:
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
        train_splits = pd.read_csv(f"results/make_splits/{cancer}_train_splits.csv")
        test_splits = pd.read_csv(f"results/make_splits/{cancer}_test_splits.csv")
        data = pd.read_csv(f"results/preprocess_data/{cancer}.csv").iloc[:, 1:]
        X_ = data.iloc[:, 3:]
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
    ).to_csv("results/tables/table_S1.csv", index=False)
