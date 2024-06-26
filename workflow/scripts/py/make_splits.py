import sys


def main(
    data_path,
    random_seed,
    n_outer_repetitions,
    n_outer_splits,
    train_splits_output_path,
    test_splits_output_path,
) -> int:

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import RepeatedStratifiedKFold

    np.random.seed(random_seed)
    data = pd.read_csv(
        data_path,
        low_memory=False,
    )

    # Exact column choice doesn't matter
    # as this is only to create the splits anyway.
    X = data[[i for i in data.columns if i not in ["OS_days", "OS"]]]
    cv = RepeatedStratifiedKFold(
        n_repeats=n_outer_repetitions, n_splits=n_outer_splits, random_state=random_seed
    )
    splits = [i for i in cv.split(X, data["OS"])]
    pd.DataFrame([i[0] for i in splits]).to_csv(
        train_splits_output_path,
        index=False,
    )
    pd.DataFrame([i[1] for i in splits]).to_csv(
        test_splits_output_path,
        index=False,
    )
    return 0


with open(snakemake.log[0], "w") as f:
    sys.stderr = sys.stdout = f
    main(
        data_path=snakemake.input["data_path"],
        random_seed=snakemake.params["random_seed"],
        n_outer_repetitions=snakemake.params["n_outer_repetitions"],
        n_outer_splits=snakemake.params["n_outer_splits"],
        train_splits_output_path=snakemake.output["train_splits_output_path"],
        test_splits_output_path=snakemake.output["test_splits_output_path"],
    )
