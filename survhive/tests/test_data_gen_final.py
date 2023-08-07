import numpy as np
import pandas as pd
import os
import torch

# Data generation functions
# TODO : Need to change this relative path, getcwd will differ. For ex my cwd is rootdir, so it throws FileNotFound Error
# use Path(__file__).parent.parent.parent OR os.path.dirname(os.path.abspath(__file__))
path = os.getcwd()


def numpy_test_data_1d(scenario="default"):
    if scenario == "default":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "first_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "last_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "high_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "low_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "all_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "no_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    linear_predictor = df.preds.to_numpy(dtype=np.float32)
    time = df.time.to_numpy(dtype=np.float32)  # .reshape(-1)
    event = df.event.to_numpy(dtype=np.float32)  # .reshape(-1)
    return linear_predictor, time, event


def numpy_test_data_2d(scenario="default"):
    if scenario == "default":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "first_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "last_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "high_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "low_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "all_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "no_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    pred_1d = df.preds.to_numpy(dtype=np.float32).reshape(25, 1)
    linear_predictor = np.hstack((pred_1d, pred_1d))
    time = df.time.to_numpy(dtype=np.float32)
    event = df.event.to_numpy(dtype=np.float32)
    return linear_predictor, time, event


def torch_test_data_1d(scenario="default"):
    if scenario == "default":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "first_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "last_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "high_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "low_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "all_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "no_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    linear_predictor = df.preds.to_numpy(dtype=np.float32)
    time = df.time.to_numpy(dtype=np.float32)  # .reshape(-1)
    event = df.event.to_numpy(dtype=np.float32)  # .reshape(-1)
    return (
        torch.from_numpy(linear_predictor),
        torch.from_numpy(time),
        torch.from_numpy(event),
    )


def torch_test_data_2d(scenario="default"):
    if scenario == "default":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "first_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "last_five_zero":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "high_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "low_event_ratio":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "all_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    elif scenario == "no_events":
        df = pd.read_csv(
            path + "/test_data/survival_simulation_25_" + scenario + ".csv"
        )
    pred_1d = df.preds.to_numpy(dtype=np.float32).reshape(25, 1)
    linear_predictor = np.hstack((pred_1d, pred_1d))
    time = df.time.to_numpy(dtype=np.float32)  # .reshape(-1)
    event = df.event.to_numpy(dtype=np.float32)  # .reshape(-1)
    return (
        torch.from_numpy(linear_predictor),
        torch.from_numpy(time),
        torch.from_numpy(event),
    )
