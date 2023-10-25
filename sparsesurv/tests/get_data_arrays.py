from pathlib import Path

import numpy as np
import pandas as pd

# use Path(__file__).parent.parent.parent for PROJECT_ROOT_DIR (OR os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = Path(__file__).parent


def get_1d_array(case="default", dims=1):
    file_path = TEST_DIR / "test_data" / f"survival_simulation_25_{case}.csv"
    df = pd.read_csv(file_path)

    linear_predictor = df.preds.to_numpy(dtype=np.float32)
    time = df.time.to_numpy(dtype=np.float32)
    event = df.event.to_numpy(dtype=np.float32)
    return linear_predictor, time, event


def get_2d_array(case="default"):
    file_path = TEST_DIR / "test_data" / f"survival_simulation_25_{case}.csv"
    df = pd.read_csv(file_path)

    pred_1d = df.preds.to_numpy(dtype=np.float32)[:, None]
    linear_predictor = np.hstack((pred_1d, pred_1d))
    time = df.time.to_numpy(dtype=np.float32)
    event = df.event.to_numpy(dtype=np.float32)
    return linear_predictor, time, event
