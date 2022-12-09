import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def inverse_transform_survival_target(
    y: np.array,
) -> tuple[np.array, np.array]:
    event = y >= 0
    event = event.flatten()
    time = np.abs(y)
    return time, event


# Utility function to calculate kernel input, coressponds to the e_i(coef) function in the paper
@jit(nopython=True, cache=True)
def e_func(
    X: np.array,
    time: np.array,
    coef: np.array,
) -> np.array:
    return np.log(time) + np.dot(X, coef.T).reshape(time.shape)
