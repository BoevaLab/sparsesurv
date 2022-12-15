from math import erf

import numpy as np
from numba import jit, vectorize, float64


@jit(nopython=True, cache=True)
def inverse_transform_survival_target(
    y: np.array,
) -> tuple[np.array, np.array]:
    event = y >= 0
    event = event.flatten()
    time = np.abs(y).flatten()
    return time, event


# Gaussian norm kernel functions
@jit(nopython=True, cache=True)
def norm_pdf(x: float) -> float:
    return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


@jit(nopython=True, cache=True)
# @vectorize([float64(float64)])
def norm_cdf(x: float) -> float:
    return 0.5 + erf(x / np.sqrt(2)) / 2


# Utility function to calculate kernel input, coressponds to the e_i(coef) function in the paper
@jit(nopython=True, cache=True)
def e_func(
    X: np.array,
    time: np.array,
    coef: np.array,
) -> np.array:
    return np.log(time) + np.dot(X, coef.T)


@jit(nopython=True, cache=True)
def e_func_i(
    X: np.array,
    time: np.array,
    coef: np.array,
    i: int,
) -> float:
    return np.log(time[i]) + np.dot(X[i], coef.T)
