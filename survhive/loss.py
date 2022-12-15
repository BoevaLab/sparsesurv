import numpy as np
from numba import jit
from utils import (
    e_func,
    e_func_i,
    inverse_transform_survival_target,
    norm_pdf,
    norm_cdf,
)


# the Kernel function chosen is the normal denisty function as used in the paper
# i.e. K(.) = norm(.)
@jit(nopython=True, cache=True)
def ah_loss(
    X: np.array,
    y: np.array,
    coef: np.array,
    bandwidth: float,
) -> np.array:

    n_samples = X.shape[0]
    time, event = inverse_transform_survival_target(y)

    e_matrix = e_func(X, time, coef)

    term1 = np.sum(e_matrix[event]) / n_samples

    term2 = 0
    term3 = 0
    for i in range(n_samples):
        term2j = 0
        term3j = 0
        for j in range(n_samples):
            kernel_input = (
                e_func_i(X, time, coef, j) - e_func_i(X, time, coef, i)
            ) / bandwidth
            term2j += event[j] * norm_pdf(kernel_input)
            term3j += np.exp(-np.dot(X[j], coef.T)) * norm_cdf(kernel_input)
        term2 += event[i] * np.log(term2j / n_samples / bandwidth)
        term3 += event[i] * np.log(term3j / n_samples)
    term2 /= n_samples
    term3 /= n_samples

    return term1 - term2 + term3


# use normal densifty function as gaussian kernel, e_func is the same as R_func in the paper
@jit(nopython=True, cache=True)
def aft_loss(
    X: np.array,
    y: np.array,
    coef: np.array,
    bandwidth: float,
) -> np.array:

    n_samples = X.shape[0]
    time, event = inverse_transform_survival_target(y)

    e_matrix = e_func(X, time, coef)

    term1 = np.sum(np.dot(X, coef.T)[event]) / n_samples
    term2 = np.sum(e_matrix[event]) / n_samples

    term3 = 0
    term4 = 0
    for i in range(n_samples):
        term3j = 0
        term4j = 0
        for j in range(n_samples):
            kernel_input = (
                e_func_i(X, time, coef, j) - e_func_i(X, time, coef, i)
            ) / bandwidth
            term3j += event[j] * norm_pdf(kernel_input)
            term4j += norm_cdf(kernel_input)
        term3 += event[i] * np.log(term3j / n_samples / bandwidth)
        term4 += event[i] * np.log(term4j / n_samples)
    term3 /= n_samples
    term4 /= n_samples

    return -term1 + term2 - term3 + term4
