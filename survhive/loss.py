import numpy as np
from numba import jit
from scipy.stats import norm
from utils import e_func, inverse_transform_survival_target


# the Kernel function chosen is the normal denisty function as used in the paper
# i.e. K(.) = norm(.)
# @jit(nopython=True, cache=True)
def ah_loss(
    X: np.array,
    y: np.array,
    coef: np.array,
    bandwidth: float,
) -> np.array:

    n_samples = X.shape[0]
    time, event = inverse_transform_survival_target(y)

    e_matrix = e_func(X, time, coef)

    term1 = np.sum(e_matrix[event, :]) / n_samples

    kernel_input_matrix = (
        np.subtract.outer(e_matrix, e_matrix).reshape(n_samples, n_samples) / bandwidth
    )

    pdf_matrix = np.vectorize(norm.pdf)(kernel_input_matrix)
    term2 = np.sum(pdf_matrix[:, event], axis=1, keepdims=True) / n_samples / bandwidth
    term2 = np.sum(np.log(term2)[event, :]) / n_samples

    cdf_matrix = np.vectorize(norm.cdf)(kernel_input_matrix)
    term3 = np.exp(-np.dot(X, coef.T)) * cdf_matrix
    term3 = np.sum(term3, axis=1, keepdims=True) / n_samples
    term3 = np.sum(np.log(term3)[event, :]) / n_samples

    return -term1 + term2 - term3


# use normal densifty function as gaussian kernel, e_func is the same as R_func in the paper
# @jit(nopython=True, cache=True)
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
    term2 = np.sum(e_matrix[event, :]) / n_samples

    kernel_input_matrix = (
        np.subtract.outer(e_matrix, e_matrix).reshape(n_samples, n_samples) / bandwidth
    )

    pdf_matrix = np.vectorize(norm.pdf)(kernel_input_matrix)
    term3 = np.sum(pdf_matrix[:, event], axis=1, keepdims=True) / n_samples / bandwidth
    term3 = np.sum(np.log(term3)[event, :]) / n_samples

    cdf_matrix = np.vectorize(norm.cdf)(kernel_input_matrix)
    term4 = np.sum(cdf_matrix, axis=1, keepdims=True) / n_samples
    term4 = np.sum(np.log(term4)[event, :]) / n_samples

    return term1 - term2 + term3 - term4
