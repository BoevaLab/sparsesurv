from math import erf, exp

import numpy as np
from numba import jit

from .constants import PDF_PREFACTOR, SQRT_TWO


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    return PDF_PREFACTOR * exp(-0.5 * (x**2))


# @jit(nopython=True, cache=True, fastmath=True)
# def kernel(a, b, bandwidth):
#     kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
#     for ix in range(a.shape[0]):
#         for qx in range(b.shape[0]):
#             kernel_matrix[ix, qx] = gaussian_kernel(
#                 (a[ix] - b[qx]) / bandwidth
#             )
#     return kernel_matrix


# @jit(nopython=True, cache=True, fastmath=True)
# def integrated_kernel(a, b, bandwidth):
#     integrated_kernel_matrix: np.array = np.empty(
#         shape=(a.shape[0], b.shape[0])
#     )
#     for ix in range(a.shape[0]):
#         for qx in range(b.shape[0]):
#             integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
#                 (a[ix] - b[qx]) / bandwidth
#             )
#     return integrated_kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def difference_kernels(a, b, bandwidth):
    difference: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix


@jit(nopython=True, cache=True)
def inverse_transform_survival(
    y: np.array,
) -> tuple[np.array, np.array]:
    event = y >= 0
    event = event.flatten().astype(np.int_)
    time = np.abs(y).flatten()
    return time, event


@jit(nopython=True, cache=True)
def transform_survival(time: np.array, event: np.array) -> np.array:
    y: np.array = np.copy(time)
    y[np.logical_not(event)] = np.negative(y[np.logical_not(event)])
    return y
