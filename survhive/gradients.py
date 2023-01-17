import numpy as np
from numba import jit
from scipy.stats import norm
from sklearn.utils.extmath import safe_sparse_dot

from .utils import (
    e_func_i,
    get_gradient_latent_overlapping_group_lasso,
    inverse_transform_survival_target,
    norm_cdf,
    norm_pdf,
    norm_pdf_prime,
)


@jit(nopython=True, cache=True)
def cox_ph_breslow_negative_gradient(
    coef: np.array,
    X: np.array,
    y: np.array,
    risk_matrix: np.array,
    groups=None,
    has_overlaps=False,
    inverse_groups=None,
):
    _: np.array
    event: np.array
    _, event = inverse_transform_survival_target(y)
    log_partial_hazard: np.array = safe_sparse_dot(X, coef.T, dense_output=True)
    death_ix: np.array = event == 1

    gradient = np.negative(
        np.sum(
            log_partial_hazard[death_ix]
            - np.log(
                np.sum(
                    np.tile(A=np.exp(log_partial_hazard), reps=np.sum(death_ix))
                    * risk_matrix
                )
            )
        )
    )
    if has_overlaps:
        gradient = get_gradient_latent_overlapping_group_lasso(
            gradient, groups, inverse_groups
        )
    return gradient


@jit(nopython=True, cache=True)
def cox_ph_efron_negative_gradient(
    coef: np.array,
    X: np.array,
    risk_matrix: np.array,
    death_matrix: np.array,
    groups=None,
    has_overlaps=False,
    inverse_groups=None,
):
    log_partial_hazard: np.array = safe_sparse_dot(X, coef.T, dense_output=True)
    death_matrix_sum = np.sum(death_matrix, axis=0)
    death_set_log_partial_hazard = np.matmul(log_partial_hazard, death_matrix)
    death_set_partial_hazard = np.matmul(np.exp(log_partial_hazard), death_matrix)
    risk_set_partial_hazard = np.matmul(np.exp(log_partial_hazard), risk_matrix.T)
    efron_matrix = np.repeat(
        np.expand_dims(np.arange(np.max(death_matrix_sum)), 1),
        repeats=death_matrix_sum.shape,
        axis=1,
    )
    helper_matrix = np.zeros(efron_matrix.shape)
    for ix, qx in enumerate(death_matrix_sum):
        efron_matrix[qx:, ix] = 0
        helper_matrix[qx:, ix] = 1
    gradient = np.sum(
        death_set_log_partial_hazard
        - np.sum(
            np.log(
                risk_set_partial_hazard
                - risk_set_partial_hazard * helper_matrix
                - (efron_matrix / death_matrix_sum) * death_set_partial_hazard
                + helper_matrix
            ),
            axis=0,
        )
    )

    if has_overlaps:
        gradient = get_gradient_latent_overlapping_group_lasso(
            gradient, groups, inverse_groups
        )
    return gradient


@jit(nopython=True, cache=True)
def ah_gradient(
    X: np.array,
    y: np.array,
    coef: np.array,
    bandwidth: float,
    groups=None,
    has_overlaps=False,
    inverse_groups=None,
) -> np.array:

    n_samples = X.shape[0]
    n_dim = X.shape[1]
    time, event = inverse_transform_survival_target(y)

    term1 = np.sum(X[event, :], axis=0) / n_samples

    term2 = np.zeros(n_dim)
    term3 = np.zeros(n_dim)
    for i in range(n_samples):
        term2j = np.zeros(n_dim)
        term3j = np.zeros(n_dim)
        term2j_prime = np.zeros(n_dim)
        term3j_prime = np.zeros(n_dim)
        for j in range(n_samples):
            kernel_input = (
                e_func_i(X, time, coef, j) - e_func_i(X, time, coef, i)
            ) / bandwidth

            exp_term = np.exp(-np.dot(X[j], coef.T))
            pdf_term = norm_pdf(kernel_input)
            cdf_term = norm_cdf(kernel_input)

            term2j += event[j] * pdf_term
            term3j += exp_term * cdf_term

            # f(g(x))' = g'(x)f'(g(x))
            kernel_input_prime = (X[j] - X[i]) / bandwidth
            term2j_prime += event[j] * kernel_input_prime * norm_pdf_prime(kernel_input)

            # (f(x)g(k(x)))' = f(x)k'(x)g'(k(x)) + f'(x)g(k(x))
            term3j_prime += (
                exp_term * kernel_input_prime * pdf_term + -X[j] * exp_term * cdf_term
            )
        # log'(f(x)) = f'(x) / f(x), the constants canceled out
        term2 += event[i] * term2j_prime / term2j
        term3 += event[i] * term3j_prime / term3j
    term2 /= n_samples
    term3 /= n_samples

    gradient = term1 - term2 + term3
    if has_overlaps:
        gradient = get_gradient_latent_overlapping_group_lasso(
            gradient, groups, inverse_groups
        )
    return gradient


@jit(nopython=True, cache=True)
def aft_gradient(
    X: np.array,
    y: np.array,
    coef: np.array,
    bandwidth: float,
    groups=None,
    has_overlaps=False,
    inverse_groups=None,
) -> np.array:

    n_samples = X.shape[0]
    n_dim = X.shape[1]
    time, event = inverse_transform_survival_target(y)

    term1 = np.sum(X[event, :], axis=0) / n_samples
    term2 = term1

    term3 = np.zeros(n_dim)
    term4 = np.zeros(n_dim)
    for i in range(n_samples):
        term3j = np.zeros(n_dim)
        term4j = np.zeros(n_dim)
        term3j_prime = np.zeros(n_dim)
        term4j_prime = np.zeros(n_dim)
        for j in range(n_samples):
            kernel_input = (
                e_func_i(X, time, coef, j) - e_func_i(X, time, coef, i)
            ) / bandwidth

            pdf_term = norm_pdf(kernel_input)

            term3j += event[j] * pdf_term
            term4j += norm_cdf(kernel_input)

            kernel_input_prime = (X[j] - X[i]) / bandwidth
            term3j_prime += event[j] * kernel_input_prime * norm_pdf_prime(kernel_input)
            term4j_prime += kernel_input_prime * pdf_term

        term3 += event[i] * term3j_prime / term3j
        term4 += event[i] * term4j_prime / term4j
    term3 /= n_samples
    term4 /= n_samples

    gradient = -term1 + term2 - term3 + term4
    if has_overlaps:
        gradient = get_gradient_latent_overlapping_group_lasso(
            gradient, groups, inverse_groups
        )
    return gradient
