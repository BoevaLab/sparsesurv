from typing import Tuple

import numpy as np
from numba import jit
from scipy.stats import norm
from sklearn.utils.extmath import safe_sparse_dot
from typeguard import typechecked

from .utils import (
    e_func_i,
    get_gradient_latent_overlapping_group_lasso,
    inverse_transform_survival_target,
    norm_cdf,
    norm_pdf,
    norm_pdf_prime,
)


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


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def update_risk_sets_breslow(
    risk_set_sum: float,
    death_set_count: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
) -> Tuple[float, float]:
    local_risk_set += 1 / (risk_set_sum / death_set_count)
    local_risk_set_hessian += 1 / ((risk_set_sum**2) / death_set_count)
    return local_risk_set, local_risk_set_hessian


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def calculate_sample_grad_hess(
    sample_partial_hazard: float,
    sample_event: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    weight: float,
) -> Tuple[float, float]:
    return (
        sample_partial_hazard * local_risk_set
    ) - sample_event * weight, sample_partial_hazard * local_risk_set - local_risk_set_hessian * (
        sample_partial_hazard**2
    )


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def breslow_numba(
    time: np.array,
    event: np.array,
    log_partial_hazard: np.array,
    sample_weight: np.array,
):
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    risk_set_sum = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time = time[0]
    local_risk_set = 0
    local_risk_set_hessian = 0
    death_set_count = 0
    censoring_set_count = 0
    accumulated_sum = 0

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i] * sample_weight[i]

        if previous_time < sample_time:
            if death_set_count:
                (local_risk_set, local_risk_set_hessian,) = update_risk_sets_breslow(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                    sample_weight[i],
                )

            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0

        if sample_event:
            death_set_count += 1
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        local_risk_set, local_risk_set_hessian = update_risk_sets_breslow(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
        )
    return grad, hess


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def calculate_sample_grad_hess_efron(
    sample_partial_hazard: float,
    sample_event: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    local_risk_set_death: float,
    local_risk_set_hessian_death: float,
    weight: float,
) -> Tuple[float, float]:
    if sample_event:
        return ((sample_partial_hazard) * (local_risk_set_death)) - (
            sample_event * weight
        ), (sample_partial_hazard) * (local_risk_set_death) - (
            (local_risk_set_hessian_death)
        ) * (
            (sample_partial_hazard) ** 2
        )
    else:
        return ((sample_partial_hazard) * local_risk_set), (
            sample_partial_hazard
        ) * local_risk_set - local_risk_set_hessian * ((sample_partial_hazard) ** 2)


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def update_risk_sets_efron_pre(
    risk_set_sum: float,
    death_set_count: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    death_set_risk: float,
) -> Tuple[float, float, float, float]:
    local_risk_set_death: float = local_risk_set
    local_risk_set_hessian_death: float = local_risk_set_hessian

    for ell in range(death_set_count):
        contribution: float = ell / death_set_count
        local_risk_set += 1 / (risk_set_sum - (contribution) * death_set_risk)
        local_risk_set_death += (1 - (ell / death_set_count)) / (
            risk_set_sum - (contribution) * death_set_risk
        )
        local_risk_set_hessian += (
            1 / ((risk_set_sum - (contribution) * death_set_risk))
        ) ** 2

        local_risk_set_hessian_death += ((1 - contribution) ** 2) / (
            ((risk_set_sum - (contribution) * death_set_risk)) ** 2
        )

    return (
        local_risk_set,
        local_risk_set_hessian,
        local_risk_set_death,
        local_risk_set_hessian_death,
    )


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def efron_numba(
    time: np.array,
    event: np.array,
    log_partial_hazard: np.array,
    sample_weight: np.array,
) -> Tuple[np.array, np.array]:
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    risk_set_sum = 0
    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time: float = time[0]
    local_risk_set: int = 0
    local_risk_set_hessian: int = 0
    death_set_count: int = 0
    censoring_set_count: int = 0
    accumulated_sum: int = 0
    death_set_risk: float = 0.0
    local_risk_set_death: float = 0.0
    local_risk_set_hessian_death: float = 0.0

    for i in range(samples):
        risk_set_sum += sample_weight[i] * partial_hazard[i]

    for i in range(samples):
        sample_time: float = time[i]
        sample_event: int = event[i]
        sample_partial_hazard: float = partial_hazard[i] * sample_weight[i]

        if previous_time < sample_time:
            if death_set_count:
                (
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                ) = update_risk_sets_efron_pre(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                    death_set_risk,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                    sample_weight[death_ix],
                )
            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0
            death_set_risk = 0
            local_risk_set_death = 0
            local_risk_set_hessian_death = 0

        if sample_event:
            death_set_count += 1
            death_set_risk += sample_partial_hazard
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        (
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        ) = update_risk_sets_efron_pre(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
            death_set_risk,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
            sample_weight[death_ix],
        )
    return grad, hess
