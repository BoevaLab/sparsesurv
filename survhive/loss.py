from math import log

import numpy as np
from numba import jit
from utils import (
    e_func,
    e_func_i,
    inverse_transform_survival_target,
    norm_cdf,
    norm_pdf,
)


# the Kernel function chosen is the normal denisty function as used in the paper
# i.e. K(.) = norm(.)
@jit(nopython=True, cache=True)
def ah_loss(
    X: np.array,
    y: np.array,
    coef: np.array,
    bandwidth: float,
) -> float:

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
) -> float:

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


@jit(nopython=True, cache=True)
def efron_likelihood(log_partial_hazard, time, event, sample_weight):
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    previous_time = time[0]
    risk_set_sum = 0
    accumulated_sum = 0
    death_set_count = 0
    death_set_risk = 0
    likelihood = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i]
        sample_partial_log_hazard = log_partial_hazard[i]
        weight = sample_weight[i]

        if previous_time < sample_time:
            for ell in range(death_set_count):
                likelihood -= np.log(
                    risk_set_sum - ((ell / death_set_count) * death_set_risk)
                )
            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            death_set_risk = 0

        if sample_event:
            death_set_count += 1
            death_set_risk += weight * sample_partial_hazard
            likelihood += weight * sample_partial_log_hazard

        accumulated_sum += weight * sample_partial_hazard
        previous_time = sample_time

    for ell in range(death_set_count):
        likelihood -= np.log(risk_set_sum - ((ell / death_set_count) * death_set_risk))
    return likelihood / samples.shape[0]


@jit(nopython=True, cache=True)
def breslow_likelihood(log_partial_hazard, time, event, sample_weight):
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    previous_time = time[0]
    risk_set_sum = 0
    likelihood = 0
    set_count = 0
    accumulated_sum = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    for k in range(samples):
        current_time = time[k]
        weight = sample_weight[i]
        if current_time > previous_time:
            # correct set-count, have to go back to set the different hazards for the ties
            likelihood -= set_count * log(risk_set_sum)
            risk_set_sum -= accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count += 1
            likelihood += log_partial_hazard[k] * weight

        previous_time = current_time
        accumulated_sum += partial_hazard[k] * weight

    likelihood -= set_count * log(risk_set_sum)
    return likelihood / samples.shape[0]


LOSS_FACTORY = {"efron": efron_likelihood, "breslow": breslow_likelihood}
