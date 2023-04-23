import math
from math import log

import numpy as np
from numba import jit
import math

from .utils import difference_kernels
from .constants import EPS


@jit(nopython=True, cache=True)
def efron_likelihood(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    """Efron approximation of the cumulative baseline hazard function to compute partial
        likelihood of the CoxPH model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).

    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    partial_hazard = np.exp(linear_predictor)
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
        sample_partial_log_hazard = linear_predictor[i]

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
            death_set_risk += sample_partial_hazard
            likelihood += sample_partial_log_hazard

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    for ell in range(death_set_count):
        likelihood -= np.log(risk_set_sum - ((ell / death_set_count) * death_set_risk))
    return -likelihood / samples


@jit(nopython=True, cache=True)
def breslow_likelihood(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    """Breslow approximation of the cumulative baseline hazard function to compute partial
        likelihood of the CoxPH model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).


    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(linear_predictor)
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
        if (
            current_time > previous_time
        ):  # if time is sorted this will always hold unless current = prev
            # correct set-count, have to go back to set the different hazards for the ties
            likelihood -= set_count * log(risk_set_sum)
            risk_set_sum -= accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count += 1
            likelihood += linear_predictor[k]

        previous_time = current_time
        accumulated_sum += partial_hazard[k]

    likelihood -= set_count * log(risk_set_sum)
    return -likelihood / samples


@jit(nopython=True, cache=True)
def logsubstractexp(a, b):
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) - np.exp(b - max_value))


@jit(nopython=True, cache=True)
def logaddexp(a, b):
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) + np.exp(b - max_value))


@jit(nopython=True, fastmath=True)
def numba_logsumexp_stable(a):
    max_ = np.max(a)
    return max_ + log(np.sum(np.exp(a - max_)))


@jit(nopython=True, cache=True)
def breslow_likelihood_stable(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    """Breslow approximation of the cumulative baseline hazard function to compute partial
        likelihood of the CoxPH model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).


    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    # Assumes times have been sorted beforehand.
    samples = time.shape[0]
    previous_time = time[0]
    likelihood = 0
    set_count = 0
    accumulated_sum = -np.inf
    log_risk_set_sum = numba_logsumexp_stable(linear_predictor)

    # for i in range(samples):
    #    risk_set_sum += partial_hazard[i]

    for k in range(samples):
        current_time = time[k]
        if (
            current_time > previous_time
        ):  # if time is sorted this will always hold unless current = prev
            # correct set-count, have to go back to set the different hazards for the ties
            likelihood -= set_count * log_risk_set_sum
            log_risk_set_sum = logsubstractexp(log_risk_set_sum, accumulated_sum)
            set_count = 0
            accumulated_sum = -np.inf

        if event[k]:
            set_count += 1
            likelihood += linear_predictor[k]

        previous_time = current_time
        accumulated_sum = logaddexp(accumulated_sum, linear_predictor[k])

    likelihood -= set_count * log_risk_set_sum
    return -likelihood / samples


@jit(nopython=True, cache=True)
def efron_likelihood_stable(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    """Efron approximation of the cumulative baseline hazard function to compute partial
        likelihood of the CoxPH model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).

    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    samples = time.shape[0]
    previous_time = time[0]
    accumulated_sum = -np.inf
    death_set_count = 0
    log_death_set_risk = -np.inf
    likelihood = 0
    log_risk_set_sum = numba_logsumexp_stable(linear_predictor)

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_log_hazard = linear_predictor[i]

        if previous_time < sample_time:
            for ell in range(death_set_count):
                if ell > 0:
                    likelihood -= logsubstractexp(
                        log_risk_set_sum,
                        log(ell) - log(death_set_count) + log_death_set_risk,
                    )
                else:
                    likelihood -= log_risk_set_sum

            log_risk_set_sum = logsubstractexp(log_risk_set_sum, accumulated_sum)
            accumulated_sum = -np.inf
            death_set_count = 0
            log_death_set_risk = -np.inf

        if sample_event:
            death_set_count += 1
            log_death_set_risk = logaddexp(
                log_death_set_risk, sample_partial_log_hazard
            )
            likelihood += sample_partial_log_hazard

        accumulated_sum = logaddexp(accumulated_sum, sample_partial_log_hazard)
        previous_time = sample_time

    for ell in range(death_set_count):
        if ell > 0:
            likelihood -= logsubstractexp(
                log_risk_set_sum,
                log(ell) - log(death_set_count) + log_death_set_risk,
            )
        else:
            likelihood -= log_risk_set_sum
    return -likelihood / samples


@jit(nopython=True, cache=True, fastmath=True)
def ah_likelihood(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    """Partial likelihood estimator for Accelerated Hazards model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).
        bandwidth_function (str, optional): _description_. Defaults to "jones_1990".

    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    linear_predictor: np.array = linear_predictor
    exp_linear_predictor: np.array = np.exp(-linear_predictor)
    R_linear_predictor: np.array = np.log(time * exp_linear_predictor)
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=R_linear_predictor,
        b=R_linear_predictor[event_mask],
        bandwidth=bandwidth,
    )
    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: np.array = kernel_matrix.sum(axis=0)

    integrated_kernel_sum: np.array = (
        integrated_kernel_matrix
        * exp_linear_predictor.repeat(np.sum(event)).reshape(-1, np.sum(event))
    ).sum(axis=0)
    likelihood: np.array = inverse_sample_size * (
        -R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


@jit(nopython=True, cache=True, fastmath=True)
def aft_likelihood(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    """Partial likelihood estimator for Accelerated Failure Time model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).
        bandwidth_function (str): _description_

    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    linear_predictor: np.array = linear_predictor
    R_linear_predictor: np.array = np.log(time * np.exp(linear_predictor))
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)
    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array
    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=R_linear_predictor,
        b=R_linear_predictor[event_mask],
        bandwidth=bandwidth,
    )

    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples
    kernel_sum: np.array = kernel_matrix.sum(axis=0)
    integrated_kernel_sum: np.array = integrated_kernel_matrix.sum(0)
    likelihood: np.array = inverse_sample_size * (
        linear_predictor[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood
