from math import log

from typing import Callable

import numpy as np
from numba import jit

from typeguard import typechecked
from .utils import difference_kernels


@jit(nopython=True, cache=True)
def efron_likelihood(linear_predictor, time, event, sample_weight):
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
    return -likelihood / samples.shape[0]


@jit(nopython=True, cache=True)
def breslow_likelihood(linear_predictor, time, event, sample_weight):
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
        weight = sample_weight[i]
        if current_time > previous_time:
            # correct set-count, have to go back to set the different hazards for the ties
            likelihood -= set_count * log(risk_set_sum)
            risk_set_sum -= accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count += 1
            likelihood += linear_predictor[k] * weight

        previous_time = current_time
        accumulated_sum += partial_hazard[k] * weight

    likelihood -= set_count * log(risk_set_sum)
    return -likelihood / samples.shape[0]


@jit(nopython=True, cache=True, fastmath=True)
def ah_likelihood(
    linear_predictor: np.array,
    time: np.array,
    event: np.array,
    sample_weight: np.array,
    bandwidth_function: float,
) -> np.array:
    n_samples: int = time.shape[0]
    linear_predictor: np.array = linear_predictor * sample_weight
    exp_linear_predictor: np.array = np.exp(linear_predictor)
    R_linear_predictor: np.array = np.log(time * exp_linear_predictor)
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )
    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: np.array = kernel_matrix.sum(axis=0)

    integrated_kernel_sum: np.array = (
        integrated_kernel_matrix
        * exp_linear_predictor.repeat(np.sum(event)).reshape(-1, np.sum(event))
    ).sum(axis=0)

    likelihood: np.array = inverse_sample_size * (
        -R_linear_predictor[event].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


@jit(nopython=True, cache=True, fastmath=True)
def aft_likelihood(
    linear_predictor: np.array,
    time: np.array,
    event: np.array,
    sample_weight: np.array,
    bandwidth_function: float,
) -> np.array:
    n_samples: int = time.shape[0]
    linear_predictor: np.array = linear_predictor * sample_weight
    R_linear_predictor: np.array = np.log(time * np.exp(linear_predictor))
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)
    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
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


LOSS_FACTORY = {
    "efron": efron_likelihood,
    "breslow": breslow_likelihood,
    "aft": aft_likelihood,
    "ah": ah_likelihood,
}
