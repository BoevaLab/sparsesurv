from math import log

import numpy as np
from numba import jit

from .utils import (
    difference_kernels,
    logaddexp,
    logsubstractexp,
    numba_logsumexp_stable,
)


@jit(nopython=True, cache=True, fastmath=True)
def breslow_negative_likelihood(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    samples = time.shape[0]
    previous_time = time[0]
    likelihood = 0
    set_count = 0
    accumulated_sum = -np.inf
    log_risk_set_sum = numba_logsumexp_stable(linear_predictor)

    for k in range(samples):
        current_time = time[k]
        if current_time > previous_time:
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


def breslow_negative_likelihood_beta(beta, X, time, event):
    return breslow_negative_likelihood(
        linear_predictor=X @ beta, time=time, event=event
    )


@jit(nopython=True, cache=True, fastmath=True)
def efron_negative_likelihood(
    linear_predictor: np.array, time: np.array, event: np.array
) -> np.array:
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    time_ix = np.argsort(time)
    linear_predictor = linear_predictor[time_ix]
    time = time[time_ix]
    event = event[time_ix]
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


def efron_negative_likelihood_beta(beta, X, time, event):
    return efron_negative_likelihood(linear_predictor=X @ beta, time=time, event=event)


@jit(nopython=True, cache=True, fastmath=True)
def aft_negative_likelihood(
    linear_predictor: np.array,
    time,
    event,
    bandwidth=None,
) -> np.array:
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    n_samples: int = time.shape[0]
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)
    linear_predictor: np.array = linear_predictor
    R_linear_predictor: np.array = np.log(time * np.exp(linear_predictor))
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (
        _,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
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


def aft_negative_likelihood_beta(
    beta,
    X,
    time,
    event,
    bandwidth=None,
) -> np.array:
    return float(
        aft_negative_likelihood(
            linear_predictor=np.matmul(X, beta),
            time=time,
            event=event,
            bandwidth=bandwidth,
        )
    )


@jit(nopython=True, cache=True, fastmath=True)
def eh_negative_likelihood(
    linear_predictor,
    time,
    event,
    bandwidth=None,
) -> np.array:
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    theta = np.exp(linear_predictor)
    n_samples: int = time.shape[0]
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)
    R_linear_predictor: np.array = np.log(time * theta[:, 0])
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (
        _,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=R_linear_predictor,
        b=R_linear_predictor[event_mask],
        bandwidth=bandwidth,
    )

    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: np.array = kernel_matrix.sum(axis=0)
    integrated_kernel_sum: np.array = (
        integrated_kernel_matrix
        * (theta[:, 1] / theta[:, 0])
        .repeat(int(np.sum(event)))
        .reshape(-1, int(np.sum(event)))
    ).sum(axis=0)
    likelihood: np.array = inverse_sample_size * (
        linear_predictor[:, 1][event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


def eh_negative_likelihood_beta(
    beta,
    X,
    time,
    event,
    bandwidth=None,
) -> np.array:
    hm = int(X.shape[1] / 2)
    return eh_negative_likelihood(
        linear_predictor=np.stack(
            (np.matmul(X[:, :hm], beta[:hm]), np.matmul(X[:, hm:], beta[hm:]))
        ).T,
        time=time,
        event=event,
        bandwidth=bandwidth,
    )
