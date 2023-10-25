from math import log

import numpy as np
import numpy.typing as npt
from numba import jit

from .utils import (
    difference_kernels,
    logaddexp,
    logsubstractexp,
    numba_logsumexp_stable,
)


@jit(nopython=True, cache=True, fastmath=True)
def breslow_negative_likelihood(
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Negative log-likelihood function with Breslow tie-correction.

    Args:
        linear_predictor (npt.NDArray[np.float64]): Linear predictor of the training data.
        time (npt.NDArray[np.float64]): Time of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.

    Raises:
        RuntimeError: Raises runtime error when there are no deaths/events in the given
            batch of samples.

    Returns:
        npt.NDArray[np.float64]: Average negative log likelihood.
    """
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
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Negative log-likelihood function with Efron tie-correction.

    Args:
        linear_predictor (npt.NDArray[np.float64]): Linear predictor of the training data.
        time (npt.NDArray[np.float64]): Time of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.

    Raises:
        RuntimeError: Raises runtime error when there are no deaths/events in the given
            batch of samples.

    Returns:
        npt.NDArray[np.float64]: Average negative log likelihood.
    """
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


def efron_negative_likelihood_beta(
    beta: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
) -> float:
    """Negative log-likelihood function with Efron tie-correction when the design matrix
        and the coefficients beta are provided instead of the linear predictor directly.

    Args:
        beta (npt.NDArray[np.float64]): Coefficient vector of length p.
        X (npt.NDArray[np.float64]): Design matrix of the training data. N rows and p columns.
        time (npt.NDArray[np.float64]): Time of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.

    Returns:
        npt.NDArray[np.float64]: Average negative log likelihood.
    """
    return efron_negative_likelihood(linear_predictor=X @ beta, time=time, event=event)


@jit(nopython=True, cache=True, fastmath=True)
def aft_negative_likelihood(
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    bandwidth: float = None,
) -> npt.NDArray[np.float64]:
    """Negative log-likelihood function for accelerated failure time model.

    Args:
        linear_predictor (npt.NDArray[np.float64]): Linear predictor of the training data.
        time (npt.NDArray[np.float64]): Time of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.
        bandwidth (float, optional): Bandwidth to kernel-smooth the profile likelihood.
            Will be estimated empirically if not specified. Defaults to None.

    Raises:
        RuntimeError: Raises runtime error when there are no deaths/events in the given
            batch of samples.

    Returns:
        npt.NDArray[np.float64]: Average negative log-likelihood.
    """
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    n_samples: int = time.shape[0]
    if bandwidth is None:
        # TODO #XXX.
        bandwidth = 1.30 * pow(n_samples, -0.2)
    linear_predictor: npt.NDArray[np.float64] = linear_predictor
    R_linear_predictor: npt.NDArray[np.float64] = np.log(
        time * np.exp(linear_predictor)
    )
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: npt.NDArray[np.float64] = event.astype(np.bool_)

    _: npt.NDArray[np.float64]
    kernel_matrix: npt.NDArray[np.float64]
    integrated_kernel_matrix: npt.NDArray[np.float64]

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
    kernel_sum: npt.NDArray[np.float64] = kernel_matrix.sum(axis=0)
    integrated_kernel_sum: npt.NDArray[np.float64] = integrated_kernel_matrix.sum(0)

    likelihood: npt.NDArray[np.float64] = inverse_sample_size * (
        linear_predictor[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


def aft_negative_likelihood_beta(
    beta: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    bandwidth: float = None,
) -> float:
    """Negative log-likelihood function for accelerated failure time model when the design matrix
        and the coefficients beta are provided instead of the linear predictor directly.

    Args:
        beta (npt.NDArray[np.float64]): Coefficient vector of length p.
        X (npt.NDArray[np.float64]): Design matrix of the training data. N rows and p columns.
        time (npt.NDArray[np.float64]): Time of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.
        bandwidth (float, optional): Bandwidth to kernel-smooth the profile likelihood.
            Will be estimated empirically if not specified. Defaults to None.


    Returns:
        npt.NDArray[np.float64]: Average negative log-likelihood.
    """
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
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    bandwidth: float = None,
) -> npt.NDArray[np.float64]:
    """Negative log-likelihood function for extended hazards model.

    Args:
        linear_predictor (npt.NDArray[np.float64]): Linear predictor of the training data.
        time (npt.NDArray[np.float64]): Time points of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.
        bandwidth (float, optional): Bandwidth to kernel-smooth the profile likelihood.
            Will be estimated empirically if not specified. Defaults to None.

    Raises:
        RuntimeError: Raises runtime error when there are no deaths/events in the given
            batch of samples.

    Returns:
        npt.NDArray[np.float64]: Average negative log-likelihood.
    """
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    theta = np.exp(linear_predictor)
    n_samples: int = time.shape[0]
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)
    R_linear_predictor: npt.NDArray[np.float64] = np.log(time * theta[:, 0])
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: npt.NDArray[np.float64] = event.astype(np.bool_)

    _: npt.NDArray[np.float64]
    kernel_matrix: npt.NDArray[np.float64]
    integrated_kernel_matrix: npt.NDArray[np.float64]

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

    kernel_sum: npt.NDArray[np.float64] = kernel_matrix.sum(axis=0)
    integrated_kernel_sum: npt.NDArray[np.float64] = (
        integrated_kernel_matrix
        * (theta[:, 1] / theta[:, 0])
        .repeat(int(np.sum(event)))
        .reshape(-1, int(np.sum(event)))
    ).sum(axis=0)
    likelihood: npt.NDArray[np.float64] = inverse_sample_size * (
        linear_predictor[:, 1][event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


def eh_negative_likelihood_beta(
    beta: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    bandwidth: float = None,
) -> npt.NDArray[np.float64]:
    """Negative log-likelihood function for extended hazards model when the design matrix
        and the coefficients beta are provided instead of the linear predictor directly.

    Args:
        beta (npt.NDArray[np.float64]): Coefficient vector of length p.
        X (npt.NDArray[np.float64]): Design matrix of the training data. N rows and p columns.
        time (npt.NDArray[np.float64]): Time of the training data of length n. Assumed to be sorted.
        event (npt.NDArray[np.float64]): Event indicator of the training data of length n.
        bandwidth (float, optional): Bandwidth to kernel-smooth the profile likelihood.
            Will be estimated empirically if not specified. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: Average negative log-likelihood.
    """
    size = int(X.shape[1] / 2)
    return eh_negative_likelihood(
        linear_predictor=np.stack(
            (np.matmul(X[:, :size], beta[:size]), np.matmul(X[:, size:], beta[size:]))
        ).T,
        time=time,
        event=event,
        bandwidth=bandwidth,
    )
