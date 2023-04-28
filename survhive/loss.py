from math import log

import numpy as np
from numba import jit

from .utils import logaddexp, logsubstractexp, numba_logsumexp_stable


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

    for k in range(samples):
        current_time = time[k]
        if current_time > previous_time:
            likelihood -= set_count * log_risk_set_sum
            log_risk_set_sum = logsubstractexp(
                log_risk_set_sum, accumulated_sum
            )
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

            log_risk_set_sum = logsubstractexp(
                log_risk_set_sum, accumulated_sum
            )
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


def breslow_preconditioning_loss(time, event, eta_hat, X, tau, coef):
    eta = X @ coef
    return tau * breslow_likelihood_stable(
        linear_predictor=eta, time=time, event=event
    ) + (1 - tau) * 1 / (2 * time.shape[0]) * np.sum(np.square(eta - eta_hat))


def efron_preconditioning_loss(time, event, eta_hat, X, tau, coef):
    eta = X @ coef
    return tau * efron_likelihood_stable(
        linear_predictor=eta, time=time, event=event
    ) + (1 - tau) * 1 / (2 * time.shape[0]) * np.sum(np.square(eta - eta_hat))
