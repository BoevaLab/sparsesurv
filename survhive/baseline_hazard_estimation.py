from math import log

import numpy as np
from numba import jit

from .bandwidth_estimation import jones_1990, jones_1991
from .constants import EPS
from .utils import gaussian_integrated_kernel, gaussian_kernel


@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_breslow(
    time: np.array,
    event: np.array,
    eta: np.array,
):
    exp_eta: np.array = np.exp(eta)
    local_risk_set: float = np.sum(exp_eta)
    event_mask: np.array = event.astype(np.bool_)
    n_unique_events: int = np.unique(time[event_mask]).shape[0]
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = time[0]

    for _ in range(len(time)):
        sample_time: float = time[_]
        sample_event: int = event[_]
        sample_eta: float = exp_eta[_]

        if sample_time > previous_time and local_death_set:
            cumulative_baseline_hazards[n_events_counted] = local_death_set / (
                local_risk_set
            )

            local_death_set = 0
            local_risk_set -= accumulated_risk_set
            accumulated_risk_set = 0
            n_events_counted += 1

        if sample_event:
            local_death_set += 1
        accumulated_risk_set += sample_eta
        previous_time = sample_time

    cumulative_baseline_hazards[n_events_counted] = local_death_set / (local_risk_set)

    return (
        np.unique(time[event_mask]),
        np.cumsum(cumulative_baseline_hazards),
    )


@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_efron(
    time: np.array,
    event: np.array,
    eta: np.array,
):
    exp_eta: np.array = np.exp(eta)
    local_risk_set: float = np.sum(exp_eta)
    event_mask: np.array = event.astype(np.bool_)
    n_unique_events: int = np.unique(time[event_mask]).shape[0]
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = time[0]
    local_death_set_risk: float = 0

    for _ in range(len(time)):
        sample_time: float = time[_]
        sample_event: int = event[_]
        sample_exp_eta: float = exp_eta[_]

        if sample_time > previous_time and local_death_set:
            for ell in range(local_death_set):

                cumulative_baseline_hazards[n_events_counted] += 1 / (
                    local_risk_set - (ell / local_death_set) * local_death_set_risk
                )

            local_risk_set -= accumulated_risk_set
            accumulated_risk_set = 0
            local_death_set_risk = 0
            local_death_set = 0
            n_events_counted += 1

        if sample_event:
            local_death_set += 1
            local_death_set_risk += sample_exp_eta
        accumulated_risk_set += sample_exp_eta
        previous_time = sample_time

    for ell in range(local_death_set):
        cumulative_baseline_hazards[n_events_counted] += 1 / (
            local_risk_set - (ell / local_death_set) * local_death_set_risk
        )

    return (
        np.unique(time[event_mask]),
        np.cumsum(cumulative_baseline_hazards),
    )


@jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_aft(
    time,
    train_time,
    train_event,
    train_eta,
    bandwidth_function: str = "jones_1990",
):
    if bandwidth_function == "jones_1990":
        bandwidth: float = jones_1990(time=train_time, event=train_event)
    else:
        bandwidth: float = jones_1991(time=train_time, event=train_event)

    n_samples: int = train_time.shape[0]
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = inverse_bandwidth * inverse_sample_size
    log_time: float = log(time + EPS)
    R_lp: np.array = np.log(train_time * np.exp(train_eta))
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0

    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]
        denominator += gaussian_integrated_kernel(difference)
        if train_event[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator

    return numerator / denominator


@jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_ah(
    time,
    train_time,
    train_event,
    train_eta,
    bandwidth_function: str = "jones_1990",
):
    if bandwidth_function == "jones_1990":
        bandwidth: float = jones_1990(time=train_time, event=train_event)
    else:
        bandwidth: float = jones_1991(time=train_time, event=train_event)
    n_samples: int = train_time.shape[0]
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = inverse_bandwidth * inverse_sample_size
    log_time: float = log(time + EPS)
    exp_linear_predictor: np.array = np.exp(train_eta)
    R_lp: np.array = np.log(train_time * exp_linear_predictor)
    difference_lp_log_time: np.array = R_lp - log_time
    numerator: float = 0.0
    denominator: float = 0.0

    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]
        denominator += exp_linear_predictor[_] * gaussian_integrated_kernel(difference)
        if train_event[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator

    return numerator / denominator
