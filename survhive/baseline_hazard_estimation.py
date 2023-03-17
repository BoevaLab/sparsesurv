from math import log

import numpy as np
from numba import jit
from .utils import gaussian_integrated_kernel, gaussian_kernel


@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_breslow(
    train_time: np.array,
    train_event: np.array,
    train_linear_predictor: np.array,
):
    train_linear_predictor: np.array = np.exp(train_linear_predictor)
    local_risk_set: float = np.sum(train_linear_predictor)
    n_unique_events: int = np.unique(train_time[train_event]).shape[0]
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = train_time[0]

    for _ in range(len(train_time)):
        sample_time: float = train_time[_]
        sample_event: int = train_event[_]
        sample_log_partial_hazard: float = train_linear_predictor[_]

        if sample_time > previous_time:
            cumulative_baseline_hazards[n_events_counted] = local_death_set / (
                local_risk_set
            )

            local_death_set = 0
            local_risk_set -= accumulated_risk_set
            accumulated_risk_set = 0
            n_events_counted += 1

        if sample_event:
            local_death_set += 1
        accumulated_risk_set += sample_log_partial_hazard
        previous_time = sample_time

    cumulative_baseline_hazards[n_events_counted] = local_death_set / (local_risk_set)

    return (
        np.unique(train_time[train_event]),
        np.cumsum(cumulative_baseline_hazards),
    )


@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_efron(
    train_time: np.array,
    train_event: np.array,
    train_linear_predictor: np.array,
):
    train_linear_predictor: np.array = np.exp(train_linear_predictor)
    local_risk_set: float = np.sum(train_linear_predictor)
    n_unique_events: int = np.unique(train_time[train_event]).shape[0]
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = train_time[0]
    local_death_set_risk: float = 0

    for _ in range(len(train_time)):
        sample_time: float = train_time[_]
        sample_event: int = train_event[_]
        sample_log_partial_hazard: float = train_linear_predictor[_]

        if sample_time > previous_time:
            for ell in range(len(local_death_set)):

                cumulative_baseline_hazards[n_events_counted] += 1 / (
                    local_risk_set - (ell / local_death_set) * local_death_set_risk
                )

            local_death_set = 0
            local_risk_set -= accumulated_risk_set
            accumulated_risk_set = 0
            local_death_set_risk = 0
            n_events_counted += 1

        if sample_event:
            local_death_set += 1
            local_death_set_risk += sample_log_partial_hazard
        accumulated_risk_set += sample_log_partial_hazard

    for ell in range(len(local_death_set)):
        cumulative_baseline_hazards[n_events_counted] += 1 / (
            local_risk_set - (ell / local_death_set) * local_death_set_risk
        )

    return (
        np.unique(train_time[train_event]),
        np.cumsum(cumulative_baseline_hazards),
    )


@jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_aft(
    time, train_time, train_event, train_linear_predictor, bandwidth_function
):
    n_samples: int = train_time.shape[0]
    bandwidth: float = bandwidth_function(time=train_time, event=train_event)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = inverse_bandwidth * inverse_sample_size
    log_time: float = log(time)
    R_lp: np.array = np.log(train_time * np.exp(train_linear_predictor))
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0

    for _ in n_samples:
        difference: float = difference_lp_log_time[_]
        denominator += gaussian_integrated_kernel(difference)
        if train_event[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator

    return numerator / denominator


@jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_ah(
    time, train_time, train_event, train_linear_predictor, bandwidth_function
):
    n_samples: int = train_time.shape[0]
    bandwidth: float = bandwidth_function(time=train_time, event=train_event)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = inverse_bandwidth * inverse_sample_size
    log_time: float = log(time)
    exp_linear_predictor: np.array = np.exp(train_linear_predictor)
    R_lp: np.array = np.log(train_time * exp_linear_predictor)
    difference_lp_log_time: np.array = R_lp - log_time
    numerator: float = 0.0
    denominator: float = 0.0

    for _ in n_samples:
        difference: float = difference_lp_log_time[_]
        denominator += exp_linear_predictor[_] * gaussian_integrated_kernel(difference)
        if train_event[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator

    return numerator / denominator
