from math import log

import numpy as np
import pandas as pd
from numba import jit
from scipy.integrate import quadrature

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
    if local_death_set:
        cumulative_baseline_hazards[n_events_counted] = local_death_set / (
            local_risk_set
        )

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
                    local_risk_set
                    - (ell / local_death_set) * local_death_set_risk
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

    if local_death_set:
        for ell in range(local_death_set):
            cumulative_baseline_hazards[n_events_counted] += 1 / (
                local_risk_set - (ell / local_death_set) * local_death_set_risk
            )

    return (
        np.unique(time[event_mask]),
        np.cumsum(cumulative_baseline_hazards),
    )


@jit(nopython=True, cache=True, fastmath=True)
def aft_baseline_hazard_estimator(
    time,
    time_train,
    event_train,
    eta_train,
):
    n_samples: int = time_train.shape[0]
    bandwidth = 1.30 * pow(n_samples, -0.2)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size_time: float = (
        inverse_sample_size * (1 / (time + EPS)) * inverse_bandwidth
    )
    log_time: float = log(time + EPS)

    R_lp: np.array = np.log(time_train * np.exp(eta_train))
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0
    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]
        denominator += gaussian_integrated_kernel(difference)
        if event_train[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size_time * numerator
    denominator = inverse_sample_size * denominator
    if denominator <= 0.0:
        return 0.0
    else:
        return numerator / denominator


# TODO DW: Rejit this later on.
# @jit(nopython=True, cache=True, fastmath=True)
def get_cumulative_hazard_function_aft(
    time_query,
    eta_query,
    time_train,
    event_train,
    eta_train,
) -> pd.DataFrame:
    time: np.array = np.unique(time_query)
    theta: np.array = np.exp(eta_query)
    n_samples: int = eta_query.shape[0]

    zero_flag: bool = False
    if 0 not in time:
        zero_flag = True
        time = np.concatenate([np.array([0]), time])
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))
    else:
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))

    def hazard_function_integrate(s):
        return aft_baseline_hazard_estimator(
            time=s,
            time_train=time_train,
            event_train=event_train,
            eta_train=eta_train,
        )

    integration_times = np.stack(
        [time * i for i in np.round(np.exp(np.ravel(eta_query)), 2)]
    )
    integration_times = np.unique((np.ravel(integration_times)))

    integration_times = np.concatenate(
        [[0], integration_times, [np.max(integration_times) + 0.01]]
    )

    integration_values = np.zeros(integration_times.shape[0])
    for _ in range(1, integration_values.shape[0]):
        integration_values[_] = (
            integration_values[_ - 1]
            + quadrature(
                func=hazard_function_integrate,
                a=integration_times[_ - 1],
                b=integration_times[_],
                vec_func=False,
            )[0]
        )
    for _ in range(n_samples):
        cumulative_hazard[_] = integration_values[
            np.digitize(x=time * theta[_], bins=integration_times, right=False)
            - 1
        ]
    if zero_flag:
        cumulative_hazard = cumulative_hazard[:, 1:]
        time = time[1:]
    return pd.DataFrame(cumulative_hazard, columns=time)


@jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_eh(
    time: np.array,
    time_train: np.array,
    event_train: np.array,
    eta_train: np.array,
):
    n_samples: int = time_train.shape[0]
    bandwidth = 1.30 * pow(n_samples, -0.2)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = (
        inverse_sample_size * (1 / (time + EPS)) * inverse_bandwidth
    )
    log_time: float = np.log(time + EPS)
    theta_train = np.exp(eta_train)
    R_lp: np.array = np.log(time_train * theta_train[:, 0])
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0
    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]

        denominator += (
            theta_train[_, 1]
            / theta_train[_, 0]
            * gaussian_integrated_kernel(difference)
        )
        if event_train[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator

    if denominator <= 0.0:
        return 0.0
    else:
        return numerator / denominator


# TODO DW: Rejit this later on.
# @jit(nopython=True, cache=True, fastmath=True)
def get_cumulative_hazard_function_eh(
    time_query,
    eta_query,
    time_train,
    event_train,
    eta_train,
) -> pd.DataFrame:
    time: np.array = np.unique(time_query)
    theta: np.array = np.exp(eta_query)
    n_samples: int = eta_query.shape[0]
    zero_flag: bool = False
    if 0 not in time:
        zero_flag = True
        time = np.concatenate([np.array([0]), time])
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))
    else:
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))

    def hazard_function_integrate(s):
        return baseline_hazard_estimator_eh(
            time=s,
            time_train=time_train,
            event_train=event_train,
            eta_train=eta_train,
        )

    integration_times = np.stack([time * i for i in np.round(theta[:, 0], 2)])
    integration_times = np.unique((np.ravel(integration_times)))

    integration_times = np.concatenate(
        [[0], integration_times, [np.max(integration_times) + 0.01]]
    )
    integration_values = np.zeros(integration_times.shape[0])
    for _ in range(1, integration_values.shape[0]):
        integration_values[_] = (
            integration_values[_ - 1]
            + quadrature(
                func=hazard_function_integrate,
                a=integration_times[_ - 1],
                b=integration_times[_],
                vec_func=False,
            )[0]
        )

    for _ in range(n_samples):
        cumulative_hazard[_] = (
            integration_values[
                np.digitize(
                    x=time * theta[_, 0], bins=integration_times, right=False
                )
                - 1
            ]
            * theta[_, 1]
            / theta[_, 0]
        )
    if zero_flag:
        cumulative_hazard = cumulative_hazard[:, 1:]
        time = time[1:]
    return pd.DataFrame(cumulative_hazard, columns=time)
