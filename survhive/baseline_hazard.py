from typing import Callable, Dict

import numpy as np
from numba import jit
from typeguard import typechecked


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_breslow(
    train_time: np.array,
    train_event: np.array,
    train_partial_hazards: np.array,
):
    local_risk_set: float = np.sum(train_partial_hazards)
    n_unique_events: int = np.unique(train_time[train_event]).shape[0]
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = train_time[0]

    for _ in range(len(train_time)):
        sample_time: float = train_time[_]
        sample_event: int = train_event[_]
        sample_log_partial_hazard: float = train_partial_hazards[_]

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


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_efron(
    train_time: np.array,
    train_event: np.array,
    train_partial_hazards: np.array,
):
    local_risk_set: float = np.sum(train_partial_hazards)
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
        sample_log_partial_hazard: float = train_partial_hazards[_]

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


CUMULATIVE_BASELINE_HAZARD_FACTORY: Dict[str, Callable] = {
    "breslow": breslow_estimator_breslow,
    "efron": breslow_estimator_efron,
}
