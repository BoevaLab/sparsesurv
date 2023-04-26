import numpy as np
from numba import jit


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
