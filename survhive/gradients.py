from typing import Tuple

import numpy as np
from numba import jit

from .constants import EPS


@jit(nopython=True, cache=True, fastmath=True)
def update_risk_sets_breslow(
    risk_set_sum: float,
    death_set_count: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
) -> Tuple[float, float]:
    """_summary_

    Args:
        risk_set_sum (float): _description_
        death_set_count (int): _description_
        local_risk_set (float): _description_
        local_risk_set_hessian (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """
    local_risk_set += 1 / (risk_set_sum / death_set_count)
    local_risk_set_hessian += 1 / ((risk_set_sum**2) / death_set_count)
    return local_risk_set, local_risk_set_hessian


@jit(nopython=True, cache=True, fastmath=True)
def calculate_sample_grad_hess(
    sample_partial_hazard: float,
    sample_event: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
) -> Tuple[float, float]:
    """_summary_

    Args:
        sample_partial_hazard (float): _description_
        sample_event (int): _description_
        local_risk_set (float): _description_
        local_risk_set_hessian (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """
    return (
        sample_partial_hazard * local_risk_set
    ) - sample_event, sample_partial_hazard * local_risk_set - local_risk_set_hessian * (
        sample_partial_hazard**2
    )


@jit(nopython=True, cache=True, fastmath=True)
def breslow_numba_stable(
    linear_predictor: np.array,
    time: np.array,
    event: np.array,
) -> Tuple[np.array, np.array]:
    """Gradient of the Breslow approximated version of CoxPH in numba-compatible form.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).

    Returns:
        Tuple[np.array, np.array]: Tuple containing the negative gradients and the hessian
            of with the linear predictor.
    """
    partial_hazard = np.exp(linear_predictor - np.max(linear_predictor))
    partial_hazard[partial_hazard < EPS] = EPS
    samples = time.shape[0]
    risk_set_sum = 0
    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time = time[0]
    local_risk_set = 0
    local_risk_set_hessian = 0
    death_set_count = 0
    censoring_set_count = 0
    accumulated_sum = 0

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i]
        if previous_time < sample_time:
            if death_set_count:
                (local_risk_set, local_risk_set_hessian,) = update_risk_sets_breslow(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                )

            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0

        if sample_event:
            death_set_count += 1
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        local_risk_set, local_risk_set_hessian = update_risk_sets_breslow(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
        )
    return grad / samples, hess / samples


@jit(nopython=True, cache=True, fastmath=True)
def calculate_sample_grad_hess_efron(
    sample_partial_hazard: float,
    sample_event: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    local_risk_set_death: float,
    local_risk_set_hessian_death: float,
) -> Tuple[float, float]:
    """_summary_

    Args:
        sample_partial_hazard (float): _description_
        sample_event (int): _description_
        local_risk_set (float): _description_
        local_risk_set_hessian (float): _description_
        local_risk_set_death (float): _description_
        local_risk_set_hessian_death (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """
    if sample_event:
        return ((sample_partial_hazard) * (local_risk_set_death)) - (sample_event), (
            sample_partial_hazard
        ) * (local_risk_set_death) - ((local_risk_set_hessian_death)) * (
            (sample_partial_hazard) ** 2
        )
    else:
        return ((sample_partial_hazard) * local_risk_set), (
            sample_partial_hazard
        ) * local_risk_set - local_risk_set_hessian * ((sample_partial_hazard) ** 2)


@jit(nopython=True, cache=True, fastmath=True)
def update_risk_sets_efron_pre(
    risk_set_sum: float,
    death_set_count: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    death_set_risk: float,
) -> Tuple[float, float, float, float]:
    """_summary_

    Args:
        risk_set_sum (float): _description_
        death_set_count (int): _description_
        local_risk_set (float): _description_
        local_risk_set_hessian (float): _description_
        death_set_risk (float): _description_

    Returns:
        Tuple[float, float, float, float]: _description_
    """
    local_risk_set_death: float = local_risk_set
    local_risk_set_hessian_death: float = local_risk_set_hessian

    for ell in range(death_set_count):
        contribution: float = ell / death_set_count
        local_risk_set += 1 / (risk_set_sum - (contribution) * death_set_risk)
        local_risk_set_death += (1 - (ell / death_set_count)) / (
            risk_set_sum - (contribution) * death_set_risk
        )
        local_risk_set_hessian += (
            1 / ((risk_set_sum - (contribution) * death_set_risk))
        ) ** 2

        local_risk_set_hessian_death += ((1 - contribution) ** 2) / (
            ((risk_set_sum - (contribution) * death_set_risk)) ** 2
        )

    return (
        local_risk_set,
        local_risk_set_hessian,
        local_risk_set_death,
        local_risk_set_hessian_death,
    )


@jit(nopython=True, cache=True, fastmath=True)
def efron_numba_stable(
    linear_predictor: np.array,
    time: np.array,
    event: np.array,
) -> Tuple[np.array, np.array]:
    """Gradient of the Efron approximated version of CoxPH in numba-compatible form.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).

    Returns:
        Tuple[np.array, np.array]: Tuple containing the negative gradients and the hessian
            of with the linear predictor.
    """
    partial_hazard = np.exp(linear_predictor - np.max(linear_predictor))
    partial_hazard[partial_hazard < EPS] = EPS
    samples = time.shape[0]
    risk_set_sum = 0
    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time: float = time[0]
    local_risk_set: int = 0
    local_risk_set_hessian: int = 0
    death_set_count: int = 0
    censoring_set_count: int = 0
    accumulated_sum: int = 0
    death_set_risk: float = 0.0
    local_risk_set_death: float = 0.0
    local_risk_set_hessian_death: float = 0.0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    for i in range(samples):
        sample_time: float = time[i]
        sample_event: int = event[i]
        sample_partial_hazard: float = partial_hazard[i]

        if previous_time < sample_time:
            if death_set_count:
                (
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                ) = update_risk_sets_efron_pre(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                    death_set_risk,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                )
            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0
            death_set_risk = 0
            local_risk_set_death = 0
            local_risk_set_hessian_death = 0

        if sample_event:
            death_set_count += 1
            death_set_risk += sample_partial_hazard
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        (
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        ) = update_risk_sets_efron_pre(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
            death_set_risk,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        )
    return grad / samples, hess / samples
