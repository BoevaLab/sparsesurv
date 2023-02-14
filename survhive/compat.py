from typing import Tuple

import numpy as np
from numba import jit
from typeguard import typechecked

from .gradients import breslow_numba, efron_numba
from .utils import (
    get_gradient_latent_overlapping_group_lasso,
    inverse_transform_survival_target,
)


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def cox_ph_breslow_negative_gradient(
    eta: np.array, y: np.array, sample_weight: np.array
) -> Tuple[np.array]:
    time: np.array
    event: np.array
    time, event = inverse_transform_survival_target(y)
    gradient: np.array
    hessian: np.array
    gradient = breslow_numba(
        log_partial_hazard=eta,
        time=time,
        event=event,
        sample_weight=sample_weight,
    )
    return gradient, hessian


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def cox_ph_efron_negative_gradient_hessian(
    eta: np.array, y: np.array, sample_weight: np.array
) -> Tuple[np.array]:
    time: np.array
    event: np.array
    time, event = inverse_transform_survival_target(y)
    gradient: np.array
    hessian: np.array
    gradient, hessian = efron_numba(
        log_partial_hazard=eta,
        time=time,
        event=event,
        sample_weight=sample_weight,
    )
    return gradient, hessian
