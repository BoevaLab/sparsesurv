import numpy as np

from numba import jit

from .utils import (
    inverse_transform_survival_target,
    get_risk_matrix,
    get_death_matrix,
)

from .utils import (
    check_y_survival,
    check_cox_tie_correction,
    check_array,
)

CUMULATIVE_BASELINE_HAZARD_FACTORY = {"breslow": np.nan, "efron": np.nan}


@jit(nopython=True, cache=True)
def breslow_estimator(
    beta: np.array, X: np.array, y: np.array, tie_correction="efron"
):
    beta: np.array = check_array(beta)
    X: np.array = check_array(X)
    check_y_survival(y)
    check_cox_tie_correction(tie_correction)

    time: np.array
    event: np.array
    time, event = inverse_transform_survival_target(y)
    risk_matrix = get_risk_matrix(time)
    partial_hazards = np.exp(np.dot(beta, X))
    if tie_correction == "efron":
        death_matrix: np.array = get_death_matrix(time, event)
        death_matrix = np.unique(
            death_matrix[:, np.any(death_matrix, axis=0)], axis=1
        )
        risk_matrix = np.unique(risk_matrix, axis=0)
        death_matrix_sum: int = np.sum(death_matrix, axis=0)
        death_set_partial_hazard = np.matmul(partial_hazards, death_matrix)
        risk_set_partial_hazard = np.matmul(partial_hazards, risk_matrix.T)
        efron_matrix = np.repeat(
            np.expand_dims(np.arange(np.max(death_matrix_sum)), 1),
            repeats=death_matrix_sum.shape,
            axis=1,
        )
        helper_matrix = np.zeros(efron_matrix.shape)
        risk_set_sums = np.prod(
            risk_set_partial_hazard
            - risk_set_partial_hazard * helper_matrix
            - (efron_matrix / death_matrix_sum) * death_set_partial_hazard
            + helper_matrix,
            axis=0,
        )
        for ix, qx in enumerate(death_matrix_sum):
            efron_matrix[qx:, ix] = 0
            helper_matrix[qx:, ix] = 1
    elif tie_correction == "breslow":
        risk_set_sums = np.sum(
            partial_hazards.repeat(time)
            .reshape((partial_hazards.shape[0], time.shape[0]))
            .T
            * risk_matrix,
            axis=0,
        )
    ix: np.array = np.argsort(time)
    sorted_time: np.array = time[ix]
    sorted_event: np.array = event[ix]
    sorted_risk_set_sums: np.array = risk_set_sums[ix]

    return (
        np.cumsum(1 / sorted_risk_set_sums[sorted_event]),
        sorted_time[sorted_event],
    )
