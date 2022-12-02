import numpy as np
from numba import jit
from sklearn.utils.extmath import safe_sparse_dot

from .utils import inverse_transform_survival_target


@jit(nopython=True, cache=True)
def cox_ph_breslow_negative_gradient(
    coef: np.array, X: np.array, y: np.array, risk_matrix: np.array
):
    _: np.array
    event: np.array
    _, event = inverse_transform_survival_target(y)
    log_partial_hazard: np.array = safe_sparse_dot(
        X, coef.T, dense_output=True
    )
    death_ix: np.array = event == 1

    return np.negative(
        np.sum(
            log_partial_hazard[death_ix]
            - np.log(
                np.sum(
                    np.tile(
                        A=np.exp(log_partial_hazard), reps=np.sum(death_ix)
                    )
                    * risk_matrix
                )
            )
        )
    )


@jit(nopython=True, cache=True)
def cox_ph_efron_negative_gradient(
    coef: np.array, X: np.array, risk_matrix: np.array, death_matrix: np.array
):
    log_partial_hazard: np.array = safe_sparse_dot(
        X, coef.T, dense_output=True
    )
    death_matrix_sum = np.sum(death_matrix, axis=0)
    death_set_log_partial_hazard = np.matmul(log_partial_hazard, death_matrix)
    death_set_partial_hazard = np.matmul(
        np.exp(log_partial_hazard), death_matrix
    )
    risk_set_partial_hazard = np.matmul(
        np.exp(log_partial_hazard), risk_matrix.T
    )
    efron_matrix = np.repeat(
        np.expand_dims(np.arange(np.max(death_matrix_sum)), 1),
        repeats=death_matrix_sum.shape,
        axis=1,
    )
    helper_matrix = np.zeros(efron_matrix.shape)
    for ix, qx in enumerate(death_matrix_sum):
        efron_matrix[qx:, ix] = 0
        helper_matrix[qx:, ix] = 1
    return np.sum(
        death_set_log_partial_hazard
        - np.sum(
            np.log(
                risk_set_partial_hazard
                - risk_set_partial_hazard * helper_matrix
                - (efron_matrix / death_matrix_sum) * death_set_partial_hazard
                + helper_matrix
            ),
            axis=0,
        )
    )
