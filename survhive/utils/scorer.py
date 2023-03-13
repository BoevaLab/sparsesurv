import numpy as np
from typeguard import typechecked

from survhive.utils import *


@typechecked
@jit(nopython=True, cache=True)
def get_partial_log_likelihood(z: np.ndarray, time_event: np.ndarray) -> float:
    """Computes the partial log likelihood of given data.

    Args:
        z (np.ndarray): 2-dimensional array of linear predictors of size (samples, num_coefs).
        time_event (np.ndarray): 2-dimensional array of time and event survival data.
            The array is of shape (samples, target), where targets are in str format.

    Returns:
        float: Scalar value summarising the total partial log-likelihood of the given
            linear predictors and their corresponding event and time data.
    """
    # INFO: this assumes no ties. Q - should we divide by n?
    time, event = inverse_transform_survival_target(time_event)
    sorted_indices = np.argsort(a=time, kind="stable")
    time_sorted = time[sorted_indices]
    event_sorted = event[sorted_indices]
    z_sorted = z[
        sorted_indices,
    ]
    sum_risk_set = 0
    log_likelihood = 0

    for i in range(len(time_sorted)):
        sum_risk_set = np.cumsum(z_sorted[i:])
        likelihood_i = np.log((np.exp(z_sorted[i]) / sum_risk_set) ** event_sorted)
        log_likelihood += likelihood_i

    return log_likelihood


@jit(nopython=True, cache=True)
def basic_cv(z_test: np.ndarray, y_test: np.ndarray, **kwargs) -> float:
    """Basic CV scoring function.

    Args:
        z_test (np.ndarray): 3-dimensional array of the linear predictors of all test folds.
            The array is of shape (folds, samples, predictors).
        y_test (np.ndarray): 3-dimensional array of survival time and event data
            corresponding to samples in all test folds. The array is of shape
            (folds, samples, target).

    Returns:
        float: Scalar value of the mean partial log-likelihoods across all test folds.
    """
    folds, _, _ = z_test.shape
    test_fold_likelihoods = []
    for k in range(folds):
        test_fold_likelihoods.append(
            get_partial_log_likelihood(z_test[k, :, :], y_test[k, :, :])
        )

    return np.mean(test_fold_likelihoods)


@jit(nopython=True, cache=True)
def vvh_cv(z_test: np.ndarray, y_test: np.ndarray, **kwargs) -> float:
    """Verweij and Van Houwelingen CV scoring function.

    Args:
        z_test (np.ndarray): 3-dimensional array of the linear predictors of all test folds.
            The array is of shape (folds, samples, predictors).
        y_test (np.ndarray): 3-dimensional array of survival time and event data
            corresponding to samples in all test folds. The array is of shape
            (folds, samples, target).
        kwargs : VVH method requires that the linear predictors and targets of the
            training folds are also passed as addiitonal keyword arguments. For linear
            predictors of the training data, use "z_train" as the keyword, and for the
            correspondig targets, use "y_train" as the keyword when calling the function.

    Returns:
        float: Scalar value of the mean partial log-likelihoods across all test folds.
    """
    z_train = kwargs["z_train"]
    y_train = kwargs["y_train"]
    folds, n_train_samples, n_coefs = z_train.shape
    test_fold_likelihoods = []

    for k in range(folds):
        z = np.append(z_train[k, :, :], z_test[k, :, :])
        y = np.append(y_train[k, :, :], y_test[k, :, :])

        z_likelihood = get_partial_log_likelihood(z, y)
        z_train_likelihood = get_partial_log_likelihood(
            z_train[k, :, :], y_train[k, :, :]
        )
        test_fold_likelihoods.append(z_likelihood - z_train_likelihood)

    return np.mean(test_fold_likelihoods)


@jit(nopython=True, cache=True)
def linear_cv(z_test: np.ndarray, y_test: np.ndarray, **kwargs) -> float:
    """CV score computation using linear predictors (Dai et. al. 2019).

    Args:
        z_test (np.ndarray): 3-dimensional array of the linear predictors of all test folds.
            The array is of shape (folds, samples, predictors).
        y_test (np.ndarray): 3-dimensional array of survival time and event data
            corresponding to samples in all test folds. The array is of shape
            (folds, samples, target).

    Returns:
        float: Scalar value of the partial log-likelihoods across all test folds.
    """

    assert z_test.ndim == 3
    # flatten folds to compute cv score
    n_hat_flattened = np.vstack(z_test)
    time_event_flattened = np.vstack(y_test)

    assert n_hat_flattened.shape == (z_test.shape[0] * z_test.shape[1], z_test.shape[2])
    assert len(n_hat_flattened) == len(time_event_flattened)

    log_likelihood = get_partial_log_likelihood(n_hat_flattened, time_event_flattened)

    return log_likelihood
