from typing import Callable

import numpy as np


def basic_cv_fold(
    z_test: np.array,
    test_time: np.array,
    test_event: np.array,
    z_train: np.array,
    train_time: np.array,
    train_event: np.array,
    scorer: Callable,
) -> float:
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

    test_fold_likelihood = -scorer(z_test, test_time, test_event)

    return test_fold_likelihood


def vvh_cv_fold(
    z_test: np.array,
    test_time: np.array,
    test_event: np.array,
    z_train: np.array,
    train_time: np.array,
    train_event: np.array,
    scorer: Callable,
) -> float:
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

    z = np.append(z_train, z_test)
    time = np.append(train_time, test_time)
    event = np.append(train_event, test_event)

    z_likelihood = -scorer(z, time, event)
    z_train_likelihood = -scorer(z_train, train_time, train_event)

    test_fold_likelihood = z_likelihood - z_train_likelihood
    return test_fold_likelihood


def linear_cv(
    z_test: np.array, time: np.array, event: np.array, scorer: Callable
) -> float:
    """CV score computation using linear predictors (Dai et. al. 2019).

    Args:
        z_test (np.ndarray): flattened array of the linear predictors of all test folds.
            The array is of shape (folds, samples, predictors).
        y_test (np.ndarray): flattened array of survival time and event data
            corresponding to samples in all test folds. The array is of shape
            (folds, samples, target).

    Returns:
        float: Scalar value of the partial log-likelihoods across all test folds.
    """
    log_likelihood = -scorer(z_test, time, event)
    return log_likelihood
