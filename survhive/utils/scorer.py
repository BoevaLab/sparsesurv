import numpy as np
from survhive.utils import *


def get_partial_log_likelihood(z, time_event) -> float:
    """Computes the partial log likelihood of given data.

    Args:
        z (_type_): _description_
        time_event (_type_): _description_

    Returns:
        float: _description_
    """
    # this assumes no ties. Q - should divide by n?
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


def basic_cv(z_test, y_test, **kwargs) -> float:
    """Basic CV scoring function.

    Args:
        z_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        float: _description_
    """
    folds, _, _ = z_test.shape
    test_fold_likelihoods = []
    for k in range(folds):
        test_fold_likelihoods.append(
            get_partial_log_likelihood(z_test[k, :, :], y_test[k, :])
        )

    return np.mean(test_fold_likelihoods)
    # return test_fold_likelihoods


def vvh_cv(z_test, y_test, **kwargs) -> float:
    """Verweij and Van Houwelingen CV scoring function.

    Args:
        z_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        float: _description_
    """
    z_train = kwargs["z_train"]
    y_train = kwargs["y_train"]
    folds, n_train_samples, n_coefs = z_train.shape
    test_fold_likelihoods = []

    for k in range(folds):
        z = np.append(z_train[k, :, :], z_test[k, :, :])
        y = np.append(y_train[k, :], y_test[k, :])

        z_likelihood = get_partial_log_likelihood(z, y)
        z_train_likelihood = get_partial_log_likelihood(z_train[k, :, :], y_train[k, :])
        test_fold_likelihoods.append(z_likelihood - z_train_likelihood)

    # return test_fold_likelihoods
    return np.mean(test_fold_likelihoods)


def linear_cv(z_test, y_test, **kwargs) -> float:
    """_summary_

    Args:
        z_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        float: _description_
    """

    assert z_test.ndim == 3
    # flatten folds to compute cv score
    n_hat_flattened = np.vstack(z_test)
    time_event_flattened = np.vstack(y_test)

    assert n_hat_flattened.shape == (z_test.shape[0] * z_test.shape[1], z_test.shape[2])
    assert len(n_hat_flattened) == len(time_event_flattened)

    log_likelihood = get_partial_log_likelihood(n_hat_flattened, time_event_flattened)

    return log_likelihood
