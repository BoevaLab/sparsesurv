from math import erf, exp, log
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from numba import jit
from scipy import sparse
from sklearn.linear_model._base import _pre_fit
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot

from .constants import PDF_PREFACTOR, SQRT_TWO


def normal_density(x):  
    density = 0.3989423*torch.exp(-0.5*torch.pow(x,2.0))
    return density

def inverse_transform_survival(
    y: np.array,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:

    return y["time"].astype(np.float_), y["event"].astype(np.int_)


def transform_survival(
    time: npt.NDArray[np.float64], event: npt.NDArray[np.int64]
) -> np.array:
    y = np.array(
        [
            (event_, time_)
            for time_, event_ in zip(
                time.astype(np.float_), event.astype(np.bool_)
            )
        ],
        dtype=[
            ("event", "?"),
            ("time", "f8"),
        ],
    )
    return y


def inverse_transform_survival_preconditioning(
    y: np.array,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    time = y["time"].astype(np.float_)
    event = y["event"].astype(np.int_)
    if len(y[0]) > 3:
        n_eta = len(y[0]) - 2
        eta_hat = []
        for eta in range(1, n_eta + 1):
            eta_hat.append(y[f"eta_hat_{eta}"].astype(np.float_))
        eta_hat = np.stack(eta_hat, axis=1)
    else:
        eta_hat = y["eta_hat"].astype(np.float_)
    return time, event, eta_hat


def transform_survival_preconditioning(
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    eta_hat: npt.NDArray[np.float64],
):
    eta_hat = eta_hat.squeeze()
    # TODO DW: This whole thing should be somewhat simplified long-term.
    if eta_hat.shape[0] > time.shape[0]:
        n_eta = int(eta_hat.shape[0] / time.shape[0])
        if n_eta > 2:
            # TODO DW: Document this better.
            raise NotImplementedError
        n = time.shape[0]
        # TODO DW: Generalise this to higher order etas.
        y = np.array(
            [
                (event_, time_, eta_hat_1_, eta_hat_2_)
                for time_, event_, eta_hat_1_, eta_hat_2_ in zip(
                    time.astype(np.float_),
                    event.astype(np.float_),
                    eta_hat.astype(np.float_)[:n],
                    eta_hat.astype(np.float_)[n:],
                )
            ],
            dtype=[
                ("event", "?"),
                ("time", "f8"),
                ("eta_hat_1", "f8"),
                ("eta_hat_2", "f8"),
            ],
        )
    elif len(eta_hat.shape) > 1:
        n = time.shape[0]
        y = np.array(
            [
                (event_, time_, eta_hat_1_, eta_hat_2_)
                for time_, event_, eta_hat_1_, eta_hat_2_ in zip(
                    time.astype(np.float_),
                    event.astype(np.float_),
                    eta_hat.astype(np.float_)[:, 0],
                    eta_hat.astype(np.float_)[:, 1],
                )
            ],
            dtype=[
                ("event", "?"),
                ("time", "f8"),
                ("eta_hat_1", "f8"),
                ("eta_hat_2", "f8"),
            ],
        )
    else:
        y = np.array(
            [
                (event_, time_, eta_hat_)
                for time_, event_, eta_hat_ in zip(
                    time.astype(np.float_),
                    event.astype(np.bool_),
                    eta_hat.astype(np.float_),
                )
            ],
            dtype=[("event", "?"), ("time", "f8"), ("eta_hat", "f8")],
        )
    return y


@jit(nopython=True, cache=True)
def logsubstractexp(a: float, b: float) -> float:
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) - np.exp(b - max_value))


@jit(nopython=True, cache=True)
def logaddexp(a: float, b: float) -> float:
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) + np.exp(b - max_value))


@jit(nopython=True, fastmath=True)
def numba_logsumexp_stable(a: npt.NDArray[np.float64]) -> float:
    max_ = np.max(a)
    return max_ + log(np.sum(np.exp(a - max_)))


def _path_predictions(
    X,
    y,
    time,
    event,
    sample_weight,
    train,
    test,
    fit_intercept,
    path,
    path_params,
    alphas=None,
    l1_ratio=1.0,
    X_order=None,
    dtype=None,
):
    """Returns the MSE for the models computed by 'path'.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    sample_weight : None or array-like of shape (n_samples,)
        Sample weights.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    path : callable
        Function returning a list of models on the path. See
        enet_path for an example of signature.

    path_params : dictionary
        Parameters passed to the path function.

    alphas : array-like, default=None
        Array of float that is used for cross-validation. If not
        provided, computed using 'path'.

    l1_ratio : float, default=1
        float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0`` the penalty is an
        L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
        < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    X_order : {'F', 'C'}, default=None
        The order of the arrays expected by the path function to
        avoid memory copies.

    dtype : a numpy dtype, default=None
        The dtype of the arrays expected by the path function to
        avoid memory copies.

    Notes
    -----
    Almost as-is adapted from `sklearn`.

    See Also
    --------
    TODO DW
    """
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    time_train = time[train]
    event_train = event[train]
    time_test = time[test]
    event_test = event[test]

    if sample_weight is None:
        sw_train, _ = None, None
    else:
        sw_train = sample_weight[train]
        n_samples = X_train.shape[0]
        # TLDR: Rescale sw_train to sum up to n_samples on the training set.
        # See TLDR and long comment inside ElasticNet.fit.
        sw_train *= n_samples / np.sum(sw_train)
        # Note: Alternatively, we could also have rescaled alpha instead
        # of sample_weight:
        #
        #     alpha *= np.sum(sample_weight) / n_samples

    if not sparse.issparse(X):
        for array, array_input in (
            (X_train, X),
            (y_train, y),
            (X_test, X),
            (y_test, y),
        ):
            if array.base is not array_input and not array.flags["WRITEABLE"]:
                # fancy indexing should create a writable copy but it doesn't
                # for read-only memmaps (cf. numpy#14132).
                array.setflags(write=True)

    precompute = False

    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
        X_train,
        y_train,
        None,
        precompute,
        normalize=False,
        fit_intercept=fit_intercept,
        copy=False,
        sample_weight=sw_train,
    )

    path_params = path_params.copy()
    path_params["Xy"] = Xy
    path_params["X_offset"] = X_offset
    path_params["X_scale"] = X_scale
    path_params["precompute"] = precompute
    path_params["copy_X"] = False
    path_params["alphas"] = alphas
    # needed for sparse cd solver
    path_params["sample_weight"] = sw_train

    if "l1_ratio" in path_params:
        path_params["l1_ratio"] = l1_ratio

    # Do the ordering and type casting here, as if it is done in the path,
    # X is copied and a reference is kept here
    X_train = check_array(
        X_train, accept_sparse="csc", dtype=dtype, order=X_order
    )
    alphas, coefs, _ = path(X_train, y_train, **path_params)

    if y.ndim == 1:
        # Doing this so that it becomes coherent with multioutput.
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    X_train_coefs = safe_sparse_dot(X_train, coefs)
    X_test_coefs = safe_sparse_dot(X_test, coefs)
    # print(coefs.shape)
    # raise ValueError
    if len(coefs.squeeze().shape) == 2:
        n_sparsity = np.sum(coefs != 0.0, axis=(0, 1))
    else:
        n_sparsity = np.sum(coefs != 0.0, axis=(0, 1)) / 2
    # print(X_test_coefs.shape)
    # raise ValueError
    # print(y_train)
    # raise ValueError
    return (
        X_train_coefs,
        X_test_coefs,
        transform_survival_preconditioning(time_train, event_train, y_train),
        transform_survival_preconditioning(time_test, event_test, y_test),
        n_sparsity,
    )


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    return PDF_PREFACTOR * exp(-0.5 * (x**2))


@jit(nopython=True, cache=True, fastmath=True)
def kernel(a, b, bandwidth):
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: np.array = np.empty(
        shape=(a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def difference_kernels(a, b, bandwidth):
    difference: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    integrated_kernel_matrix: np.array = np.empty(
        shape=(a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix


def basic_cv_fold(
    test_linear_predictor: np.array,
    test_time: np.array,
    test_event: np.array,
    test_eta_hat: np.array,
    train_linear_predictor: np.array,
    train_time: np.array,
    train_event: np.array,
    score_function: Callable,
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

    test_fold_likelihood = -score_function(
        test_linear_predictor, test_time, test_event
    )

    return test_fold_likelihood


def basic_mse(
    test_linear_predictor: np.array,
    test_time: np.array,
    test_event: np.array,
    test_eta_hat: np.array,
    train_linear_predictor: np.array,
    train_time: np.array,
    train_event: np.array,
    score_function: Callable,
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

    test_negative_mse = np.negative(
        np.mean(np.square(test_eta_hat - test_linear_predictor))
    )

    return test_negative_mse


def vvh_cv_fold(
    test_linear_predictor: np.array,
    test_time: np.array,
    test_event: np.array,
    test_eta_hat: np.array,
    train_linear_predictor: np.array,
    train_time: np.array,
    train_event: np.array,
    score_function: Callable,
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
    z = np.append(train_linear_predictor, test_linear_predictor, axis=0)
    time = np.append(train_time, test_time)
    event = np.append(train_event, test_event)

    z_likelihood = -score_function(z, time, event)
    z_train_likelihood = -score_function(
        train_linear_predictor, train_time, train_event
    )

    test_fold_likelihood = z_likelihood - z_train_likelihood
    return test_fold_likelihood


def linear_cv(
    test_linear_predictor: np.array,
    test_time: np.array,
    test_event: np.array,
    score_function: Callable,
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
    log_likelihood = -score_function(
        test_linear_predictor, test_time, test_event
    )
    return log_likelihood
