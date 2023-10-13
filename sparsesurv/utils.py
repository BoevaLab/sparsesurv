from math import erf, exp, log
from typing import Callable, Tuple, List

import numpy as np
import numpy.typing as npt
import torch
from numba import jit
from scipy import sparse
from sklearn.linear_model._base import _pre_fit
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot

from .constants import PDF_PREFACTOR, SQRT_TWO


def normal_density(x: torch.Tensor) -> torch.Tensor:
    """Calculate Gaussian kernel.

    Args:
        x (torch.Tensor): Input of differences.

    Returns:
        torch.Tensor: Gaussian kernel value.
    """
    # PDF_PREFACTOR = 0.3989423 (old version - rounded up)
    density = PDF_PREFACTOR * torch.exp(-0.5 * torch.pow(x, 2.0))
    return density


def inverse_transform_survival(
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Transform input variable into time and event variable.

    Args:
        y (np.array): One dimensional array containing time and censoring events.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: Survival time and event array.
    """
    return y["time"].astype(np.float_), y["event"].astype(np.int_)


def transform_survival(
    time: npt.NDArray[np.float64], event: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Transform time and event variables into one variable.

    Args:
        time (npt.NDArray[np.float64]): Survival times.
        event (npt.NDArray[np.int64]): Censoring information.

    Returns:
        npt.NDArray[np.float64]: Structured array containing survival times and right-censored survival information.
    """
    y = np.array(
        [
            (event_, time_)
            for time_, event_ in zip(time.astype(np.float_), event.astype(np.bool_))
        ],
        dtype=[
            ("event", "?"),
            ("time", "f8"),
        ],
    )
    return y


def inverse_transform_survival_preconditioning(
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Obtain survival times, censoring information and eta (e.g. y train) from structuted array.

    Args:
        y (npt.NDArray[np.float64]): Structured array containing survival times, censoring information.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]: survival times, censoring information, eta.
    """
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
) -> np.array[npt.NDArray[np.float64], npt.NDArray[np.int], npt.NDArray[np.float64]]:
    """Transform survival times, censoring information and eta (e.g. y train) into one array.

    Args:
        time (npt.NDArray[np.float64]): Survival times.
        event (npt.NDArray[np.int64]): Censoring information.
        eta_hat (npt.NDArray[np.float64]): Estimated dependent variable.

    Raises:
        NotImplementedError: Checking for dimensions.

    Returns:
        np.array: Structured array containing survival times and censoring information.
    """
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
    """Apply log-sum-exp trick when calculating the log difference for numerical stability.

    Args:
        a (npt.NDArray[np.float64]): Subtraction value first entity.
        b (npt.NDArray[np.float64]): Subtraction value second entity.

    Returns:
        npt.NDArray[np.float64]: Result of substraction with log-sum-exp trick.
    """
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) - np.exp(b - max_value))


@jit(nopython=True, cache=True)
def logaddexp(a: float, b: float) -> float:
    """Apply log-sum-exp trick when calculating the log addition for numerical stability.

    Args:
        a (float): Addition value first entity.
        b (float): Addition value second entity.

    Returns:
        float: Result of addition with log-sum-exp trick.
    """
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) + np.exp(b - max_value))


@jit(nopython=True, fastmath=True)
def numba_logsumexp_stable(a: npt.NDArray[np.float64]) -> float:
    """Apply log-sum-exp trick.

    Args:
        a (npt.NDArray[np.float64]): Input array to which the sum and then
        log will be applied.

    Returns:
        float: Result of log-sum-exp trick.
    """
    max_ = np.max(a)
    return max_ + log(np.sum(np.exp(a - max_)))


def _path_predictions(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    sample_weight: npt.ArrayLike,
    train: List,
    test: List,
    fit_intercept: bool,
    path: Callable,
    path_params: dict,
    alphas: npt.ArrayLike = None,
    l1_ratio: float = 1.0,
    X_order: str = None,
    dtype: np.dtype = None,
) -> Tuple:
    """Returns linear predictors for the models computed by 'path'.

    Note:
        Almost as-is adapted from `sklearn`.
    Args:
        X (npt.ArrayLike): Training data of shape (n_samples, n_features).
        y (npt.ArrayLike): Target values of shape (n_samples,) or (n_samples, n_targets).
        time (npt.NDArray[np.float64]): Survival times.
        event (npt.NDArray[np.int64]): Censoring information.
        sample_weight (npt.ArrayLike): Sample weights of shape (n_samples,). Defaults to None.
        train (List): The indices of the train set.
        test (List): The indices of the test set.
        fit_intercept (bool): Whether to fit intercept or not.
        path (Callable): Function returning a list of models on the path. See enet_path
            for an example of signature.
        path_params (dict): Parameters passed to the path function.
        alphas (npt.ArrayLike, optional): Array of float that is used for cross-validation.
            If not provided, computed using 'path'. Defaults to None.
        l1_ratio (float, optional): float between 0 and 1 passed to ElasticNet (scaling
            between l1 and l2 penalties). For ``l1_ratio = 0`` the penalty is an L2
            penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0 < l1_ratio < 1``,
            the penalty is a combination of L1 and L2. Defaults to 1.0.
        X_order (str, optional): The order of the arrays expected by the path function to
            avoid memory copies. One of {'F', 'C'}. Defaults to None.
        dtype (np.dtype, optional): The dtype of the arrays expected by the path function to
        avoid memory copies. Defaults to None.

    Returns:
        Tuple: Tuple of linear predictors of both train and test data including transformed
            target values (time, event) and number of non-zero coefficients for each model
            on the 'path'.
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
    X_train = check_array(X_train, accept_sparse="csc", dtype=dtype, order=X_order)
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
    """Obtain result of the integration of the Gaussian kernel.

    Args:
        x (float): Difference of hazard predictions.

    Returns:
        float: Integration value of Gaussian kernel.
    """
    return 0.5 * (1 + erf(x / SQRT_TWO))


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    """Obtain result of Gaussian kernel.

    Args:
        x (float): Difference of hazard predictions.

    Returns:
        float: Value of Gaussian kernel.
    """
    return PDF_PREFACTOR * exp(-0.5 * (x**2))


@jit(nopython=True, cache=True, fastmath=True)
def kernel(a, b, bandwidth) -> npt.NDArray[np.float64]:
    """Subtract predictor values from each other and calculate Gaussian kernel.

    Args:
        a (npt.NDArray[np.float64]): First predictor value (hazard prediction).
        b (npt.NDArray[np.float64]): Second predictor value (hazard prediction).
        bandwidth (float): Fixed kernel bandwith.

    Returns:
        npt.NDArray[np.float64]: Kernel matrix.
    """

    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel((a[ix] - b[qx]) / bandwidth)
    return kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def integrated_kernel(a, b, bandwidth) -> npt.NDArray[np.float64]:
    """Subtract predictor values from each other and calculate integrated Gaussian kernel.

    Args:
        a (npt.NDArray[np.float64]): First predictor value (hazard prediction).
        b (npt.NDArray[np.float64]): Second predictor value (hazard prediction).
        bandwidth (float): Fixed kernel bandwith.

    Returns:
        npt.NDArray[np.float64]: Integrated kernel matrix.
    """
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def difference_kernels(a, b, bandwidth) -> Tuple:
    """Subtract predictor values from each other as well as calculate (integrated) Gaussian kernel.

    Args:
        a (npt.NDArray[np.float64]): First predictor value (hazard prediction).
        b (npt.NDArray[np.float64]): Second predictor value (hazard prediction).
        bandwidth (float): Fixed kernel bandwith.

    Returns:
        Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]: Predictor difference, kernel matrix, integrated kernel matrix
    """
    difference: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix


"""

    Args:
        z_test (np.ndarray): 3-dimensional array of the linear predictors of all test folds.
            The array is of shape (folds, samples, predictors).
        y_test (np.ndarray): 3-dimensional array of survival time and event data
            corresponding to samples in all test folds. The array is of shape
            (folds, samples, target).

    Returns:
        float: 
    """


def basic_cv_fold(
    test_linear_predictor: npt.NDArray[np.float64],
    test_time: npt.NDArray[np.float64],
    test_event: npt.NDArray[np.int64],
    test_eta_hat: npt.NDArray[np.float64],
    train_linear_predictor: npt.NDArray[np.float64],
    train_time: npt.NDArray[np.float64],
    train_event: npt.NDArray[np.int64],
    score_function: Callable,
) -> float:
    """Basic CV scoring function based on the scoring function used.

    Args:
        test_linear_predictor (np.array): Linear predictors of a given test fold. X@beta.
        test_time (np.array): Sorted time points of the test fold.
        test_event (np.array): Event indicator of the test fold.
        test_eta_hat (np.array): Predicted linear predictors of a given test fold.
        train_linear_predictor (np.array): Linear predictors of the training fold.
        train_time (np.array): Sorted time points of the training fold.
        train_event (np.array): Event indicator of the training fold.
        score_function (Callable): Scoring function used to compute the negative log-likelihood.

    Returns:
        float: Scalar value of the mean partial log-likelihood for a given test fold.
    """

    test_fold_likelihood = -score_function(test_linear_predictor, test_time, test_event)

    return test_fold_likelihood


def basic_mse(
    test_linear_predictor: npt.NDArray[np.float64],
    test_time: npt.NDArray[np.float64],
    test_event: npt.NDArray[np.int64],
    test_eta_hat: npt.NDArray[np.float64],
    train_linear_predictor: npt.NDArray[np.float64],
    train_time: npt.NDArray[np.float64],
    train_event: npt.NDArray[np.int64],
    score_function: Callable,
) -> float:
    """Mean-squared error based CV scoring function.

    Args:
        test_linear_predictor (np.array): Linear predictors of a given test fold. X@beta.
        test_time (np.array): Sorted time points of the test fold.
        test_event (np.array): Event indicator of the test fold.
        test_eta_hat (np.array): Predicted linear predictors of a given test fold.
        train_linear_predictor (np.array): Linear predictors of the training fold.
        train_time (np.array): Sorted time points of the training fold.
        train_event (np.array): Event indicator of the training fold.
        score_function (Callable): Scoring function used to compute the negative log-likelihood.

    Returns:
        float: Scalar value of the mean partial log-likelihood for a given test fold.
    """

    test_negative_mse = np.negative(
        np.mean(np.square(test_eta_hat - test_linear_predictor))
    )

    return test_negative_mse


def vvh_cv_fold(
    test_linear_predictor: npt.NDArray[np.float64],
    test_time: npt.NDArray[np.float64],
    test_event: npt.NDArray[np.int64],
    test_eta_hat: npt.NDArray[np.float64],
    train_linear_predictor: npt.NDArray[np.float64],
    train_time: npt.NDArray[np.float64],
    train_event: npt.NDArray[np.int64],
    score_function: Callable,
) -> float:
    """Verweij and Van Houwelingen CV scoring function.

    Args:
        test_linear_predictor (np.array): Linear predictors of a given test fold. X@beta.
        test_time (np.array): Sorted time points of the test fold.
        test_event (np.array): Event indicator of the test fold.
        test_eta_hat (np.array): Predicted linear predictors of a given test fold.
        train_linear_predictor (np.array): Linear predictors of the training fold.
        train_time (np.array): Sorted time points of the training fold.
        train_event (np.array): Event indicator of the training fold.
        score_function (Callable): Scoring function used to compute the negative log-likelihood.

    Returns:
        float: Scalar value of the mean partial log-likelihood for a given test fold.
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
    test_linear_predictor: npt.NDArray[np.float64],
    test_time: npt.NDArray[np.float64],
    test_event: npt.NDArray[np.int64],
    score_function: Callable,
) -> float:
    """CV score computation using linear predictors (Dai et. al. 2019).

    Args:
        test_linear_predictor (np.array): Linear predictors of a given test fold. X@beta.
        test_time (np.array): Sorted time points of the test fold.
        test_event (np.array): Event indicator of the test fold.
        score_function (Callable): Scoring function used to compute the negative log-likelihood.

    Returns:
        float: Scalar value of the mean partial log-likelihood for a given test fold.
    """
    log_likelihood = -score_function(test_linear_predictor, test_time, test_event)
    return log_likelihood
