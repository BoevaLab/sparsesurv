import numbers
from abc import abstractmethod
from collections.abc import Iterable
from functools import partial
from numbers import Real
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from celer import ElasticNet as CelerElasticNet
from sklearn.linear_model._coordinate_descent import LinearModelCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_consistent_length, check_scalar

from .compat import LOSS_FACTORY
from .opt import fmin_cgprox
from .screening import StrongScreener
from .utils import (
    _soft_threshold,
    inverse_transform_preconditioning,
    inverse_transform_survival,
)


def _get_alpha_max_l1(gradient: np.array, X: np.array):
    return np.max(np.abs(np.matmul(gradient.T, X)))


def _alpha_grid_l1_preconditioning(
    gradient: np.array,
    eps: float = 0.05,
    n_alphas: int = 100,
) -> np.array:
    """Compute the grid of alpha values for model parameter search
    Parameters

    Args:
        X (np.array): Array-like object of training data of shape (n_samples, n_features).
        y (np.array): Array-like object of target values of shape (n_samples,) or (n_samples, n_outputs).
        gradient ():
        hessian ():
        Xy (np.array, optional): Dot product of X and y arrays having shape (n_features,)
        or (n_features, n_outputs). Defaults to None.
        eps (float, optional): Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``. Defaults to 1e-1.
        n_alphas (int, optional): Number of alphas along the regularization path. Defaults to 100.

    Returns:
        np.array: Regularisation parameters to try for the model.
    """
    alpha_max = np.max(np.abs(gradient))
    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas
    alphas = np.round(
        np.logspace(
            np.log10((alpha_max) * eps),
            np.log10(alpha_max),
            num=n_alphas,
        )[::-1],
        decimals=10,
    )
    return alphas


def _alpha_grid_l1(
    X: np.array,
    gradient: np.array,
    hessian: np.array,
    l1_ratio: float = 1.0,
    eps: float = 0.05,
    n_alphas: int = 100,
) -> np.array:
    """Compute the grid of alpha values for model parameter search
    Parameters

    Args:
        X (np.array): Array-like object of training data of shape (n_samples, n_features).
        y (np.array): Array-like object of target values of shape (n_samples,) or (n_samples, n_outputs).
        gradient ():
        hessian ():
        Xy (np.array, optional): Dot product of X and y arrays having shape (n_features,)
        or (n_features, n_outputs). Defaults to None.
        eps (float, optional): Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``. Defaults to 1e-1.
        n_alphas (int, optional): Number of alphas along the regularization path. Defaults to 100.

    Returns:
        np.array: Regularisation parameters to try for the model.
    """
    alpha_max = _get_alpha_max_l1(gradient=gradient, X=X) / l1_ratio
    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    alphas = np.round(
        np.logspace(
            np.log10((alpha_max) * eps),
            np.log10(alpha_max),
            num=n_alphas,
        )[::-1],
        decimals=10,
    ) / (np.sum(hessian))
    return alphas


def regularisation_path_precond(
    X: np.array,
    y: np.array,
    X_test: np.array,
    model,
    tau: float,
    eps: float = 0.05,
    n_alphas: int = 100,
    alphas: np.array = None,
    max_first: bool = True,
) -> Tuple:
    """Compute estimator path with coordinate descent.

    Args:
        X (np.array): Training data of shape (n_samples, n_features).
        y (np.array): Target values of shape (n_samples,) or (n_samples, n_targets).
        X_test (np.array): Test data of shape (n_samples, n_features).
        model (object): The model object pre-initialised to fit the data for each alpha
            and learn the coefficients.
        l1_ratio (Union[float, np.array], optional): Scaling between l1 and l2 penalties.
            ``l1_ratio=1`` corresponds to the Lasso. Defaults to 0.5.
        eps (float, optional) : Length of the path. Defaults to 1e-3.
        n_alphas (int, optional): Number of alphas along the regularization path.
            Defaults to 100.
        alphas (np.array, optional): List of alphas where to compute the models.
            Defaults to None. If None alphas are set automatically.


    Returns:
        Tuple: Tuple of the dot products of train and test samples with the coefficients
            learned during training.
    """

    n_samples, n_features = X.shape
    test_samples, _ = X_test.shape
    time, event, eta_hat = inverse_transform_preconditioning(y)
    beta_previous = np.zeros(X.shape[1])
    gradient = model.gradient(
        coef=beta_previous,
        time=time,
        event=event,
        eta_hat=eta_hat,
        X=X,
        tau=tau,
    )
    if alphas is None:
        alphas = _alpha_grid_l1_preconditioning(
            gradient=gradient,
            eps=eps,
            n_alphas=n_alphas,
        )

    elif len(alphas) > 1:
        alphas = np.sort(alphas)[::-1]
    n_alphas = len(alphas)

    strong_screener = StrongScreener(p=X.shape[1], l1_ratio=1.0)
    eta_previous = np.zeros(X.shape[0])
    beta_previous = np.zeros(X.shape[1])
    coefs = np.zeros((n_features, n_alphas), dtype=X.dtype)
    train_eta = np.empty((n_samples, n_alphas), dtype=X.dtype)
    test_eta = np.empty((test_samples, n_alphas), dtype=X.dtype)
    active_variables = np.empty(0, dtype=int)
    for i, alpha in enumerate(alphas):
        # If this is a path, we skip the first alpha
        # since it is always sparse.
        if i == 0 and max_first:
            eta_new = eta_previous
            beta_new = beta_previous
            alpha_previous = alpha
            coefs[..., i] = beta_previous
            train_eta[..., i] = eta_new
            test_eta[..., i] = np.matmul(
                X_test[:, active_variables], beta_previous[active_variables]
            )
            continue
        previous_working_set = strong_screener.working_set
        if i > 0:
            strong_screener.compute_strong_set(
                gradient=model.gradient(
                    time=time,
                    event=event,
                    eta_hat=eta_hat,
                    X=X,
                    tau=tau,
                    coef=beta_previous,
                ),
                alpha=alpha,
                alpha_previous=alpha_previous,
            )

        if i == 0:
            _ = fmin_cgprox(
                f=partial(model.loss, time, event, eta_hat, X, tau),
                f_prime=partial(model.gradient, time, event, eta_hat, X, tau),
                g_prox=partial(_soft_threshold, alpha),
                x0=np.zeros(X.shape[1]),
                rtol=model.rtol,
                maxiter=model.maxiter,
                verbose=model.verbose,
                default_step_size=model.default_step_size,
            )
            beta_new = _.x
            X_working_set = X
            eta_new = X @ beta_new

        elif strong_screener.working_set.shape[0] == 0:
            if strong_screener.strong_set.shape[0] == 0:
                eta_new = eta_previous
                beta_new = beta_previous
                alpha_previous = alpha
                break

            else:
                warm_start_coef = np.zeros(strong_screener.strong_set.shape[0])
                X_working_set = X[:, strong_screener.strong_set]
                _ = fmin_cgprox(
                    f=partial(
                        model.loss, time, event, eta_hat, X_working_set, tau
                    ),
                    f_prime=partial(
                        model.gradient,
                        time,
                        event,
                        eta_hat,
                        X_working_set,
                        tau,
                    ),
                    g_prox=partial(_soft_threshold, alpha),
                    x0=warm_start_coef,
                    rtol=model.rtol,
                    maxiter=model.maxiter,
                    verbose=model.verbose,
                    default_step_size=model.default_step_size,
                )
                beta_new = np.zeros(X.shape[1])
                beta_new[strong_screener.strong_set] = _.x
                strong_screener.expand_working_set(np.where(beta_new != 0)[0])

        else:
            X_working_set = X[:, strong_screener.working_set]
            warm_start_coef = beta_previous[strong_screener.working_set]
            _ = fmin_cgprox(
                f=partial(
                    model.loss, time, event, eta_hat, X_working_set, tau
                ),
                f_prime=partial(
                    model.gradient, time, event, eta_hat, X_working_set, tau
                ),
                g_prox=partial(_soft_threshold, alpha),
                x0=warm_start_coef,
                rtol=model.rtol,
                maxiter=model.maxiter,
                verbose=model.verbose,
                default_step_size=model.default_step_size,
            )

            beta_new = np.zeros(X.shape[1])
            beta_new[strong_screener.working_set] = _.x

            while strong_screener.strong_set.shape[0] > 0:
                active_variables = np.where(beta_new != 0)[0]
                strong_screener.check_kkt_strong(
                    gradient=model.gradient(
                        time=time,
                        event=event,
                        eta_hat=eta_hat,
                        X=X[:, strong_screener.strong_set],
                        tau=tau,
                        coef=beta_new[strong_screener.strong_set],
                    ),
                    alpha=alpha,
                )
                if strong_screener.strong_kkt_violated.shape[0] == 0:
                    break

                warm_start_coef = np.zeros(X.shape[1])
                warm_start_coef[strong_screener.working_set] = _.x
                strong_screener.expand_working_set_with_kkt_violations()
                warm_start_coef = warm_start_coef[strong_screener.working_set]
                X_working_set = X[:, strong_screener.working_set]
                _ = fmin_cgprox(
                    f=partial(
                        model.loss, time, event, eta_hat, X_working_set, tau
                    ),
                    f_prime=partial(
                        model.gradient,
                        time,
                        event,
                        eta_hat,
                        X_working_set,
                        tau,
                    ),
                    g_prox=partial(_soft_threshold, alpha),
                    x0=warm_start_coef,
                    rtol=model.rtol,
                    maxiter=model.maxiter,
                    verbose=model.verbose,
                    default_step_size=model.default_step_size,
                )
            beta_new = np.zeros(X.shape[1])
            beta_new[strong_screener.working_set] = _.x

            if model.check_global_kkt and i > 0:
                while True:
                    active_variables = np.where(beta_new != 0)[0]
                    strong_screener.check_kkt_all(
                        gradient=model.gradient(
                            time=time,
                            event=event,
                            eta_hat=eta_hat,
                            X=X,
                            tau=tau,
                            coef=beta_new,
                        ),
                        alpha=alpha,
                    )
                    if strong_screener.any_kkt_violated.shape[0] == 0:
                        break
                    warm_start_coef = np.zeros(X.shape[1])
                    warm_start_coef[strong_screener.working_set] = _.x
                    strong_screener.expand_working_set_with_overall_violations()
                    warm_start_coef = warm_start_coef[
                        strong_screener.working_set
                    ]
                    X_working_set = X[:, strong_screener.working_set]
                    _ = fmin_cgprox(
                        f=partial(
                            model.loss,
                            time,
                            event,
                            eta_hat,
                            X_working_set,
                            tau,
                        ),
                        f_prime=partial(
                            model.gradient,
                            time,
                            event,
                            eta_hat,
                            X_working_set,
                            tau,
                        ),
                        g_prox=partial(_soft_threshold, alpha),
                        x0=warm_start_coef,
                        rtol=model.rtol,
                        maxiter=model.maxiter,
                        verbose=model.verbose,
                        default_step_size=model.default_step_size,
                    )

        active_variables = np.where(beta_new != 0)[0]
        eta_previous = eta_new
        beta_previous = beta_new
        strong_screener.working_set = previous_working_set
        strong_screener.expand_working_set(active_variables)
        alpha_previous = alpha
        coefs[..., i] = beta_previous
        train_eta[..., i] = eta_previous
        test_eta[..., i] = np.matmul(
            X_test[:, active_variables], beta_previous[active_variables]
        )
    return coefs, train_eta, test_eta


def regularisation_path(
    X: np.array,
    y: np.array,
    X_test: np.array,
    model,
    l1_ratio: float = 1.0,
    eps: float = 0.05,
    n_alphas: int = 100,
    alphas: np.array = None,
    max_first: bool = True,
) -> Tuple:
    """Compute estimator path with coordinate descent.

    Args:
        X (np.array): Training data of shape (n_samples, n_features).
        y (np.array): Target values of shape (n_samples,) or (n_samples, n_targets).
        X_test (np.array): Test data of shape (n_samples, n_features).
        model (object): The model object pre-initialised to fit the data for each alpha
            and learn the coefficients.
        l1_ratio (Union[float, np.array], optional): Scaling between l1 and l2 penalties.
            ``l1_ratio=1`` corresponds to the Lasso. Defaults to 0.5.
        eps (float, optional) : Length of the path. Defaults to 1e-3.
        n_alphas (int, optional): Number of alphas along the regularization path.
            Defaults to 100.
        alphas (np.array, optional): List of alphas where to compute the models.
            Defaults to None. If None alphas are set automatically.


    Returns:
        Tuple: Tuple of the dot products of train and test samples with the coefficients
            learned during training.
    """

    n_samples, n_features = X.shape
    test_samples, _ = X_test.shape
    time, event = inverse_transform_survival(y)
    eta_previous = np.zeros(X.shape[0])

    gradient, hessian = model.gradient(
        linear_predictor=eta_previous,
        time=time,
        event=event,
    )

    if alphas is None:
        alphas = _alpha_grid_l1(
            X=X,
            gradient=gradient,
            hessian=hessian,
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
        )
    elif len(alphas) > 1:
        alphas = np.sort(alphas)[::-1]
    n_alphas = len(alphas)
    optimiser = CelerElasticNet(
        l1_ratio=l1_ratio,
        fit_intercept=False,
        warm_start=True,
        verbose=model.verbose,
        max_iter=model.inner_solver_max_iter,
        max_epochs=model.inner_solver_max_epochs,
        p0=model.inner_solver_p0,
        prune=model.inner_solver_prune,
    )
    strong_screener = StrongScreener(p=X.shape[1], l1_ratio=l1_ratio)
    eta_previous = np.zeros(X.shape[0])
    beta_previous = np.zeros(X.shape[1])
    coefs = np.zeros((n_features, n_alphas), dtype=X.dtype)
    train_eta = np.empty((n_samples, n_alphas), dtype=X.dtype)
    test_eta = np.empty((test_samples, n_alphas), dtype=X.dtype)
    eta_previous_alpha = np.zeros(X.shape[0])
    active_variables = np.empty(0, dtype=int)
    for i, alpha in enumerate(alphas):
        # If this is a path, we skip the first alpha
        # since it is always sparse.
        if i == 0 and max_first:
            eta_new = eta_previous
            beta_new = beta_previous
            alpha_previous = alpha
            coefs[..., i] = beta_previous
            train_eta[..., i] = eta_new
            test_eta[..., i] = np.matmul(
                X_test[:, active_variables], beta_previous[active_variables]
            )
            continue
        previous_working_set = strong_screener.working_set
        optimiser.__setattr__("alpha", alpha)
        for q in range(model.n_irls_iter):
            gradient: np.array
            hessian: np.array
            gradient, hessian = model.gradient(
                linear_predictor=eta_previous,
                time=time,
                event=event,
            )
            hessian_mask = np.logical_and(
                (hessian > 0).astype(bool), np.logical_not(np.isnan(hessian))
            )
            inverse_hessian = np.zeros(gradient.shape[0])
            inverse_hessian[hessian_mask] = 1 / hessian[hessian_mask]
            weights = hessian
            weights_scaled = weights * (
                hessian_mask.shape[0] / np.sum(weights)
            )
            y_irls = eta_previous - inverse_hessian * gradient
            # Calculate strong set if this is not the first alpha
            # and the first IRLS iteration for this alpha.
            if i > 0 and q < 1:
                strong_screener.compute_strong_set(
                    gradient=1
                    / n_samples
                    * np.abs(
                        np.matmul(
                            X.T,
                            (y_irls - eta_previous_alpha) * weights_scaled,
                        )
                    ),
                    alpha=alpha,
                    alpha_previous=alpha_previous,
                )

            # For the first alpha, we calculate our optimiser
            # without any screening.
            if i == 0:
                optimiser.fit(
                    X=X
                    * np.sqrt(weights_scaled)
                    .repeat(X.shape[1])
                    .reshape((X.shape[0], X.shape[1])),
                    y=np.sqrt(weights_scaled) * y_irls,
                )
                beta_new = optimiser.coef_
                X_working_set = X
            # For the any alpha after the maximum,
            # (indicated by a sparse working set), we first
            # check whether there is a non-sparse strong set.
            # If not: We break without checking KKT to save time.
            # If yes: We fit only on the strong set.
            elif strong_screener.working_set.shape[0] == 0:
                if strong_screener.strong_set.shape[0] == 0:
                    eta_new = eta_previous
                    beta_new = beta_previous
                    alpha_previous = alpha
                    break

                else:
                    warm_start_coef = np.zeros(
                        strong_screener.strong_set.shape[0]
                    )
                    optimiser.coef_ = warm_start_coef
                    X_working_set = X[:, strong_screener.strong_set]
                    optimiser.fit(
                        X=X_working_set
                        * np.sqrt(weights_scaled)
                        .repeat(X_working_set.shape[1])
                        .reshape(
                            (X_working_set.shape[0], X_working_set.shape[1])
                        ),
                        y=np.sqrt(weights_scaled) * y_irls,
                    )
                    beta_new = np.zeros(X.shape[1])
                    beta_new[strong_screener.strong_set] = optimiser.coef_
                    strong_screener.expand_working_set(
                        np.where(beta_new != 0)[0]
                    )

            # For any alpha that is not the first non-sparse alpha,
            # we first fit only with the working
            # set and only then check potential strong set violations.
            else:
                X_working_set = X[:, strong_screener.working_set]
                warm_start_coef = beta_previous[strong_screener.working_set]
                optimiser.coef_ = warm_start_coef
                optimiser.fit(
                    X=X_working_set
                    * np.sqrt(weights_scaled)
                    .repeat(X_working_set.shape[1])
                    .reshape((X_working_set.shape[0], X_working_set.shape[1])),
                    y=np.sqrt(weights_scaled) * y_irls,
                )

                beta_new = np.zeros(X.shape[1])
                beta_new[strong_screener.working_set] = optimiser.coef_

                # Check KKT for strong set here.
                while True:
                    eta_new = optimiser.predict(X_working_set)
                    strong_screener.check_kkt_strong(
                        gradient=1
                        / n_samples
                        * np.abs(
                            np.matmul(
                                X[:, strong_screener.strong_set].T,
                                (y_irls - eta_new) * weights_scaled,
                            )
                        ),
                        alpha=alpha,
                    )
                    if strong_screener.strong_kkt_violated.shape[0] > 0:
                        warm_start_coef = np.zeros(X.shape[1])
                        warm_start_coef[
                            strong_screener.working_set
                        ] = optimiser.coef_
                        strong_screener.expand_working_set_with_kkt_violations()
                        warm_start_coef = warm_start_coef[
                            strong_screener.working_set
                        ]
                        optimiser.coef_ = warm_start_coef
                        X_working_set = X[:, strong_screener.working_set]
                        optimiser.fit(
                            X=X_working_set
                            * np.sqrt(weights_scaled)
                            .repeat(X_working_set.shape[1])
                            .reshape(
                                (
                                    X_working_set.shape[0],
                                    X_working_set.shape[1],
                                )
                            ),
                            y=np.sqrt(weights_scaled) * y_irls,
                        )
                        continue
                    else:
                        break
                beta_new = np.zeros(X.shape[1])
                beta_new[strong_screener.working_set] = optimiser.coef_

            # We stop running IRLS if:
            # - The convergence threshold is reached.
            # - The coefficients are fully sparse.
            # - The number of maximum of IRLS iterations
            # has been reached.
            if (
                np.max(np.abs(beta_new - beta_previous))
                < model.tol * np.max(np.abs(beta_new))
                or np.max(np.abs(beta_new)) == 0.0
                or (q == (model.n_irls_iter - 1))
            ):
                # We check the complete set for KKT violations if all
                # of the below hold:
                # - Beta is not fully sparse.
                # - The user has indicated that they want to check it.
                if (
                    np.max(np.abs(beta_new)) > 0.0
                    and model.check_global_kkt
                    and i > 0
                ):
                    while True:
                        eta_new: np.array = optimiser.predict(X_working_set)
                        strong_screener.check_kkt_all(
                            gradient=1
                            / n_samples
                            * np.abs(
                                np.matmul(
                                    X.T,
                                    (y_irls - eta_new) * weights_scaled,
                                )
                            ),
                            alpha=alpha,
                        )
                        if strong_screener.any_kkt_violated.shape[0] > 0:
                            warm_start_coef = np.zeros(X.shape[1])
                            warm_start_coef[
                                strong_screener.working_set
                            ] = optimiser.coef_
                            strong_screener.expand_working_set_with_overall_violations()
                            warm_start_coef = warm_start_coef[
                                strong_screener.working_set
                            ]
                            optimiser.coef_ = warm_start_coef
                            X_working_set = X[:, strong_screener.working_set]
                            optimiser.fit(
                                X=X_working_set
                                * np.sqrt(weights_scaled)
                                .repeat(X_working_set.shape[1])
                                .reshape(
                                    (
                                        X_working_set.shape[0],
                                        X_working_set.shape[1],
                                    )
                                ),
                                y=np.sqrt(weights_scaled) * y_irls,
                            )
                            continue
                        else:
                            break
                if strong_screener.working_set.shape[0] > 0:
                    beta_new = np.zeros(X.shape[1])
                    beta_new[strong_screener.working_set] = optimiser.coef_
                eta_previous = eta_new
                eta_previous_alpha = eta_previous
                beta_previous = beta_new
                active_variables = np.where(beta_new != 0)[0]
                strong_screener.working_set = previous_working_set
                strong_screener.expand_working_set(active_variables)
                alpha_previous = alpha
                break
            # If IRLS has not yet converged, we update all relevant
            # variables and move on to the next IRLS iteration.
            else:
                eta_new: np.array = optimiser.predict(X_working_set)
                eta_previous = eta_new
                beta_previous = beta_new
                active_variables = np.where(beta_new != 0)[0]
                alpha_previous = alpha
        coefs[..., i] = beta_previous
        train_eta[..., i] = eta_new
        test_eta[..., i] = np.matmul(
            X_test[:, active_variables], beta_previous[active_variables]
        )
    return coefs, train_eta, test_eta


def alpha_path_eta(
    X: np.array,
    y: np.array,
    model: object,
    gradient: np.array,
    hessian: np.array,
    train: List[int],
    test: List[int],
    alphas: np.array = None,
    n_alphas: int = 100,
    l1_ratio: float = 1.0,
    eps: float = 0.05,
) -> Tuple:
    """Returns the dot product of samples and coefs for the models computed by 'path'.

    Args:
        X (np.array): Training data of shape (n_samples, n_features).
        y (np.array): Target values of shape (n_samples,) or (n_samples, n_targets).
        model (object): The model object pre-initialised to fit the data for each alpha
            and learn the coefficients.
        sample_weight (np.array): Sample weights of shape (n_samples,). Pass None if
            there are no weights.
        train (List): The indices of the train set.
        test (List): The indices of the test set.
        alphas (np.array, optional): Array of float that is used for cross-validation. If not
        provided, computed using 'path'. Defaults to None.
        l1_ratio (Union[float,List], optional): Scaling between
        l1 and l2 penalties. For ``l1_ratio = 0`` the penalty is an
        L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
        < l1_ratio < 1``, the penalty is a combination of L1 and L2. Defaults to 1.

    Returns:
        Tuple: Tuple of the dot products of train and test samples with the coefficients
            learned during training, and the associated target values for train and test.
    """
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    train_order = np.argsort(np.abs(y_train), kind="stable")
    X_train = X_train[train_order, :]
    y_train = y_train[train_order]

    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    for array, array_input in (
        (X_train, X),
        (y_train, y),
        (X_test, X),
        (y_test, y),
    ):
        if array.base is not array_input and not array.flags["WRITEABLE"]:
            array.setflags(write=True)

    if alphas is None:
        alphas = _alpha_grid_l1(
            X=X,
            gradient=gradient,
            hessian=hessian,
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
        )
    elif len(alphas) > 1:
        alphas = np.sort(alphas)[::-1]

    train_eta = np.empty((n_samples_train, n_alphas), dtype=X.dtype)
    test_eta = np.empty((n_samples_test, n_alphas), dtype=X.dtype)

    model.__setattr__("warm_start", True)
    model.__setattr__("l1_ratio", l1_ratio)

    _, train_eta, test_eta = regularisation_path(
        X=X_train,
        y=y_train,
        X_test=X_test,
        model=model,
        l1_ratio=l1_ratio,
        eps=eps,
        n_alphas=n_alphas,
        alphas=alphas,
        max_first=True,
    )

    return train_eta, test_eta, y_train, y_test


def alpha_path_eta_precond(
    X: np.array,
    y: np.array,
    model: object,
    gradient: np.array,
    train: List[int],
    test: List[int],
    alphas: np.array = None,
    n_alphas: int = 100,
    tau: float = 1.0,
    eps: float = 0.05,
) -> Tuple:
    """Returns the dot product of samples and coefs for the models computed by 'path'.

    Args:
        X (np.array): Training data of shape (n_samples, n_features).
        y (np.array): Target values of shape (n_samples,) or (n_samples, n_targets).
        model (object): The model object pre-initialised to fit the data for each alpha
            and learn the coefficients.
        sample_weight (np.array): Sample weights of shape (n_samples,). Pass None if
            there are no weights.
        train (List): The indices of the train set.
        test (List): The indices of the test set.
        alphas (np.array, optional): Array of float that is used for cross-validation. If not
        provided, computed using 'path'. Defaults to None.
        l1_ratio (Union[float,List], optional): Scaling between
        l1 and l2 penalties. For ``l1_ratio = 0`` the penalty is an
        L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
        < l1_ratio < 1``, the penalty is a combination of L1 and L2. Defaults to 1.

    Returns:
        Tuple: Tuple of the dot products of train and test samples with the coefficients
            learned during training, and the associated target values for train and test.
    """
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    time, _, _ = inverse_transform_preconditioning(y_train)

    train_order = np.argsort(time, kind="stable")
    X_train = X_train[train_order, :]
    y_train = y_train[train_order]

    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    for array, array_input in (
        (X_train, X),
        (y_train, y),
        (X_test, X),
        (y_test, y),
    ):
        if array.base is not array_input and not array.flags["WRITEABLE"]:
            array.setflags(write=True)

    if alphas is None:
        alphas = _alpha_grid_l1(
            X=X, gradient=gradient, eps=eps, n_alphas=n_alphas
        )
    elif len(alphas) > 1:
        alphas = np.sort(alphas)[::-1]

    train_eta = np.empty((n_samples_train, n_alphas), dtype=X.dtype)
    test_eta = np.empty((n_samples_test, n_alphas), dtype=X.dtype)

    model.__setattr__("warm_start", True)
    model.__setattr__("tau", tau)

    _, train_eta, test_eta = regularisation_path_precond(
        X=X_train,
        y=y_train,
        X_test=X_test,
        model=model,
        tau=tau,
        eps=eps,
        n_alphas=n_alphas,
        alphas=alphas,
        max_first=True,
    )

    return train_eta, test_eta, y_train, y_test


class RegularizedLinearSurvivalModelCV(LinearModelCV):
    """Cross validation class with custom scoring functions."""

    @abstractmethod
    def __init__(
        self,
        eps: float = 0.05,
        n_alphas: int = 100,
        alphas: np.array = None,
        l1_ratios: Union[float, np.array] = None,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        """Constructor.

        Args:
            optimiser (str): Optimiser to use for model fitting. See OPTIMISERFACTORY for
                options.
            cv_score_method (str): CV scoring method to use for model selection. One of
                ["linear_predictor","regular","vvh"]. Defaults to "linear_predictor".
            eps (float, optional): Length of the path. ``eps=1e-3`` means that
                ``alpha_min / alpha_max = 1e-3``. Defaults to 1e-3.
            n_alphas (int, optional): Number of alphas along the regularization path.
                Defaults to 100.
            alphas (np.array, optional): Array of float that is used for cross-validation. If not
                provided, computed using 'path'. Defaults to None.
            l1_ratios (Union[float,np.array], optional): Scaling between
                l1 and l2 penalties. For ``l1_ratio = 0`` the penalty is an
                L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
                < l1_ratio < 1``, the penalty is a combination of L1 and L2. Defaults to None.
            tol (float, optional): The tolerance for the optimization. Defaults to 1e-4.
            copy_X (bool, optional): Creates a copy of X if True. Defaults to True.
            cv (Union[int,object], optional): Cross validation splitting strategy.
                Defaults to None, which uses the default 5-fold cv. Can also pass cv-generator.
            n_jobs (int, optional): Number of CPUs to use during the cross validation. Defaults to None.
            random_state (int, optional): The seed of the pseudo random number generator that selects a random
                feature to update. Defaults to None.
        """

        super().__init__(
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=False,
            max_iter=None,
            tol=None,
            copy_X=None,
            cv=None,
            n_jobs=None,
            random_state=None,
        )

        self.l1_ratios = l1_ratios
        if isinstance(self.l1_ratios, float):
            self.l1_ratios = list(self.l1_ratios)

        cv = 5 if cv is None else cv
        if isinstance(cv, numbers.Integral):
            self.cv = StratifiedKFold(
                cv, shuffle=True, random_state=random_state
            )
        elif isinstance(cv, Iterable):
            self.cv = _CVIterableWrapper(cv)
        elif hasattr(cv, "split"):
            self.cv = cv
        else:
            raise ValueError(
                "Expected cv to be an integer, sklearn model selection object or an iterable"
            )

        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self,
        X: np.array,
        y: np.array,
    ) -> object:
        """Fit linear model.
        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Args:
            X (np.array): Training data of shape (n_samples, n_features).
            y (np.array): Target values of shape (n_samples,) or (n_samples, n_targets).
            sample_weight (Union[float,np.array]): Sample weights used for fitting and evaluation of the weighted
                mean squared error of each cv-fold. Has shape (n_samples,) and defaults
                to None.

        Returns:
            self(object): Returns an instance of fitted model.
        """
        time: np.array
        event: np.array
        time, event = inverse_transform_survival(y=y)
        sorted_indices: np.array = np.argsort(a=time, kind="stable")
        time_sorted: np.array = time[sorted_indices]
        event_sorted: np.array = event[sorted_indices]
        X_sorted: np.array = X[sorted_indices, :]
        y_sorted: np.array = y[sorted_indices]
        self._validate_params()
        check_consistent_length(X_sorted, y_sorted)

        model = self._get_estimator()
        path_params = self.get_params()
        path_params.pop("fit_intercept", None)

        if "l1_ratios" in path_params:
            l1_ratios = np.atleast_1d(path_params["l1_ratios"])

            path_params["l1_ratios"] = l1_ratios
        else:
            l1_ratios = [
                1.0,
            ]

        path_params.pop("cv", None)
        path_params.pop("n_jobs", None)

        alphas = self.alphas
        n_l1_ratios = len(l1_ratios)

        check_scalar_alpha = partial(
            check_scalar,
            target_type=Real,
            min_val=0.0,
            include_boundaries="left",
        )

        gradient, hessian = model.gradient(
            linear_predictor=np.zeros(X_sorted.shape[0]),
            time=time_sorted,
            event=event_sorted,
        )
        if alphas is None:
            alphas = [
                _alpha_grid_l1(
                    X=X_sorted,
                    gradient=gradient,
                    hessian=hessian,
                    l1_ratio=l1_ratio,
                    eps=self.eps,
                    n_alphas=self.n_alphas,
                )
                for l1_ratio in l1_ratios
            ]
        else:
            for index, alpha in enumerate(alphas):
                check_scalar_alpha(alpha, f"alphas[{index}]")

            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratios, 1))

        n_alphas = len(alphas[0])
        path_params.update({"n_alphas": n_alphas})

        folds = list(self.cv.split(X_sorted, (y_sorted > 0).astype(int)))
        best_pl_score = np.inf

        jobs = (
            delayed(alpha_path_eta)(
                X=X_sorted,
                y=y_sorted,
                model=model,
                gradient=gradient,
                hessian=hessian,
                train=train,
                test=test,
                alphas=this_alphas,
                n_alphas=self.n_alphas,
                l1_ratio=this_l1_ratio,
                eps=self.eps,
            )
            for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
            for train, test in folds
        )
        eta_path = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(jobs)
        # TODO DW: Since we're not using train eta now,
        # we can actually remove this.
        train_eta_folds, test_eta_folds, _, test_y_folds = zip(*eta_path)
        n_folds = int(len(train_eta_folds) / len(l1_ratios))

        mean_cv_score_l1 = []
        mean_cv_score = []

        for i in range(len(l1_ratios)):
            mean_cv_score_l1 = []
            test_eta = test_eta_folds[n_folds * i : n_folds * (i + 1)]
            test_y = test_y_folds[n_folds * i : n_folds * (i + 1)]
            test_eta_method = np.concatenate(test_eta)
            test_y_method = np.concatenate(test_y)
            test_time, test_event = inverse_transform_survival(test_y_method)
            for j in range(len(alphas[i])):
                likelihood = model.loss(
                    test_eta_method[:, j], test_time, test_event
                )
                if np.isnan(likelihood):
                    mean_cv_score_l1.append(np.inf)
                else:
                    mean_cv_score_l1.append(likelihood)
            mean_cv_score.append(mean_cv_score_l1)

        self.pl_path_ = mean_cv_score
        for l1_ratio, l1_alphas, pl_alphas in zip(
            l1_ratios, alphas, mean_cv_score
        ):
            i_best_alpha = np.argmin(pl_alphas)
            this_best_pl = pl_alphas[i_best_alpha]
            if this_best_pl < best_pl_score:
                best_alpha = l1_alphas[i_best_alpha]
                best_l1_ratio = l1_ratio
                best_pl_score = this_best_pl

        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
            if n_l1_ratios == 1:
                self.alphas_ = self.alphas_[0]

        else:
            self.alphas_ = np.asarray(alphas[0])

        # Refit the model with the parameters selected
        common_params = {
            name: value
            for name, value in self.get_params().items()
            if name in model.get_params()
        }
        model.set_params(**common_params)
        model.alpha = best_alpha
        model.l1_ratio = best_l1_ratio
        model.fit(X_sorted, y_sorted)
        model.check_global_kkt = False
        if not hasattr(self, "l1_ratio"):
            del self.l1_ratio_
        self.model = model
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

        return self

    def predict_survival_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
        return self.model.predict_survival_function(X, time)


class RegularizedPreconditionedLinearSurvivalModelCV(LinearModelCV):
    """Cross validation class with custom scoring functions."""

    @abstractmethod
    def __init__(
        self,
        eps: float = 0.05,
        n_alphas: int = 100,
        alphas: np.array = None,
        taus: Union[float, np.array] = None,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        """Constructor.

        Args:
            optimiser (str): Optimiser to use for model fitting. See OPTIMISERFACTORY for
                options.
            cv_score_method (str): CV scoring method to use for model selection. One of
                ["linear_predictor","regular","vvh"]. Defaults to "linear_predictor".
            eps (float, optional): Length of the path. ``eps=1e-3`` means that
                ``alpha_min / alpha_max = 1e-3``. Defaults to 1e-3.
            n_alphas (int, optional): Number of alphas along the regularization path.
                Defaults to 100.
            alphas (np.array, optional): Array of float that is used for cross-validation. If not
                provided, computed using 'path'. Defaults to None.
            l1_ratios (Union[float,np.array], optional): Scaling between
                l1 and l2 penalties. For ``l1_ratio = 0`` the penalty is an
                L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
                < l1_ratio < 1``, the penalty is a combination of L1 and L2. Defaults to None.
            tol (float, optional): The tolerance for the optimization. Defaults to 1e-4.
            copy_X (bool, optional): Creates a copy of X if True. Defaults to True.
            cv (Union[int,object], optional): Cross validation splitting strategy.
                Defaults to None, which uses the default 5-fold cv. Can also pass cv-generator.
            n_jobs (int, optional): Number of CPUs to use during the cross validation. Defaults to None.
            random_state (int, optional): The seed of the pseudo random number generator that selects a random
                feature to update. Defaults to None.
        """

        super().__init__(
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=False,
            max_iter=None,
            tol=None,
            copy_X=None,
            cv=None,
            n_jobs=None,
            random_state=None,
        )

        self.taus = taus
        if isinstance(self.taus, float):
            self.taus = list(self.taus)

        cv = 5 if cv is None else cv
        if isinstance(cv, numbers.Integral):
            self.cv = StratifiedKFold(
                cv, shuffle=True, random_state=random_state
            )
        elif isinstance(cv, Iterable):
            self.cv = _CVIterableWrapper(cv)
        elif hasattr(cv, "split"):
            self.cv = cv
        else:
            raise ValueError(
                "Expected cv to be an integer, sklearn model selection object or an iterable"
            )

        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self,
        X: np.array,
        y: np.array,
    ) -> object:
        """Fit linear model.
        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Args:
            X (np.array): Training data of shape (n_samples, n_features).
            y (np.array): Target values of shape (n_samples,) or (n_samples, n_targets).
            sample_weight (Union[float,np.array]): Sample weights used for fitting and evaluation of the weighted
                mean squared error of each cv-fold. Has shape (n_samples,) and defaults
                to None.

        Returns:
            self(object): Returns an instance of fitted model.
        """
        time: np.array
        event: np.array
        time, event, y_teacher = inverse_transform_preconditioning(y=y)
        sorted_indices: np.array = np.argsort(a=time, kind="stable")
        time_sorted: np.array = time[sorted_indices]
        event_sorted: np.array = event[sorted_indices]
        y_teacher_sorted = y_teacher[sorted_indices]
        X_sorted: np.array = X[sorted_indices, :]
        y_sorted: np.array = y[sorted_indices]
        self._validate_params()
        check_consistent_length(X_sorted, y_sorted)

        model = self._get_estimator()
        path_params = self.get_params()
        path_params.pop("fit_intercept", None)

        if "taus" in path_params:
            taus = np.atleast_1d(path_params["taus"])

            path_params["taus"] = taus
        else:
            taus = [
                1.0,
            ]

        path_params.pop("cv", None)
        path_params.pop("n_jobs", None)

        alphas = self.alphas
        n_taus = len(taus)

        check_scalar_alpha = partial(
            check_scalar,
            target_type=Real,
            min_val=0.0,
            include_boundaries="left",
        )

        if alphas is None:
            alphas = [
                _alpha_grid_l1_preconditioning(
                    gradient=model.gradient(
                        coef=np.zeros(X.shape[1]),
                        time=time_sorted,
                        event=event_sorted,
                        eta_hat=y_teacher_sorted,
                        X=X_sorted,
                        tau=tau,
                    ),
                    eps=self.eps,
                    n_alphas=self.n_alphas,
                )
                for tau in taus
            ]
        else:
            for index, alpha in enumerate(alphas):
                check_scalar_alpha(alpha, f"alphas[{index}]")

            alphas = np.tile(np.sort(alphas)[::-1], (n_taus, 1))

        n_alphas = len(alphas[0])
        path_params.update({"n_alphas": n_alphas})

        folds = list(self.cv.split(X_sorted, (event).astype(int)))
        best_pl_score = np.inf

        jobs = (
            delayed(alpha_path_eta_precond)(
                X=X_sorted,
                y=y_sorted,
                model=model,
                gradient=model.gradient(
                    coef=np.zeros(X.shape[1]),
                    time=time_sorted,
                    event=event_sorted,
                    eta_hat=y_teacher_sorted,
                    X=X_sorted,
                    tau=this_tau,
                ),
                train=train,
                test=test,
                alphas=this_alphas,
                n_alphas=self.n_alphas,
                tau=this_tau,
                eps=self.eps,
            )
            for this_tau, this_alphas in zip(taus, alphas)
            for train, test in folds
        )
        eta_path = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(jobs)
        # TODO DW: Since we're not using train eta now,
        # we can actually remove this.
        train_eta_folds, test_eta_folds, _, test_y_folds = zip(*eta_path)
        n_folds = int(len(train_eta_folds) / n_taus)

        mean_cv_score_tau = []
        mean_cv_score = []

        for i in range(n_taus):
            mean_cv_score_tau = []
            test_eta = test_eta_folds[n_folds * i : n_folds * (i + 1)]
            test_y = test_y_folds[n_folds * i : n_folds * (i + 1)]
            test_eta_method = np.concatenate(test_eta)
            test_y_method = np.concatenate(test_y)
            test_time, test_event, _ = inverse_transform_preconditioning(
                test_y_method
            )
            for j in range(len(alphas[i])):
                likelihood = LOSS_FACTORY[model.tie_correction](
                    test_eta_method[:, j], test_time, test_event
                )
                if np.isnan(likelihood):
                    mean_cv_score_tau.append(np.inf)
                else:
                    mean_cv_score_tau.append(likelihood)
            mean_cv_score.append(mean_cv_score_tau)

        self.pl_path_ = mean_cv_score
        for tau, tau_alphas, pl_alphas in zip(taus, alphas, mean_cv_score):
            i_best_alpha = np.argmin(pl_alphas)
            this_best_pl = pl_alphas[i_best_alpha]
            if this_best_pl < best_pl_score:
                best_alpha = tau_alphas[i_best_alpha]
                best_tau = tau
                best_pl_score = this_best_pl

        self.tau = best_tau
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
            if n_taus == 1:
                self.alphas_ = self.alphas_[0]

        else:
            self.alphas_ = np.asarray(alphas[0])

        # Refit the model with the parameters selected
        common_params = {
            name: value
            for name, value in self.get_params().items()
            if name in model.get_params()
        }
        model.set_params(**common_params)
        model.alpha = best_alpha
        model.tau = best_tau
        model.check_global_kkt = False
        model.fit(X_sorted, y_sorted)
        self.model = model
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

        return self

    def predict_survival_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
        return self.model.predict_survival_function(X, time)
