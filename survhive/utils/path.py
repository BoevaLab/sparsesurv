import sys
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from cv import _alpha_grid
from numba import jit
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_random_state
from typeguard import typechecked


@typechecked
@jit(nopython=True, cache=True)
def regularisation_path(
    X: ArrayLike,
    y: ArrayLike,
    *,
    l1_ratio: Union[float, ArrayLike] = 0.5,
    eps: float = 1e-3,
    n_alphas: int = 100,
    alphas: np.ndarray = None,
    Xy: ArrayLike = None,
    coef_init: np.ndarray = None,
    verbose: bool = False,
    return_n_iter: bool = False,
    positive: bool = False,
    **params,
) -> Tuple:
    """Compute estimator path with coordinate descent.

    Args:
        X (ArrayLike): Training data of shape (n_samples, n_features).
        y (ArrayLike): Target values of shape (n_samples,) or (n_samples, n_targets).
        l1_ratio (Union[float, ArrayLike], optional): Scaling between l1 and l2 penalties. 
            ``l1_ratio=1`` corresponds to the Lasso. Defaults to 0.5.
        eps (float, optional) : Length of the path. Defaults to 1e-3.
        n_alphas (int, optional): Number of alphas along the regularization path. 
            Defaults to 100.
        alphas (np.ndarray, optional): List of alphas where to compute the models.
            Defaults to None. If None alphas are set automatically.
        Xy (ArrayLike, optional): Dot product between X and y, of shape (n_features,) or 
            (n_features, n_targets). Defaults to None.
        coef_init (np.ndarray, optional): ndarray of shape (n_features, ), The initial values of the coefficients.
            Defaults to None.
        verbose (bool, optional): Verbosity .Defaults to False.
        return_n_iter (bool, optional): Whether to return the number of iterations or not.
            Defaults to False.
        positive (bool, optional): If set to True, forces coefficients to be positive.
            Defaults to False.
        **params : kwargs
            Keyword arguments passed to the coordinate descent solver.
    
    Returns:
        alphas : ndarray of shape (n_alphas,)
            The alphas along the path where models are computed.
        coefs : ndarray of shape (n_features, n_alphas) or \
                (n_targets, n_features, n_alphas)
            Coefficients along the path.
        dual_gaps : ndarray of shape (n_alphas,)
            The dual gaps at the end of the optimization for each alpha.
        n_iters : list of int
            The number of iterations taken by the coordinate descent optimizer to
            reach the specified tolerance for each alpha.
            (Is returned when ``return_n_iter`` is set to True).

    """

    sample_weight = params.pop("sample_weight", None)
    tol = params.pop("tol", 1e-4)
    max_iter = params.pop("max_iter", 1000)
    random_state = params.pop("random_state", None)

    if len(params) > 0:
        raise ValueError("Unexpected parameters in params", params.keys())

    n_samples, n_features = X.shape

    if alphas is None:
        alphas = _alpha_grid(
            X,
            y,
            Xy=Xy,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            eps=eps,
            n_alphas=n_alphas,
            copy_X=False,
        )
    elif len(alphas) > 1:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(random_state)

    coefs = np.empty((n_features, n_alphas), dtype=X.dtype)

    if coef_init is None:
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order="F")
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        # account for n_samples scaling in objectives between here and cd_fast
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples

        # TODO: modify this to match our model+optimiser function
        model = cd_fast.sparse_enet_coordinate_descent(
            w=coef_,
            alpha=l1_reg,
            beta=l2_reg,
            X_data=X.data,
            X_indices=X.indices,
            X_indptr=X.indptr,
            y=y,
            sample_weight=sample_weight,
            max_iter=max_iter,
            tol=tol,
            rng=rng,
            positive=positive,
        )

        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        # we correct the scale of the returned dual gap, as the objective
        # in cd_fast is n_samples * the objective in this docstring.
        dual_gaps[i] = dual_gap_ / n_samples
        n_iters.append(n_iter_)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print("Path: %03i out of %03i" % (i, n_alphas))
            else:
                sys.stderr.write(".")

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps
