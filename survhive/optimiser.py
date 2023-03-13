import warnings
from typing import Callable, List, Optional, Union, Dict

import numpy as np
from celer import ElasticNet as CelerElasticNet
from celer import GroupLasso as CelerGroupLasso
from numba import jit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet as ScikitElasticNet
from typeguard import typechecked
from sklearn.linear_model.base import LinearModel

OPTIMISERS: Dict[str, LinearModel] = {
    "scikit_elastic_net": ScikitElasticNet,
    "scikit_lasso": ScikitElasticNet,
    "celer_elastic_net": CelerElasticNet,
    "celer_lasso": CelerElasticNet,
    "celer_group_lasso": CelerGroupLasso,
}


@typechecked
@jit(nopython=True, cache=True, fastmath=True)
def backtracking_line_search(
    loss: Callable,
    sample_weight: np.array,
    y: np.array,
    gradient_direction: np.array,
    current_prediction: np.array,
    previous_prediction: np.array,
    previous_loss: float,
    reduction_factor: float = 0.5,
    max_learning_rate: float = 1.0,
) -> float:
    current_learning_rate: float = max_learning_rate
    while loss(
        eta=(1 - current_learning_rate) * previous_prediction + current_prediction,
        y=y,
        sample_weight=sample_weight,
    ) > previous_loss - (current_learning_rate**2 / 2) * np.sum(
        np.square(gradient_direction)
    ):
        current_learning_rate *= reduction_factor
    return current_learning_rate


@typechecked
def _validate_optimiser(
    optimiser: str,
    alpha: float,
    l1_ratio: Optional[float],
    groups: Optional[List[int]],
    warm_start: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7,
    verbose: int = 0,
) -> LinearModel:
    optimiser_: LinearModel
    if alpha is not None:
        if optimiser not in ["scikit_elastic_net", "celer_elastic_net"]:
            raise ValueError(
                f"Expected `l1_ratio` to be in [0, 1], since `optimiser` is {optimiser}."
            )
        optimiser_: LinearModel = OPTIMISERS[optimiser](
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            warm_start=warm_start,
        )
    else:
        if groups is not None:
            if optimiser not in ["celer_group_lasso"]:
                raise ValueError(
                    f"Expected `groups` to be not `None`, since `optimiser` is {optimiser}."
                )
            optimiser_ = OPTIMISERS[optimiser](
                alpha=alpha,
                groups=groups,
                fit_intercept=False,
                verbose=verbose,
                tol=tol,
                max_iter=max_iter,
                warm_start=warm_start,
            )

        optimiser_ = OPTIMISERS[optimiser](
            alpha=alpha,
            l1_ratio=1.0,
            fit_intercept=False,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            warm_start=warm_start,
        )
    return optimiser_


@typechecked
class Optimiser:
    def __init__(
        self,
        grad: Callable,
        loss: Callable,
        alpha: float,
        optimiser: str,
        l1_ratio: float,
        groups: Optional[List[List[int]]] = None,
        scale_group: Optional[str] = "group_length",
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 100,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.grad: Callable = grad
        self.loss: Callable = loss
        self.alpha: float = alpha
        self.optimiser: str = optimiser
        self.l1_ratio: float = l1_ratio
        self.groups: Optional[List[List[int]]] = groups
        self.scale_group: Optional[str] = scale_group
        self.line_search: bool = line_search
        self.line_search_reduction_factor: float = line_search_reduction_factor
        self.warm_start: bool = warm_start
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.verbose: int = verbose
        self.random_state: Optional[Union[int, np.random.RandomState]] = random_state
        self.history = []

    def track_history(
        self,
        previous_beta: np.array,
        new_beta: np.array,
        loss: float,
        divergence: float,
    ):
        self.history.append(
            {
                "loss": loss,
                "gap": np.max(np.abs(new_beta - previous_beta)),
                "divergence": divergence,
                "beta": new_beta,
            }
        )

    def optimise(self, X, y, sample_weight, previous_fit=None) -> np.array:
        optimiser: LinearModel = _validate_optimiser(
            self.optimiser, self.alpha, self.groups
        )
        beta: np.array
        if self.warm_start and previous_fit is not None:
            optimiser.coef_ = previous_fit
            beta = previous_fit
        else:
            beta = np.zeros(X.shape[1])

        for _ in self.max_iter:
            eta: np.array = np.matmul(X, beta)
            gradient: np.array
            hessian: np.array
            gradient, hessian = self.grad(
                X=X, y=y, eta=eta, sample_weight=sample_weight
            )
            optimiser.fit(X=X, y=(eta - hessian * gradient), sample_weights=hessian)
            beta_new: np.array = optimiser.coef_
            eta_new: np.array = np.matmul(X, beta_new)
            learning_rate: float = backtracking_line_search(
                loss=self.loss,
                sample_weight=sample_weight,
                current_prediction=eta_new,
                previous_prediction=eta,
                previous_loss=self.history[-1]["loss"],
                reduction_factor=self.reduction_factor,
                max_learning_rate=1.0,
                gradient_direction=np.matmul(X.T, gradient),
            )
            beta_updated: np.array = (1 - learning_rate) * beta + (
                learning_rate
            ) * beta_new
            self.track_history(
                y=y,
                previous_beta=beta,
                new_beta=beta_updated,
                loss=self.loss(eta=eta_new, y=y, sample_weight=sample_weight),
            )
            beta = beta_updated
            if self.history[-1]["gap"] < self.tol * np.max(np.abs(beta)):
                return beta

        warnings.warn(
            f"Convergence not reached after {self.max_iter + 1} iterations."
            + "Consider increasing `max_iter` or decreasing `tol`.",
            ConvergenceWarning,
        )
        return beta
