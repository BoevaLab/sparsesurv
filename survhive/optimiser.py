import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from celer import ElasticNet as CelerElasticNet
from celer import GroupLasso as CelerGroupLasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet as ScikitElasticNet
from sklearn.linear_model._base import LinearModel
from typeguard import typechecked

from .utils import inverse_transform_survival

OPTIMISERS: Dict[str, LinearModel] = {
    "scikit_elastic_net": ScikitElasticNet,
    "scikit_lasso": ScikitElasticNet,
    "celer_elastic_net": CelerElasticNet,
    "celer_lasso": CelerElasticNet,
    "celer_group_lasso": CelerGroupLasso,
}


def backtracking_line_search(
    loss: Callable,
    time: np.array,
    event: np.array,
    search_direction: np.array,
    gradient_direction: np.array,
    current_prediction: np.array,
    previous_prediction: np.array,
    previous_loss: float,
    reduction_factor: float = 0.5,
    max_learning_rate: float = 1.0,
    max_iter: int = 5,
    dampening: float = 0.5,
) -> float:
    current_learning_rate: float = max_learning_rate
    current_iter: int = 0
    while (
        loss(
            linear_predictor=(1 - current_learning_rate) * previous_prediction
            + current_learning_rate * current_prediction,
            time=time,
            event=event,
        )
        > previous_loss
        + (
            dampening
            * current_learning_rate
            * np.matmul(gradient_direction.T, search_direction)
        )
        - np.finfo(float).eps
    ) and current_iter < max_iter:
        current_learning_rate *= reduction_factor
        current_iter += 1
    print(current_learning_rate)
    return current_learning_rate


@typechecked
def _validate_optimiser(
    optimiser: str,
    alpha: float,
    l1_ratio: Optional[float] = None,
    groups: Optional[List[List[int]]] = None,
    warm_start: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7,
    verbose: int = 0,
) -> LinearModel:
    optimiser_: LinearModel
    if l1_ratio is not None:
        if (
            optimiser not in ["scikit_elastic_net", "celer_elastic_net"]
            and l1_ratio != 1.0
        ):
            raise ValueError(
                f"Expected `l1_ratio` to be one, since `optimiser` is {optimiser}."
            )

        if "celer" not in optimiser:
            optimiser_: LinearModel = OPTIMISERS[optimiser](
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=False,
                tol=tol,
                max_iter=max_iter,
                warm_start=warm_start,
            )
        else:
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
    return optimiser_


@typechecked
class Optimiser:
    def __init__(
        self,
        grad: Callable,
        loss: Callable,
        alpha: float,
        optimiser: str,
        l1_ratio: Optional[float],
        groups: Optional[List[List[int]]] = None,
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
    ):
        self.history.append(
            {
                "loss": loss,
                "gap": np.max(np.abs(new_beta - previous_beta)),
                "beta": new_beta,
            }
        )

    def optimise(self, X, y, previous_fit=None) -> np.array:
        optimiser: LinearModel = _validate_optimiser(
            optimiser=self.optimiser,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            groups=self.groups,
            warm_start=self.warm_start,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )
        beta: np.array
        if self.warm_start and previous_fit is not None:
            optimiser.coef_ = previous_fit
            beta = previous_fit
        else:
            beta = np.zeros(X.shape[1])

        time: np.array
        event: np.array
        time, event = inverse_transform_survival(y)
        self.history.append(
            {
                "loss": self.loss(
                    linear_predictor=np.matmul(X, beta),
                    time=time,
                    event=event,
                ),
                "gap": np.nan,
                "beta": beta,
            }
        )
        for _ in range(self.max_iter):
            eta: np.array = np.matmul(X, beta)
            gradient: np.array
            hessian: np.array
            gradient, hessian = self.grad(
                linear_predictor=eta,
                time=time,
                event=event,
            )
            inverse_hessian = 1 / hessian
            optimiser.fit(
                X=X,
                y=(eta - inverse_hessian * gradient),
                sample_weight=hessian,
            )
            beta_new: np.array = optimiser.coef_
            eta_new: np.array = np.matmul(X, beta_new)
            learning_rate: float
            learning_rate = backtracking_line_search(
                loss=self.loss,
                time=time,
                event=event,
                current_prediction=eta_new,
                previous_prediction=eta,
                previous_loss=self.history[-1]["loss"],
                reduction_factor=self.line_search_reduction_factor,
                max_learning_rate=1.0,
                gradient_direction=np.matmul(X.T, gradient),
                search_direction=(beta_new - beta),
            )
            beta_updated: np.array = (1 - learning_rate) * beta + (
                learning_rate
            ) * beta_new
            eta_new = np.matmul(X, beta_updated)
            self.track_history(
                previous_beta=beta,
                new_beta=beta_updated,
                loss=self.loss(linear_predictor=eta_new, time=time, event=event),
            )
            beta = beta_updated
            if (
                self.history[-1]["gap"]
                < self.tol * np.max(np.abs(beta)) + np.finfo(float).eps
            ):
                return beta
        if self.verbose:
            warnings.warn(
                f"Convergence not reached after {self.max_iter + 1} iterations."
                + "Consider increasing `max_iter` or decreasing `tol`.",
                ConvergenceWarning,
            )
        return beta
