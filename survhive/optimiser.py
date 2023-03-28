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
from .constants import EPS
from numba import jit

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
    if np.linalg.norm((previous_prediction - current_prediction), 2) < EPS:
        return 0.0
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
    p0=10,
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
                p0=p0,
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
        p0=10,
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
        self.p0 = p0

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

    def optimise_path(self, X, y, alphas) -> np.array:
        optimiser: LinearModel = _validate_optimiser(
            optimiser=self.optimiser,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            groups=self.groups,
            warm_start=self.warm_start,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            p0=self.p0,
        )
        beta: np.array
        time: np.array
        event: np.array
        time, event = inverse_transform_survival(y)
        # self.history.append(
        #     {
        #         "loss": self.loss(
        #             linear_predictor=np.matmul(X, beta),
        #             time=time,
        #             event=event,
        #         ),
        #         "gap": np.nan,
        #         "beta": beta,
        #     }
        # )
        coef = np.zeros((X.shape[1], alphas.shape[0]))
        for _ in range(self.max_iter):
            print(_)
            eta: np.array = np.matmul(X, coef[:, 1])
            # print(eta)
            gradient: np.array
            hessian: np.array
            gradient, hessian = self.grad(
                linear_predictor=eta,
                time=time,
                event=event,
            )
            inverse_hessian = hessian.copy()
            hessian_mask = (hessian > 0).astype(bool)
            inverse_hessian[np.logical_not(hessian_mask)] = np.inf
            inverse_hessian = 1 / inverse_hessian
            inverse_hessian[inverse_hessian == 1 / np.inf] = 0
            weights = hessian[hessian_mask]
            correction_factor = np.sum(weights)
            weights = weights * (np.sum(hessian_mask) / np.sum(weights))
            weights_sqrt = np.sqrt(weights)
            X_irls = X[hessian_mask, :] * weights_sqrt.repeat(X.shape[1]).reshape(
                (np.sum(hessian_mask), X.shape[1])
            )
            y_irls = weights_sqrt * (eta - inverse_hessian * gradient)[hessian_mask]
            print(np.sum(y_irls))
            print(np.sum(X_irls))
            print(np.sum(weights))
            # print(weights.repeat(X.shape[1]).reshape((weights.shape[0], X.shape[1])).T)
            # print(weights.repeat(X.shape[1]).reshape((weights.shape[0], X.shape[1])).shape)
            # print(alphas)
            # raise ValueError
            # print((X * weights.repeat(X.shape[1]).reshape((weights.shape[0], X.shape[1]))))
            # print(((eta - inverse_hessian * gradient)*weights)[hessian_mask])

            new_coef = optimiser.path(
                # X=(X * weights.repeat(X.shape[1]).reshape((weights.shape[0], X.shape[1])))[hessian_mask, ],
                # y=((eta - inverse_hessian * gradient)*weights)[hessian_mask],
                X_irls,
                y_irls,
                # sample_weight=hessian[hessian_mask],
                alphas=alphas,
                coef_init=coef[:, 0],
            )[1]

            print(np.sum(new_coef[:, 0]))

            # beta_new: np.array = optimiser.coef_
            # eta_new: np.array = np.matmul(X, beta_new)
            # learning_rate: float
            # learning_rate = backtracking_line_search(
            #     loss=self.loss,
            #     time=time,
            #     event=event,
            #     current_prediction=eta_new,
            #     previous_prediction=eta,
            #     previous_loss=self.history[-1]["loss"],
            #     reduction_factor=self.line_search_reduction_factor,
            #     max_learning_rate=1.0,
            #     gradient_direction=np.matmul(X.T, gradient),
            #     search_direction=(beta_new - beta),
            # )
            # beta_updated: np.array = (1 - learning_rate) * beta + (
            #     learning_rate
            # ) * beta_new
            # eta_new = np.matmul(X, beta_updated)
            # self.track_history(
            #     previous_beta=beta,
            #     new_beta=beta_updated,
            #     loss=self.loss(linear_predictor=eta_new, time=time, event=event),
            # )
            # beta = beta_updated
            # print(self.history[-1]["gap"])
            # if (
            #    self.history[-1]["gap"]
            #    < self.tol * np.max(np.abs(beta)) + np.finfo(float).eps
            # ):
            # if self.history[-1]["gap"] < 0.001:
            if np.max(np.abs(new_coef - coef)) < 0.001:
                print(_)
                return new_coef
            else:
                coef = new_coef
        if self.verbose:
            warnings.warn(
                f"Convergence not reached after {self.max_iter + 1} iterations."
                + "Consider increasing `max_iter` or decreasing `tol`.",
                ConvergenceWarning,
            )
        return beta

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
            p0=self.p0,
        )
        beta: np.array
        if self.warm_start and previous_fit is not None:
            optimiser.coef_ = previous_fit
            beta = previous_fit
        else:
            beta = np.zeros(X.shape[1])
            optimiser.coef_ = beta

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
            print(time)

            gradient, hessian = self.grad(
                linear_predictor=eta,
                time=time,
                event=event,
            )

            # if _ == 0:
            #    alpha_max = np.max(np.abs(np.matmul(gradient.T, X[hessian_mask, :] * np.sqrt(hessian[hessian_mask]).repeat(X.shape[1]).reshape((np.sum(hessian_mask), X.shape[1]))))) #/ (np.sum(hessian))
            inverse_hessian = hessian.copy()
            hessian_mask = (hessian > 0).astype(bool)
            inverse_hessian[np.logical_not(hessian_mask)] = np.inf
            inverse_hessian = 1 / inverse_hessian
            # print("HEYO")
            # print(hessian)
            # print(inverse_hessian)
            inverse_hessian[inverse_hessian == 1 / np.inf] = 0
            # print(optimiser)
            # TODO: Strong screening here.
            print(np.sum(hessian[hessian_mask]))
            print(np.sum(gradient[hessian_mask]))
            weights = hessian[hessian_mask]
            correction_factor = np.sum(weights)
            weights = weights * (np.sum(hessian_mask) / np.sum(weights))
            # if _ == 0:
            # print((np.abs(np.matmul(gradient.T[hessian_mask], X[hessian_mask, :]))))
            alpha_max = (
                np.max(np.abs(np.matmul(gradient.T[hessian_mask], X[hessian_mask, :])))
                / correction_factor
            )
            # print(alpha_max)
            # strong_variables = np.where(np.abs(np.matul(X[hessian_mask, :].T  (inverse_hessian * gradient)[hessian_mask])) >= 2 * alphas[i] - alphas[i - 1])[0]
            # print(np.sqrt(hessian[hessian_mask]).repeat(X.shape[1]).reshape((np.sum(hessian_mask), X.shape[1])))
            # print(X[hessian_mask, :] * np.sqrt(weights).repeat(X.shape[1]).reshape((np.sum(hessian_mask), X.shape[1])))
            # print(np.sqrt(weights) * (eta - inverse_hessian * gradient)[hessian_mask])
            # print(np.sqrt(weights) * (eta - inverse_hessian * gradient)[hessian_mask])
            # print(X[hessian_mask, :] * np.sqrt(weights).repeat(X.shape[1]).reshape((np.sum(hessian_mask), X.shape[1])))
            # print(optimiser)
            # print((X[hessian_mask, :] * np.sqrt(weights).repeat(X.shape[1]).reshape((np.sum(hessian_mask), X.shape[1]))).shape)
            # print((np.sqrt(weights) * (eta - inverse_hessian * gradient)[hessian_mask]).shape)
            print(
                np.sum(
                    X[hessian_mask, :]
                    * np.sqrt(weights)
                    .repeat(X.shape[1])
                    .reshape((np.sum(hessian_mask), X.shape[1]))
                )
            )
            print(
                np.sum(
                    np.sqrt(weights) * (eta - inverse_hessian * gradient)[hessian_mask]
                )
            )
            print(np.sum(weights))
            optimiser.fit(
                X=X[hessian_mask, :]
                * np.sqrt(weights)
                .repeat(X.shape[1])
                .reshape((np.sum(hessian_mask), X.shape[1])),
                y=np.sqrt(weights) * (eta - inverse_hessian * gradient)[hessian_mask],
                # sample_weight=hessian[hessian_mask],
            )
            # TODO: Double check strong screening here.
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
            # print(learning_rate)
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
            # print(self.history[-1]["gap"])
            # if (
            #    self.history[-1]["gap"]
            #    < self.tol * np.max(np.abs(beta)) + np.finfo(float).eps
            # ):
            if self.history[-1]["gap"] < 0.00001:
                return beta
        if self.verbose:
            warnings.warn(
                f"Convergence not reached after {self.max_iter + 1} iterations."
                + "Consider increasing `max_iter` or decreasing `tol`.",
                ConvergenceWarning,
            )
        return beta
