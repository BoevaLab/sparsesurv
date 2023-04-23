from functools import partial
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from typeguard import typechecked

from ._base import RegularizedLinearSurvivalModel
from .compat import BASELINE_HAZARD_FACTORY, GRADIENT_FACTORY, LOSS_FACTORY


@typechecked
class AH(RegularizedLinearSurvivalModel):
    def __init__(
        self,
        alpha: float,
        optimiser: str,
        l1_ratio: Optional[float] = 1.0,
        groups: Optional[List[List[int]]] = None,
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        bandwidth_function: str = "jones_1990",
    ) -> None:
        super().__init__(
            alpha=alpha,
            optimiser=optimiser,
            l1_ratio=l1_ratio,
            groups=groups,
            line_search=line_search,
            line_search_reduction_factor=line_search_reduction_factor,
            warm_start=warm_start,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            is_convex=False,
        )
        self.gradient: Callable = partial(
            GRADIENT_FACTORY["accelerated_hazards"],
            bandwidth_function=bandwidth_function,
        )
        self.loss: Callable = partial(
            LOSS_FACTORY["accelerated_hazards"],
            bandwidth_function=bandwidth_function,
        )
        self.baseline_hazard: Callable = partial(
            BASELINE_HAZARD_FACTORY["accelerated_hazards"],
            bandwidth_function=bandwidth_function,
        )
        self.bandwidth_function = bandwidth_function

    def predict_baseline_hazard_function(self, time):
        baseline_hazard: np.array = np.empty(time.shape[0])
        for _ in range(time.shape[0]):
            baseline_hazard[_] = self.baseline_hazard(
                time=time[_],
                train_time=self.train_time,
                train_event=self.train_event,
                train_eta=self.train_eta,
                bandwidth_function=self.bandwidth_function,
            )
        return baseline_hazard

    def predict_cumulative_hazard_function(self, X, time):
        theta: np.array = np.exp(self.predict(X))
        n_samples: int = X.shape[0]
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0] + 1))
        zero_flag: bool = False
        if 0 not in time:
            zero_flag = True
            time = np.concatenate([np.array([0]), time])

        for _ in range(n_samples):
            cumulative_hazard[_, :] = cumtrapz(
                y=self.predict_baseline_hazard_function(time * theta[_]) * theta[_],
                x=time,
                initial=0,
            )
        if zero_flag:
            cumulative_hazard = cumulative_hazard[:, 1:]
            time = time[1:]
        return pd.DataFrame(cumulative_hazard, columns=time)


@typechecked
class AHLasso(AH):
    def __init__(
        self,
        alpha: float,
        optimiser: str,
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        bandwidth_function: str = "jones_1990",
    ) -> None:
        super().__init__(
            alpha=alpha,
            optimiser=optimiser,
            l1_ratio=1.0,
            groups=None,
            line_search=line_search,
            line_search_reduction_factor=line_search_reduction_factor,
            warm_start=warm_start,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            bandwidth_function=bandwidth_function,
        )


@typechecked
class AHElasticNet(AH):
    def __init__(
        self,
        alpha: float,
        optimiser: str,
        l1_ratio: float,
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        bandwidth_function: str = "jones_1990",
    ) -> None:
        super().__init__(
            alpha=alpha,
            optimiser=optimiser,
            l1_ratio=l1_ratio,
            groups=None,
            line_search=line_search,
            line_search_reduction_factor=line_search_reduction_factor,
            warm_start=warm_start,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            bandwidth_function=bandwidth_function,
        )


@typechecked
class AHGroupLasso(AH):
    def __init__(
        self,
        alpha: float,
        optimiser: str,
        groups: List[List[int]],
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        bandwidth_function: str = "jones_1990",
    ) -> None:
        super().__init__(
            alpha=alpha,
            optimiser=optimiser,
            l1_ratio=None,
            groups=groups,
            line_search=line_search,
            line_search_reduction_factor=line_search_reduction_factor,
            warm_start=warm_start,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            bandwidth_function=bandwidth_function,
        )
