from typing import List, Union, Optional, Callable
from functools import partial

import copt as cp
import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from scipy.stats import norm
from typeguard import typechecked


from ._base import RegularizedLinearSurvivalModel
from .compat import (
    LOSS_FACTORY,
    GRADIENT_FACTORY,
    BASELINE_HAZARD_FACTORY,
)


@typechecked
class AFT(RegularizedLinearSurvivalModel):
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
        )
        self.gradient: Callable = partial(
            GRADIENT_FACTORY["accelerated_failure_time"],
            bandwidth_function=bandwidth_function,
        )
        self.loss: Callable = partial(
            LOSS_FACTORY["accelerated_failure_time"],
            bandwidth_function=bandwidth_function,
        )
        self.baseline_hazard: Callable = partial(
            BASELINE_HAZARD_FACTORY["accelerated_failure_time"],
            bandwidth_function=bandwidth_function,
        )
        self.bandwidth_function = bandwidth_function

    def predict_baseline_hazard_function(self, time):
        return self.baseline_hazard(
            time=time, event=self.train_event, eta=self.train_eta
        )

    def predict_cumulative_hazard_function(self, X, time):
        theta: np.array = np.exp(self.predict(X))
        return np.trapz(
            y=self.predict_baseline_hazard_function(time * theta) * theta,
            x=time,
        )


@typechecked
class AFTLasso(AFT):
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
class AFTElasticNet(AFT):
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
class AFTGroupLasso(AFT):
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
