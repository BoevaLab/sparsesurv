from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from typeguard import typechecked

from ._base import RegularizedLinearSurvivalModel
from .compat import BASELINE_HAZARD_FACTORY, GRADIENT_FACTORY, LOSS_FACTORY


@typechecked
class CoxPH(RegularizedLinearSurvivalModel):
    def __init__(
        self,
        alpha: float,
        optimiser: str,
        l1_ratio: float = 1.0,
        groups: Optional[List[List[int]]] = None,
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        tie_correction: str = "efron",
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
        if tie_correction not in ["efron", "breslow"]:
            raise ValueError(
                "Expected `tie_correction` to be in ['efron', 'breslow']."
                + f"Found {tie_correction} instead."
            )
        self.gradient: Callable = GRADIENT_FACTORY[tie_correction]
        self.loss: Callable = LOSS_FACTORY[tie_correction]
        self.baseline_hazard_estimator: Callable = BASELINE_HAZARD_FACTORY[
            tie_correction
        ]
        self.tie_correction = tie_correction

    def predict_cumulative_hazard_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
        if np.min(time) < 0:
            raise ValueError(
                "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
                + f"Minimum time found was {np.min(time)}."
                + "Please remove any times strictly less than zero."
            )
        cumulative_baseline_hazards_times: np.array
        cumulative_baseline_hazards: np.array
        (
            cumulative_baseline_hazards_times,
            cumulative_baseline_hazards,
        ) = BASELINE_HAZARD_FACTORY[self.tie_correction](
            time=self.train_time, event=self.train_event, eta=self.train_eta
        )
        cumulative_baseline_hazards = np.concatenate(
            [np.array([0.0]), cumulative_baseline_hazards]
        )
        cumulative_baseline_hazards_times: np.array = np.concatenate(
            [np.array([0.0]), cumulative_baseline_hazards_times]
        )
        cumulative_baseline_hazards: np.array = np.tile(
            A=cumulative_baseline_hazards[
                np.digitize(x=time, bins=cumulative_baseline_hazards_times, right=False)
                - 1
            ],
            reps=X.shape[0],
        ).reshape((X.shape[0], time.shape[0]))
        log_hazards: np.array = (
            np.tile(
                A=self.predict(X),
                reps=time.shape[0],
            )
            .reshape((time.shape[0], X.shape[0]))
            .T
        )
        cumulative_hazard_function: pd.DataFrame = pd.DataFrame(
            cumulative_baseline_hazards * np.exp(log_hazards),
            columns=time,
        )
        return cumulative_hazard_function


@typechecked
class CoxPHLasso(CoxPH):
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
        tie_correction: str = "efron",
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
            tie_correction=tie_correction,
        )


@typechecked
class CoxPHElasticNet(CoxPH):
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
        tie_correction: str = "efron",
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
            tie_correction=tie_correction,
        )


@typechecked
class CoxPHGroupLasso(CoxPH):
    def __init__(
        self,
        alpha: float,
        optimiser: str,
        l1_ratio: float,
        groups: List[List[int]],
        line_search: bool = True,
        line_search_reduction_factor: float = 0.5,
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        tie_correction: str = "efron",
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
            tie_correction=tie_correction,
        )
