from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from ._base import RegularizedLinearSurvivalModel
from .gradients import GRADIENT_FACTORY
from .proximal_operators import ProximalOperator
from .utils import get_closest


class CoxPH(RegularizedLinearSurvivalModel):
    def __init__(
        self,
        alpha: Optional[float],
        threshold: float,
        groups: List[Union[int, List[int]]],
        proximal_operator: ProximalOperator,
        scale_group: Optional[str] = "group_length",
        solver: str = "copt",
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        tie_correction: str = "efron",
        baseline_hazard_estimator: str = "breslow",
    ) -> None:
        super().__init__(
            alpha=alpha,
            threshold=threshold,
            groups=groups,
            proximal_operator=proximal_operator,
            scale_group=scale_group,
            solver=solver,
            warm_start=warm_start,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            tie_correction=tie_correction,
            baseline_hazard_estimator=baseline_hazard_estimator,
        )
        if not isinstance(tie_correction, str):
            raise ValueError(
                f"`tie_correction` must be of type `str`. Found type {type(tie_correction)} instead."
            )
        if tie_correction not in ["efron", "breslow"]:
            raise ValueError(
                f"`tie_correction` must be one of ['efron', 'breslow']. Found {tie_correction} instead."
            )
        self.tie_correction: str = tie_correction
        self.grad: Callable[
            [np.array, pd.DataFrame, np.array], np.array
        ] = GRADIENT_FACTORY[tie_correction]
        if not isinstance(baseline_hazard_estimator, str):
            raise ValueError(
                f"`baseline_hazard_estimator` must be of type `str`. Found type {type(baseline_hazard_estimator)} instead."
            )
        if baseline_hazard_estimator not in ["breslow"]:
            raise ValueError(
                f"`baseline_hazard_estimator` must be one of ['breslow']. Found {baseline_hazard_estimator} instead."
            )
        self.baseline_hazard_estimator: str = baseline_hazard_estimator

    def predict_cumulative_hazard_function(
        self, X: pd.DataFrame, time: np.array
    ) -> pd.DataFrame:
        super().predict_cumulative_hazard_function(X, time)
        baseline_cumulative_hazard: np.array = np.tile(
            # TODO: Actually implement this - should get closest,
            # (or always next/previous) cum. hazard value from
            # a pd.Series.
            A=get_closest(self.baseline_cumulative_hazard, time),
            reps=X.shape[0],
        ).reshape((X.shape[0], time.shape[0]))
        log_hazards: np.array = (
            np.tile(A=self.predict(X), reps=time.shape[0])
            .reshape((time.shape[0], X.shape[0]))
            .T
        )
        cumulative_hazard_function: pd.DataFrame = pd.DataFrame(
            baseline_cumulative_hazard * np.exp(log_hazards), columns=time
        )
        return cumulative_hazard_function
