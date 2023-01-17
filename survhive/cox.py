from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from ._base import RegularizedLinearSurvivalModel
from .gradients import GRADIENT_FACTORY
from .utils import (
    get_closest,
    get_death_matrix,
    get_risk_matrix,
    _check_groups,
)


class CoxPH(RegularizedLinearSurvivalModel):
    def __init__(
        self,
        threshold: float,
        groups: List[Union[int, List[int]]],
        proximal_operator: str = "lasso",
        scale_group: Optional[str] = "group_length",
        solver: str = "copt",
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
        psi: float = 0.0,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 4.0,
        tau: Optional[float] = 1 / 3,
        tie_correction: str = "efron",
        baseline_hazard_estimator: str = "breslow",
    ) -> None:
        super().__init__(
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
            psi=psi,
            alpha=alpha,
            gamma=gamma,
            tau=tau,
        )

        self.tie_correction: str = tie_correction
        self.baseline_hazard_estimator: str = baseline_hazard_estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        sample_weight: np.array = None,
        check_input: bool = True,
    ) -> None:
        if not isinstance(self.tie_correction, str):
            raise ValueError(
                f"`tie_correction` must be of type `str`. Found type {type(self.tie_correction)} instead."
            )
        if self.tie_correction not in ["efron", "breslow"]:
            raise ValueError(
                f"`tie_correction` must be one of ['efron', 'breslow']. Found {self.tie_correction} instead."
            )
        if not isinstance(self.baseline_hazard_estimator, str):
            raise ValueError(
                f"`baseline_hazard_estimator` must be of type `str`. Found type {type(self.baseline_hazard_estimator)} instead."
            )
        if self.baseline_hazard_estimator not in ["breslow"]:
            raise ValueError(
                f"`baseline_hazard_estimator` must be one of ['breslow']. Found {self.baseline_hazard_estimator} instead."
            )
        self.original_groups: List[List[int]] = self.groups
        self.groups: np.array
        self.has_overlaps: bool
        self.inverse_groups: List[List[int]]
        self.groups, self.has_overlaps, self.inverse_groups = _check_groups(self.groups)
        # Freeze all arguments of the gradient that we are optimising
        # except for beta. Specifically, cache objects such as
        # the risk and death matrix which are semi-expensive to compute.
        if self.tie_correction == "efron":
            risk_matrix: np.array = get_risk_matrix()
            death_matrix: np.array = get_death_matrix()
            self.grad: Callable[[np.array], np.array] = partial(
                GRADIENT_FACTORY[self.tie_correction],
                X=X,
                risk_matrix=risk_matrix,
                death_matrix=death_matrix,
                groups=self.groups,
                has_overlaps=self.has_overlaps,
                inverse_groups=self.inverse_groups,
            )
        elif self.tie_correction == "breslow":
            risk_matrix = get_risk_matrix()
            self.grad = partial(
                GRADIENT_FACTORY[self.tie_correction],
                X=X,
                y=y,
                risk_matrix=risk_matrix,
            )
        return super().fit(
            X=X, y=y, sample_weight=sample_weight, check_input=check_input
        )

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
