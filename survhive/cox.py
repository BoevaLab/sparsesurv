from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from typeguard import typechecked

from ._base import RegularizedLinearSurvivalModel
from .compat import BASELINE_HAZARD_FACTORY, GRADIENT_FACTORY, LOSS_FACTORY
from .utils import calculate_sgl_groups


@typechecked
class CoxPH(RegularizedLinearSurvivalModel):
    def __init__(
        self,
        alpha: float,
        type: str,
        l1_ratio: float = 1.0,
        groups: Optional[List[List[int]]] = None,
        group_weights: Optional[str] = "scale_by_group_group_lasso",
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        tie_correction: str = "efron",
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
    ) -> None:
        super().__init__(
            alpha=alpha,
            type=type,
            l1_ratio=l1_ratio,
            groups=groups,
            group_weights=group_weights,
            warm_start=warm_start,
            n_irls_iter=n_irls_iter,
            tol=tol,
            verbose=verbose,
            inner_solver_max_iter=inner_solver_max_iter,
            inner_solver_max_epochs=inner_solver_max_epochs,
            inner_solver_p0=inner_solver_p0,
            inner_solver_prune=inner_solver_prune,
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
class CoxPHElasticNet(CoxPH):
    def __init__(
        self,
        alpha: float,
        l1_ratio: float,
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        tie_correction: str = "efron",
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
    ) -> None:
        super().__init__(
            alpha=alpha,
            type="elastic_net",
            l1_ratio=l1_ratio,
            groups=None,
            group_weights=None,
            warm_start=warm_start,
            n_irls_iter=n_irls_iter,
            tol=tol,
            verbose=verbose,
            tie_correction=tie_correction,
            inner_solver_max_iter=inner_solver_max_iter,
            inner_solver_max_epochs=inner_solver_max_epochs,
            inner_solver_p0=inner_solver_p0,
            inner_solver_prune=inner_solver_prune,
        )


@typechecked
class CoxPHLasso(CoxPHElasticNet):
    def __init__(
        self,
        alpha: float,
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        tie_correction: str = "efron",
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
    ) -> None:
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,
            warm_start=warm_start,
            n_irls_iter=n_irls_iter,
            tol=tol,
            verbose=verbose,
            tie_correction=tie_correction,
            inner_solver_max_iter=inner_solver_max_iter,
            inner_solver_max_epochs=inner_solver_max_epochs,
            inner_solver_p0=inner_solver_p0,
            inner_solver_prune=inner_solver_prune,
        )


@typechecked
class CoxPHGroupLasso(CoxPH):
    def __init__(
        self,
        alpha: float,
        groups: List[List[int]],
        group_weights: str = "scale_by_group_group_lasso",
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        tie_correction: str = "efron",
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
    ) -> None:
        super().__init__(
            alpha=alpha,
            type="group_lasso",
            l1_ratio=None,
            groups=groups,
            group_weights=group_weights,
            warm_start=warm_start,
            n_irls_iter=n_irls_iter,
            tol=tol,
            verbose=verbose,
            tie_correction=tie_correction,
            inner_solver_max_iter=inner_solver_max_iter,
            inner_solver_max_epochs=inner_solver_max_epochs,
            inner_solver_p0=inner_solver_p0,
            inner_solver_prune=inner_solver_prune,
        )


@typechecked
class CoxPHSparseGroupLasso(CoxPH):
    def __init__(
        self,
        alpha: float,
        l1_ratio: float,
        groups: List[List[int]],
        group_weights: str = "scale_by_group_sparse_group_lasso",
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        tie_correction: str = "efron",
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
    ) -> None:
        groups = calculate_sgl_groups(groups=groups)
        super().__init__(
            alpha=alpha,
            type="group_lasso",
            l1_ratio=l1_ratio,
            groups=groups,
            group_weights=group_weights,
            warm_start=warm_start,
            n_irls_iter=n_irls_iter,
            tol=tol,
            verbose=verbose,
            tie_correction=tie_correction,
            inner_solver_max_iter=inner_solver_max_iter,
            inner_solver_max_epochs=inner_solver_max_epochs,
            inner_solver_p0=inner_solver_p0,
            inner_solver_prune=inner_solver_prune,
        )
