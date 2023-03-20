from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from ._base import RegularizedLinearSurvivalModel
from .compat import BASELINE_HAZARD_FACTORY, GRADIENT_FACTORY, LOSS_FACTORY
from .utils import transform_survival, inverse_transform_survival

from typeguard import typechecked


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

    def fit(self, X: np.array, y: np.array, sample_weight=None) -> None:
        """Fit model with proximal gradient descent.

        Parameters
        ----------
        X : pd.DataFrame of (n_samples, n_features)
            Data.
        y : np.array of shape (n_samples,)
            Target. Will be cast to X's dtype if necessary.
        sample_weight : np.array of shape (n_samples,), default=None
            Sample weights. Internally, the `sample_weight` vector will be
            rescaled to sum to `n_samples`.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Notes
        ---
        To be implemented in each child class.
        """

        time: np.array
        event: np.array
        time, event = inverse_transform_survival(y=y)
        sorted_indices: np.array = np.argsort(a=time, kind="stable")
        time_sorted: np.array = time[sorted_indices]
        event_sorted: np.array = event[sorted_indices]
        X_sorted: np.array = X[
            sorted_indices,
        ]
        sample_weight: float
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        # sample_weight = sample_weight / np.sum(sample_weight)

        super().fit(
            X=X_sorted,
            y=transform_survival(time=time_sorted, event=event_sorted),
            sample_weight=sample_weight,
        )
        self.train_time: np.array = sorted
        self.train_event: np.array = event
        self.train_partial_hazards: np.array = np.matmul(X_sorted, self.coef_)

    def score(self, X, y, sample_weight=None):
        time: np.array
        event: np.array
        time, event = inverse_transform_survival(y=y)
        sorted_indices: np.array = np.argsort(a=time, kind="stable")
        time_sorted: np.array = time[sorted_indices]
        event_sorted: np.array = event[sorted_indices]
        X_sorted: np.array = X[
            sorted_indices,
        ]
        if sample_weight is None:
            sample_weight: np.array = np.ones(time.shape[0]) / time.shape[0]
        # Flip the loss by turning it into the likelihood
        # again since score implies higher values are better.
        return np.negative(
            self.loss(
                log_partial_hazard=self.predict(X_sorted),
                time=time_sorted,
                event=event_sorted,
                sample_weight=sample_weight,
            )
        )

    def predict_cumulative_hazard_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
        if np.min(time) < 0:
            raise ValueError(
                "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
                + f"Minimum time found was {np.min(time)}."
                + "Please remove any times strictly less than zero."
            )
        cumulative_baseline_hazards: np.array = BASELINE_HAZARD_FACTORY[
            self.tie_correction
        ](
            train_time=self.train_time,
            train_event=self.train_event,
            train_partial_hazards=self.train_partial_hazards,
        )
        cumulative_baseline_hazards = np.concatenate(
            [np.array(0.0), cumulative_baseline_hazards]
        )
        train_times: np.array = np.concatenate([np.array(0.0), self.train_time])

        cumulative_baseline_hazards: np.array = np.tile(
            A=np.cumulative_baseline_hazards[
                np.digitize(x=time, bins=train_times, right=False)
            ],
            reps=X.shape[0],
        ).reshape((X.shape[0], time.shape[0]))
        log_hazards: np.array = (
            np.tile(A=self.predict(X), reps=time.shape[0])
            .reshape((time.shape[0], X.shape[0]))
            .T
        )
        cumulative_hazard_function: pd.DataFrame = pd.DataFrame(
            cumulative_baseline_hazards * np.exp(log_hazards), columns=time
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
