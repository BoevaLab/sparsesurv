from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from typeguard import typechecked

from .optimiser import Optimiser
from .utils import inverse_transform_survival, transform_survival


@typechecked
class RegularizedLinearSurvivalModel(LinearModel):
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
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
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
        self.intercept_ = 0
        self.coef_ = None

    def set_coef(self, coef: np.array):
        self.coef_ = coef
        return None

    def fit(self, X: np.array, y: np.array) -> None:
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
        y_sorted: np.array = transform_survival(time=time_sorted, event=event_sorted)
        optimiser: Optimiser = Optimiser(
            grad=self.gradient,
            loss=self.loss,
            alpha=self.alpha,
            optimiser=self.optimiser,
            l1_ratio=self.l1_ratio,
            groups=self.groups,
            line_search=self.line_search,
            line_search_reduction_factor=self.line_search_reduction_factor,
            warm_start=self.warm_start,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        if self.warm_start and self.coef_ is not None:
            self.coef_: np.array = optimiser.optimise(
                X=X_sorted,
                y=y_sorted,
                previous_fit=self.coef_,
            )
        else:
            self.coef_: np.array = optimiser.optimise(X=X_sorted, y=y_sorted)
        self.train_time = time_sorted
        self.train_event = event_sorted
        self.train_eta = self.predict(X_sorted)
        self.optimiser_state = optimiser

    def predict_cumulative_hazard_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
        """Predict cumulative hazard function for each sample and each requested time.

        Parameters
        ----------
        X : pd.DataFrame of (n_samples, n_features)
            Data.
        time : np.array of (n_times)
            Times at which hazard function predictions are desired.

        Returns
        ---
        cumulative_hazard_function : NDArray[Shape["*", "*"], Float32] of (n_samples, n_times)
            Cumulative hazard function for each sample and each requested time.

        Notes
        ---
        To be implemented in each child class.
        """
        raise NotImplementedError

    def predict_survival_function(self, X: np.array, time: np.array) -> pd.DataFrame:
        """Predict survival function for each sample and each requested time.

        Parameters
        ----------
        X : pd.DataFrame of (n_samples, n_features)
            Data.
        time : np.array of (n_times)
            Times at which hazard function predictions are desired.

        Returns
        ---
        survival_function : pd.DataFrame of (n_samples, n_times)
            Survival function for each sample and each requested time.

        Notes
        ---
        We exclusively rely on `predict_cumulative_hazard_function`
        and simply transform this to the survival function.
        """
        time_sorted: np.array = np.sort(a=time, kind="stable")
        return np.exp(
            np.negative(self.predict_cumulative_hazard_function(X=X, time=time_sorted))
        )

    def score(self, X, y):
        time: np.array
        event: np.array
        time, event = inverse_transform_survival(y=y)
        sorted_indices: np.array = np.argsort(a=time, kind="stable")
        time_sorted: np.array = time[sorted_indices]
        event_sorted: np.array = event[sorted_indices]
        X_sorted: np.array = X[
            sorted_indices,
        ]
        return np.negative(
            self.loss(
                linear_predictor=self.predict(X_sorted),
                time=time_sorted,
                event=event_sorted,
            )
        )
