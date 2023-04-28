import inspect
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel

from .cv import regularisation_path, regularisation_path_precond
from .utils import (
    inverse_transform_preconditioning,
    inverse_transform_survival,
)


class RegularizedLinearSurvivalModel(LinearModel):
    def __init__(
        self,
        alpha: float,
        l1_ratio: Optional[float] = 1.0,
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
        check_global_kkt=True,
    ):
        self.alpha: float = alpha
        self.l1_ratio: float = l1_ratio
        self.warm_start: bool = warm_start
        self.n_irls_iter: int = n_irls_iter
        self.tol: float = tol
        self.verbose: int = verbose
        self.intercept_ = 0
        self.coef_ = None
        self.inner_solver_max_iter = inner_solver_max_iter
        self.inner_solver_max_epochs = inner_solver_max_epochs
        self.inner_solver_p0 = inner_solver_p0
        self.inner_solver_prune = inner_solver_prune
        self.check_global_kkt = check_global_kkt

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
        X_sorted = X[sorted_indices, :]
        y_sorted = y[sorted_indices]
        self.coef_ = np.squeeze(
            regularisation_path(
                X=X_sorted,
                y=y_sorted,
                X_test=X,
                model=self,
                l1_ratio=self.l1_ratio,
                eps=None,
                n_alphas=None,
                alphas=np.array([self.alpha]),
                max_first=False,
            )[0]
        )
        self.train_time = time[sorted_indices]
        self.train_event = event[sorted_indices]
        self.train_eta = self.predict(X_sorted)

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

    def predict_survival_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
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
            np.negative(
                self.predict_cumulative_hazard_function(X=X, time=time_sorted)
            )
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


class PreconditionedLinearSurvivalModel(LinearModel):
    def __init__(
        self,
        alpha: float,
        tau: Optional[float] = 1.0,
        maxiter=1000,
        rtol=1e-6,
        verbose=0,
        default_step_size=1.0,
        check_global_kkt=True,
    ):
        self.alpha: float = alpha
        self.type = type
        self.tau: float = tau

        self.rtol = rtol
        self.rtol = rtol
        self.verbose = verbose
        self.maxiter = maxiter
        self.default_step_size = default_step_size
        self.check_global_kkt = check_global_kkt
        self.intercept_ = 0
        self.coef_ = None

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
        time, event, _ = inverse_transform_preconditioning(y=y)
        sorted_indices: np.array = np.argsort(a=time, kind="stable")
        X_sorted = X[sorted_indices, :]
        y_sorted = y[sorted_indices]

        self.coef_ = np.squeeze(
            regularisation_path_precond(
                X=X_sorted,
                y=y_sorted,
                X_test=X,
                model=self,
                tau=self.tau,
                eps=None,
                n_alphas=None,
                alphas=np.array([self.alpha]),
                max_first=False,
            )[0]
        )
        self.train_time = time[sorted_indices]
        self.train_event = event[sorted_indices]
        self.train_eta = self.predict(X_sorted)

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

    def predict_survival_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
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
            np.negative(
                self.predict_cumulative_hazard_function(X=X, time=time_sorted)
            )
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
