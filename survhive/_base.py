from typing import List, Optional, Union

import copt as cp
import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel

from .baseline_hazard import CUMULATIVE_BASELINE_HAZARD_FACTORY
from .proximal_operators import ElasticNetProximal
from .utils import (
    _check_max_iter,
    _check_nu,
    _check_penalty_factors,
    _check_proximal_operator,
    _check_solver,
    _check_tol,
    _check_verbose,
)


class RegularizedLinearSurvivalModel(LinearModel):
    def __init__(
        self,
        proximal_operator: str,
        groups: List[Union[int, List[int]]],
        threshold: float,
        scale_group: Optional[str] = "group_length",
        penalty_factors: Optional[np.array] = None,
        solver: str = "copt",
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        psi: float = 0.0,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 4.0,
        tau: Optional[float] = 1 / 3,
        nu: Optional[float] = 1.0,
    ):
        self.proximal_operator = proximal_operator
        self.groups = groups
        self.threshold = threshold
        self.scale_group = scale_group
        self.penalty_factors = penalty_factors
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.psi = psi
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.nu = nu

        # By default, we have not calculated the baseline
        # cumulative hazard yet, so we set the caching tracker
        # to false.
        self.baseline_cumulative_hazard_cached = False

    def fit(self, X: pd.DataFrame, y: np.array) -> None:
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
        penalty_factors = _check_penalty_factors(self.penalty_factors)
        proximal_operator = _check_proximal_operator(
            proximal_operator=self.proximal_operator,
            groups=self.groups,
            scale_group=self.scale_group,
            penalty_factors=penalty_factors,
            threshold=self.threshold,
            alpha=self.alpha,
            gamma=self.gamma,
            tau=self.tau,
        )
        nu = _check_nu(nu=self.nu)
        if nu:
            proximal_operator = ElasticNetProximal(
                proximal_operator=proximal_operator,
                threshold=self.threshold * penalty_factors,
                nu=nu,
            )

        solver = _check_solver(self.solver)
        max_iter = _check_max_iter(self.max_iter)
        tol = _check_tol(self.max_tol)
        verbose = _check_verbose(self.verbose)
        self.n_features_ = X.shape[1]
        coef_ = np.zeros(self.n_features_)
        if solver == "copt":
            pgd = cp.minimize_proximal_gradient(
                self.grad,
                coef_,
                proximal_operator,
                jac=True,
                step="backtracking",
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                callback=None,
                accelerated=False,
            )
        elif solver == "numba":
            raise NotImplementedError

        self.coef_: np.array = pgd.x
        self.is_fitted: bool = True
        self.baseline_cumulative_hazard: np.array
        self.baseline_cumulative_hazard_times: np.array
        (
            self.baseline_cumulative_hazard,
            self.baseline_cumulative_hazard_times,
        ) = CUMULATIVE_BASELINE_HAZARD_FACTORY[self.baseline_hazard_estimator](
            X=X, y=y, coef=self.coef_
        )
        self.baseline_cumulative_hazard_cached = True

    def predict_hazard_function(self, X: pd.DataFrame, time: np.array) -> pd.DataFrame:
        """Predict hazard function for each sample and each requested time.

        Parameters
        ----------
        X : pd.DataFrame of (n_samples, n_features)
            Data.
        time : np.array of (n_times)
            Times at which hazard function predictions are desired.

        Returns
        ---
        hazard_function : pd.DataFrame of (n_samples, n_times)
            Hazard function for each sample and each requested time.

        Notes
        ---
        To be implemented in each child class.
        """
        raise NotImplementedError

    def predict_cumulative_hazard_function(
        self, X: pd.DataFrame, time: np.array
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
        self, X: pd.DataFrame, time: np.array
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
        return np.exp(
            np.negative(self.predict_cumulative_hazard_function(X=X, time=time))
        )
