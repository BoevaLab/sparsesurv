from typing import List, Union, Optional

import copt as cp
import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_X_y

from proximal_operators import ProximalOperator

# TODO:
# - Look into warnings and catching/turning these off or do we not
# want to catch them by default?
# - Look into sparse matrix handling; I think these would be really
# nice to have.
# - Look into how we actually predict, because I think kicking
# non-zero features is slightly faster and numerically more stable?
# --> Look at how scikit does it for Lasso/LassoCV.
# - Handle all TODOs in code below.
# - Look into parallelism (refer to scikit Lasso function).
# - Include check sample weight.
# - Actually seed the random state.


class RegularizedLinearSurvivalModel(LinearModel):
    def __init__(
        self,
        alpha: Optional[float],
        _lambda: float,
        groups: List[Union[int, List[int]]],
        # TODO: Need to actually write this interface class
        # to make it work with this.
        prox: ProximalOperator,
        scale_group: Optional[str] = "group_length",
        solver: str = "copt",
        warm_start: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self.alpha = alpha
        self._lambda = _lambda
        self.groups = groups
        self.prox = prox
        self.scale_group = scale_group
        self.solver = solver
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        # TODO: Briefly verify that this holds for all models.
        # Survival models (generally do not have an intercept).
        self.intercept_ = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        sample_weight: np.array = None,
        check_input: bool = True,
    ) -> None:
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
        # Instance checks for `__init__` arguments.
        if not isinstance(self.alpha, float) and self.alpha is not None:
            raise ValueError(
                f"`alpha` must be either of type float or None. Got {self.alpha} of type {type(self.alpha)} instead."
            )

        if not isinstance(self._lambda, float):
            if isinstance(self._lambda, int):
                self.lambd = float(self._lambda)
            else:
                raise ValueError(
                    f"`_lambda` must be of type float. Got type {type(self._lambda)} instead."
                )

        if not isinstance(self.groups, list):
            raise ValueError(
                f"`groups` must be of type list. Got type {type(self.groups)} instead."
            )

        if not isinstance(self.scale_group, str):
            if self.scale_group is None:
                self.scale_group = "group_length"
            else:
                raise ValueError(
                    f"`scale_group` must be either of type str or None. Got {self.scale_group} of type {type(self.scale_group)} instead."
                )

        # TODO: Need to implement this interface for ProximalOperator.
        if not isinstance(self.prox, ProximalOperator):
            raise ValueError(
                f"`prox` must be of type ProximalOperator. Got type {type(self.prox)} instead."
            )

        if not isinstance(self.solver, str):
            raise ValueError(
                f"`solver` must be of type str. Got type {type(self.solver)} instead."
            )

        if not isinstance(self.warm_start, bool):
            raise ValueError(
                f"`warm_start` must be of type bool. Got type {type(self.warm_start)} instead."
            )

        if not isinstance(self.max_iter, int):
            raise ValueError(
                f"`max_iter` must be of type int. Got type {type(self.max_iter)} instead."
            )

        if not isinstance(self.tol, float):
            raise ValueError(
                f"`tol` must be of type float. Got type {type(self.tol)} instead."
            )

        if not isinstance(self.verbose, int):
            raise ValueError(
                f"`verbose` must be of type int. Got type {type(self.verbose)} instead."
            )

        if (
            not isinstance(self.random_state, int)
            and self.random_state is not None
        ):
            raise ValueError(
                f"`random_state` must be of type int or None. Got type {type(self.random_state)} instead."
            )

        # Logic checks for `__init__` arguments.
        if self.alpha is not None:
            if self.alpha < 0.0 or self.alpha > 1.0:
                raise ValueError(
                    f"`alpha` must be in [0, 1]. Got {self.alpha} instead."
                )

        # TODO: Which other scaling factors could make sense here?
        # TODO: Look at this a bit.
        if self.scale_group not in ["scale_by_group"]:
            raise ValueError(
                f"`scale_group` must be one of ['scale_by_group']. Got {self.scale_group} instead."
            )

        if self.solver not in ["copt", "cython"]:
            raise ValueError(
                f"`solver` must be one of ['copt', 'cython']. Got {self.solver} instead."
            )

        # TODO: Need to actually write this function in utils
        # and adjust the arguments here if necessary.
        self.groups = check_groups(self.groups)

        # Logic checks for `fit` arguments.
        X, y = check_X_y(
            X,
            y,
            accept_sparse="csr",
            accept_large_sparse=True,
            dtype=[np.float32],
            y_numeric=True,
            multi_output=False,
        )

        # TODO: Need to actually write this function in utils
        # and adjust the arguments here if necessary.
        y = check_survival_y(
            y,
        )
        # TODO: Need to actually write this function in utils
        # and adjust the arguments here if necessary.
        time, event = transform_survival_target(y)
        self.n_features_ = X.shape[1]
        coef_ = np.zeros(self.n_features_)
        if self.solver == "copt":
            pgd = cp.minimize_proximal_gradient(
                self.model(X, y).grad,
                coef_,
                self.prox,
                jac=True,
                step="backtracking",
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                callback=None,
                accelerated=False,
            )
        elif self.solver == "cython":
            # TODO: Actually implement this and exchange this for
            # the proper call. Try to be as similar as possible
            # to the copt API (while [hopefully] gaining speed).
            raise NotImplementedError

        self.coef_ = pgd.x
        self.is_fitted = True

    def predict_hazard_function(
        self, X: pd.DataFrame, time: np.array
    ) -> pd.DataFrame:
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
            np.negative(self.predict_cumulative_hazard_function(X, time))
        )
