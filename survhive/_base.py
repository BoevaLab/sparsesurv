import inspect
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from typeguard import typechecked

from .utils import (
    calculate_sgl_groups,
    estimate_group_weights,
    has_overlaps,
    inverse_transform_survival,
    resolve_overlaps,
    summarise_overlapping_coefs,
)


@typechecked
class RegularizedLinearSurvivalModel(LinearModel):
    def __init__(
        self,
        alpha: float,
        type: str,
        l1_ratio: Optional[float] = 1.0,
        groups: Optional[List[List[int]]] = None,
        group_weights: Optional[str] = None,
        warm_start: bool = True,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
        verbose: int = 0,
        inner_solver_max_iter: int = 100,
        inner_solver_max_epochs: int = 50000,
        inner_solver_p0: int = 10,
        inner_solver_prune: bool = True,
    ):
        self.alpha: float = alpha
        self.type = type
        self.l1_ratio: float = l1_ratio
        if groups is not None:
            if l1_ratio is not None:
                (
                    self.group_mapping,
                    self.group_reverse_mapping,
                    self.group_overlap,
                    groups,
                    l1_ratio,
                ) = calculate_sgl_groups(groups)
            else:
                if has_overlaps(groups):
                    (
                        self.group_mapping,
                        self.group_reverse_mapping,
                        self.group_overlap,
                        groups,
                        l1_ratio,
                    ) = resolve_overlaps(groups)
                else:
                    self.group_reverse_mapping = None
                    self.group_overlap = False
        else:
            self.group_reverse_mapping = None
            self.group_overlap = None

        self.groups: Optional[List[List[int]]] = groups
        self.group_weights = group_weights
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

    def set_coef(self, coef: np.array) -> None:
        self.coef_ = coef
        return self

    def _get_param_names(cls):
        """Get parameter names for the estimator From official scikit-learn Base class."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator. From official scikit-learn Base class.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator. From official scikit-learn Base class.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def fit(self, X: np.array, y: np.array, previous_fit=None, alphas=None) -> None:
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
        if self.type == "group_lasso":
            if self.has_overlaps:
                X_sorted = X_sorted[:, self.group_mapping]
            group_weights = estimate_group_weights(
                groups=self.groups,
                strategy=self.group_weights,
                l1_ratio=self.l1_ratio,
            )

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
                n_irls_iter=100,
                tol=0.0001,
                check_global_kkt=True,
                max_first=False,
                group_weights=group_weights,
            )[0]
        )
        if self.type == "group_lasso":
            if self.has_overlaps:
                self.coef_ = summarise_overlapping_coefs(
                    coef=self.coef_,
                    group_reverse_mapping=self.group_reverse_mapping,
                )

        assert self.coef_.shape[0] == X.shape[1]
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


@typechecked
class PreconditionedSurvivalModel:
    def __init__(self, teacher_pipe, student_pipe, teacher, student, tau) -> None:
        super().__init__()
        self.teacher_pipe = teacher_pipe
        self.student_pipe = student_pipe
        self.teacher = teacher
        self.student = student
        self.tau = tau

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
        time_sorted = time[sorted_indices]
        event_sorted = event[sorted_indices]
        X_sorted = X[sorted_indices, :]
        y_sorted = y[sorted_indices]
        X_sorted_teacher = self.teacher_pipe.fit_transform(X_sorted)
        X_sorted_student = self.student_pipe.fit_transform(X_sorted)
        self.teacher.fit(X_sorted_teacher, y_sorted)
        self.student.fit(
            X_sorted_student,
            self.teacher.predict(X_sorted_teacher),
            time_sorted,
            event_sorted,
        )
        self.train_time = time[sorted_indices]
        self.train_event = event[sorted_indices]
        self.train_eta = self.student.predict(X_sorted_student)

    def predict(self, X: np.array):
        return self.student.predict(self.student_pipe.transform(X))

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
            self.teacher.loss(
                linear_predictor=(
                    self.student.predict(self.student_pipe.transform(X_sorted))
                ),
                time=time_sorted,
                event=event_sorted,
            )
        )
