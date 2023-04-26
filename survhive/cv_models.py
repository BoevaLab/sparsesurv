from typing import Union, List

import numpy as np
from typeguard import typechecked

from survhive._utils_.cv import CrossValidation, alpha_path_eta
from survhive.cox import CoxPHElasticNet, CoxPHGroupLasso


@typechecked
class CoxPHElasticNetCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        tie_correction: str = "efron",
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: np.array = None,
        l1_ratios: Union[float, np.array] = [1.0],
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
    ) -> None:
        super().__init__(
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.tie_correction = tie_correction
        self.n_irls_iter = n_irls_iter
        self.tol = tol

    def _get_estimator(self):
        return CoxPHElasticNet(
            alpha=0.01,
            tie_correction=self.tie_correction,
            l1_ratio=self.l1_ratios[0],
            tol=self.tol,
            n_irls_iter=self.n_irls_iter,
        )

    def _is_multitask(self):
        return False


@typechecked
class CoxPHLassoCV(CoxPHElasticNetCV):
    def __init__(
        self,
        tie_correction: str = "efron",
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: np.array = None,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
    ) -> None:
        super().__init__(
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=[1.0],
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            tie_correction=tie_correction,
            n_irls_iter=n_irls_iter,
            tol=tol,
        )


@typechecked
class CoxPHGroupLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        groups: List[List[int]],
        group_weights: str = "scale_by_group",
        cv_score_method: str = "linear_predictor",
        eps: float = 0.01,
        n_alphas: int = 100,
        alphas: np.array = None,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
    ) -> None:
        super().__init__(
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_irls_iter = n_irls_iter
        self.tol = tol
        self.groups = groups
        self.group_weights = group_weights

    def _get_estimator(self):
        return CoxPHGroupLasso(
            alpha=0.01,
            groups=self.groups,
            group_weights=self.group_weights,
            n_irls_iter=self.n_irls_iter,
            tol=self.tol,
        )

    def _is_multitask(self):
        return False
