from typing import Union

import numpy as np

from .cox import CoxPHElasticNet, CoxPHPrecond
from .cv import (
    RegularizedLinearSurvivalModelCV,
    RegularizedPreconditionedLinearSurvivalModelCV,
    alpha_path_eta,
    alpha_path_eta_precond,
)


class CoxPHElasticNetCV(RegularizedLinearSurvivalModelCV):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        tie_correction: str = "efron",
        eps: float = 0.05,
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


class CoxPHLassoCV(CoxPHElasticNetCV):
    def __init__(
        self,
        tie_correction: str = "efron",
        eps: float = 0.05,
        n_alphas: int = 100,
        alphas: np.array = None,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
        n_irls_iter: int = 5,
        tol: float = 0.0001,
    ) -> None:
        super().__init__(
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


class CoxPHPrecondCV(RegularizedPreconditionedLinearSurvivalModelCV):
    path = staticmethod(alpha_path_eta_precond)

    def __init__(
        self,
        tie_correction: str = "efron",
        eps: float = 0.05,
        n_alphas: int = 100,
        alphas: np.array = None,
        taus: Union[float, np.array] = [1.0],
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
        maxiter=1000,
        rtol=1e-6,
        verbose=0,
        default_step_size=1.0,
    ) -> None:
        super().__init__(
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            taus=taus,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.tie_correction = tie_correction
        self.maxiter = maxiter
        self.rtol = rtol
        self.verbose = verbose
        self.default_step_size = default_step_size

    def _get_estimator(self):
        return CoxPHPrecond(
            tie_correction=self.tie_correction,
            alpha=0.01,
            tau=self.taus[0],
            maxiter=self.maxiter,
            rtol=self.rtol,
            verbose=self.verbose,
            default_step_size=self.default_step_size,
        )

    def _is_multitask(self):
        return False
