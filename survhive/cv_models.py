from typing import Union

from numpy.typing import ArrayLike
from typing import Union
from typing import Tuple, List
from typeguard import typechecked

from survhive._utils_.cv import CrossValidation, alpha_path_eta
from survhive.aft import *
from survhive.ah import *
from survhive.cox import *


@typechecked
class AHLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 0.05,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = [1.0],
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self.optimiser = optimiser

    def _get_estimator(self):
        return AHLasso(alpha=0.01, optimiser=self.optimiser)

    def _is_multitask(self):
        return False


@typechecked
class AHElasticNetCV(CrossValidation):

    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _get_estimator(self):
        return AHElasticNet(alpha=0.01, optimiser=self.optimiser, l1_ratio=0.5)

    def _is_multitask(self):
        return False


@typechecked
class AHGroupLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _get_estimator(self):
        return AHGroupLasso(alpha=0.01, optimiser=self.optimiser, groups=[[0, 1]])

    def _is_multitask(self):
        return False


class AFTLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = [1.0],
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self.optimiser = optimiser

    def _get_estimator(self):
        return AFTLasso(alpha=0.01, optimiser=self.optimiser)

    def _is_multitask(self):
        return False


@typechecked
class AFTElasticNetCV(CrossValidation):

    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _get_estimator(self):
        return AFTElasticNet(alpha=0.01, optimiser=self.optimiser, l1_ratio=0.5)

    def _is_multitask(self):
        return False


@typechecked
class AFTGroupLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _get_estimator(self):
        return AFTGroupLasso(alpha=0.01, optimiser=self.optimiser, groups=[[0, 1]])

    def _is_multitask(self):
        return False


@typechecked
class CoxPHLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = [1.0],
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self.optimiser = optimiser

    def _get_estimator(self):
        return CoxPHLasso(alpha=0.01, optimiser=self.optimiser, warm_start=True)

    def _is_multitask(self):
        return False


@typechecked
class CoxPHElasticNetCV(CrossValidation):

    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        tie_correction: str = "efron",
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.tie_correction = tie_correction

    def _get_estimator(self):
        return CoxPHElasticNet(
            alpha=0.01,
            optimiser=self.optimiser,
            l1_ratio=0.5,
            tie_correction=self.tie_correction,
        )

    def _is_multitask(self):
        return False


@typechecked
class CoxPHGroupLassoCV(CrossValidation):
    path = staticmethod(alpha_path_eta)

    def __init__(
        self,
        optimiser: str,
        cv_score_method: str = "linear_predictor",
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        l1_ratios: Union[float, ArrayLike] = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: Union[int, object] = None,
        n_jobs: int = None,
        random_state: int = None,
    ) -> None:
        super().__init__(
            optimiser=optimiser,
            cv_score_method=cv_score_method,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _get_estimator(self):
        return CoxPHGroupLasso(alpha=0.01, optimiser=self.optimiser, groups=[[0, 1]])

    def _is_multitask(self):
        return False
