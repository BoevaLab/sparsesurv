from typing import Dict, Optional, Union, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.optimize._hessian_update_strategy import BFGS
from typeguard import typechecked

from ._base import SurvivalMixin
from .baseline_hazard_estimation import get_cumulative_hazard_function_aft
from .exceptions import ConvergenceException
from .gradients import aft_gradient_beta
from .loss import aft_negative_likelihood_beta
from .utils import inverse_transform_survival


#@typechecked
class AFT(SurvivalMixin):
    """Linear Accelerated Failure Time (AFT) model based on kernel-smoothed PL [1].

    Fits a linear AFT model based on the kernel smoothed profile likelihood
    as proposed by [2]. Uses the `trust-ncg` algorithm implementation
    from 'scipy.optimize.minimize` for optimization using a BFGS [2]
    quasi-Newton strategy. Gradients are JIT-compiled using numba
    and implemented in an efficient manner (see `pcsurv.gradients`).

    Parameters
    ----------
    bandwidth: Optional[float]
        Bandwidth to be used for kernel smoothing the profile likelihood.
        If left unspecified (i.e., `None`), optimal bandwidth will be
        estimted empirically, similar to previous work [3, 4].
    tol: Optional[float]
        Tolerance for terminating the `trust-ncg` algorithm in scipy.
    options: Optional[Dict[str, Union[bool, int, float]]]
        Solver-specific configuration options of the `trust-ncg`
        solver in scipy.

    Attributes
    ----------
    bandwidth: Optional[float]
        Bandwidth to be used for kernel smoothing the profile likelihood.
        If left unspecified (i.e., `None`), optimal bandwidth will be
        estimted empirically, similar to previous work [3, 4].
    tol: Optional[float]
        Tolerance for terminating the `trust-ncg` algorithm in scipy.
    options: Optional[Dict[str, Union[bool, int, float]]]
        Solver-specific configuration options of the `trust-ncg`
        solver in scipy.

    References
    ----------
    [1] Zeng, Donglin, and D. Y. Lin. "Efficient estimation for the accelerated failure time model." Journal of the American Statistical Association 102.480 (2007): 1387-1396.
    [2] Fletcher, Roger. Practical methods of optimization. John Wiley & Sons, 2000.
    [3] Sheather, Simon J., and Michael C. Jones. "A reliable data‐based bandwidth selection method for kernel density estimation." Journal of the Royal Statistical Society: Series B (Methodological) 53.3 (1991): 683-690.
    [4] Zhong, Qixian, Jonas W. Mueller, and Jane-Ling Wang. "Deep extended hazard models for survival analysis." Advances in Neural Information Processing Systems 34 (2021): 15111-15124.
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        tol: Optional[float] = None,
        options: Optional[Dict[str, Union[bool, int, float]]] = None,
    ) -> None:
        self.bandwidth: Optional[float] = bandwidth
        self.tol: Optional[float] = tol
        self.options: Optional[Dict[str, Union[bool, int, float]]] = options

    def init_coefs(self, X) -> npt.NDArray[np.float64]:
        """Initializes the coefficients of the AFT model at all zeros.

        Parameters
        ----------
        X: npt.NDArray[np.float64]
            Training design matrix with n rows and p columns.

        Returns
        -------
        npt.NDArray[np.float64]
            Initialized coefficients with p rows and 2 columns.
        """
        return np.zeros(X.shape[1])

    def fit(self, X, y, sample_weight=None):
        """Fits the linear AFT model using quasi-Newton methods.

        Parameters
        ----------
        X: np.NDArray[np.float64]
            Design matrix.
        y: np.array
            Structured array containing right-censored survival information.
        sample_weight: Optional[np.NDarray[np.float64]]
            Sample weight used during model fitting. Currently unused
            and kept for sklearn compatibility.

        Returns
        -------
        None
        """
        # TODO DW: Add some additional arg checks here.
        time: npt.NDArray[np.float64]
        event: npt.NDArray[np.int64]
        time, event = inverse_transform_survival(y)
        # TODO DW: Add correct typing here.
        res: Any = minimize(
            fun=aft_negative_likelihood_beta,
            x0=self.init_coefs(X=X),
            args=(X, time, event),
            method="trust-ncg",
            jac=aft_gradient_beta,
            hess=BFGS(),
            hessp=None,
            bounds=None,
            constraints=None,
            tol=self.tol,
            callback=None,
            options=self.options,
        )
        if res.success:
            self.coef_: npt.NDArray[np.float64] = res.x
        else:
            self.coef_: npt.NDArray[np.float64] = res.x
            # raise ConvergenceException(
            #     msg="Convergence of the AFT model failed.", code=1
            # )
        # Cache training eta, time and event for calculating
        # the cumulative hazard (or, rather, survival) function later.
        self.train_eta: npt.NDArray[np.float64] = X @ self.coef_
        self.train_time: npt.NDArray[np.float64] = time
        self.train_event: npt.NDArray[np.int64] = event
        return None

    def predict_cumulative_hazard_function(
        self, X: npt.NDArray[np.float64], time: npt.NDArray[np.float64]
    ):
        """Calculate cumulative hazard function for the AFT model.

        Parameters
        ----------
        X: npt.NDArray[np.float64]
            Query design matrix with u rows and p columns.
        time: npt.NDArray[np.float64]
            Query times of dimension k. Assumed to be unique and ordered.

        Returns
        -------
        cumulative_hazard_function: npt.NDArray[np.float64]
            Query cumulative hazard function for samples 1, ..., u
            and times 1, ..., k. Thus, has u rows and k columns.
        """
        if not np.array_equal(time, np.unique(time)):
            raise ValueError("Times passed to ")
        cumulative_hazard_function: npt.NDArray[
            np.float64
        ] = get_cumulative_hazard_function_aft(
            time_query=time,
            eta_query=self.predict(X=X),
            time_train=self.train_time,
            event_train=self.train_event,
            eta_train=self.train_eta,
        )
        return cumulative_hazard_function
