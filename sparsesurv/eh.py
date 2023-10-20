import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.optimize._hessian_update_strategy import BFGS
from sklearn.exceptions import ConvergenceWarning

from ._base import SurvivalMixin
from .baseline_hazard_estimation import get_cumulative_hazard_function_eh
from .gradients import eh_gradient_beta
from .loss import eh_negative_likelihood_beta
from .utils import inverse_transform_survival


class EH(SurvivalMixin):
    """Linear Extended Hazards (EH) model based on kernel-smoothed PL [1].

    Fits a linear EH model based on the kernel smoothed profile likelihood
    as proposed by [1]. Uses the `trust-ncg` algorithm implementation
    from 'scipy.optimize.minimize` for optimization using a BFGS [2]
    quasi-Newton strategy. Gradients are JIT-compiled using numba
    and implemented in an efficient manner (see `pcsurv.gradients`).

    References:
        [1] Tseng, Yi-Kuan, and Ken-Ning Shu. "Efficient estimation for a semiparametric extended hazards model." Communications in Statistics—Simulation and Computation® 40.2 (2011): 258-273.

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
        """Constructor.

        Args:
            bandwidth (Optional[float], optional): Bandwidth to be used for kernel
                smoothing the profile likelihood. If left unspecified (i.e., `None`),
                optimal bandwidth will be estimted empirically, similar to previous work [[3]_, [4]_].
                Defaults to None.
            tol (float, optional): Tolerance for terminating the `trust-ncg`
                algorithm in scipy. Defaults to None.
            options (Dict[str, Union[bool, int, float]], optional):  Solver-specific
                configuration options of the `trust-ncg` solver in scipy. Defaults to None.
        """
        self.bandwidth: Optional[float] = bandwidth
        self.tol: Optional[float] = tol
        self.options: Optional[Dict[str, Union[bool, int, float]]] = options

    def init_coefs(self, X) -> npt.NDArray[np.float64]:
        """Initializes the coefficients of the EH model at all zeros.

        Args:
            X (_type_): Training design matrix with n rows and p columns.

        Returns:
            npt.NDArray[np.float64]: Initialized coefficients with p rows and 2 columns.
        """

        return np.zeros(X.shape[1])

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: np.array,
        sample_weight: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Fits the linear AFT model using the `trust-ncg` implementation from `scipy`.

        Args:
            X (npt.NDArray[np.float64]): Design matrix.
            y (np.array): Structured array containing right-censored survival information.
            sample_weight (npt.NDArray[np.float64], optional): Sample weight used during model fitting.
                Currently unused and kept for sklearn compatibility. Defaults to None.
            sample_weight (npt.NDArray[np.float64]): Kept for API compatibility.
        """

        time: npt.NDArray[np.float64]
        event: npt.NDArray[np.float64]
        time, event = inverse_transform_survival(y)
        # Since `minimize` works solely with vectors (i.e.,
        # one-dimensional) matrices, we stack the coefficients
        # into a vector. For this, we also duplicate X for easier
        # `sklearn`` compatability.
        X: npt.NDArray[np.float64] = np.concatenate((X, X), axis=1)
        res: Any = minimize(
            fun=eh_negative_likelihood_beta,
            x0=self.init_coefs(X=X),
            args=(X, time, event, self.bandwidth),
            method="trust-ncg",
            jac=eh_gradient_beta,
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
            warnings.warn("Convergence of the EH model failed.", ConvergenceWarning)
            self.coef_: npt.NDArray[np.float64] = res.x

        # Cache training eta, time and event for calculating
        # the cumulative hazard (or, rather, survival) function later.
        self.train_eta: npt.NDArray[np.float64] = X @ self.coef_
        self.train_time: npt.NDArray[np.float64] = time
        self.train_event: npt.NDArray[np.float64] = event
        return None

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate linear predictor for the EH model.

        Note:
            Since the EH model has two coefficient vectors, we need a slightly different
                signature relative to the standard X @ beta approach.

        Args:
            X (npt.NDArray[np.float64]): Query design matrix with u rows and p columns.

        Returns:
            npt.NDArray[np.float64]: Query linear predictor with u rows and 2 columns.
        """

        n_features: int = X.shape[1]
        query_eta: npt.NDArray[np.float64] = np.concatenate(
            (X @ self.coef_[:n_features], X @ self.coef_[n_features:])
        )
        return query_eta

    def predict_cumulative_hazard_function(
        self, X: npt.NDArray[np.float64], time: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predict cumulative hazard function for patients in `X` at times `time`.

        Args:
            X (npt.NDArray[np.float64]): Query design matrix with u rows and p columns.
            time (npt.NDArray[np.float64]): Query times of dimension k. Assumed to be unique and ordered.

        Raises:
            ValueError: Raises ValueError when the event times are not unique and sorted in ascending order.

        Returns:
            npt.NDArray[np.float64]: Query cumulative hazard function for samples 1, ..., u
                and times 1, ..., k. Thus, has u rows and k columns.
        """

        if not np.array_equal(time, np.unique(time)):
            raise ValueError(
                "Expected `time` to be unique and sorted in ascending order."
            )
        cumulative_hazard_function: npt.NDArray[
            np.float64
        ] = get_cumulative_hazard_function_eh(
            time_query=time,
            eta_query=self.predict(X=X),
            time_train=self.time_train,
            event_train=self.event_train,
            eta_train=self.eta_train,
        )
        return cumulative_hazard_function
