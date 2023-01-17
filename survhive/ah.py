import imp
from typing import List, Union

import copt as cp
import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from scipy.stats import norm
# from sklearn.utils.validation import check_X_y

from sksurv.util import check_array_survival

# TODO:
# depends on the shape of beta X@beta may change to beta@X
# get beta, the input to loss and predictions


class AHSurvivalModel(LinearModel):
    """
    Implement the Accerlated Hazard Model in [1]
    Parameters
    ----------
    alpha : float, optional, default: 1.0
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.
    distribution : one of "Lognormal", "Weibull", "Exponential", "Loglogistic", default: "Exponential"
        Different AFT models with various types of known distribution from prior knowledge.
    bandwith: float, default: 1.0
        the computed bandwidth will be n^(-1/bandwidth) ie.n^-1 by default
    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Weight vector.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    References
    ----------
    [1] https://onlinelibrary.wiley.com/doi/full/10.1111/j.1541-0420.2011.01592.x
    """

    def __init__(
        self,
        alpha: float = 1.0,
        groups: List[Union[int, List[int]]],
        prox: ProximalOperator,
        scale_group: Union[None, str] = "group_length",
        max_iter: int = 1000,
        tol: float = 1e-3,
        solver: str = "copt",
        random_state: Union[None, int] = None,
        bandwidth: float = 1.0,
    ):
        self.alpha = alpha
        self.groups = groups
        self.prox = prox
        self.scale_group = scale_group
        # self.fit_intercept = fit_intercept
        # self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state
        self.bandwdith = bandwidth

        self.beta = np.zero()

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
        """
        self.n_samples = X.shape[0]
        # event, time = check_array_survival(X, y)

        I = y >= 0
        Y = np.abs(y)

        #the Kernel function chosen is the normal denisty function as in the paper
        #i.e. K(.) = norm(.)
        def ah_loss(beta):
            term1 = -1/self.n_samples * np.cumsum(I@self.e_func(beta))

            term2 = 0
            term3 = 0
            for i in range(self.n_samples):
                term2j = 0
                term3j = 0
                for j in range(1):
                    kernel_input = (self.e_func_i(j, beta) - self.e_func_i(i, beta)) / self.bandwidth
                    term2j = I[j] * norm.pdf(kernel_input) 
                    term3j = np.exp(-X[j]@beta) * norm.cdf(kernel_input)
                term2 += I[i] * np.log(1/(self.n_samples * self.bandwidth) * term2j)
                term2 += I[i] * np.log(1/self.n_samples * term2j)
            term2 *= 1/self.n_samples
            term3 *= 1/self.n_samples

            return term1 + term2 - term3

        # super().fit(X, np.log(time), sample_weight=weights)

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

    def predict_hazard_function(
        self, X: pd.DataFrame, y: np.array, time: np.array,
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

        term1 = 0
        term2 = 0
        for i in range(self.n_samples):
            kernel_input = (e_func_i(i, self.beta) - np.log()) / self.bandwidth
            term1 = I[i] * norm.pdf(kernel_input)
            term2 = np.exp(-X[i]@self.beta) * norm.cdf(kernel_input)
        term1 *= 1/(self.n_samples * self.bandwidth * time)
        term2 *= 1/self.n_samples

        return term1 @ np.inverse(term2)


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
            
        return self.predict_hazard_function(X, time).cumsum(axis=1)[np.log(time)]

    def predict_survival_function(
        self, X: pd.DataFrame, time: np.array,
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

    #Utility function to calculate kernel input, coressponds to the e_i(beta) function in the paper
    def e_func_i(self, X: pd.DataFrame, y: np.array, i: int, beta: np.array,) -> np.array:
        return np.log(y[i]) + X[i]@beta

    def e_func(self, X: pd.DataFrame, y: np.array, beta:np.array,) -> np.array:
        return np.log(y) + X@beta
