import warnings

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from .utils import inverse_transform_survival, transform_survival, transform_survival_kd


class SurvivalMixin(BaseEstimator):
    """Mixin to calculate the estimated survival function via the estimated cumulative hazard function."""

    def predict_survival_function(
        self, X: npt.NDArray[np.float64], time: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate and return the survival function of the student at times `time` based on the design matrix `X`.

        Args:
            X (npt.NDArray[np.float64]): Design matrix. Has n rows, where
            n corresponds to the number of samples and p columns, where
            p corresponds to the number of covariates.
            time (npt.NDArray[np.float64]): Times at which survival
            function predictions are to be made. Must be unique
            and sorted in ascending order.

        Returns:
            npt.NDArray[np.float64]: Survival function as predicted by
            the student, at times `time` and based on design matrix
            `X`. Has `len(time)` columns and `X.shape[0]` rows.
        """
        if np.unique(time).shape[0] < time.shape[0]:
            warnings.warn(
                "Requested times are non-unique."
                + " Returning predictions for unique times only.",
                UserWarning,
            )
        # Take unique time elements and sort them.
        time = np.unique(time)
        return np.exp(
            np.negative(self.predict_cumulative_hazard_function(X=X, time=time))
        )

    def predict(self, X: npt.NDArray[np.float64]):
        """Calculate and return the linear predictor of the student based on the design matrix `X`.

        Args:
            X (npt.NDArray[np.float64]): Design matrix. Has n rows, where
            n corresponds to the number of samples and p columns, where
            p corresponds to the number of covariates.

        Returns:
            npt.NDArray[np.float64]: Linear predictor.
        """
        return X @ self.coef_


class KDSurv:
    """Wrapper class to perform knowledge distillation (KD) for sparse survival models [1, 2].

    References:
        [1] Our manuscript is still under review.
        [2] Paul, Debashis, et al. "“Preconditioning” for feature selection and regression in high-dimensional problems." (2008): 1595-1618.
    """

    def __init__(self, teacher: BaseEstimator, student: BaseEstimator) -> None:
        """Constructor.

        Args:
            teacher (BaseEstimator): Teacher model that is to be used to generate
            a predictor that the student will try to approximate with as few covariates
            as possible.
            student (BaseEstimator): Student model that aims to achieve sparsity
            while approximating the predictor of the teacher.

        See also:
            sksurv.linear_model.CoxPHSurvivalAnalysis
            sparsesurv.aft.AFT
            sparsesurv.eh.EH
            sparsesurv.cv.KDPHElasticNetCV
            sparsesurv.cv.KDAFTElasticNetCV
            sparsesurv.cv.KDEHMultiTaskLassoCV
        """
        self.teacher = teacher
        self.student = student

    def fit(self, X: npt.NDArray[np.float64], y: np.array) -> None:
        """Fit a model for knowledge distillation in sparse survival analyis.

        In particular, first estimate the teacher model on the right-censored survival
        time `y` and the design matrix `X`. Afterward, estimate the student model
        by fitting an appropiately regularized linear regression on model on `X` and
        `z`, where correspond to the predictions of the teacher on `X`.

        Args:
            X (npt.NDArray[np.float64]): Design matrix. Has n rows, where
                n corresponds to the number of samples and p columns, where
                p corresponds to the number of covariates.
            y (np.array): Structured array containing right-censored survival information.
        """
        time: npt.NDArray[np.float64]
        event: npt.NDArray[np.float64]
        time, event = inverse_transform_survival(y=y)
        time_ix: npt.NDArray[np.int64] = np.argsort(time)

        X: npt.NDArray[np.float64] = X[time_ix, :]
        time = time[time_ix]
        event = event[time_ix]
        self.teacher.fit(X=X, y=transform_survival(time=time, event=event))
        self.student.fit(
            X=X,
            y=transform_survival_kd(
                time=time, event=event, eta_hat=self.pc_pipe.predict(X=X)
            ),
        )

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate and return the linear predictor of the student based on the design matrix `X`.

        Args:
            X (npt.NDArray[np.float64]): Design matrix. Has n rows, where
            n corresponds to the number of samples and p columns, where
            p corresponds to the number of covariates.

        Returns:
            npt.NDArray[np.float64]: Linear predictor as predicted by
            the student, based on design matrix `X`. Has length
            `X.shape[0]`.
        """
        return self.student.predict(X=X)

    def predict_survival_function(
        self, X: npt.NDArray[np.float64], time: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate and return the survival function of the student at times `time` based on the design matrix `X`.

        Args:
            X (npt.NDArray[np.float64]): Design matrix. Has n rows, where
            n corresponds to the number of samples and p columns, where
            p corresponds to the number of covariates.
            time (npt.NDArray[np.float64]): Times at which survival
            function predictions are to be made. Must be unique
            and sorted in ascending order.

        Returns:
            npt.NDArray[np.float64]: Survival function as predicted by
            the student, at times `time` and based on design matrix
            `X`. Has `len(time)` columns and `X.shape[0]` rows.
        """
        return self.student.predict_survival_function(X=X, time=time)
