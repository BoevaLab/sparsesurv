import warnings

import numpy as np
from sklearn.base import BaseEstimator

from .utils import inverse_transform_survival, transform_survival, transform_survival_kd


class SurvivalMixin(BaseEstimator):
    def predict_survival_function(self, X, time):
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

    def predict(self, X):
        return X @ self.coef_


class KDSurv:
    def __init__(self, teacher, student) -> None:
        self.teacher = teacher
        self.student = student

    def fit(self, X, y):
        time, event = inverse_transform_survival(y=y)
        time_ix = np.argsort(time)

        X = X[time_ix, :]
        time = time[time_ix]
        event = event[time_ix]
        self.teacher.fit(X=X, y=transform_survival(time=time, event=event))
        self.student.fit(
            X=X,
            y=transform_survival_kd(
                time=time, event=event, eta_hat=self.teacher.predict(X=X)
            ),
        )

    def predict(self, X):
        return self.student.predict(X=X)

    def predict_survival_function(self, X, time):
        return self.student.predict_survival_function(X=X, time=time)
