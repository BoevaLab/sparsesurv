import warnings

import numpy as np
from sklearn.base import BaseEstimator

from .utils import (
    inverse_transform_survival,
    transform_survival,
    transform_survival_preconditioning,
)


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
            np.negative(
                self.predict_cumulative_hazard_function(X=X, time=time)
            )
        )

    def predict(self, X):
        return X @ self.coef_


class PCSurvCV:
    def __init__(self, pc_pipe, model_pipe) -> None:
        self.pc_pipe = pc_pipe
        self.model_pipe = model_pipe

    def fit(self, X, y):
        time, event = inverse_transform_survival(y=y)
        time_ix = np.argsort(time)

        X = X[time_ix, :]
        time = time[time_ix]
        event = event[time_ix]
        self.pc_pipe.fit(X=X, y=transform_survival(time=time, event=event))
        self.model_pipe.fit(
            X=X,
            y=transform_survival_preconditioning(
                time=time, event=event, eta_hat=self.pc_pipe.predict(X=X)
            ),
        )

    def predict(self, X):
        return self.model_pipe.predict(X=X)

    def predict_survival_function(self, X, time):
        return self.model_pipe.predict_survival_function(X=X, time=time)


class PCSurv:
    def __init__(self, pc_pipe, model_pipe) -> None:
        self.pc_pipe = pc_pipe
        self.model_pipe = model_pipe

    def fit(self, X, y):
        time, event = inverse_transform_survival(y=y)
        time_ix = np.argsort(time)

        X = X[time_ix, :]
        time = time[time_ix]
        event = event[time_ix]
        self.pc_pipe.fit(X=X, y=transform_survival(time=time, event=event))
        self.model_pipe.fit(
            X=X,
            y=transform_survival_preconditioning(
                time=time, event=event, eta_hat=self.pc_pipe.predict(X=X)
            ),
        )

    def predict(self, X):
        return self.model_pipe.predict(X=X)

    def predict_survival_function(self, X, time):
        return self.model_pipe.predict_survival_function(X=X, time=time)
