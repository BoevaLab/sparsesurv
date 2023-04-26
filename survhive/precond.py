import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typeguard import typechecked

from ._base import PreconditionedSurvivalModel
from .compat import (
    BASELINE_HAZARD_FACTORY,
    COX_REFERENCE,
    ElasticNetCVPreconditioner,
)


class CoxElasticNetCVPreconditioner(ElasticNetCVPreconditioner):
    def __init__(
        self,
        tie_correction,
        l1_ratio=1.0,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        max_iter=100,
        tol=1e-4,
        cv=None,
        verbose=0,
        max_epochs=50000,
        p0=10,
        prune=True,
        precompute="auto",
        positive=False,
        n_jobs=None,
    ):
        super().__init__(
            model_type=tie_correction,
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            precompute=precompute,
            positive=positive,
            n_jobs=n_jobs,
        )
        self.tie_correction = tie_correction


@typechecked
class CoxPHPreconditionedElasticNet(PreconditionedSurvivalModel):
    def __init__(
        self,
        teacher_pipe=make_pipeline(StandardScaler(), PCA(n_components=16)),
        tie_correction: str = "efron",
        l1_ratio=1.0,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        max_iter=100,
        tol=1e-4,
        cv=None,
        verbose=0,
        max_epochs=50000,
        p0=10,
        prune=True,
        precompute="auto",
        n_jobs=None,
    ) -> None:
        if tie_correction not in ["efron", "breslow"]:
            raise ValueError(
                "Expected `tie_correction` to be in ['efron', 'breslow']."
                + f"Found {tie_correction} instead."
            )
        super().__init__(
            teacher_pipe=teacher_pipe,
            teacher=COX_REFERENCE[tie_correction],
            student=make_pipeline(
                StandardScaler(),
                CoxElasticNetCVPreconditioner(
                    tie_correction=tie_correction,
                    l1_ratio=l1_ratio,
                    eps=eps,
                    n_alphas=n_alphas,
                    alphas=alphas,
                    max_iter=max_iter,
                    tol=tol,
                    cv=cv,
                    verbose=verbose,
                    max_epochs=max_epochs,
                    p0=p0,
                    prune=prune,
                    precompute=precompute,
                    positive=False,
                    n_jobs=n_jobs,
                ),
            ),
        )

        self.tie_correction = tie_correction
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.precompute = precompute
        self.n_jobs = n_jobs

    def predict_cumulative_hazard_function(
        self, X: np.array, time: np.array
    ) -> pd.DataFrame:
        if np.min(time) < 0:
            raise ValueError(
                "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
                + f"Minimum time found was {np.min(time)}."
                + "Please remove any times strictly less than zero."
            )
        cumulative_baseline_hazards_times: np.array
        cumulative_baseline_hazards: np.array
        (
            cumulative_baseline_hazards_times,
            cumulative_baseline_hazards,
        ) = BASELINE_HAZARD_FACTORY[self.student.tie_correction](
            time=self.train_time, event=self.train_event, eta=self.train_eta
        )
        cumulative_baseline_hazards = np.concatenate(
            [np.array([0.0]), cumulative_baseline_hazards]
        )
        cumulative_baseline_hazards_times: np.array = np.concatenate(
            [np.array([0.0]), cumulative_baseline_hazards_times]
        )
        cumulative_baseline_hazards: np.array = np.tile(
            A=cumulative_baseline_hazards[
                np.digitize(x=time, bins=cumulative_baseline_hazards_times, right=False)
                - 1
            ],
            reps=X.shape[0],
        ).reshape((X.shape[0], time.shape[0]))
        log_hazards: np.array = (
            np.tile(
                A=self.student.predict(X),
                reps=time.shape[0],
            )
            .reshape((time.shape[0], X.shape[0]))
            .T
        )
        cumulative_hazard_function: pd.DataFrame = pd.DataFrame(
            cumulative_baseline_hazards * np.exp(log_hazards),
            columns=time,
        )
        return cumulative_hazard_function


@typechecked
class CoxPHPreconditionedElasticNet(CoxPHPreconditionedElasticNet):
    def __init__(
        self,
        teacher_pipe=make_pipeline(StandardScaler(), PCA(n_components=16)),
        tie_correction: str = "efron",
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        max_iter=100,
        tol=1e-4,
        cv=None,
        verbose=0,
        max_epochs=50000,
        p0=10,
        prune=True,
        precompute="auto",
        n_jobs=None,
    ) -> None:
        super().__init__(
            teacher_pipe=teacher_pipe,
            tie_correction=tie_correction,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            precompute=precompute,
            n_jobs=n_jobs,
        )
