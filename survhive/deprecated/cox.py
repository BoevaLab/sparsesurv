from typing import Callable, Optional

import numpy as np
import pandas as pd

from ._base import PreconditionedLinearSurvivalModel
from .compat import BASELINE_HAZARD_FACTORY, GRADIENT_FACTORY, LOSS_FACTORY


class CoxPHPrecond(PreconditionedLinearSurvivalModel):
    def __init__(
        self,
        tie_correction,
        alpha: float,
        tau: Optional[float] = 1.0,
        maxiter=1000,
        rtol=1e-6,
        verbose=0,
        default_step_size=1.0,
        check_global_kkt=True,
    ) -> None:
        super().__init__(
            alpha=alpha,
            tau=tau,
            maxiter=maxiter,
            rtol=rtol,
            verbose=verbose,
            default_step_size=default_step_size,
            check_global_kkt=check_global_kkt,
        )
        if tie_correction not in ["efron", "breslow"]:
            raise ValueError(
                "Expected `tie_correction` to be in ['efron', 'breslow']."
                + f"Found {tie_correction} instead."
            )
        self.gradient: Callable = GRADIENT_FACTORY[
            f"{tie_correction}_preconditioning"
        ]
        self.loss: Callable = LOSS_FACTORY[f"{tie_correction}_preconditioning"]
        self.baseline_hazard_estimator: Callable = BASELINE_HAZARD_FACTORY[
            tie_correction
        ]
        self.tie_correction = tie_correction

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
        ) = BASELINE_HAZARD_FACTORY[self.tie_correction](
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
                np.digitize(
                    x=time, bins=cumulative_baseline_hazards_times, right=False
                )
                - 1
            ],
            reps=X.shape[0],
        ).reshape((X.shape[0], time.shape[0]))
        log_hazards: np.array = (
            np.tile(
                A=self.predict(X),
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
