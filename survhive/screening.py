import numpy as np
import numpy as np
from numba import int32, float32  # import the types
from numba.experimental import jitclass
from numba import njit

from .constants import EPS


def check_kkt(a, b):
    return np.where(a >= (b - EPS))[0]


class StrongScreener(object):
    def __init__(self, p: int, l1_ratio: float):
        self.working_set: np.array = np.array([]).astype(np.int_)
        self.complete_set: np.array = np.arange(p).astype(np.int_)
        self.strong_set: np.array = np.array([]).astype(np.int_)
        self.strong_kkt_violated = np.array([]).astype(np.int_)
        self.any_kkt_violated = np.array([]).astype(np.int_)
        self.ever_active_set = np.array([]).astype(np.int_)
        self.l1_ratio = l1_ratio
        return None

    def compute_strong_set(
        self,
        X,
        y,
        eta_previous,
        alpha,
        alpha_previous,
        correction_factor,
        active,
    ):
        self.strong_set = np.setdiff1d(
            np.where(
                1
                / (X.shape[0])
                * np.abs(np.matmul(X.T, (y - eta_previous) * correction_factor))
                >= self.l1_ratio * (2 * alpha - alpha_previous)
            )[0],
            active,
        )
        return None

    def check_kkt_strong(self, X, y, eta, alpha, correction_factor):
        self.strong_kkt_violated = check_kkt(
            a=1
            / X.shape[0]
            * np.abs(np.matmul(X[:, self.strong_set].T, (y - eta) * correction_factor)),
            b=self.l1_ratio * alpha,
        )
        return None

    def calculate_non_strong_non_working_set(self):
        return np.setdiff1d(
            self.complete_set, np.union1d(self.strong_set, self.working_set)
        )

    def check_kkt_all(self, X, y, eta, alpha, correction_factor):
        self.any_kkt_violated = self.calculate_non_strong_non_working_set()[
            check_kkt(
                a=1
                / X.shape[0]
                * np.abs(
                    np.matmul(
                        X[:, self.calculate_non_strong_non_working_set()].T,
                        (y - eta) * correction_factor,
                    )
                ),
                b=self.l1_ratio * alpha,
            )
        ]
        return None

    def expand_working_set(self, a):
        self.working_set = np.append(self.working_set, a)
        return None

    def expand_working_set_with_kkt_violations(self):
        self.working_set = np.append(
            self.working_set, self.strong_set[self.strong_kkt_violated]
        )
        self.strong_set = np.delete(self.strong_set, self.strong_kkt_violated)
        return None

    def expand_working_set_with_overall_violations(self):
        self.working_set = np.union1d(self.working_set, self.any_kkt_violated)
        self.strong_set = np.setdiff1d(self.strong_set, self.any_kkt_violated)
        return None
