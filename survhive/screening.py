import numpy as np

from .constants import EPS


def check_kkt(a, b):
    return np.where(a >= (b - EPS))[0]


class StrongScreener(object):
    def __init__(
        self,
        p: int,
        l1_ratio: float,
    ):
        self.working_set: np.array = np.array([]).astype(np.int_)
        self.complete_set: np.array = np.arange(p).astype(np.int_)
        self.strong_set: np.array = np.array([]).astype(np.int_)
        self.strong_kkt_violated = np.array([]).astype(np.int_)
        self.any_kkt_violated = np.array([]).astype(np.int_)
        self.l1_ratio = l1_ratio

    def compute_strong_set(
        self,
        gradient,
        alpha,
        alpha_previous,
    ):
        self.strong_set = np.setdiff1d(
            np.where(
                np.abs(gradient)
                >= self.l1_ratio * (2 * alpha - alpha_previous)
            )[0],
            self.working_set,
        )

    def check_kkt_strong(self, gradient, alpha):
        self.strong_kkt_violated = check_kkt(
            a=np.abs(gradient), b=self.l1_ratio * alpha
        )

    def check_kkt_all(self, gradient, alpha):
        self.any_kkt_violated = np.setdiff1d(
            check_kkt(np.abs(gradient), (self.l1_ratio * alpha)),
            self.working_set,
        )

    def expand_working_set(self, a):
        self.working_set = np.union1d(self.working_set, a)

    def expand_working_set_with_kkt_violations(self):
        self.working_set = np.append(
            self.working_set, self.strong_set[self.strong_kkt_violated]
        )
        self.strong_set = np.delete(self.strong_set, self.strong_kkt_violated)

    def expand_working_set_with_overall_violations(self):
        self.working_set = np.union1d(self.working_set, self.any_kkt_violated)
        self.strong_set = np.setdiff1d(self.strong_set, self.any_kkt_violated)
