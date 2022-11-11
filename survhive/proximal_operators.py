from typing import List

import numpy as np
from numba import jit

# TODO:
# - Figure out whether the latent group lasso stuff will work
# for everything?
# - Other biselection regularizers


@jit(nopython=True, cache=True)
def _soft_threshold(x: np.array, threshold: float) -> np.array:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@jit(nopython=True, cache=True)
def _soft_threshold_group(x: np.array, threshold: float) -> np.array:
    return np.sign(x) * np.maximum(
        np.abs(x) - threshold * (x / np.linalg.norm(x, ord=2)), 0.0
    )


@jit(nopython=True, cache=True)
def _scad_thresh(x: np.array, threshold: float, a: float) -> np.array:
    lower_mask: np.array = np.abs(x) <= (2.0 * threshold)
    upper_mask: np.array = np.abs(x) > (threshold * a)
    middle_mask: np.array = (
        np.ones(lower_mask.shape) - upper_mask - middle_mask
    )
    return (
        _soft_threshold(x=x, threshold=threshold) * lower_mask
        + ((a - 1) / (a - 2))
        * _soft_threshold(x=x, threshold=(a * threshold) / (a - 1))
        * middle_mask
        + x * upper_mask
    )


@jit(nopython=True, cache=True)
def _scad_thresh_group(x: np.array, threshold: float, a: float) -> np.array:
    lower_mask: np.array = np.abs(x) <= (2.0 * threshold)
    upper_mask: np.array = np.abs(x) > (threshold * a)
    middle_mask: np.array = (
        np.ones(lower_mask.shape) - upper_mask - middle_mask
    )
    return (
        _soft_threshold_group(x=x, threshold=threshold) * lower_mask
        + ((a - 1) / (a - 2))
        * _soft_threshold_group(x=x, threshold=(a * threshold) / (a - 1))
        * middle_mask
        + x * upper_mask
    )


@jit(nopython=True, cache=True)
def _mcp_thresh(x: np.array, threshold: float, gamma: float):
    mask: np.array = np.sign(x) > (threshold * gamma)
    return (gamma / (gamma - 1)) * _soft_threshold(
        x=x, threshold=threshold
    ) * (1 - mask) + x * mask


@jit(nopython=True, cache=True)
def _mcp_thresh_group(x: np.array, threshold: float, gamma: float) -> np.array:
    mask: np.array = np.sign(x) > (threshold * gamma)
    return (gamma / (gamma - 1)) * _soft_threshold_group(
        x=x, threshold=threshold
    ) * (1 - mask) + x * mask


class ProximalOperator:
    def __init__(self, threshold) -> None:
        self.threshold: float = threshold

    def __call__(self, coef: np.array) -> np.array:
        return NotImplementedError


class LassoProximal(ProximalOperator):
    def __init__(self, threshold: float) -> None:
        super().__init__(threshold)

    def __call__(self, coef: np.array) -> np.array:
        return _soft_threshold(coef, self.threshold)


class GroupLassoProximal(ProximalOperator):
    def __init__(self, threshold: float, groups: List[np.array]) -> None:
        super().__init__(threshold)
        self.groups: List[np.array] = groups

    def __call__(self, coef: np.array) -> np.array:
        for group in self.groups:
            coef[group] = _soft_threshold_group(coef[group], self.threshold)
        return coef


class SparseGroupLassoProximal(ProximalOperator):
    def __init__(
        self, threshold: float, alpha: float, groups: List[np.array]
    ) -> None:
        super().__init__(threshold)
        self.alpha: float = alpha
        self.groups: List[np.array] = groups

    def __call__(self, coef: np.array) -> np.array:
        return GroupLassoProximal(
            threshold=self.threshold * (1 - self.alpha), groups=self.groups
        )(coef) * LassoProximal(threshold=self.threshold * self.alpha)(coef)


class SCADProximal(ProximalOperator):
    def __init__(self, threshold: float, a: float = 3.7) -> None:
        super().__init__(threshold)
        self.a: float = a

    def __call__(self, coef: np.array) -> np.array:
        _scad_thresh(coef, threshold=self.threshold, a=self.a)


class GroupSCADProximal(ProximalOperator):
    def __init__(
        self, threshold: float, groups: List[np.array], a: float = 3.7
    ) -> None:
        super().__init__(threshold)
        self.a: float = a
        self.groups: List[np.array] = groups

    def __call__(self, coef: np.array) -> np.array:
        for group in self.groups:
            coef[group] = _scad_thresh_group(
                coef[group], threshold=self.threshold, gamma=self.gamma
            )
        return coef


class MPCProximal(ProximalOperator):
    def __init__(self, threshold: float, gamma=4.0) -> None:
        super().__init__(threshold)
        self.gamma: float = gamma

    def __call__(self, coef: np.array) -> np.array:
        _mcp_thresh(coef, threshold=self.threshold, gamma=self.gamma)


class MPCGroupProximal(ProximalOperator):
    def __init__(self, threshold: float, gamma=4.0) -> None:
        super().__init__(threshold)
        self.gamma: float = gamma

    def __call__(self, coef: np.array) -> np.array:
        for group in self.groups:
            coef[group] = _mcp_thresh_group(
                coef[group], threshold=self.threshold, gamma=self.gamma
            )
        return coef

