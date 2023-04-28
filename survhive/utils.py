from math import log

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def inverse_transform_survival(
    y: np.array,
) -> tuple[np.array, np.array]:
    event = y >= 0
    event = event.flatten().astype(np.int_)
    time = np.abs(y).flatten()
    return time, event


@jit(nopython=True, cache=True)
def transform_survival(time: np.array, event: np.array) -> np.array:
    y: np.array = np.copy(time)
    y[np.logical_not(event)] = np.negative(y[np.logical_not(event)])
    return y


def inverse_transform_preconditioning(
    y: np.array,
) -> tuple[np.array, np.array]:
    y_survival = np.array([i.rsplit("|")[0] for i in y]).astype(float)
    y_teacher = np.array([i.rsplit("|")[1] for i in y]).astype(float)
    time, event = inverse_transform_survival(y_survival)
    return time, event, y_teacher


def transform_preconditioning(time, event, y_teacher):
    y_survival = transform_survival(time=time, event=event).astype(str)
    y_teacher = y_teacher.astype(str)
    y = np.array(
        [f"{y_survival[i]}|{y_teacher[i]}" for i in range(y_teacher.shape[0])]
    )
    return y


@jit(nopython=True, cache=True)
def logsubstractexp(a, b):
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) - np.exp(b - max_value))


@jit(nopython=True, cache=True)
def logaddexp(a, b):
    max_value = max(a, b)
    return max_value + np.log(np.exp(a - max_value) + np.exp(b - max_value))


@jit(nopython=True, fastmath=True)
def numba_logsumexp_stable(a):
    max_ = np.max(a)
    return max_ + log(np.sum(np.exp(a - max_)))


def _soft_threshold(thresh, a, step_size):
    return np.maximum(np.abs(a) - thresh * step_size, 0.0) * np.sign(a)
