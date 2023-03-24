import numpy as np
from numba import jit

from .constants import JONES_1990_PREFACTOR, JONES_1991_PREFACTOR


# NB: We jit these such that the gradients can be run in
# nopython mode.
@jit(nopython=True, cache=True, fastmath=True)
def jones_1990(time: np.array, event: np.array):
    n: int = time.shape[0]
    sigma: float = np.std(np.log(time[event.astype(np.bool_)]))
    return JONES_1990_PREFACTOR * sigma * (n**-0.2)


@jit(nopython=True, cache=True, fastmath=True)
def jones_1991(time: np.array, event: np.array):
    n: int = time.shape[0]
    sigma: float = np.std(np.log(time))
    return JONES_1991_PREFACTOR * sigma * (n**-1 / 3)
