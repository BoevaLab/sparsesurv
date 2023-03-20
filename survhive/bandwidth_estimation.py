import numpy as np
from numba import jit

JONES_1990_PREFACTOR: float = 1.30406
JONES_1991_PREFACTOR: float = 1.5874


@jit(nopython=True)
def jones_1990(time: np.array, event: np.array):
    n: int = time.shape[0]
    sigma: float = np.std(np.log(time[event]))
    print(sigma)
    print(n)
    return JONES_1990_PREFACTOR * sigma * (n**-0.2)


@jit(nopython=True)
def jones_1991(time: np.array, event: np.array):
    n: int = time.shape[0]
    sigma: float = np.std(time)
    return JONES_1991_PREFACTOR * sigma * (n**-1 / 3)
