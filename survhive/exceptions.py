class ConvergenceException(Exception):
    """Signifies that an algorithm failed to converge.

    Parameters
    ----------
    msg: str
        Message indicating failed convergence.
    code: int
        Exception code.

    Attributes
    ----------
    msg: str
        Message indicating failed convergence.
    code: int
        Exception code.

    Notes
    -----
    Some models implemented in `pcsurv`, notably the Accelerated Failure
    Time (AFT) and Extended Hazards (EH) model, are implemented using
    kernel-smoothed profile likelihoods [1, 2]. Since these models
    are implemented using quasi-Newton methods, there may be (rare)
    cases of failed convergence.

    References
    ----------
    [1] Tseng, Yi-Kuan, and Ken-Ning Shu. "Efficient estimation for a semiparametric extended hazards model." Communications in Statistics—Simulation and Computation® 40.2 (2011): 258-273.
    [2] Zeng, Donglin, and D. Y. Lin. "Efficient estimation for the accelerated failure time model." Journal of the American Statistical Association 102.480 (2007): 1387-1396.
    """

    def __init__(self, msg: str, code: int) -> None:
        self.msg: str = msg
        self.code: int = code
