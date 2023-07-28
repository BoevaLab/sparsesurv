from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import jit

from .constants import CDF_ZERO, PDF_PREFACTOR
from .utils import difference_kernels


@jit(nopython=True, cache=True, fastmath=True)
def aft_gradient(
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    bandwidth: Optional[float] = None,
):
    """Calculates the negative gradient of the AFT model wrt eta.

    Parameters
    ----------
    linear_predictor: npt.NDArray[np.float64]
        Linear predictor of the training data. Of dimension n.
    time: npt.NDArray[np.float64]
        Time of the training data. Of dimension n. Assumed to be sorted
        (does not matter here, but regardless).
    event: npt.NDArray[np.int64]
        Event indicator of the training data. Of dimension n.
    bandwidth: Optional[float]
        Bandwidth to kernel-smooth the profile likelihood. Will
        be estimated if not specified.

    Returns
    -------
    gradient: npt.NDArray[np.float64]
        Negative gradient of the AFT model wrt eta. Of dimensionality n.
    """
    linear_predictor: npt.NDArray[np.float64] = np.exp(linear_predictor)
    linear_predictor = np.log(time * linear_predictor)
    n_samples: int = time.shape[0]

    # Estimate bandwidth using an estimate proportional to the
    # the optimal bandwidth.
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)
    gradient: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size_bandwidth: float = (
        inverse_sample_size * inverse_bandwidth
    )

    zero_kernel: float = PDF_PREFACTOR
    event_count: int = 0

    # Cache various calculated quantities to reuse during later
    # calculation of the gradient.
    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)

    integrated_kernel_denominator: np.array = integrated_kernel_matrix.sum(
        axis=0
    )

    for _ in range(n_samples):

        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                kernel_matrix[_, :]
                * inverse_bandwidth
                / integrated_kernel_denominator
            ).sum()
        )

        if sample_event:
            gradient_correction_factor = (
                inverse_sample_size_bandwidth
                * zero_kernel
                / integrated_kernel_denominator[event_count]
            )

            gradient_one = -(
                inverse_sample_size
                * (
                    kernel_numerator_full[
                        _,
                    ]
                    / kernel_denominator
                ).sum()
            )

            prefactor: float = kernel_numerator_full[
                event_mask, event_count
            ].sum() / (kernel_denominator[event_count])

            gradient_two = inverse_sample_size * prefactor

            prefactor = (
                (kernel_matrix[:, event_count].sum() - zero_kernel)
                * inverse_bandwidth
                / integrated_kernel_matrix[:, event_count].sum()
            )
            gradient_four = inverse_sample_size * prefactor

            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            )

            event_count += 1

        else:
            gradient[_] = gradient_three

    # Flip the gradient sign since we are performing minimization.
    gradient = np.negative(gradient)
    return gradient


def aft_gradient_beta(
    beta: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    bandwidth: Optional[float] = None,
):
    """Calculates the negative gradient of the AFT model wrt beta.

    Utility function to be used with off-the-shelf optimisers (e.g., scipy).
    Since the main gradient function calculates the gradient wrt eta
    (see `aft_gradient`), we recover the gradient wrt beta through a
    matrix multiplication.

    Parameters
    ----------
    beta: npt.NDArray[np.float64]
        Coefficient vector. Length p.
    X: npt.NDarray[np.float64]
        Design matrix of the training data. N rows and p columns.
    time: npt.NDArray[np.float64]
        Time of the training data. Length n. Assumed to be sorted
        (does not matter here, but regardless).
    event: npt.NDArray[np.int64]
        Event indicator of the training data. Length n.
    bandwidth: Optional[float]
        Bandwidth to kernel-smooth the profile likelihood. Will
        be estimated empirically if not specified.

    Returns
    -------
    beta_gradient: npt.NDArray[np.float64]
        Negative gradient of the AFT model wrt beta. Length p.
    """
    # if not np.array_equal(np.sort(time), time):
    #     raise ValueError(
    #         "`time` is expected to be sorted (ascending). Unsorted `time` found instead."
    #     )
    eta_gradient: npt.NDArray[np.float64] = aft_gradient(
        linear_predictor=np.matmul(X, beta),
        time=time,
        event=event,
        bandwidth=bandwidth,
    )
    beta_gradient: npt.NDArray[np.float64] = np.matmul(X.T, eta_gradient)
    return beta_gradient


@jit(nopython=True, cache=True, fastmath=True)
def eh_gradient(
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    bandwidth: Optional[float] = None,
) -> np.array:
    """Calculates the negative gradient of the EH model wrt eta.

    Parameters
    ----------
    linear_predictor: npt.NDArray[np.float64]
        Linear predictor of the training data. N rows and 2 columns.
    time: npt.NDArray[np.float64]
        Time of the training data. Of dimension n. Assumed to be sorted
        (does not matter here, but regardless).
    event: npt.NDArray[np.int64]
        Event indicator of the training data. Of dimension n.
    bandwidth: Optional[float]
        Bandwidth to kernel-smooth the profile likelihood. Will
        be estimated if not specified.

    Returns
    -------
    gradient: npt.NDArray[np.float64]
        Negative gradient of the EH model wrt eta. Of dimensionality 2n.
    """
    n_samples: int = time.shape[0]
    n_events: int = np.sum(event)

    # Estimate bandwidth using an estimate proportional to the
    # the optimal bandwidth.
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)

    # Cache various calculated quantities to reuse during later
    # calculation of the gradient.
    theta = np.exp(linear_predictor)

    linear_predictor_misc = np.log(time * theta[:, 0])

    linear_predictor_vanilla: np.array = theta[:, 1] / theta[:, 0]

    # Calling these cox and aft respectively, since setting
    # the respectively other coefficient to zero recovers
    # the (kernel-smoothed PL) model of the other one (e.g.,
    # setting Cox to zero recovers AFT and vice-versa).
    gradient_eta_cox: np.array = np.empty(n_samples)
    gradient_eta_aft: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    zero_kernel: float = PDF_PREFACTOR
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor_misc,
        b=linear_predictor_misc[event_mask],
        bandwidth=bandwidth,
    )

    sample_repeated_linear_predictor: np.array = (
        linear_predictor_vanilla.repeat(n_events).reshape(
            (n_samples, n_events)
        )
    )

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)

    integrated_kernel_denominator: np.array = (
        integrated_kernel_matrix * sample_repeated_linear_predictor
    ).sum(axis=0)

    for _ in range(n_samples):
        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                (
                    -linear_predictor_vanilla[_]
                    * integrated_kernel_matrix[_, :]
                    + linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        gradient_five = -(
            inverse_sample_size
            * (
                (linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :])
                / integrated_kernel_denominator
            ).sum()
        )

        if sample_event:
            gradient_correction_factor = inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    + linear_predictor_vanilla[_]
                    * zero_kernel
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
            )

            gradient_one = -(
                inverse_sample_size
                * (
                    kernel_numerator_full[
                        _,
                    ]
                    / kernel_denominator
                ).sum()
            )

            prefactor: float = kernel_numerator_full[
                event_mask, event_count
            ].sum() / (kernel_denominator[event_count])

            gradient_two = inverse_sample_size * prefactor

            prefactor = (
                (
                    (
                        linear_predictor_vanilla
                        * kernel_matrix[:, event_count]
                    ).sum()
                    - linear_predictor_vanilla[_] * zero_kernel
                )
                * inverse_bandwidth
                - (linear_predictor_vanilla[_] * zero_integrated_kernel)
            ) / integrated_kernel_denominator[event_count]

            gradient_four = inverse_sample_size * prefactor

            gradient_eta_cox[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            ) - inverse_sample_size

            gradient_eta_aft[_] = gradient_five + inverse_sample_size

            event_count += 1

        else:
            gradient_eta_cox[_] = gradient_three
            gradient_eta_aft[_] = gradient_five
    # Flip the gradient sign since we are performing minimization and
    # concatenate the two gradients since we stack both coefficients
    # into a vector.
    gradient_eta_eh = np.negative(
        np.concatenate((gradient_eta_cox, gradient_eta_aft))
    )
    return gradient_eta_eh


def eh_gradient_beta(
    beta: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    bandwidth: Optional[float] = None,
):
    """Calculates the negative gradient of the EH model wrt beta.

    Utility function to be used with off-the-shelf optimisers (e.g., scipy).
    Since the main gradient function calculates the gradient wrt eta
    (see `eh_gradient`), we recover the gradient wrt beta through a
    matrix multiplication.

    Parameters
    ----------
    beta: npt.NDArray[np.float64]
        Coefficient vector. Length 2p to account for the two
        coefficients that were stacked into one vector (see
        `pcsurv.eh.EH` for details).
    X: npt.NDarray[np.float64]
        Design matrix of the training data. N rows and 2p columns.
    time: npt.NDArray[np.float64]
        Time of the training data. Length n. Assumed to be sorted
        (does not matter here, but regardless).
    event: npt.NDArray[np.int64]
        Event indicator of the training data. Length n.
    bandwidth: Optional[float]
        Bandwidth to kernel-smooth the profile likelihood. Will
        be estimated empirically if not specified.

    Returns
    -------
    beta_eh_gradient: npt.NDArray[np.float64]
        Negative gradient of the AFT model wrt beta. Length 2p.
    """
    # if not np.array_equal(np.sort(time), time):
    #     raise ValueError(
    #         "`time` is expected to be sorted (ascending). Unsorted `time` found instead."
    #     )
    # Calculate original feature size (i.e., get rid of coefficient
    # stacking).
    p: int = int(X.shape[1] / 2)
    n: int = int(X.shape[0])
    beta_cox_gradient: npt.NDArray[np.float64] = np.matmul(
        X[:, :p].T,
        eh_gradient(
            linear_predictor=np.stack(
                (
                    np.matmul(X[:, :p], beta[:p]),
                    np.matmul(X[:, p:], beta[p:]),
                )
            ).T,
            time=time,
            event=event,
            bandwidth=bandwidth,
        )[:n],
    )
    beta_aft_gradient: npt.NDArray[np.float64] = np.matmul(
        X[:, p:].T,
        eh_gradient(
            linear_predictor=np.stack(
                (
                    np.matmul(X[:, :p], beta[:p]),
                    np.matmul(X[:, p:], beta[p:]),
                )
            ).T,
            time=time,
            event=event,
            bandwidth=bandwidth,
        )[n:],
    )
    beta_eh_gradient: npt.NDArray[np.float64] = np.concatenate(
        (beta_cox_gradient, beta_aft_gradient)
    )
    return beta_eh_gradient
