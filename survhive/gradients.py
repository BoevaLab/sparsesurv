from typing import Tuple

import numpy as np
from numba import jit

from .bandwidth_estimation import jones_1990, jones_1991
from .constants import CDF_ZERO, PDF_PREFACTOR, SQRT_EPS
from .utils import difference_kernels


@jit(nopython=True, cache=True, fastmath=True)
def modify_hessian(hessian: np.array, hessian_modification_strategy: str):
    if not np.any(hessian < 0):
        return hessian
    if hessian_modification_strategy == "ignore":
        hessian[hessian < 0] = 0
    elif hessian_modification_strategy == "eps":
        hessian[hessian < 0] = SQRT_EPS
    elif hessian_modification_strategy == "flip":
        # hessian[hessian < 0] = np.negative(hessian[hessian < 0])
        hessian += np.abs(np.min(hessian)) + SQRT_EPS
    return hessian


@jit(nopython=True, cache=True, fastmath=True)
def aft_numba(
    time: np.array,
    event: np.array,
    linear_predictor: np.array,
    bandwidth_function: str,
    hessian_modification_strategy: str = "flip",
):
    # print(linear_predictor)
    # print(time)
    # print(event)
    if bandwidth_function == "jones_1990":
        bandwidth: float = jones_1990(time=time, event=event)
    else:
        bandwidth: float = jones_1991(time=time, event=event)
    # bandwidth = time.shape[0]  ** (-1/7)
    bandwidth = np.std(np.log((time))) * (time.shape[0] ** (-1 / 7))
    # print(bandwidth)
    linear_predictor: np.array = np.exp(linear_predictor)
    linear_predictor = np.log(time * linear_predictor)
    n_samples: int = time.shape[0]
    gradient: np.array = np.empty(n_samples)
    hessian: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    squared_inverse_bandwidth: float = inverse_bandwidth**2
    inverse_sample_size_bandwidth: float = inverse_sample_size * inverse_bandwidth

    zero_kernel: float = PDF_PREFACTOR
    event_count: int = 0
    squared_zero_kernel: float = zero_kernel**2

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )

    squared_kernel_matrix: np.array = np.square(kernel_matrix)
    squared_difference_outer_product: np.array = np.square(difference_outer_product)
    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )
    squared_kernel_numerator: np.array = np.square(kernel_numerator_full[event_mask, :])

    squared_difference_kernel_numerator: np.array = kernel_matrix[event_mask, :] * (
        squared_difference_outer_product[event_mask, :] * squared_inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)
    squared_kernel_denominator: np.array = np.square(kernel_denominator)

    integrated_kernel_denominator: np.array = integrated_kernel_matrix.sum(axis=0)
    squared_integrated_kernel_denominator: np.array = np.square(
        integrated_kernel_denominator
    )

    for _ in range(n_samples):

        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                kernel_matrix[_, :] * inverse_bandwidth / integrated_kernel_denominator
            ).sum()
        )
        hessian_five = (
            inverse_sample_size
            * (
                squared_kernel_matrix[_, :]
                * squared_inverse_bandwidth
                / squared_integrated_kernel_denominator
            ).sum()
        )
        hessian_six = (
            inverse_sample_size
            * (
                kernel_numerator_full[_, :]
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

            hessian_correction_factor = -inverse_sample_size * (
                squared_zero_kernel
                * squared_inverse_bandwidth
                / squared_integrated_kernel_denominator[event_count]
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
            hessian_one = -(
                inverse_sample_size
                * (
                    squared_kernel_numerator[
                        event_count,
                    ]
                    / squared_kernel_denominator
                ).sum()
            )

            hessian_two = inverse_sample_size * (
                (
                    (
                        squared_difference_kernel_numerator[event_count, :]
                        - (
                            kernel_matrix[
                                _,
                            ]
                            * squared_inverse_bandwidth
                        )
                    )
                    / kernel_denominator
                ).sum()
                + (
                    zero_kernel
                    * squared_inverse_bandwidth
                    / kernel_denominator[event_count]
                )
            )

            prefactor: float = kernel_numerator_full[event_mask, event_count].sum() / (
                kernel_denominator[event_count]
            )

            gradient_two = inverse_sample_size * prefactor
            hessian_three = -inverse_sample_size * (prefactor**2)

            hessian_four = inverse_sample_size * (
                (
                    ((squared_difference_kernel_numerator[:, event_count]).sum())
                    - (
                        squared_inverse_bandwidth
                        * ((kernel_matrix[event_mask, event_count]).sum() - zero_kernel)
                    )
                )
                / (kernel_denominator[event_count])
            )
            prefactor = (
                (kernel_matrix[:, event_count].sum() - zero_kernel)
                * inverse_bandwidth
                / integrated_kernel_matrix[:, event_count].sum()
            )
            gradient_four = inverse_sample_size * prefactor

            hessian_seven = inverse_sample_size * (prefactor**2)
            hessian_eight = inverse_sample_size * (
                (kernel_numerator_full[:, event_count] * inverse_bandwidth).sum()
                / integrated_kernel_denominator[event_count]
            )

            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            )
            hessian[_] = (
                hessian_one
                + hessian_two
                + hessian_three
                + hessian_four
                + hessian_five
                + hessian_six
                + hessian_seven
                + hessian_eight
                + hessian_correction_factor
            )
            event_count += 1

        else:
            gradient[_] = gradient_three
            hessian[_] = hessian_five + hessian_six
    return np.negative(gradient), modify_hessian(
        hessian=np.negative(hessian),
        hessian_modification_strategy=hessian_modification_strategy,
    )


@jit(nopython=True, cache=True, fastmath=True)
def ah_numba(
    time: np.array,
    event: np.array,
    linear_predictor: np.array,
    bandwidth_function: str = "jones_1990",
    hessian_modification_strategy: str = "flip",
):
    if bandwidth_function == "jones_1990":
        bandwidth: float = jones_1990(time=time, event=event)
    else:
        bandwidth: float = jones_1991(time=time, event=event)

    linear_predictor_vanilla: np.array = np.exp(linear_predictor)
    linear_predictor = np.log(time * linear_predictor_vanilla)
    n_samples: int = time.shape[0]
    n_events: int = np.sum(event)
    gradient: np.array = np.empty(n_samples)
    hessian: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    squared_inverse_bandwidth: float = inverse_bandwidth**2

    zero_kernel: float = PDF_PREFACTOR
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )

    squared_difference_outer_product: np.array = np.square(difference_outer_product)

    sample_repeated_linear_predictor: np.array = linear_predictor_vanilla.repeat(
        n_events
    ).reshape((n_samples, n_events))

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )
    squared_kernel_numerator: np.array = np.square(kernel_numerator_full[event_mask, :])

    squared_difference_kernel_numerator: np.array = kernel_matrix[event_mask, :] * (
        squared_difference_outer_product[event_mask, :] * squared_inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)
    squared_kernel_denominator: np.array = np.square(kernel_denominator)

    integrated_kernel_denominator: np.array = (
        integrated_kernel_matrix * sample_repeated_linear_predictor
    ).sum(axis=0)

    for _ in range(n_samples):

        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                (
                    linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :]
                    + linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        hessian_five = inverse_sample_size * (
            (
                np.square(
                    (
                        linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :]
                        + linear_predictor_vanilla[_]
                        * kernel_matrix[_, :]
                        * inverse_bandwidth
                    )
                    / integrated_kernel_denominator
                )
            ).sum()
        )
        hessian_six = -(
            inverse_sample_size
            * (
                (
                    linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :]
                    + 2
                    * linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                    - linear_predictor_vanilla[_]
                    * kernel_numerator_full[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        if sample_event:
            gradient_correction_factor = inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    + linear_predictor_vanilla[_] * zero_kernel * inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
            )

            hessian_correction_factor = -inverse_sample_size * (
                (
                    (
                        linear_predictor_vanilla[_] * zero_integrated_kernel
                        + linear_predictor_vanilla[_] * zero_kernel * inverse_bandwidth
                    )
                    / integrated_kernel_denominator[event_count]
                )
                ** 2
                - (
                    (
                        linear_predictor_vanilla[_] * zero_integrated_kernel
                        + 2
                        * linear_predictor_vanilla[_]
                        * zero_kernel
                        * inverse_bandwidth
                    )
                    / (integrated_kernel_denominator[event_count])
                )
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
            hessian_one = -(
                inverse_sample_size
                * (
                    squared_kernel_numerator[
                        event_count,
                    ]
                    / squared_kernel_denominator
                ).sum()
            )

            hessian_two = inverse_sample_size * (
                (
                    (
                        squared_difference_kernel_numerator[event_count, :]
                        - (
                            kernel_matrix[
                                _,
                            ]
                            * squared_inverse_bandwidth
                        )
                    )
                    / kernel_denominator
                ).sum()
                + (
                    zero_kernel
                    * squared_inverse_bandwidth
                    / kernel_denominator[event_count]
                )
            )

            prefactor: float = kernel_numerator_full[event_mask, event_count].sum() / (
                kernel_denominator[event_count]
            )

            gradient_two = inverse_sample_size * prefactor
            hessian_three = -inverse_sample_size * (prefactor**2)

            hessian_four = inverse_sample_size * (
                (
                    ((squared_difference_kernel_numerator[:, event_count]).sum())
                    - (
                        squared_inverse_bandwidth
                        * ((kernel_matrix[event_mask, event_count]).sum() - zero_kernel)
                    )
                )
                / (kernel_denominator[event_count])
            )
            prefactor = (
                (
                    (linear_predictor_vanilla * kernel_matrix[:, event_count]).sum()
                    - linear_predictor_vanilla[_] * zero_kernel
                )
                * inverse_bandwidth
                - (linear_predictor_vanilla[_] * zero_integrated_kernel)
            ) / integrated_kernel_denominator[event_count]
            gradient_four = inverse_sample_size * prefactor

            hessian_seven = inverse_sample_size * (prefactor**2)
            hessian_eight = inverse_sample_size * (
                (
                    (
                        linear_predictor_vanilla
                        * kernel_numerator_full[:, event_count]
                        * inverse_bandwidth
                    ).sum()
                    - linear_predictor_vanilla[_] * zero_integrated_kernel
                )
                / integrated_kernel_denominator[event_count]
            )

            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            ) - inverse_sample_size

            hessian[_] = (
                hessian_one
                + hessian_two
                + hessian_three
                + hessian_four
                + hessian_five
                + hessian_six
                + hessian_seven
                + hessian_eight
                + hessian_correction_factor
            )
            event_count += 1

        else:
            gradient[_] = gradient_three
            hessian[_] = hessian_five + hessian_six
    return np.negative(gradient), modify_hessian(
        hessian=np.negative(hessian),
        hessian_modification_strategy=hessian_modification_strategy,
    )


@jit(nopython=True, cache=True, fastmath=True)
def update_risk_sets_breslow(
    risk_set_sum: float,
    death_set_count: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
) -> Tuple[float, float]:
    local_risk_set += 1 / (risk_set_sum / death_set_count)
    local_risk_set_hessian += 1 / ((risk_set_sum**2) / death_set_count)
    return local_risk_set, local_risk_set_hessian


@jit(nopython=True, cache=True, fastmath=True)
def calculate_sample_grad_hess(
    sample_partial_hazard: float,
    sample_event: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
) -> Tuple[float, float]:
    return (
        sample_partial_hazard * local_risk_set
    ) - sample_event, sample_partial_hazard * local_risk_set - local_risk_set_hessian * (
        sample_partial_hazard**2
    )


@jit(nopython=True, cache=True, fastmath=True)
def breslow_numba(
    linear_predictor: np.array,
    time: np.array,
    event: np.array,
):
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(linear_predictor)
    samples = time.shape[0]
    risk_set_sum = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time = time[0]
    local_risk_set = 0
    local_risk_set_hessian = 0
    death_set_count = 0
    censoring_set_count = 0
    accumulated_sum = 0

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i]

        if previous_time < sample_time:
            if death_set_count:
                (local_risk_set, local_risk_set_hessian,) = update_risk_sets_breslow(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                )

            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0

        if sample_event:
            death_set_count += 1
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        local_risk_set, local_risk_set_hessian = update_risk_sets_breslow(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
        )
    return grad / samples, hess / samples


@jit(nopython=True, cache=True, fastmath=True)
def calculate_sample_grad_hess_efron(
    sample_partial_hazard: float,
    sample_event: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    local_risk_set_death: float,
    local_risk_set_hessian_death: float,
) -> Tuple[float, float]:
    if sample_event:
        return ((sample_partial_hazard) * (local_risk_set_death)) - (sample_event), (
            sample_partial_hazard
        ) * (local_risk_set_death) - ((local_risk_set_hessian_death)) * (
            (sample_partial_hazard) ** 2
        )
    else:
        return ((sample_partial_hazard) * local_risk_set), (
            sample_partial_hazard
        ) * local_risk_set - local_risk_set_hessian * ((sample_partial_hazard) ** 2)


@jit(nopython=True, cache=True, fastmath=True)
def update_risk_sets_efron_pre(
    risk_set_sum: float,
    death_set_count: int,
    local_risk_set: float,
    local_risk_set_hessian: float,
    death_set_risk: float,
) -> Tuple[float, float, float, float]:
    local_risk_set_death: float = local_risk_set
    local_risk_set_hessian_death: float = local_risk_set_hessian

    for ell in range(death_set_count):
        contribution: float = ell / death_set_count
        local_risk_set += 1 / (risk_set_sum - (contribution) * death_set_risk)
        local_risk_set_death += (1 - (ell / death_set_count)) / (
            risk_set_sum - (contribution) * death_set_risk
        )
        local_risk_set_hessian += (
            1 / ((risk_set_sum - (contribution) * death_set_risk))
        ) ** 2

        local_risk_set_hessian_death += ((1 - contribution) ** 2) / (
            ((risk_set_sum - (contribution) * death_set_risk)) ** 2
        )

    return (
        local_risk_set,
        local_risk_set_hessian,
        local_risk_set_death,
        local_risk_set_hessian_death,
    )


@jit(nopython=True, cache=True, fastmath=True)
def efron_numba(
    linear_predictor: np.array,
    time: np.array,
    event: np.array,
) -> Tuple[np.array, np.array]:
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(linear_predictor)
    samples = time.shape[0]
    risk_set_sum = 0
    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time: float = time[0]
    local_risk_set: int = 0
    local_risk_set_hessian: int = 0
    death_set_count: int = 0
    censoring_set_count: int = 0
    accumulated_sum: int = 0
    death_set_risk: float = 0.0
    local_risk_set_death: float = 0.0
    local_risk_set_hessian_death: float = 0.0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    for i in range(samples):
        sample_time: float = time[i]
        sample_event: int = event[i]
        sample_partial_hazard: float = partial_hazard[i]

        if previous_time < sample_time:
            if death_set_count:
                (
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                ) = update_risk_sets_efron_pre(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                    death_set_risk,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                )
            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0
            death_set_risk = 0
            local_risk_set_death = 0
            local_risk_set_hessian_death = 0

        if sample_event:
            death_set_count += 1
            death_set_risk += sample_partial_hazard
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        (
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        ) = update_risk_sets_efron_pre(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
            death_set_risk,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        )
    return grad / samples, hess / samples
