from math import log
from typing import List, Set

import numpy as np
from numba import jit
from typeguard import typechecked


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


@typechecked
def is_partition(groups: Set[int]):
    return np.array_equal(np.array(list(groups)), np.arange(np.max(groups)))


@typechecked
def has_overlaps(groups: List[List[int]]):
    flattened_groups = [
        group_membership for group in groups for group_membership in group
    ]
    flattened_groups_set = set(flattened_groups)
    assert is_partition(flattened_groups)
    return len(flattened_groups) > len(flattened_groups_set)


@typechecked
def resolve_overlaps(groups: List[List[int]]):
    flattened_groups = [
        group_membership for group in groups for group_membership in group
    ]
    covariates, group_membership_count = np.unique(flattened_groups, return_counts=True)

    # `groups` is the group vector in the new, covariate duplicated
    # space. `group_mapping` should expand the covariates as needed.
    # `group_reverse_mapping` can be used to later summarise the
    # coefficients to get them on the original covariate scale.
    group_mapping = np.repeat(covariates, group_membership_count)
    group_reverse_mapping = []
    for covariate in covariates:
        group_reverse_mapping.append(np.where(group_mapping == covariate)[0])
    already_seen_covariate = {covariate: 0 for covariate in covariates}
    expanded_groups = []
    for group in groups:
        current_group = []
        for covariate in group:
            current_group.append(covariate)
            already_seen_covariate[covariate] += 1
        expanded_groups.append(current_group)

    return group_mapping, group_reverse_mapping, True, expanded_groups


@typechecked
def calculate_sgl_groups(groups: List[List[int]]):
    if has_overlaps(groups=groups):
        (
            group_mapping,
            group_reverse_mapping,
            _,
            expanded_groups,
        ) = resolve_overlaps(groups=groups)
    else:
        flattened_groups = [
            group_membership for group in groups for group_membership in group
        ]
        group_mapping = flattened_groups
        expanded_groups = groups
        group_reverse_mapping = list(np.arange(len(flattened_groups)))

    covariate_size = np.max(group_mapping) + 1
    for covariate in len(flattened_groups):
        group_mapping.append(covariate)
        expanded_groups.append([covariate_size])
        group_reverse_mapping.append(group_reverse_mapping + [covariate_size])
        covariate_size += 1

    return group_mapping, group_reverse_mapping, True, expanded_groups


@typechecked
def estimate_group_weights(
    groups: List[List[int]], strategy: np.array, l1_ratio: float
):
    flattened_groups = [
        group_membership for group in groups for group_membership in group
    ]
    n_features = len(flattened_groups)
    group_weights = np.zeros(n_features)
    group_sizes = np.array([len(group) for group in groups])
    if "sparse_group_lasso" in strategy:
        assert l1_ratio is not None
        if strategy == "group_size_sparse_group_lasso":
            group_weights = np.repeat(np.sqrt(group_sizes), group_sizes)
        elif strategy == "inverse_group_size_sparse_group_lasso":
            group_weights = np.repeat(1 / np.sqrt(group_sizes), group_sizes)
        sgl_mask = np.repeat(group_sizes == 1, group_sizes)
        group_weights[sgl_mask] *= l1_ratio
        group_weights[np.logical_not(sgl_mask)] *= 1 - l1_ratio
    else:
        if strategy == "group_size_group_lasso":
            group_weights = np.repeat(np.sqrt(group_sizes), group_sizes)
        elif strategy == "inverse_group_size_group_lasso":
            group_weights = np.repeat(1 / np.sqrt(group_sizes), group_sizes)
    return group_weights


@typechecked
def summarise_overlapping_coefs(
    coef: np.array, group_reverse_mapping: List[List[int]]
) -> np.array:
    summarised_coef: np.array = np.zeros(group_reverse_mapping.shape[0])
    for ix in range(len(group_reverse_mapping)):
        summarised_coef[ix] = np.sum(coef[group_reverse_mapping[ix]])
    return summarised_coef
