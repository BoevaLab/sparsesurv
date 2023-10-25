from math import log

import numpy as np
import pytest
from sparsesurv.tests.get_data_arrays import get_1d_array

from sparsesurv.loss import efron_negative_likelihood


def efron_calculation(linear_predictor, time, event):
    if np.sum(event) <= 0:
        raise RuntimeError("No events detected!")
    sorted_ix = np.argsort(time)
    linear_predictor_sorted = linear_predictor[sorted_ix]
    linear_predictor_sorted_exp = np.exp(linear_predictor_sorted)
    time_sorted = time[sorted_ix]
    event_sorted = event[sorted_ix]
    ll = np.sum(linear_predictor_sorted[event_sorted.astype(bool)])
    previous_time = 0.0
    efron_counter = 0
    risk_set = np.sum(linear_predictor_sorted_exp)
    efron_death_sum = 0.0
    efron_censor_sum = 0.0
    for i in range(sorted_ix.shape[0]):
        current_linear_predictor = linear_predictor_sorted_exp[i]
        current_time = time_sorted[i]
        current_event = event_sorted[i]

        if current_time == previous_time and current_event:
            efron_counter += 1
            efron_death_sum += current_linear_predictor
        elif current_time == previous_time:
            efron_censor_sum += current_linear_predictor
        else:
            for ell in range(efron_counter):
                ll -= log(risk_set - ell / efron_counter * efron_death_sum)
            risk_set -= efron_death_sum + efron_censor_sum
            efron_counter = int(current_event)
            efron_death_sum = current_event * current_linear_predictor
            efron_censor_sum = (1 - current_event) * current_linear_predictor
        previous_time = current_time
    for ell in range(efron_counter):
        ll -= log(risk_set - ell / efron_counter * efron_death_sum)
    return -ll / sorted_ix.shape[0]


class TestEfronLoss:
    def test_default(self):
        linear_predictor, time, event = get_1d_array("default")

        efron_formula_computation = efron_calculation(linear_predictor, time, event)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            efron_loss, efron_formula_computation, atol=1e-2
        ), f"Computed efron loss is {efron_loss} but formula yields {efron_formula_computation} for default data."

    def test_first_five_zero(self):
        linear_predictor, time, event = get_1d_array("first_five_zero")

        efron_formula_computation = efron_calculation(linear_predictor, time, event)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            efron_loss, efron_formula_computation, atol=1e-2
        ), f"Computed efron loss is {efron_loss} but formula yields {efron_formula_computation} for edge case: first five zero events."

    def test_last_five_zero(self):
        linear_predictor, time, event = get_1d_array("last_five_zero")

        efron_formula_computation = efron_calculation(linear_predictor, time, event)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            efron_loss, efron_formula_computation, atol=1e-2
        ), f"Computed efron loss is {efron_loss} but formula yields {efron_formula_computation} for edge case: last five zero events."

    def test_high_event_ratio(self):
        linear_predictor, time, event = get_1d_array("high_event_ratio")

        efron_formula_computation = efron_calculation(linear_predictor, time, event)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            efron_loss, efron_formula_computation, atol=1e-2
        ), f"Computed efron loss is {efron_loss} but formula yields {efron_formula_computation} for edge case: high event ratio."

    def test_low_event_ratio(self):
        linear_predictor, time, event = get_1d_array("low_event_ratio")

        efron_formula_computation = efron_calculation(linear_predictor, time, event)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            efron_loss, efron_formula_computation, atol=1e-2
        ), f"Computed efron loss is {efron_loss} but formula yields {efron_formula_computation} for edge case: low event ratio."

    def test_all_events(self):
        linear_predictor, time, event = get_1d_array("all_events")

        efron_formula_computation = efron_calculation(linear_predictor, time, event)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            efron_loss, efron_formula_computation, atol=1e-2
        ), f"Computed efron loss is {efron_loss} but formula yields {efron_formula_computation} for edge case: all(100%) events."

    def test_no_events(self):
        linear_predictor, time, event = get_1d_array("no_events")

        with pytest.raises(RuntimeError) as excinfo:
            efron_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(
            excinfo.value
        ), "Events detected in data. Check data or the function <efron_negative_likelihood> to make sure data is processed correctly."
