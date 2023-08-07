import os
import sys
from math import log

import numpy as np
import pandas as pd
import pytest

from test_data_gen_final import numpy_test_data_1d

from survhive.loss import breslow_negative_likelihood


def breslow_calculation(linear_predictor, time, event):
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    sorted_ix = np.argsort(time)
    linear_predictor_sorted = linear_predictor[sorted_ix]
    linear_predictor_sorted_exp = np.exp(linear_predictor_sorted)
    time_sorted = time[sorted_ix]
    event_sorted = event[sorted_ix].astype(int)
    ll = np.sum(linear_predictor_sorted[event_sorted.astype(bool)])
    previous_time = 0.0
    risk_set = np.sum(linear_predictor_sorted_exp)
    breslow_sum = 0.0
    breslow_count = 0.0

    for i in range(sorted_ix.shape[0]):
        current_linear_predictor = linear_predictor_sorted_exp[i]
        current_time = time_sorted[i]
        current_event = event_sorted[i]
        if current_time == previous_time:
            breslow_count += current_event
            breslow_sum += current_linear_predictor
        else:
            ll -= breslow_count * log(risk_set)
            risk_set -= breslow_sum
            breslow_count = current_event
            breslow_sum = current_linear_predictor
        previous_time = current_time

    if breslow_count:
        ll -= breslow_count * log(risk_set)

    return -ll / sorted_ix.shape[0]


class TestBreslowLoss:
    def test_default(self):
        linear_predictor, time, event = numpy_test_data_1d("default")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for default data."

    def test_first_five_zero(self):
        linear_predictor, time, event = numpy_test_data_1d("first_five_zero")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: first five zero events."

    def test_last_five_zero(self):
        linear_predictor, time, event = numpy_test_data_1d("last_five_zero")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: last five zero events."

    def test_high_event_ratio(self):
        linear_predictor, time, event = numpy_test_data_1d("high_event_ratio")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: high event ratio."

    def test_low_event_ratio(self):
        linear_predictor, time, event = numpy_test_data_1d("low_event_ratio")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: low event ratio."

    def test_all_events(self):
        linear_predictor, time, event = numpy_test_data_1d("all_events")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: all(100%) events."

    def test_no_events(self):
        linear_predictor, time, event = numpy_test_data_1d("no_events")

        with pytest.raises(RuntimeError) as excinfo:
            breslow_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(
            excinfo.value
        ), "Events detected in data. Check data or the function <breslow_negative_likelihood> to make sure data is processed correctly."
