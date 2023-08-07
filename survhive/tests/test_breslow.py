import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from test_data_gen_final import numpy_test_data_1d, numpy_test_data_2d

from survhive.loss import (
    # aft_negative_likelihood,
    breslow_negative_likelihood,
    # efron_negative_likelihood,
    # eh_negative_likelihood,
)

def breslow_calculation(linear_predictor, time, event):
    """Breslow loss Moeschberger page 259."""
    numerator = []
    denominator = []
    n_samples = len(linear_predictor)

    for idx, t in enumerate(np.unique(time[event.astype(bool)])):
        numerator.append(np.exp(np.sum(np.where(t==time, linear_predictor, 0))))
    riskset = (np.outer(time,time)<=np.square(time)).astype(int)
    linear_predictor_exp = np.exp(linear_predictor)
    riskset = riskset*linear_predictor_exp
    uni, idx, counts = np.unique(time[event.astype(bool)], return_index=True, return_counts=True)
    denominator = np.sum(riskset[event.astype(bool)], axis=1)[idx]
    return -np.log(np.prod(numerator/(denominator**counts)))/n_samples

class TestBreslowLoss:

    def test_default(self):
        linear_predictor, time, event = numpy_test_data_1d("default")
        
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(breslow_loss,breslow_formula_computation, atol=1e-2), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for default data."


    def test_first_five_zero(self):
        
        linear_predictor, time, event = numpy_test_data_1d("first_five_zero")
        
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(breslow_loss,breslow_formula_computation, atol=1e-2), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: first five zero events."

    def test_last_five_zero(self):

        linear_predictor, time, event = numpy_test_data_1d("last_five_zero")
        
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(breslow_loss,breslow_formula_computation, atol=1e-2), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: last five zero events."

    def test_high_event_ratio(self):

        linear_predictor, time, event = numpy_test_data_1d("high_event_ratio")
        
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(breslow_loss,breslow_formula_computation, atol=1e-2), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: high event ratio."

    def test_low_event_ratio(self):

        linear_predictor, time, event = numpy_test_data_1d("low_event_ratio")
        
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(breslow_loss,breslow_formula_computation, atol=1e-2), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: low event ratio."

    def test_all_events(self):

        linear_predictor, time, event = numpy_test_data_1d("all_events")
        
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(breslow_loss,breslow_formula_computation, atol=1e-2), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: all(100%) events."

    def test_no_events(self):

        linear_predictor, time, event = numpy_test_data_1d("no_events")
        
        with pytest.raises(RuntimeError) as excinfo:
            breslow_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(excinfo.value), f"Events detected in data. Check data or the function <breslow_negative_likelihood> to make sure data is processed correctly."