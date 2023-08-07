import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from test_data_gen_final import numpy_test_data_1d, numpy_test_data_2d

from survhive.loss import (
    aft_negative_likelihood,
    breslow_negative_likelihood,
    efron_negative_likelihood,
    eh_negative_likelihood,
)

# assume you are in survhive tests directory
path = os.getcwd()
dim = 25

# Scenarios

scenarios = [
    "default",
    "first_five_zero",
    "last_five_zero",
    "high_event_ratio",
    "low_event_ratio",
    "all_events",
    #'no_events'
]

# 1. Test Loss Functions


def test_breslow_loss():
    print(os.path.dirname(os.path.abspath(__file__)))
    df_breslow = pd.read_csv("./test_data/breslow_losses.csv")
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_1d(scenario)
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        print(scenario, breslow_loss * 25, df_breslow[scenario].values[0])
        # *25 due sample size divison
        assert np.allclose(breslow_loss * 25, df_breslow[scenario], atol=1e-2)


# test_breslow_loss()


def test_efron_loss():
    df_efron = pd.read_csv("./test_data/efron_losses.csv")
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_1d(scenario)
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)
        print(scenario, efron_loss * 25, df_efron[scenario].values[0])
        # *10 due sample size divison
        assert np.allclose(efron_loss * 25, df_efron[scenario])


# test_efron_loss()


def test_aft_loss():
    df_aft = pd.read_csv("./test_data/aft_losses.csv")
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_1d(scenario)
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        print(aft_loss, df_aft[scenario].values[0])
        assert np.allclose(aft_loss, df_aft[scenario].values[0])


# test_aft_loss()


def test_eh_loss():
    df_eh = pd.read_csv("./test_data/eh_losses.csv")
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_2d(scenario)
        print(linear_predictor, time, event)
        eh_loss = eh_negative_likelihood(linear_predictor.reshape(dim, 2), time, event)
        print(eh_loss, df_eh[scenario].values[0])
        assert np.allclose(eh_loss, df_eh[scenario].values[0])


# test_eh_loss()

# 2. Test no events runtimeerror

# scenarios = [#'default',
#'first_five_zero',
#'last_five_zero',
#'high_event_ratio',
#'low_event_ratio',
#'all_events',
#      'no_events'
#      ]


def test_breslow_loss_exception():
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_1d(scenario)
        with pytest.raises(RuntimeError) as excinfo:
            breslow_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(excinfo.value)


# test_breslow_loss_exception()


def test_efron_loss_exception():
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_1d(scenario)
        with pytest.raises(RuntimeError) as excinfo:
            efron_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(excinfo.value)


# test_efron_loss_exception()


def test_aft_loss_exception():
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_1d(scenario)
        with pytest.raises(RuntimeError) as excinfo:
            aft_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(excinfo.value)


# test_aft_loss_exception()


def test_eh_loss_exception():
    for scenario in scenarios:
        linear_predictor, time, event = numpy_test_data_2d(scenario)
        with pytest.raises(RuntimeError) as excinfo:
            eh_negative_likelihood(linear_predictor.reshape(dim, 2), time, event)
        assert "No events detected!" in str(excinfo.value)


# test_eh_loss_exception()
