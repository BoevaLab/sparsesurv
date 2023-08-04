import numpy as np
import pandas as pd
from scipy.optimize import check_grad
import sys
import os
#sys.path.append('/Users/JUSC/Documents/survhive/survhive')
from survhive.loss import (aft_negative_likelihood, eh_negative_likelihood,
                           breslow_negative_likelihood, efron_negative_likelihood)
#from loss_functions_test import eh_gradient_test, eh_negative_likelihood_test, eh_negative_likelihood_torch
from test_data_gen_final import (numpy_test_data_1d,
                                 numpy_test_data_2d)
import pytest
from survhive.gradients import aft_gradient, eh_gradient

# assume you are in survhive tests directory
path = os.getcwd()
dim = 25

# Scenarios

scenarios = ['default', 
             'first_five_zero',
             'last_five_zero',
             'high_event_ratio',
             'low_event_ratio',
             'all_events',
             #'no_events'
             ]

# 1. Test Gradients

def test_aft_gradient():
        for scenario in scenarios:
                linear_predictor, time, event = numpy_test_data_1d(scenario)
                diff = check_grad(lambda x: aft_negative_likelihood(x, time, event), lambda x: aft_gradient(x, time, event ), linear_predictor)
                print(scenario, diff)
                assert diff < 1e-6

#test_aft_gradient()


# cannot use scipy for this as 2d

def test_eh_gradient():
        df_eh = pd.read_csv('test_data/eh_gradients.csv')
        for scenario in scenarios:
                linear_predictor, time, event = numpy_test_data_2d(scenario)
                eh_gradient_np = eh_gradient(
                                linear_predictor,
                                time,
                                event,
                                bandwidth=None,
                                ).reshape(25,2, order='F').flatten()
                assert np.allclose(eh_gradient_np, df_eh[scenario].values)

#test_eh_gradient()
