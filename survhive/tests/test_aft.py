import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import check_grad
from test_data_gen_final import numpy_test_data_1d

from survhive.loss import aft_negative_likelihood
from survhive.utils import normal_density

def aft_calculation(linear_predictor, time, event):
    """Accelerated Failure Time Loss."""
    assert isinstance(linear_predictor,torch.Tensor), f"<linear_predictor> should be a Tensor, but is {type(linear_predictor)} instead."
    if torch.sum(event)==0:
          raise RuntimeError('No events detected!')
    
    n_samples = len(event)

    if linear_predictor.dim()!=2:
        linear_predictor = linear_predictor[:,None]
        # .reshape(n_samples,1)
    
    h = 1.30*math.pow(n_samples,-0.2)
    #h 1.304058*math.pow(n_samples,-0.2)  ## 1.304058*n_samples^(-1/5) or 1.587401*math.pow(n_samples,-0.333333) 1.587401*n_samples^(-1/3)
    time = time.view(n_samples,1)
    event = event.view(n_samples,1)
    
    # R = g(Xi) + log(Oi)
    R = torch.add(linear_predictor,torch.log(time)) 
    
    # Rj - Ri
    rawones = torch.ones([1,n_samples], dtype = linear_predictor.dtype)
    R1 = torch.mm(R,rawones)
    R2 = torch.mm(torch.t(rawones),torch.t(R))
    DR = R1 - R2 
    
    # K[(Rj-Ri)/h]
    K = normal_density(DR/h)
    Del = torch.mm(event, rawones)
    DelK = Del*K 
    
    # (1/nh) *sum_j eventj * K[(Rj-Ri)/h]
    Dk = torch.sum(DelK, dim=0)/(n_samples*h)
    
    # log {(1/nh) * eventj * K[(Rj-Ri)/h]}    
    log_Dk = torch.log(Dk)     
    A = torch.t(event)*log_Dk/n_samples   
    S1 = A.sum()  
    
    ncdf=torch.distributions.normal.Normal(torch.tensor([0.0], dtype = linear_predictor.dtype), torch.tensor([1.0], dtype = linear_predictor.dtype)).cdf
    P = ncdf(DR/h)
    CDF_sum = torch.sum(P, dim=0)/n_samples
    Q = torch.log(CDF_sum)
    S2 = -(event*Q.view(n_samples,1)).sum()/n_samples
            
    S0 = -(event*torch.log(time)).sum()/n_samples
    
    S = S0 + S1 + S2 
    S = -S
    return S

class TestAFTLoss:

    def test_default(self):
        
        linear_predictor, time, event = numpy_test_data_1d("default")
        
        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = torch.from_numpy(linear_predictor), torch.from_numpy(time), torch.from_numpy(event)
        aft_formula_computation = aft_calculation(linear_predictor_tensor, time_tensor, event_tensor)
        
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(aft_loss,aft_formula_computation, atol=1e-2), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for default data."


    def test_first_five_zero(self):
        
        linear_predictor, time, event = numpy_test_data_1d("first_five_zero")
        
        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = torch.from_numpy(linear_predictor), torch.from_numpy(time), torch.from_numpy(event)
        aft_formula_computation = aft_calculation(linear_predictor_tensor, time_tensor, event_tensor)
        
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(aft_loss,aft_formula_computation, atol=1e-2), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: first five zero events."

    def test_last_five_zero(self):

        linear_predictor, time, event = numpy_test_data_1d("last_five_zero")
        
        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = torch.from_numpy(linear_predictor), torch.from_numpy(time), torch.from_numpy(event)
        aft_formula_computation = aft_calculation(linear_predictor_tensor, time_tensor, event_tensor)
        
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(aft_loss,aft_formula_computation, atol=1e-2), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: last five zero events."

    def test_high_event_ratio(self):

        linear_predictor, time, event = numpy_test_data_1d("high_event_ratio")
        
        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = torch.from_numpy(linear_predictor), torch.from_numpy(time), torch.from_numpy(event)
        aft_formula_computation = aft_calculation(linear_predictor_tensor, time_tensor, event_tensor)
        
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(aft_loss,aft_formula_computation, atol=1e-2), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: high event ratio."

    def test_low_event_ratio(self):

        linear_predictor, time, event = numpy_test_data_1d("low_event_ratio")
        
        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = torch.from_numpy(linear_predictor), torch.from_numpy(time), torch.from_numpy(event)
        aft_formula_computation = aft_calculation(linear_predictor_tensor, time_tensor, event_tensor)
        
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(aft_loss,aft_formula_computation, atol=1e-2), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: low event ratio."

    def test_all_events(self):

        linear_predictor, time, event = numpy_test_data_1d("all_events")
        
        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = torch.from_numpy(linear_predictor), torch.from_numpy(time), torch.from_numpy(event)
        aft_formula_computation = aft_calculation(linear_predictor_tensor, time_tensor, event_tensor)
        
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        
        assert np.allclose(aft_loss,aft_formula_computation, atol=1e-2), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: all(100%) events."

    def test_no_events(self):

        linear_predictor, time, event = numpy_test_data_1d("no_events")
        
        with pytest.raises(RuntimeError) as excinfo:
            aft_negative_likelihood(linear_predictor, time, event)
        assert "No events detected!" in str(excinfo.value), f"Events detected in data. Check data or the function <aft_negative_likelihood> to make sure data is processed correctly."