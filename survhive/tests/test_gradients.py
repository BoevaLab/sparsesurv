import numpy as np
from scipy.optimize import check_grad
import sys
#sys.path.append('/Users/JUSC/Documents/survhive/survhive')
from survhive.loss import (aft_negative_likelihood, eh_negative_likelihood,
                           breslow_negative_likelihood, efron_negative_likelihood)
from survhive.gradients import aft_gradient, eh_gradient
from loss_functions_test import eh_gradient_test, eh_negative_likelihood_test, eh_negative_likelihood_torch
import pytest
import torch

# pytest data fixture
@pytest.fixture
def numpy_test_data():
        linear_predictor = np.array([0.67254923,
        0.86077982,
        0.43557393,
        0.94059047,
        0.8446509 ,
        0.23657039,
        0.74629685,
        0.99700768,
        0.28182768,
        0.44495038]) #.reshape(1,10)
        time = np.array([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]]).reshape(-1)
        event = np.array([[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]],dtype=np.float32).reshape(-1)
        return linear_predictor, time, event


# 1. Test Loss Functions

def test_breslow_loss(numpy_test_data):
        linear_predictor, time, event = numpy_test_data
        breslow_loss = breslow_negative_likelihood(linear_predictor, time, event)
        #*10 due sample size divison
        assert np.allclose(breslow_loss*10,10.799702318875216)

def test_efron_loss(numpy_test_data):
        linear_predictor, time, event = numpy_test_data
        efron_loss = efron_negative_likelihood(linear_predictor, time, event)
        assert np.allclose(efron_loss*10, 10.683134406156332)

def test_aft_loss(numpy_test_data):
        linear_predictor, time, event = numpy_test_data
        aft_loss = aft_negative_likelihood(linear_predictor, time, event)
        assert np.allclose(aft_loss, 1.8701611757278442)

def test_eh_loss():
        linear_predictor = np.array([[0.67254923, 0.03356795],
       [0.86077982, 0.65922692],
       [0.43557393, 0.75447972],
       [0.94059047, 0.30572004],
       [0.8446509 , 0.07916267],
       [0.23657039, 0.44693716],
       [0.74629685, 0.32637245],
       [0.99700768, 0.10225456],
       [0.28182768, 0.05405025],
       [0.44495038, 0.08454563]])#.reshape(10,1)
        time = np.array([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]],dtype=np.float32).reshape(-1)
        event = np.array([[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]], dtype=int).reshape(-1)
        eh_loss = eh_negative_likelihood(linear_predictor, time, event)
        assert np.allclose(eh_loss, 1.7399981021881104)


# 2. Test Gradients

def test_aft_gradient(numpy_test_data):
        linear_predictor, time, event = numpy_test_data
        diff = check_grad(lambda x: aft_negative_likelihood(x, time, event), lambda x: aft_gradient(x, time, event ), linear_predictor)
        assert diff < 1e-7

def test_eh_torch_loss():
        linear_predictor = np.array([[0.67254923, 0.03356795],
       [0.86077982, 0.65922692],
       [0.43557393, 0.75447972],
       [0.94059047, 0.30572004],
       [0.8446509 , 0.07916267],
       [0.23657039, 0.44693716],
       [0.74629685, 0.32637245],
       [0.99700768, 0.10225456],
       [0.28182768, 0.05405025],
       [0.44495038, 0.08454563]], dtype=np.float32) 
        time = np.array([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]]).reshape(-1)
        event = np.array([[1, 0, 0, 0, 0, 1, 1, 0, 1, 1]],dtype=int).reshape(-1)
        eh_loss_np = eh_negative_likelihood(linear_predictor, time, event)
        linear_predictor = torch.tensor([
        [0.67254923, 0.03356795],
        [0.86077982, 0.65922692],
        [0.43557393, 0.75447972],
        [0.94059047, 0.30572004],
        [0.8446509 , 0.07916267],
        [0.23657039, 0.44693716],
        [0.74629685, 0.32637245],
        [0.99700768, 0.10225456],
        [0.28182768, 0.05405025],
        [0.44495038, 0.08454563]], requires_grad=True)#.reshape(10,1)
        time = torch.tensor([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]])
        event = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 0, 1, 1]],dtype=torch.float)
        eh_loss_torch = eh_negative_likelihood_torch(linear_predictor, time.reshape(-1), event.reshape(-1))
        assert np.allclose(eh_loss_np, eh_loss_torch.detach().numpy())

def test_eh_gradient():
        linear_predictor = np.array([[0.67254923, 0.03356795],
       [0.86077982, 0.65922692],
       [0.43557393, 0.75447972],
       [0.94059047, 0.30572004],
       [0.8446509 , 0.07916267],
       [0.23657039, 0.44693716],
       [0.74629685, 0.32637245],
       [0.99700768, 0.10225456],
       [0.28182768, 0.05405025],
       [0.44495038, 0.08454563]], dtype=np.float32) 
        time = np.array([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]]).reshape(-1)
        event = np.array([[1, 0, 0, 0, 0, 1, 1, 0, 1, 1]],dtype=int).reshape(-1)
        eh_gradient_np = eh_gradient(
        linear_predictor,
        time,
        event,
        bandwidth=None,
        ).reshape(10,2, order='F')
        linear_predictor = torch.tensor([
        [0.67254923, 0.03356795],
        [0.86077982, 0.65922692],
        [0.43557393, 0.75447972],
        [0.94059047, 0.30572004],
        [0.8446509 , 0.07916267],
        [0.23657039, 0.44693716],
        [0.74629685, 0.32637245],
        [0.99700768, 0.10225456],
        [0.28182768, 0.05405025],
        [0.44495038, 0.08454563]], requires_grad=True)#.reshape(10,1)
        time = torch.tensor([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]])
        event = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 0, 1, 1]],dtype=torch.float)
        eh_loss_own_torch = eh_negative_likelihood_torch(linear_predictor, time.reshape(-1), event.reshape(-1))
        eh_loss_own_torch.backward()
        eh_gradient_torch = linear_predictor.grad
        assert np.allclose(eh_gradient_np, eh_gradient_torch)