import numpy as np
import pandas as pd
from scipy.optimize import check_grad
import sys
import os
#sys.path.append('/Users/JUSC/Documents/survhive/survhive')
import numpy.typing as npt
from typing import Optional
import math
from test_data_gen_final import (numpy_test_data_1d,
                                 numpy_test_data_2d,
                                 torch_test_data_1d,
                                 torch_test_data_2d)

                                 
from survhive.constants import CDF_ZERO, PDF_PREFACTOR
from survhive.utils import (
    difference_kernels,
    logaddexp,
    logsubstractexp,
    numba_logsumexp_stable,
)
import torch
from math import erf, exp, log


# Scenarios

scenarios = ['default', 
             'first_five_zero',
             'last_five_zero',
             'high_event_ratio',
             'low_event_ratio',
             'all_events',
             #'no_events'
             ]

dim = 25

# Manual Breslow and Efron loss calculations

# Breslow Handling of Ties

def breslow_calculation(linear_predictor, time, event):
    """Breslow loss Moeschberger page 259."""
    nominator = []
    denominator = []
    for idx, t in enumerate(np.unique(time[event.astype(bool)])):
        nominator.append(np.exp(np.sum(np.where(t==time, linear_predictor, 0))))
    riskset = (np.outer(time,time)<=np.square(time)).astype(int)
    linear_predictor_exp = np.exp(linear_predictor)
    riskset = riskset*linear_predictor_exp
    uni, idx, counts = np.unique(time[event.astype(bool)], return_index=True, return_counts=True)
    denominator = np.sum(riskset[event.astype(bool)], axis=1)[idx]
    return -np.log(np.prod(nominator/(denominator**counts)))

breslow_losses = {}
for scenario in scenarios:
    linear_predictor, time, event = numpy_test_data_1d(scenario)
    breslowloss = breslow_calculation(linear_predictor, time, event)
    breslow_losses[scenario] = breslowloss
df_breslow = pd.DataFrame(breslow_losses, index=[0])
df_breslow.to_csv('test_data/breslow_losses.csv', index=False)
print(df_breslow)


# Efron

def efron_calculation(linear_predictor, time, event):
    """Efron loss Moeschberger page 259."""
    
    ## ADD EFRON CALCULATION HERE ##

    return 0

efron_losses = {}
for scenario in scenarios:
    linear_predictor, time, event = numpy_test_data_1d(scenario)
    efronloss = efron_calculation(linear_predictor, time, event)
    efron_losses[scenario] = efronloss
df_efron = pd.DataFrame(breslow_losses, index=[0])
df_efron.to_csv('test_data/efron_losses.csv')
print(df_efron)

# aft and eh loss calculation from paper for comparison

def eaftloss(out, time, delta): ##loss function for AFT or EH
    if torch.sum(delta)==0:
          raise RuntimeError('No events detected!')
    ia, ib = out.size()
    if ib == 1: ###loss function for AFT
        n = len(delta)
        print('aft')
        h = 1.30*math.pow(n,-0.2)
        #h 1.304058*math.pow(n,-0.2)  ## 1.304058*n^(-1/5) or 1.587401*math.pow(n,-0.333333) 1.587401*n^(-1/3)
        time = time.view(n,1)
        delta = delta.view(n,1)
        
        # R = g(Xi) + log(Oi)
        R = torch.add(out,torch.log(time)) 
        
        # Rj - Ri
        rawones = torch.ones([1,n], dtype = out.dtype)
        R1 = torch.mm(R,rawones)
        R2 = torch.mm(torch.t(rawones),torch.t(R))
        DR = R1 - R2 
        
        # K[(Rj-Ri)/h]
        K = normal_density(DR/h)
        Del = torch.mm(delta, rawones)
        DelK = Del*K 
        
        # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
        Dk = torch.sum(DelK, dim=0)/(n*h)
        
        # log {(1/nh) * Deltaj * K[(Rj-Ri)/h]}    
        log_Dk = torch.log(Dk)     
        A = torch.t(delta)*log_Dk/n   
        S1 = A.sum()  
        
        ncdf=torch.distributions.normal.Normal(torch.tensor([0.0], dtype = out.dtype), torch.tensor([1.0], dtype = out.dtype)).cdf
        P = ncdf(DR/h)
        CDF_sum = torch.sum(P, dim=0)/n
        Q = torch.log(CDF_sum)
        S2 = -(delta*Q.view(n,1)).sum()/n
             
        S0 = -(delta*torch.log(time)).sum()/n
        
        S = S0 + S1 + S2 
        S = -S
    else: ### loss function for Extended hazard model
        print('eh model')
        n = len(out[:,0])
        h = 1.30*math.pow(n,-0.2)  ## or 1.59*n^(-1/3)
        print('bandwidth', h)
        time = time.view(n,1)
        delta = delta.view(n,1)
        g1 = out[:,0].view(n,1)
        g2 = out[:,1].view(n,1)
        
        # R = g(Xi) + log(Oi)
        R = torch.add(g1,torch.log(time)) 
        
        S1 =  (delta*g2).sum()/n
        S2 = -(delta*R).sum()/n
        print('S1,S2', S1, S2)
        
        # Rj - Ri
        rawones = torch.ones(1,n)
        R1 = torch.mm(R,rawones)
        R2 = torch.mm(torch.t(rawones),torch.t(R))
        DR = R1 - R2 
        
        # K[(Rj-Ri)/h]
        K = normal_density(DR/h)
        Del = torch.mm(delta, rawones)
        DelK = Del*K 
        
        # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
        Dk = torch.sum(DelK, dim=0)/(n*h)  ## Dk would be zero as learning rate too large!
        
        # log {(1/nh) * Deltaj * K[(Rj-Ri)/h]}    
        log_Dk = torch.log(Dk)    
        
        S3 = (torch.t(delta)*log_Dk).sum()/n    
        
        # Phi((Rj-Ri)/h)
        ncdf=torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])).cdf
        P = ncdf(DR/h) 
        L = torch.exp(g2-g1)
        LL = torch.mm(L,rawones)
        LP_sum = torch.sum(LL*P, dim=0)/n
        Q = torch.log(LP_sum)
        
        S4 = -(delta*Q.view(n,1)).sum()/n
        print(S1+ S2, S3, S4)
        S = S1 + S2 + S3 + S4  
        S = -S
    return S

def normal_density(a):  
    b = 0.3989423*torch.exp(-0.5*torch.pow(a,2.0))
    return b

#aft results


aft_losses = {}
for scenario in scenarios:
    linear_predictor, time, event = torch_test_data_1d(scenario)
    aftloss = eaftloss(linear_predictor.reshape(dim,1), time, event.reshape(dim,1))
    aft_losses[scenario] = aftloss.numpy()
df_aft = pd.DataFrame(aft_losses, index=[0])
df_aft.to_csv('test_data/aft_losses.csv', index=False)
print(df_aft)



# eh results

eh_losses = {}
for scenario in scenarios:
    linear_predictor, time, event = torch_test_data_2d(scenario)
    ehloss = eaftloss(linear_predictor.reshape(dim,2), time, event.reshape(dim,1))
    eh_losses[scenario] = ehloss.numpy()
df_eh = pd.DataFrame(eh_losses, index=[0])
df_eh.to_csv('test_data/eh_losses.csv', index=False)
print(df_eh)

