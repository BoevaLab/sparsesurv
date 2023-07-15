import numpy as np
from scipy.optimize import check_grad
import sys
#sys.path.append('/Users/JUSC/Documents/survhive/survhive')
import numpy.typing as npt
from typing import Optional
import math
from survhive.constants import CDF_ZERO, PDF_PREFACTOR
from survhive.utils import (
    difference_kernels,
    logaddexp,
    logsubstractexp,
    numba_logsumexp_stable,
)
import torch
from math import erf, exp, log

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
    denominator = np.sum(riskset[event.astype(bool)],axis=1)[idx]
    return -np.log(np.prod(nominator/(denominator**counts)))

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

breslowloss = breslow_calculation(linear_predictor, time, event)
print(f'Breslow Loss: {breslowloss}')

# Efron


def efron_calculation(linear_predictor, time, event):
    """Efron loss Moeschberger page 259."""
    nominator = []
    denominator = []
    for idx, t in enumerate(np.unique(time[event.astype(bool)])):
        nominator.append(np.exp(np.sum(np.where(t==time, linear_predictor, 0))))

    riskset = (np.outer(time,time)<=np.square(time)).astype(int)
    linear_predictor_exp = np.exp(linear_predictor)
    riskset = riskset*linear_predictor_exp

    uni, idx, counts = np.unique(time[event.astype(bool)], return_index=True, return_counts=True)
    denominator = np.sum(riskset[event.astype(bool)],axis=1)[idx]
    # adapt one tie condition in data, if time make nicer
    denominator[1] = 17.77170772*(17.77170772 - 0.5*(np.exp(0.86077982)+
    np.exp(0.43557393)))
    return -np.log(np.prod(nominator/(denominator)))

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

efronloss = efron_calculation(linear_predictor, time, event)
print(f'Efron Loss: {efronloss}')

# aft and eh loss calculation from paper for comparison

def eaftloss(out, time, delta): ##loss function for AFT or EH
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
linear_predictor = torch.tensor(
    [0.67254923,
    0.86077982,
    0.43557393,
    0.94059047,
    0.8446509 ,
    0.23657039,
    0.74629685,
    0.99700768,
    0.28182768,
    0.44495038], requires_grad=True).reshape(10,1)

time = torch.tensor([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]],dtype=torch.float32)
event = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.float32).reshape(10,1)
aftloss = eaftloss(linear_predictor, time, event)
print(f'AFT Loss: {aftloss}')

# eh results
linear_predictor = torch.tensor([[0.67254923, 0.03356795],
       [0.86077982, 0.65922692],
       [0.43557393, 0.75447972],
       [0.94059047, 0.30572004],
       [0.8446509 , 0.07916267],
       [0.23657039, 0.44693716],
       [0.74629685, 0.32637245],
       [0.99700768, 0.10225456],
       [0.28182768, 0.05405025],
       [0.44495038, 0.08454563]], requires_grad=True)#.reshape(10,1)
time = torch.tensor([[ 1,  3,  3,  4,  7,  8,  9, 11, 13, 16]],dtype=torch.float32)
event = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.float32).reshape(10,1)
ehloss = eaftloss(linear_predictor, time, event)
print(f'EH Loss: {ehloss}')


def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))


def gaussian_kernel(x):
    return PDF_PREFACTOR * exp(-0.5 * (x**2))



def kernel(a, b, bandwidth):
    kernel_matrix: torch.tensor = torch.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return kernel_matrix


def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: torch.tensor = torch.empty(
        shape=(a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix


def difference_kernels(a, b, bandwidth):
    difference: torch.tensor = torch.empty(shape=(a.shape[0], b.shape[0]))
    kernel_matrix: torch.tensor = torch.empty(shape=(a.shape[0], b.shape[0]))
    integrated_kernel_matrix: torch.tensor = torch.empty(
        shape=(a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix


def eh_negative_likelihood_torch(
    linear_predictor: torch.tensor,
    time: torch.tensor,
    event: torch.tensor,
    bandwidth: torch.tensor = None
) -> torch.tensor:
    #y1 = y[:,0]
    #time, event = transform_back_torch(y1)
    # need two predictors here
    linear_predictor_1: torch.tensor = linear_predictor[:, 0]# * sample_weight
    linear_predictor_2: torch.tensor = linear_predictor[:, 1]# * sample_weight
    exp_linear_predictor_1 = torch.exp(linear_predictor_1)
    exp_linear_predictor_2 = torch.exp(linear_predictor_2)

    n_events: int = torch.sum(event)
    n_samples: int = time.shape[0]
    if not bandwidth:
        bandwidth = 1.30 * torch.pow(n_samples, torch.tensor(-0.2))
    R_linear_predictor: torch.tensor = torch.log(time * exp_linear_predictor_1)
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: torch.tensor = event.bool()

    _: torch.tensor
    kernel_matrix: torch.tensor
    integrated_kernel_matrix: torch.tensor

    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask = event.bool()
    rv = torch.distributions.normal.Normal(0, 1, validate_args=None)
    sample_repeated_linear_predictor = (
        (exp_linear_predictor_2 / exp_linear_predictor_1).repeat((int(n_events.item()), 1)).T
    )
    diff = (
        R_linear_predictor.reshape(-1, 1) - R_linear_predictor[event_mask]
    ) / bandwidth

    kernel_matrix = torch.exp(
        -1 / 2 * torch.square(diff[event_mask, :])
    ) / torch.sqrt(torch.tensor(2) * torch.pi)
    integrated_kernel_matrix = rv.cdf(diff)
    
    inverse_sample_size: float = 1 / n_samples
    kernel_sum = kernel_matrix.sum(axis=0)
    integrated_kernel_sum = (
        sample_repeated_linear_predictor * integrated_kernel_matrix
    ).sum(axis=0)
    #print('integrated_kernel_sum', integrated_kernel_sum)
    print(linear_predictor_2[event_mask].sum()/n_samples
        , R_linear_predictor[event_mask].sum()/n_samples
        , torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()/n_samples
        , torch.log(inverse_sample_size * integrated_kernel_sum).sum()/n_samples)
    likelihood: torch.tensor = inverse_sample_size * (
        linear_predictor_2[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - torch.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood

def eh_negative_likelihood_test(
    linear_predictor,
    time,
    event,
    bandwidth=None,
) -> np.array:
    linear_predictor = linear_predictor.reshape(10,2)
    theta = np.exp(linear_predictor)
    n_samples: int = time.shape[0]
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)
    R_linear_predictor: np.array = np.log(time * theta[:, 0])
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=R_linear_predictor,
        b=R_linear_predictor[event_mask],
        bandwidth=bandwidth,
    )

    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: np.array = kernel_matrix.sum(axis=0)

    integrated_kernel_sum: np.array = (
        integrated_kernel_matrix
        * (theta[:, 1] / theta[:, 0])
        .repeat(np.sum(event))
        .reshape(-1, np.sum(event))
    ).sum(axis=0)
    likelihood: np.array = inverse_sample_size * (
        linear_predictor[:, 1][event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


#@jit(nopython=True, cache=True, fastmath=True)
def eh_gradient_test(
    linear_predictor: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int64],
    bandwidth: Optional[float] = None,
) -> np.array:
    """Calculates the negative gradient of the EH model wrt eta.

    Parameters
    ----------
    linear_predictor: npt.NDArray[np.float64]
        Linear predictor of the training data. N rows and 2 columns.
    time: npt.NDArray[np.float64]
        Time of the training data. Of dimension n. Assumed to be sorted
        (does not matter here, but regardless).
    event: npt.NDArray[np.int64]
        Event indicator of the training data. Of dimension n.
    bandwidth: Optional[float]
        Bandwidth to kernel-smooth the profile likelihood. Will
        be estimated if not specified.

    Returns
    -------
    gradient: npt.NDArray[np.float64]
        Negative gradient of the EH model wrt eta. Of dimensionality 2n.
    """
    n_samples: int = time.shape[0]
    n_events: int = np.sum(event)

    # Estimate bandwidth using an estimate proportional to the
    # the optimal bandwidth.
    if bandwidth is None:
        bandwidth = 1.30 * pow(n_samples, -0.2)

    # Cache various calculated quantities to reuse during later
    # calculation of the gradient.
    theta = np.exp(linear_predictor)

    linear_predictor_misc = np.log(time * theta[:, 0])

    linear_predictor_vanilla: np.array = theta[:, 1] / theta[:, 0]

    # Calling these cox and aft respectively, since setting
    # the respectively other coefficient to zero recovers
    # the (kernel-smoothed PL) model of the other one (e.g.,
    # setting Cox to zero recovers AFT and vice-versa).
    gradient_eta_cox: np.array = np.empty(n_samples)
    gradient_eta_aft: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    zero_kernel: float = PDF_PREFACTOR
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor_misc,
        b=linear_predictor_misc[event_mask],
        bandwidth=bandwidth,
    )

    sample_repeated_linear_predictor: np.array = (
        linear_predictor_vanilla.repeat(n_events).reshape(
            (n_samples, n_events)
        )
    )

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)

    integrated_kernel_denominator: np.array = (
        integrated_kernel_matrix * sample_repeated_linear_predictor
    ).sum(axis=0)

    for _ in range(n_samples):
        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                (
                    -linear_predictor_vanilla[_]
                    * integrated_kernel_matrix[_, :]
                    + linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        gradient_five = -(
            inverse_sample_size
            * (
                (linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :])
                / integrated_kernel_denominator
            ).sum()
        )

        if sample_event:
            gradient_correction_factor = inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    + linear_predictor_vanilla[_]
                    * zero_kernel
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
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

            prefactor: float = kernel_numerator_full[
                event_mask, event_count
            ].sum() / (kernel_denominator[event_count])

            gradient_two = inverse_sample_size * prefactor

            prefactor = (
                (
                    (
                        linear_predictor_vanilla
                        * kernel_matrix[:, event_count]
                    ).sum()
                    - linear_predictor_vanilla[_] * zero_kernel
                )
                * inverse_bandwidth
                - (linear_predictor_vanilla[_] * zero_integrated_kernel)
            ) / integrated_kernel_denominator[event_count]

            gradient_four = inverse_sample_size * prefactor

            gradient_eta_cox[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            ) - inverse_sample_size

            gradient_eta_aft[_] = gradient_five + inverse_sample_size

            event_count += 1

        else:
            gradient_eta_cox[_] = gradient_three
            gradient_eta_aft[_] = gradient_five
    # Flip the gradient sign since we are performing minimization and
    # concatenate the two gradients since we stack both coefficients
    # into a vector.
    gradient_eta_eh = np.negative(
        np.concatenate((gradient_eta_cox, gradient_eta_aft))
    )
    return gradient_eta_eh