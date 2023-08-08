import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from get_data_arrays import get_2d_array

from survhive.gradients import eh_gradient

from .test_eh import eh_calculation


def get_eh_gradient(case="default"):
    linear_predictor, time, event = get_2d_array(case)
    linear_predictor_tensor, time_tensor, event_tensor = (
        torch.from_numpy(linear_predictor),
        torch.from_numpy(time),
        torch.from_numpy(event),
    )
    linear_predictor_tensor.requires_grad_()
    eh_loss_torch = eh_calculation(linear_predictor_tensor, time_tensor, event_tensor)
    eh_loss_torch.backward()
    eh_gradient_torch = linear_predictor_tensor.grad
    return eh_gradient_torch.numpy().flatten()


# cannot use scipy for this as 2d


class TestEHGradient:
    dims = 2
    abs_tol = 1.05e-8

    def test_default(self):
        linear_predictor, time, event = get_2d_array(case="default")
        eh_gradient_computed = (
            eh_gradient(
                linear_predictor,
                time,
                event,
                bandwidth=None,
            )
            .reshape(len(event), self.dims, order="F")
            .flatten()
        )
        eh_torch_gradient = get_eh_gradient(case="default")
        pd.DataFrame(eh_gradient_computed).to_csv("/Users/nja/Desktop/eh_grad_comp.csv")
        pd.DataFrame(eh_torch_gradient).to_csv("/Users/nja/Desktop/eh_grad_torch.csv")
        print("EHTorch", eh_torch_gradient)
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)

    def test_first_five_zero(self):
        linear_predictor, time, event = get_2d_array(case="first_five_zero")
        eh_gradient_computed = (
            eh_gradient(
                linear_predictor,
                time,
                event,
                bandwidth=None,
            )
            .reshape(len(event), self.dims, order="F")
            .flatten()
        )
        eh_torch_gradient = get_eh_gradient(case="first_five_zero")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)

    def test_last_five_zero(self):
        linear_predictor, time, event = get_2d_array(case="last_five_zero")
        eh_gradient_computed = (
            eh_gradient(
                linear_predictor,
                time,
                event,
                bandwidth=None,
            )
            .reshape(len(event), self.dims, order="F")
            .flatten()
        )
        eh_torch_gradient = get_eh_gradient(case="last_five_zero")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)

    def test_high_event_ratio(self):
        linear_predictor, time, event = get_2d_array(case="high_event_ratio")
        eh_gradient_computed = (
            eh_gradient(
                linear_predictor,
                time,
                event,
                bandwidth=None,
            )
            .reshape(len(event), self.dims, order="F")
            .flatten()
        )
        eh_torch_gradient = get_eh_gradient(case="high_event_ratio")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)

    def test_low_event_ratio(self):
        linear_predictor, time, event = get_2d_array(case="low_event_ratio")
        eh_gradient_computed = (
            eh_gradient(
                linear_predictor,
                time,
                event,
                bandwidth=None,
            )
            .reshape(len(event), self.dims, order="F")
            .flatten()
        )
        eh_torch_gradient = get_eh_gradient(case="low_event_ratio")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)

    def test_all_events(self):
        linear_predictor, time, event = get_2d_array(case="all_events")

        eh_gradient_computed = (
            eh_gradient(
                linear_predictor,
                time,
                event,
                bandwidth=None,
            )
            .reshape(len(event), self.dims, order="F")
            .flatten()
        )
        eh_torch_gradient = get_eh_gradient(case="all_events")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)
