from scipy.optimize import check_grad
from .get_data_arrays import get_1d_array

from sparsesurv.gradients import aft_gradient
from sparsesurv.loss import aft_negative_likelihood


class TestAFTGradients:
    tolerance = 1e-6

    def test_default(self):
        linear_predictor, time, event = get_1d_array("default")
        diff = check_grad(
            lambda x: aft_negative_likelihood(x, time, event),
            lambda x: aft_gradient(x, time, event),
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_first_five_zero(self):
        linear_predictor, time, event = get_1d_array("first_five_zero")
        diff = check_grad(
            lambda x: aft_negative_likelihood(x, time, event),
            lambda x: aft_gradient(x, time, event),
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_last_five_zero(self):
        linear_predictor, time, event = get_1d_array("last_five_zero")
        diff = check_grad(
            lambda x: aft_negative_likelihood(x, time, event),
            lambda x: aft_gradient(x, time, event),
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_high_event_ratio(self):
        linear_predictor, time, event = get_1d_array("high_event_ratio")
        diff = check_grad(
            lambda x: aft_negative_likelihood(x, time, event),
            lambda x: aft_gradient(x, time, event),
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_low_event_ratio(self):
        linear_predictor, time, event = get_1d_array("low_event_ratio")
        diff = check_grad(
            lambda x: aft_negative_likelihood(x, time, event),
            lambda x: aft_gradient(x, time, event),
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_all_events(self):
        linear_predictor, time, event = get_1d_array("all_events")
        diff = check_grad(
            lambda x: aft_negative_likelihood(x, time, event),
            lambda x: aft_gradient(x, time, event),
            linear_predictor,
        )
        assert diff < self.tolerance
