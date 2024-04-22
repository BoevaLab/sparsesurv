import torch
from skorch.callbacks import Callback
from skorch.net import NeuralNet
from skorch.utils import to_numpy

from sparsesurv.neuralsurv.python.model.sksurv_imports import BreslowEstimator
from sparsesurv.neuralsurv.python.utils.misc_utils import seed_torch, transform
from sparsesurv.utils import inverse_transform_survival


class BaseSurvivalNet(NeuralNet):
    def predict(self, X):
        log_hazard_ratios = to_numpy(self.forward(X))
        return log_hazard_ratios


class CoxPHNeuralNet(BaseSurvivalNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        time, event = inverse_transform_survival(y)
        y_neural = transform(time=time, event=event)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y_neural, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X))
            .detach()
            .numpy()
            .ravel()
            .astype(float),
            time,
            event,
        )
        return self

    def fit_breslow(self, log_hazard_ratios, time, event):
        self.breslow = BreslowEstimator().fit(log_hazard_ratios, event, time)

    def predict_survival_function(self, X):
        log_hazard_ratios = self.forward(X).detach().numpy().ravel().astype(float)
        survival_function = self.breslow.get_survival_function(log_hazard_ratios)
        return survival_function


class FixSeed(Callback):
    def __init__(self, generator):
        self.generator = generator

    def initialize(self):
        seed = self.generator.integers(low=0, high=262144, size=1)[0]
        seed_torch(seed)
        return super().initialize()
