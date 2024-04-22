from sparsesurv.neuralsurv.python.model.fusion import EarlyFusion
from sparsesurv.neuralsurv.python.model.skorch_infra import CoxPHNeuralNet
from sparsesurv.neuralsurv.python.utils.misc_utils import BreslowLoss

FUSION_FACTORY = {
    "early": EarlyFusion,
}

CRITERION_FACTORY = {
    "cox": BreslowLoss,
}


SKORCH_NET_FACTORY = {
    "cox": CoxPHNeuralNet,
}
