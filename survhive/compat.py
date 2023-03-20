import numpy as np
from .gradients import ah_numba, aft_numba, breslow_numba, efron_numba
from .loss import (
    ah_likelihood,
    aft_likelihood,
    breslow_likelihood,
    efron_likelihood,
)
from .bandwidth_estimation import jones_1990, jones_1991
from .baseline_hazard_estimation import (
    breslow_estimator_breslow,
    breslow_estimator_efron,
)


GRADIENT_FACTORY = {
    "breslow": breslow_numba,
    "efron": efron_numba,
    "accelerated_hazards": ah_numba,
    "accelerated_failure_time": aft_numba,
}

LOSS_FACTORY = {
    "breslow": breslow_likelihood,
    "efron": efron_likelihood,
    "accelerated_failure_time": aft_likelihood,
    "accelerated_hazards": ah_likelihood,
}

BANDWIDTH_FUNCTION_FACTORY = {
    "jones_1990": jones_1990,
    "jones_1991": jones_1991,
}

BASELINE_HAZARD_FACTORY = {
    "breslow": breslow_estimator_breslow,
    "efron": breslow_estimator_efron,
    "accelerated_failure_time": np.array,
    "accelerated_hazards": np.array,
}
