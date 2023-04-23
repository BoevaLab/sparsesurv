from .bandwidth_estimation import jones_1990, jones_1991
from .baseline_hazard_estimation import (
    baseline_hazard_estimator_aft,
    baseline_hazard_estimator_ah,
    breslow_estimator_breslow,
    breslow_estimator_efron,
)
from .gradients import aft_numba, ah_numba, breslow_numba_stable, efron_numba_stable
from .loss import (
    aft_likelihood,
    ah_likelihood,
    breslow_likelihood_stable,
    efron_likelihood_stable,
)

GRADIENT_FACTORY = {
    "breslow": breslow_numba_stable,
    "efron": efron_numba_stable,
    "accelerated_hazards": ah_numba,
    "accelerated_failure_time": aft_numba,
}

LOSS_FACTORY = {
    "breslow": breslow_likelihood_stable,
    "efron": efron_likelihood_stable,
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
    "accelerated_failure_time": baseline_hazard_estimator_aft,
    "accelerated_hazards": baseline_hazard_estimator_ah,
}
