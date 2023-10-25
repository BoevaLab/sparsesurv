from collections.abc import Callable
from typing import Dict

from .baseline_hazard_estimation import (
    breslow_estimator_breslow,
    breslow_estimator_efron,
    get_cumulative_hazard_function_aft,
    get_cumulative_hazard_function_eh,
)
from .loss import (
    aft_negative_likelihood,
    breslow_negative_likelihood,
    efron_negative_likelihood,
    eh_negative_likelihood,
)
from .utils import basic_cv_fold, basic_mse, linear_cv, vvh_cv_fold

# Minimal type hinting here, since signatures are not always
# perfectly aligned for different model types (in particular
# Cox versus kernel-smoothed methods).

LOSS_FACTORY: Dict[str, Callable] = {
    "breslow": breslow_negative_likelihood,
    "efron": efron_negative_likelihood,
    "aft": aft_negative_likelihood,
    "eh": eh_negative_likelihood,
}


BASELINE_HAZARD_FACTORY: Dict[str, Callable] = {
    "breslow": breslow_estimator_breslow,
    "efron": breslow_estimator_efron,
    "aft": get_cumulative_hazard_function_aft,
    "eh": get_cumulative_hazard_function_eh,
}


CVSCORERFACTORY: Dict[str, Callable] = {
    "linear_predictor": linear_cv,
    "vvh": vvh_cv_fold,
    "basic": basic_cv_fold,
    "mse": basic_mse,
}
