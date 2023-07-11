from .baseline_hazard_estimation import (
    breslow_estimator_breslow,
    breslow_estimator_efron,
)
from .gradients import (
    breslow_numba_stable,
    breslow_preconditioning,
    efron_numba_stable,
    efron_preconditioning,
)
from .loss import (
    breslow_likelihood_stable,
    breslow_preconditioning_loss,
    efron_likelihood_stable,
    efron_preconditioning_loss,
)

GRADIENT_FACTORY = {
    "breslow": breslow_numba_stable,
    "breslow_preconditioning": breslow_preconditioning,
    "efron": efron_numba_stable,
    "efron_preconditioning": efron_preconditioning,
}

LOSS_FACTORY = {
    "breslow": breslow_likelihood_stable,
    "breslow_preconditioning": breslow_preconditioning_loss,
    "efron": efron_likelihood_stable,
    "efron_preconditioning": efron_preconditioning_loss,
}


BASELINE_HAZARD_FACTORY = {
    "breslow": breslow_estimator_breslow,
    "efron": breslow_estimator_efron,
}


def predict_cumulative_hazard_function_cox(
        query_eta, query_time,
        train_time, train_event, train_eta, 
                                           ):
    pass