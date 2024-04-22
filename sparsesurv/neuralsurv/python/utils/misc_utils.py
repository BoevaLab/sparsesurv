import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from skorch.dataset import ValidSplit, get_len
from skorch.utils import to_numpy
from torch.nn.modules.loss import _Loss


# Adapted from https://github.com/pytorch/pytorch/issues/7068.
def seed_torch(seed=42):
    """Sets all seeds within torch and adjacent libraries.

    Args:
        seed: Random seed to be used by the seeding functions.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


def create_risk_matrix(observed_survival_time):
    observed_survival_time = observed_survival_time.squeeze()
    return (
        (
            torch.outer(observed_survival_time, observed_survival_time)
            >= torch.square(observed_survival_time)
        )
        .long()
        .T
    )


def negative_partial_log_likelihood(
    predicted_log_hazard_ratio,
    observed_survival_time,
    observed_event_indicator,
):
    if torch.sum(observed_event_indicator) <= 0.0:
        return torch.tensor(0.0, requires_grad=True)
    risk_matrix = create_risk_matrix(observed_survival_time)
    loss = -torch.sum(
        observed_event_indicator.float().squeeze()
        * (
            predicted_log_hazard_ratio.squeeze()
            - torch.log(
                torch.sum(
                    risk_matrix.float()
                    * torch.exp(predicted_log_hazard_ratio.squeeze()),
                    axis=1,
                )
            )
        )
    ) / torch.sum(observed_event_indicator)
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError
    return loss


def transform_torch(time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    event_mod = event.clone()
    event_mod[event_mod == 0] = -1
    if (time == 0).any():
        raise RuntimeError("Data contains zero time value!")
    y = event_mod * time
    return y.float()


def transform_back_torch(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transforms XGBoost digestable format variable y into time and event.

    Parameters
    ----------
    y : npt.NDArray[float]
        Array containing survival time and event where negative value is taken as censored event.

    Returns
    -------
    tuple[npt.NDArray[float],npt.NDArray[int]]
        Survival time and event.
    """
    time = torch.abs(y)
    event = (torch.abs(y) == y).float()
    return time, event


def transform(time, event):
    event_mod = np.copy(event)
    event_mod[event_mod == 0] = -1
    if np.any(time == 0):
        raise RuntimeError("Data contains zero time value!")
    y = event_mod * time
    return y


def transform_back(y):
    time = np.abs(y)
    event = np.abs(y) == y
    event = event.astype(np.int64)
    return time, event


def breslow_likelihood_torch(
    y: torch.Tensor, log_partial_hazard: torch.Tensor
) -> torch.Tensor:
    """Generate negative loglikelihood (loss) according to Breslow.
    Assumes times have been sorted beforehand.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    npt.NDArray[float]
        Negative loglikelihood (loss) according to Breslow.
    """

    if isinstance(log_partial_hazard, np.ndarray):
        log_partial_hazard = torch.from_numpy(log_partial_hazard)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    log_partial_hazard = log_partial_hazard.double()
    time = torch.abs(y)
    event = (torch.abs(y) == y).float()
    log_partial_hazard = torch.clamp(log_partial_hazard, -75, 75)
    partial_hazard = torch.exp(log_partial_hazard)
    n_events = torch.sum(event)
    n_samples = time.shape[0]
    previous_time = time[0]
    risk_set_sum = 0
    likelihood = 0
    set_count = 0
    accumulated_sum = 0
    risk_set_sum = torch.sum(partial_hazard)

    for k in range(n_samples):
        current_time = time[k]
        if current_time > previous_time:
            if set_count > 0:
                likelihood = likelihood - (set_count * torch.log(risk_set_sum))
            risk_set_sum = risk_set_sum - accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count = set_count + 1
            likelihood = likelihood + log_partial_hazard[k]

        previous_time = current_time
        accumulated_sum = accumulated_sum + partial_hazard[k]
    if set_count > 0:
        likelihood = likelihood - (set_count * torch.log(risk_set_sum))
    if torch.isnan(likelihood) or torch.isinf(likelihood):
        raise ValueError
    return -likelihood / n_events


class BreslowLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, prediction, input):
        time, event = transform_back_torch(input)
        loss = negative_partial_log_likelihood(prediction, time, event)
        return loss


def calculate_log_hazard_input_size(fusion_method, blocks, modality_dimension):
    match fusion_method:
        case "early":
            return sum([len(block) for block in blocks])
        case "early_ae":
            return modality_dimension
        case "late_mean":
            raise ValueError
        case "late_moe":
            raise ValueError
        case "intermediate_mean":
            return modality_dimension
        case "intermediate_max":
            return modality_dimension
        case "intermediate_concat":
            return modality_dimension * len(blocks)
        case "intermediate_ae":
            return modality_dimension * len(blocks)
        case "intermediate_embrace":
            return modality_dimension
        case "intermediate_attention":
            return modality_dimension


class StratifiedSkorchSurvivalSplit(ValidSplit):
    def __call__(self, dataset, y=None, groups=None):
        if y is not None:
            if np.min(y) < 0.0:
                y = (np.abs(y) == y).astype(float)

        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y."
        )

        if (y is None) and self.stratified:
            raise bad_y_error

        cv = self.check_cv(y)
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if len_dataset != len_y:
                raise ValueError(
                    "Cannot perform a CV split if dataset and y "
                    "have different lengths."
                )

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid


class StratifiedSurvivalKFold(StratifiedKFold):
    """Adapt `StratifiedKFold` to make it usable with our adapted
    survival target format.
    """

    def _make_test_folds(self, X, y=None):
        if y is not None and isinstance(y, np.ndarray):
            #if np.min(y) < 0.0:
            if y.dtype.names is not None:
                #y = (np.abs(y) == y).astype(float)
                y = y["event"].astype(np.float64)

        return super()._make_test_folds(X=X, y=y)

    def _iter_test_masks(self, X, y=None, groups=None):
        if y is not None and isinstance(y, np.ndarray):
            if y.dtype.names is not None:
                y = y["event"].astype(np.float64)
        return super()._iter_test_masks(X, y=y)

    def split(self, X, y, groups=None):
        return super().split(X=X, y=y, groups=groups)
