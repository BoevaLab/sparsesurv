import torch

from sparsesurv.constants import PDF_PREFACTOR


def normal_density(x: torch.Tensor) -> torch.Tensor:
    """Calculate Gaussian kernel.

    Args:
        x (torch.Tensor): Input of differences.

    Returns:
        torch.Tensor: Gaussian kernel value.
    """
    density = PDF_PREFACTOR * torch.exp(-0.5 * torch.pow(x, 2.0))
    return density
