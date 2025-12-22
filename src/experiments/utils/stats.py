# src/experiments/utils/stats.py

from typing import Tuple

import torch
from torch import Tensor


def compute_mean_and_covariance(
    particles: Tensor,  # (M, num_particles, n)
    weights: Tensor,  # (M, num_particles)
) -> Tuple[Tensor, Tensor]:
    """
    Compute weighted mean and covariance from particles.

    Parameters
    ----------
    particles : Tensor, shape (M, N, n)
    weights : Tensor, shape (M, N)

    Returns
    -------
    mean : Tensor, shape (M, n)
    cov  : Tensor, shape (M, n, n)
    """
    mean = torch.einsum("MNn,MN->Mn", particles, weights)  # (M, n)
    diff = particles - mean.unsqueeze(1)  # (M, N, n)
    diff_outer = torch.einsum("MNn,MNm->MNnm", diff, diff)  # (M, N, n, n)
    cov = torch.einsum("MNnm,MN->Mnm", diff_outer, weights)  # (M, n, n)
    return mean, cov


def compute_mean_and_std_of_mse_by_state(
    X_true: Tensor,  # (T, M, n)
    X_est: Tensor,  # (T, M, n)
) -> Tensor:
    squared_errors = (X_true - X_est) ** 2  # (T, M, n)
    mse_by_trial_and_state = torch.mean(squared_errors, dim=0)  # (M, n)

    mean_mse_by_state = torch.mean(mse_by_trial_and_state, dim=0)  # (n,)
    std_mse_by_state = torch.std(mse_by_trial_and_state, dim=0, unbiased=True)  # (n,)

    return mean_mse_by_state.cpu().numpy(), std_mse_by_state.cpu().numpy()


def compute_overall_mean_and_std_of_mse(
    X_true: Tensor,  # (T, M, n)
    X_est: Tensor,  # (T, M, n)
) -> Tuple[float, float]:
    squared_errors = (X_true - X_est) ** 2  # (T, M, n)
    mse_by_trial = torch.mean(squared_errors, dim=(0, 2))  # (M,)

    mean_mse = torch.mean(mse_by_trial)  # scalar
    std_mse = torch.std(mse_by_trial, unbiased=True)  # scalar

    return mean_mse.cpu().numpy(), std_mse.cpu().numpy()
