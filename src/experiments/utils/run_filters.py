# src/experiments/utils/run_filters.py
# cd lapf-ifacjsc2026

import torch

from ...lapf_project.filters.lapf_for_clamped_linear_gaussian import (
    ClampedLinearGaussianLAPF,
    ClampedLinearGaussianLAPFConfig,
)


def run_no_observation(
    config: ClampedLinearGaussianLAPFConfig,
    T: int,
    M: int,
    A: torch.Tensor,
    Q: torch.Tensor,
    mu_w: torch.Tensor,
    state_prior_mean: torch.Tensor,
    state_prior_cov: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    no_ob = ClampedLinearGaussianLAPF(
        config=config,
        A=A,
        Q=Q,
        mu_w=mu_w,
        state_prior_mean=state_prior_mean,
        state_prior_cov=state_prior_cov,
    )

    # Initialize particles
    no_ob.initialize(batch_size=M)

    # Observations: None
    observations = None

    particles_hist, weights_hist = no_ob.run(
        T=T,
        observations=observations,
    )

    return particles_hist.cpu(), weights_hist.cpu()


def run_edapf(
    config: ClampedLinearGaussianLAPFConfig,
    T: int,
    M: int,
    A: torch.Tensor,
    Q: torch.Tensor,
    mu_w: torch.Tensor,
    state_prior_mean: torch.Tensor,
    state_prior_cov: torch.Tensor,
    C_human: torch.Tensor,
    R_human: torch.Tensor,
    Y_human_tilde: torch.Tensor,
    R_tilde: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edapf = ClampedLinearGaussianLAPF(
        config=config,
        A=A,
        Q=Q,
        mu_w=mu_w,
        state_prior_mean=state_prior_mean,
        state_prior_cov=state_prior_cov,
        C=C_human,
        R=R_human + R_tilde,
    )

    # Initialize particles
    edapf.initialize(batch_size=M)

    # Observations: list of dicts with only predicted cognitive values
    observations = []
    for t in range(T):
        observations.append({"y_phys": Y_human_tilde[t, :, :]})  # (M, p_human)

    particles_hist, weights_hist = edapf.run(
        T=T,
        observations=observations,
    )

    return particles_hist.cpu(), weights_hist.cpu()


def run_lapf(
    config: ClampedLinearGaussianLAPFConfig,
    T: int,
    M: int,
    A: torch.Tensor,
    Q: torch.Tensor,
    mu_w: torch.Tensor,
    state_prior_mean: torch.Tensor,
    state_prior_cov: torch.Tensor,
    C_human: torch.Tensor,
    R_human: torch.Tensor,
    Y_phys: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lapf = ClampedLinearGaussianLAPF(
        config=config,
        A=A,
        Q=Q,
        mu_w=mu_w,
        state_prior_mean=state_prior_mean,
        state_prior_cov=state_prior_cov,
        C_human=C_human,
        R_human=R_human,
    )

    # Initialize particles
    lapf.initialize(batch_size=M)

    # Observations: list of dicts with only human_label_probs
    observations = []
    for t in range(T):
        y_phys = Y_phys[t, :, :] if Y_phys is not None else None
        human_label_probs = probs[t, :, :, :] if probs is not None else None
        observations.append(
            {
                "y_phys": y_phys,  # (M, p_human)
                "human_label_probs": human_label_probs,  # (M, p_human, num_labels)
            }
        )

    particles_hist, weights_hist = lapf.run(
        T=T,
        observations=observations,
    )

    return particles_hist.cpu(), weights_hist.cpu()
