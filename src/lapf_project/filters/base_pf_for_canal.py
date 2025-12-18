# src/lapf_project/filters/base_pf_for_canal.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from .base_pf import BaseParticleFilter, PFConfig


@dataclass
class PFConfigForCanal(PFConfig):
    """Configuration for a particle filter for canal systems.

    Attributes
    ----------
    num_particles : int
        Number of particles N.
    resample_threshold : Optional[float]
        Threshold for ESS / N. If None, resample at every step.
    device : torch.device or str
        Device on which tensors are stored.
    state_clamp_min : float
        Lower bound for state clamping.
    state_clamp_max : float
        Upper bound for state clamping.
    """

    state_clamp_min: float = 0.0
    state_clamp_max: float = 5.0


class BaseParticleFilterForCanal(BaseParticleFilter):
    """Particle filter base class for the canal system.

    This class implements a particle filter for the following linear system with state clamping:

        x_{k+1} = proj_{[x_min, x_max]}(A x_k + B u_k + w_k),   w_k ~ N(mu_w, Q).

    Subclasses must implement:

        - `compute_log_weights`

    Parameters
    ----------
    config : PFConfigForCanal
        Particle filter configuration.
    A : Tensor, shape (n, n)
        State transition matrix.
    B : Tensor, shape (n, k)
        Control input matrix.
    Q : Tensor, shape (n, n)
        Covariance matrix of the process noise.
    mu_w : Optional[Tensor], shape (n,)
        Mean of the process noise. If None, assumed to be zero.
    state_prior_mean : Tensor, shape (n,)
        Mean of the initial state prior.
    state_prior_cov : Tensor, shape (n, n)
        Covariance of the initial state prior.

    """

    def __init__(
        self,
        config: PFConfigForCanal,
        A: Tensor,
        B: Tensor,
        Q: Tensor,
        mu_w: Optional[Tensor] = None,
        state_prior_mean: Optional[Tensor] = None,
        state_prior_cov: Optional[Tensor] = None,
    ) -> None:
        super().__init__(config)

        self.A = A.to(self.device)  # (n, n)
        self.B = B.to(self.device)  # (n, k)
        self.Q = Q.to(self.device)  # (n, n)
        self.mu_w = (
            mu_w.to(self.device)
            if mu_w is not None
            else torch.zeros(Q.shape[0], device=self.device)
        )  # (n,)

        # Prior distribution parameters
        self.state_prior_mean = (
            state_prior_mean.to(self.device) if state_prior_mean is not None else None
        )  # (n,)
        self.state_prior_cov = (
            state_prior_cov.to(self.device) if state_prior_cov is not None else None
        )  # (n, n)

        if (self.state_prior_mean is None) != (self.state_prior_cov is None):
            raise ValueError(
                "state_prior_mean and state_prior_cov must both be provided or both be None."
            )

    # ------------------------------------------------------------------
    # Implement abstract methods from BaseParticleFilter
    # ------------------------------------------------------------------
    def initialize_particles(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample initial particles from the Gaussian prior.

        Parameters
        ----------
        batch_size : int
            Number of independent filter runs M.

        Returns
        -------
        particles : Tensor, shape (M, N, n)
            Initial particles drawn from N(state_prior_mean, state_prior_cov).
        weights : Tensor, shape (M, N)
            Uniform initial weights (1 / N).
        """
        if self.state_prior_mean is None or self.state_prior_cov is None:
            raise RuntimeError(
                "state_prior_mean and state_prior_cov must be set to initialize particles."
            )

        n = self.state_prior_mean.shape[0]
        if self.state_prior_cov.shape != (n, n):
            raise ValueError("state_prior_cov must have shape (n, n).")

        N = self.config.num_particles
        M = batch_size

        prior_dist = MultivariateNormal(self.state_prior_mean, self.state_prior_cov)
        # (M, N, n)
        particles = prior_dist.sample((M, N)).to(self.device)

        # Uniform initial weights: (M, N)
        weights = torch.full((M, N), 1.0 / N, device=self.device)

        return particles, weights

    def predict_step(self, particles: Tensor, control: Optional[Tensor]) -> Tensor:
        """Linear state transition with Gaussian process noise.

        Parameters
        ----------
        particles : Tensor, shape (M, N, n)
            Particles at time t.
        control : Optional[Tensor]
            Control input at time t. Supported shapes:
                - None: no control input,
                - (k,) : shared control for all batches and particles,
                - (M, k): batch-wise control (shared over particles).

        Returns
        -------
        next_particles : Tensor, shape (M, N, n)
            Predicted particles at time t+1.
        """
        M, N, n = particles.shape
        device = self.device

        # A x_k
        # A: (n, n), particles: (M, N, n) -> (M, N, n)
        Ax = torch.einsum("ij,MNj->MNi", self.A, particles)

        # B u_k
        if control is None:
            Bu = torch.zeros((M, N, n), device=device)
        else:
            control = control.to(device)
            if control.dim() == 1:
                # (k,)
                Bu_single = torch.matmul(self.B, control)  # (n,)
                Bu = Bu_single.view(1, 1, n).expand(M, N, n)
            elif control.dim() == 2 and control.shape[0] == M:
                # (M, k)
                # (M, k) @ (k, n) -> (M, n)
                Bu_batch = torch.matmul(control, self.B.T)  # (M, n)
                Bu = Bu_batch.view(M, 1, n).expand(M, N, n)
            else:
                raise ValueError(
                    "Unsupported control shape. Expected None, (k,) or (M, k), "
                    f"but got {tuple(control.shape)}."
                )

        # Process noise w_k ~ N(mu_w, Q)
        process_dist = MultivariateNormal(
            self.mu_w,
            self.Q,
        )
        w = process_dist.sample((M, N))  # (M, N, n)

        # x_{k+1} = A x_k + B u_k + w_k
        x_next = Ax + Bu + w

        # State clamping (e.g., water level in [0, 5])
        clamp_min = self.config.state_clamp_min
        clamp_max = self.config.state_clamp_max

        if clamp_min is None or clamp_max is None:
            raise ValueError("Both clamp_min and clamp_max must be specified.")

        x_next = torch.clamp(x_next, clamp_min, clamp_max)

        return x_next
