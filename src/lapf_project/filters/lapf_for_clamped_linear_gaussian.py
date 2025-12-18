# src/lapf_project/filters/lapf_for_clamped_linear_gaussian.py

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor
from torch.distributions import Normal

from .base_pf_for_clamped_linear_gaussian import (
    ClampedLinearGaussianPF,
    ClampedLinearGaussianPFConfig,
)


@dataclass
class ClampedLinearGaussianLAPFConfig(ClampedLinearGaussianPFConfig):
    """Configuration for ClampedLinearGaussianLAP.

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
    observation_clamp_min : float
        Lower bound for observation clamping.
    observation_clamp_max : float
        Upper bound for observation clamping.
    """

    observation_clamp_min: float = 0.0
    observation_clamp_max: float = 5.0


class ClampedLinearGaussianLAPF(ClampedLinearGaussianPF):
    """Clamped linear Gaussian particle filter with optional human sensor (LAPF).

    This class *extends* `ClampedLinearGaussianPF` by adding observation models (physics and human).
    The dynamics/prior parameters (A, B, Q, mu_w, state_prior_*) are accepted here and forwarded to the parent constructor.

    Parameters
    ----------
    config : ClampedLinearGaussianLAPFConfig
        Particle filter configuration.
    A : Tensor, shape (n, n)
        State transition matrix.
    B : Tensor, shape (n, k)
        Input matrix.
    Q : Tensor, shape (n, n)
        Covariance matrix of the process noise.
    mu_w : Optional[Tensor], shape (n,)
        Mean of the process noise. If None, assumed to be zero.
    state_prior_mean : Tensor, shape (n,)
        Mean of the initial state prior.
    state_prior_cov : Tensor, shape (n, n)
        Covariance of the initial state prior.
    C : Optional[Tensor], shape (p, n)
        Observation matrix for the physics sensor. If None, the physics
        sensor is disabled.
    R : Optional[Tensor], shape (p, p)
        Covariance matrix of the physics sensor noise. Must be provided
        if C is not None.
    mu_v : Optional[Tensor], shape (n,)
        Mean of the process noise. If None, assumed to be zero.
    C_human : Optional[Tensor], shape (p_human, n)
        Observation matrix for the human sensor. If None, the human
        sensor is disabled.
    R_human : Optional[Tensor], shape (p_human, p_human)
        Covariance matrix of the human sensor noise. Must be provided
        if C_human is not None.
    mu_v_human : Optional[Tensor], shape (n,)
        Mean of the human sensor noise. If None, assumed to be zero.

    Notes
    -----
    - If both `C`/`R` and `C_human`/`R_human` are None, the filter
      performs pure prediction (no measurement update).
    - Observations are expected to be passed as dictionaries:

        observation = {
            "y_phys": y_phys,                       # Tensor, shape (M, p)
            "human_label_probs": human_label_probs, # Tensor, shape (M, p_human, num_labels)
        }

      Each entry can be omitted (or set to None) if the corresponding
      sensor is not used at a given time step.
    """

    def __init__(
        self,
        config: ClampedLinearGaussianLAPFConfig,
        A: Tensor,
        B: Tensor,
        Q: Tensor,
        mu_w: Optional[Tensor] = None,
        state_prior_mean: Optional[Tensor] = None,
        state_prior_cov: Optional[Tensor] = None,
        C: Optional[Tensor] = None,
        R: Optional[Tensor] = None,
        mu_v: Optional[Tensor] = None,
        C_human: Optional[Tensor] = None,
        R_human: Optional[Tensor] = None,
        mu_v_human: Optional[Tensor] = None,
    ) -> None:
        super().__init__(config, A, B, Q, mu_w, state_prior_mean, state_prior_cov)

        self.C = C.to(self.device) if C is not None else None  # (p, n)
        self.R = R.to(self.device) if R is not None else None  # (p, p)
        self.mu_v = (
            (
                mu_v.to(self.device)
                if mu_v is not None
                else torch.zeros(R.shape[0], device=self.device)
            )
            if R is not None
            else None
        )  # (p,)
        self.C_human = (
            C_human.to(self.device) if C_human is not None else None
        )  # (p_human, n)
        self.R_human = (
            R_human.to(self.device) if R_human is not None else None
        )  # (p_human, p_human)
        self.mu_v_human = (
            (
                mu_v_human.to(self.device)
                if mu_v_human is not None
                else torch.zeros(R_human.shape[0], device=self.device)
            )
            if R_human is not None
            else None
        )  # (p_human,)

        # Basic consistency checks for sensor matrices
        if (self.C is None) != (self.R is None):
            raise ValueError(
                "Physics sensor: C and R must be provided together or both be None."
            )
        if (self.C_human is None) != (self.R_human is None):
            raise ValueError(
                "Human sensor: C_human and R_human must be provided together or both be None."
            )

    # ------------------------------------------------------------------
    # Implement abstract methods from BaseParticleFilter
    # ------------------------------------------------------------------
    def compute_log_weights(
        self,
        particles: Tensor,
        observation: Any,
    ) -> Tensor:
        """Compute log-weights from physics and/or human observations.

        Parameters
        ----------
        particles : Tensor, shape (M, N, n)
            Particles at the current time step.
        observation : Any
            Observation at the current time step. Typically a dict:

                observation = {
                    "y_phys": y_phys,                         # (M, p)
                    "human_label_probs": human_label_probs,   # (M, p_human, num_labels)
                }

            Each entry can be omitted or set to None if the sensor is
            not available at the current time step.

        Returns
        -------
        log_weights : Tensor, shape (M, N)
            Log-weights (log-likelihood) for each particle.
        """
        device = self.device
        M, N, _ = particles.shape

        # Default: no contribution (log 1 = 0)
        log_weights = torch.zeros((M, N), device=device)

        # Unpack observation
        y_phys: Optional[Tensor] = None
        human_label_probs: Optional[Tensor] = None

        if isinstance(observation, dict):
            y_phys = observation.get("y_phys", None)
            human_label_probs = observation.get("human_label_probs", None)
        elif observation is not None:
            # Allow passing only physics observations directly as a tensor
            if observation.ndim == 2:
                y_phys = observation

            # Allow passing only human observations directly as a tensor
            elif observation.ndim == 3:
                human_label_probs = observation

            else:
                raise ValueError(
                    "If observation is not a dict, it must be a tensor "
                    "of shape (M, p) for physics or (M, p_human, num_labels) for human."
                )

        # Physics sensor
        if y_phys is not None:
            if self.C is None or self.R is None:
                raise ValueError(
                    "Physics sensor matrices (C, R) are not provided, "
                    "but 'y_phys' was given."
                )
            log_weights += self._compute_loglikelihood_physics(particles, y_phys)

        # Human sensor
        if human_label_probs is not None:
            if self.C_human is None or self.R_human is None:
                raise ValueError(
                    "Human sensor matrices (C_human, R_human) are not provided, "
                    "but 'human_label_probs' was given."
                )
            log_weights += self._compute_loglikelihood_human(
                particles,
                human_label_probs,
            )

        # If both sensors are disabled, log_weights remains zero,
        # corresponding to a pure prediction step (no observation update).
        return log_weights

    # ------------------------------------------------------------------
    # Sensor-specific log-likelihoods
    # ------------------------------------------------------------------
    def _compute_loglikelihood_physics(
        self,
        particles: Tensor,  # (M, N, n)
        y_phys: Tensor,  # (M, p)
    ) -> Tensor:
        """Compute log p(y_phys | x) for the physics sensor.

        The observation model is

            y_phys ~ N(C x + mu_v, R)

        with clamped observations in [y_min, y_max].
        """
        if self.C is None or self.R is None:
            raise RuntimeError("Physics sensor matrices C and R are not set.")

        device = self.device
        C = self.C
        R = self.R

        p = C.shape[0]  # observation dimension

        # Mean of y | x: (M, N, p)
        mu_y = torch.einsum("pm,MNm->MNp", C, particles) + self.mu_v

        # Standard deviation per dimension: (p,)
        std = torch.sqrt(torch.diag(R)).to(device)

        # Expand shapes for broadcasting
        std_exp = std.view(1, 1, p)  # (1, 1, p)
        y_phys_exp = y_phys.unsqueeze(1)  # (M, 1, p)
        y_min = (
            torch.tensor(self.observation_clamp_min, device=device)
            .view(1, 1, 1)
            .expand_as(y_phys_exp)
        )
        y_max = (
            torch.tensor(self.observation_clamp_max, device=device)
            .view(1, 1, 1)
            .expand_as(y_phys_exp)
        )

        # Endpoint classification (with tolerance)
        is_lower = torch.isclose(y_phys_exp, y_min, atol=1e-9, rtol=0.0)
        is_upper = torch.isclose(y_phys_exp, y_max, atol=1e-9, rtol=0.0)
        is_mid = ~(is_lower | is_upper)

        # Normal distribution N(mu_y, std^2)
        normals = Normal(loc=mu_y, scale=std_exp)

        # Mid points: log pdf
        log_pdf_mid = normals.log_prob(y_phys_exp) * is_mid

        # Lower bound: log Φ(y_min)
        log_cdf_low = torch.log(torch.clamp(normals.cdf(y_min), min=1e-12)) * is_lower

        # Upper bound: log (1 - Φ(y_max))
        log_cdf_up = (
            torch.log(torch.clamp(1.0 - normals.cdf(y_max), min=1e-12)) * is_upper
        )

        # Sum over observation dimensions
        loglik_per_dim = log_pdf_mid + log_cdf_low + log_cdf_up  # (M, N, p)
        loglik = loglik_per_dim.sum(dim=-1)  # (M, N)

        return loglik

    def _compute_loglikelihood_human(
        self,
        particles: Tensor,  # (M, N, n)
        human_label_probs: Tensor,  # (M, p_human, num_labels)
    ) -> Tensor:
        """Compute log p(s | x) for the human sensor using a quantized Gaussian model.

        The model assumes that the human observes a latent continuous
        quantity

            y_human ~ N(C_human x + mu_v_human, R_human)

        which is then quantized into num_labels bins over [y_min, y_max].
        The quantized-label classification model provides p(q | s) as `human_label_probs`. This method computes p(q | x) and combines it with p(q | s).
        """
        if self.C_human is None or self.R_human is None:
            raise RuntimeError("Human sensor matrices C_human and R_human are not set.")

        device = self.device
        C_h = self.C_human
        R_h = self.R_human

        p_human = C_h.shape[0]  # human sensor dimension
        num_labels = human_label_probs.shape[-1]  # number of quantization labels

        # Mean of y_human | x: (M, N, p_human)
        mu_y = torch.einsum("pm,MNm->MNp", C_h, particles) + self.mu_v_human

        # Standard deviation per dimension: (p_human,)
        std = torch.sqrt(torch.diag(R_h)).to(device)

        # Quantization thresholds in [y_min, y_max]
        y_min = self.observation_clamp_min
        y_max = self.observation_clamp_max
        thresholds = torch.linspace(y_min, y_max, num_labels + 1, device=device)[
            1:num_labels
        ]
        # (num_labels-1,)

        # Broadcast shapes
        mean_ = mu_y.unsqueeze(-1)  # (M, N, p_human, 1)
        std_ = std.view(1, 1, p_human, 1)  # (1, 1, p_human, 1)
        thresholds_ = thresholds.view(1, 1, 1, -1)  # (1, 1, 1, num_labels-1)

        # Standardized thresholds
        z = (thresholds_ - mean_) / std_  # (M, N, p_human, num_labels-1)

        # Standard normal CDF
        standard_normal = Normal(0.0, 1.0)
        cdf_thresholds = standard_normal.cdf(z)  # (M, N, p_human, num_labels-1)
        # Pad with 0 and 1 to get CDF at bin edges
        left_pad = torch.zeros_like(cdf_thresholds[..., :1], device=device)  # (..., 1)
        right_pad = torch.ones_like(cdf_thresholds[..., :1], device=device)  # (..., 1)
        cdf = torch.cat(
            [left_pad, cdf_thresholds, right_pad], dim=-1
        )  # (M, N, p_human, num_labels+1)

        # Bin probabilities p(q = ℓ | x): difference of neighboring CDFs
        # Result: (M, N, p_human, num_labels)
        prob_q_given_x = cdf[..., 1:] - cdf[..., :-1]
        prob_q_given_x = torch.clamp(prob_q_given_x, min=1e-8)

        # human_label_probs: (M, p_human, num_labels) = p(q | s)
        # We combine them via:
        #
        #   w_{m,i,p} ∝ sum_{ℓ} p(q=ℓ | x_{m,i}) * p(q=ℓ | s_{m,p})
        #
        # einsum indices:
        #   prob_q_given_x: (M, N, p_human, num_labels)
        #   human_label_probs: (M, p_human, num_labels)
        # -> weights_by_dim: (M, N, p_human)
        weights_by_dim = torch.einsum(
            "MNpL,MpL->MNp", prob_q_given_x, human_label_probs.to(device)
        )  # (M, N, p_human)

        # Convert to log domain and sum over human sensor dimensions
        log_weights_by_dim = torch.log(weights_by_dim + 1e-8)  # (M, N, p_human)
        log_weights_human = log_weights_by_dim.sum(dim=-1)  # (M, N)

        return log_weights_human
