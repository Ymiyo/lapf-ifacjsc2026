# src/lapf_project/filters/base_pf.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class PFConfig:
    """Configuration for a particle filter.

    Attributes
    ----------
    num_particles : int
        Number of particles N.
    resample_threshold : Optional[float]
        Threshold for the effective sample size (ESS) relative to N.
        If None, resampling is performed at every time step.
        If not None, resampling is performed when ESS / N < resample_threshold.
    device : torch.device or str
        Device on which particles and weights are stored.
    """

    num_particles: int
    resample_threshold: Optional[float] = None
    device: torch.device | str


class BaseParticleFilter(ABC):
    """Abstract base class for batched particle filters with PyTorch.

    This class implements the generic SMC loop for a batched particle filter:

        1. Initialize particles and weights.
        2. For each time step:
            - Predict: propagate particles through the dynamics model.
            - Update: compute log-weights using the likelihood model.
            - Normalize weights.
            - Resample (always or when ESS is low).

    Subclasses must implement:

        - `initialize_particles`
        - `predict_step`
        - `compute_log_weights`

    Notes
    -----
    - Particles: shape (M, N, n)
    - Weights: shape (M, N)
    """

    def __init__(self, config: PFConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.particles: Optional[Tensor] = None  # (M, N, n)
        self.weights: Optional[Tensor] = None  # (M, N)

    # ------------------------------------------------------------------
    # Methods that subclasses must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize_particles(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Initialize particles and weights at time t = 0.

        This method should sample particles from the prior distribution
        and return (particles, weights) with uniform weights.

        Parameters
        ----------
        batch_size : int
            Number of independent filter runs M.

        Returns
        -------
        particles : Tensor, shape (M, N, n)
            Initial particles.
        weights : Tensor, shape (M, N)
            Initial (normalized) weights.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, particles: Tensor, control: Optional[Tensor]) -> Tensor:
        """Propagate particles through the dynamics model for one time step.

        Parameters
        ----------
        particles : Tensor, shape (M, N, n)
            Particles at time t.
        control : Optional[Tensor]
            Control input u_t. The shape and meaning depend on the
            specific system (e.g., shape=(k,) or (M, k)).

        Returns
        -------
        next_particles : Tensor, shape (M, N, n)
            Particles at time t+1.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_log_weights(
        self,
        particles: Tensor,
        observation: Any,
    ) -> Tensor:
        """Compute log-weights (log-likelihoods) for each particle.

        This method should combine all available sensors (physics,
        human, etc.) and return log p(y_t | x_t^{(i)}) up to an additive
        constant.

        Parameters
        ----------
        particles : Tensor, shape (M, N, n)
            Particles at the current time step.
        observation : Any
            Observation at the current time step. Typical examples:
            - physics-only:
                observation = {"y_phys": y_phys}
            - LAPF (physics + human):
                observation = {
                    "y_phys": y_phys,
                    "human_label_probs": human_label_probs,
                }

        Returns
        -------
        log_weights : Tensor, shape (M, N)
            Log-weights (unnormalized) for each particle.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common particle-filter operations
    # ------------------------------------------------------------------
    def initialize(self, batch_size: int) -> None:
        """Initialize particles and weights.

        This method calls `initialize_particles` implemented by the
        subclass and sets internal attributes.

        Parameters
        ----------
        batch_size : int
            Number of independent filter runs M.
        """
        particles, weights = self.initialize_particles(batch_size)
        particles = particles.to(self.device)
        weights = weights.to(self.device)

        if particles.ndim != 3:
            raise ValueError("Particles must have shape (M, N, n).")
        if weights.ndim != 2:
            raise ValueError("Weights must have shape (M, N).")

        M, N, n = particles.shape
        if N != self.config.num_particles:
            raise ValueError(
                f"Expected {self.config.num_particles} particles, but got {N}."
            )
        if weights.shape != (M, N):
            raise ValueError(
                f"Weights must have shape (M, N), but got {weights.shape}."
            )

        # Normalize weights just in case
        weights = self._normalize_weights(weights)

        self.particles = particles
        self.weights = weights

    def predict(self, control: Optional[Tensor]) -> None:
        """Propagate particles through the dynamics model.

        Parameters
        ----------
        control : Optional[Tensor]
            Control input u_t for the current time step.
        """
        if self.particles is None:
            raise RuntimeError(
                "Particles are not initialized. Call `initialize()` first."
            )

        self.particles = self.predict_step(self.particles, control)

    def update(self, observation: Any) -> None:
        """Update particle weights using the given observation.

        Parameters
        ----------
        observation : Any
            Observation at the current time step.
        """
        if self.particles is None or self.weights is None:
            raise RuntimeError("Filter is not initialized. Call `initialize()` first.")

        log_w = self.compute_log_weights(self.particles, observation)
        if log_w.shape != self.weights.shape:
            raise ValueError(
                f"Shape mismatch between log-weights {log_w.shape} "
                f"and weights {self.weights.shape}."
            )

        # Numerically stable weight update:
        # w_t^{(i)} ∝ w_{t-1}^{(i)} * exp(log p(y_t | x_t^{(i)}))
        log_prev = torch.log(self.weights + 1e-30)
        log_new = log_prev + log_w
        self.weights = self._normalize_log_weights(log_new)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    def effective_sample_size(self) -> Tensor:
        """Compute the effective sample size (ESS) for each batch.

        Returns
        -------
        ess : Tensor, shape (M,)
            ESS for each batch m, defined as 1 / sum_i w_{m,i}^2.
        """
        if self.weights is None:
            raise RuntimeError("Weights are not initialized.")

        # (M,)
        ess = 1.0 / torch.sum(self.weights**2, dim=1)
        return ess

    def maybe_resample(self) -> None:
        """Perform resampling if necessary.

        - If `resample_threshold` is None: always resample.
        - Otherwise: resample only when ESS / N < resample_threshold.
        """
        if self.particles is None or self.weights is None:
            raise RuntimeError("Filter is not initialized.")

        M, N, _ = self.particles.shape
        if self.config.resample_threshold is None:
            do_resample = torch.ones(M, dtype=torch.bool, device=self.device)
        else:
            ess = self.effective_sample_size()  # (M,)
            threshold = self.config.resample_threshold * N
            do_resample = ess < threshold  # (M,)

        if not torch.any(do_resample):
            return

        # Resample batch by batch
        new_particles_list = []
        new_weights_list = []

        for m in range(M):
            if do_resample[m]:
                indices = torch.multinomial(
                    self.weights[m], N, replacement=True
                )  # (N,)
                new_particles = self.particles[m, indices]  # (N, n)
                new_weights = torch.full((N,), 1.0 / N, device=self.device)
            else:
                new_particles = self.particles[m]
                new_weights = self.weights[m]

            new_particles_list.append(new_particles)
            new_weights_list.append(new_weights)

        self.particles = torch.stack(new_particles_list, dim=0)  # (M, N, n)
        self.weights = torch.stack(new_weights_list, dim=0)  # (M, N)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def step(
        self,
        control: Optional[Tensor],
        observation: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Perform a single predict-update-resample step.

        Parameters
        ----------
        control : Optional[Tensor]
            Control input at the current time step.
        observation : Any
            Observation at the current time step.

        Returns
        -------
        particles : Tensor, shape (M, N, n)
            Particles after the update (and optional resampling).
        weights : Tensor, shape (M, N)
            Corresponding weights.
        """
        self.predict(control)
        self.update(observation)
        self.maybe_resample()

        assert self.particles is not None
        assert self.weights is not None
        return self.particles.clone(), self.weights.clone()

    def run(
        self,
        controls: Optional[Tensor],
        observations: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Run the filter over a sequence of observations.

        Parameters
        ----------
        controls : Optional[Tensor]
            Sequence of control inputs. Can be:
                - None (no control),
                - shape (T, k),
                - or shape (T, M, k), depending on the system.
        observations : Any
            Sequence of observations (length T). For example:
                - a list of tensors,
                - a list of dicts {'y_phys': ..., 'human_label_probs': ...}, etc.

        Returns
        -------
        particles_hist : Tensor, shape (T, M, N, n)
            History of particles.
        weights_hist : Tensor, shape (T, M, N)
            History of weights.
        """
        if self.particles is None or self.weights is None:
            raise RuntimeError("Filter is not initialized. Call `initialize()` first.")

        T = len(observations)
        M, N, n = self.particles.shape

        particles_hist = torch.zeros((T, M, N, n), device=self.device)
        weights_hist = torch.zeros((T, M, N), device=self.device)

        for t in range(T):
            if controls is None:
                u_t = None
            elif controls.dim() == 2:
                # (T, k)
                u_t = controls[t]
            elif controls.dim() == 3:
                # (T, M, k)
                u_t = controls[t]
            else:
                raise ValueError(f"Unexpected control shape: {controls.shape}")

            obs_t = observations[t]
            particles_t, weights_t = self.step(u_t, obs_t)

            particles_hist[t] = particles_t
            weights_hist[t] = weights_t

        return particles_hist, weights_hist

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_weights(weights: Tensor) -> Tensor:
        """Normalize weights along the particle dimension."""
        # weights: (M, N)
        sums = weights.sum(dim=1, keepdim=True)  # (M, 1)
        return weights / (sums + 1e-30)

    @staticmethod
    def _normalize_log_weights(log_weights: Tensor) -> Tensor:
        """Normalize log-weights in a numerically stable way."""
        # log_weights: (M, N)
        max_log, _ = torch.max(log_weights, dim=1, keepdim=True)  # (M, 1)
        w = torch.exp(log_weights - max_log)  # (M, N)
        sums = w.sum(dim=1, keepdim=True)  # (M, 1)
        return w / (sums + 1e-30)
