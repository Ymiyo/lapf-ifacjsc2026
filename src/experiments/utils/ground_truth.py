# src/experiments/utils/simulation.py

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from ...lapf_project.data.text_templates import TEST_DICT


class PlantSystem:
    """True linear time-invariant (LTI) system used to generate trajectories."""

    def __init__(
        self,
        A: Tensor,
        B: Tensor,
    ) -> None:
        """
        Parameters
        ----------
        A : Tensor, shape (n, n)
            State transition matrix.
        B : Tensor, shape (n, k)
            Input matrix.
        """
        # Do not move to a specific device here.
        # Assume A and B are already on the desired device.
        self.A = A
        self.B = B

    def get_next_state(
        self,
        xb: Tensor,
        control: Tensor,
        wb: Tensor,
        x_min: float,
        x_max: float,
    ) -> Tensor:
        """
        State transition equation:

            x_{k+1} = proj_{[x_min, x_max]}(A x_k + B u_k + w_k)

        Parameters
        ----------
        xb : Tensor, shape (M, n)
            State vectors (batched).
        control : Tensor
            Control input vector. Supported shape: (k,) or (M, k)
        wb : Tensor, shape (M, n)
            Process noise.
        x_min : float
            Minimum value for x.
        x_max : float
            Maximum value for x.

        Returns
        -------
        Tensor, shape (M, n)
            Next state vectors (batched), clamped to [0, x_max].
        """
        M, n = xb.shape
        device = self.A.device

        Ax = torch.einsum("nm,Mm->Mn", self.A, xb)  # (M, n)

        if control is None:
            Bu = torch.zeros((M, n), device=device)  # (M, n)
        else:
            control = control.to(device)
            if self.B is None:
                raise ValueError("Control input provided but B matrix is None.")
            if control.dim() == 1:
                # (k,)
                Bu_single = torch.matmul(self.B, control)  # (n,)
                Bu = Bu_single.view(1, n).expand(M, n)  # (M, n)
            elif control.dim() == 2 and control.shape[0] == M:
                # (M, k)
                # (M, k) @ (k, n) -> (M, n)
                Bu = torch.matmul(control, self.B.T)  # (M, n)
            else:
                raise ValueError(
                    "Unsupported control shape. Expected None, (k,) or (M, k), "
                    f"but got {tuple(control.shape)}."
                )

        xb_next = Ax + Bu + wb  # (M, n)

        if x_min is None or x_max is None:
            raise ValueError("x_min and x_max must be provided for state clamping.")

        xb_next = torch.clamp(xb_next, x_min, x_max)
        return xb_next


class HumanSensor:
    """Pseudo human sensor consisting of cognitive and expression modules."""

    def __init__(self, C_human: Tensor) -> None:
        """
        Parameters
        ----------
        C_human : Tensor, shape (p_human, n)
            Observation matrix for the human sensor.
        """
        self.C_human = C_human

    def cognitive_module(
        self,
        xb: Tensor,
        vb: Tensor,
        y_min: float,
        y_max: float,
    ) -> Tensor:
        """
        Continuous internal representation:

            y_human = C_human x + v

        Parameters
        ----------
        xb : Tensor, shape (M, n)
            State vectors (batched).
        vb : Tensor, shape (M, p_human)
            Observation noise.
        y_min : float
            Minimum value for y.
        y_max : float
            Maximum value for y.

        Returns
        -------
        Tensor, shape (M, p_human)
            Cognitive values, clamped to [y_min, y_max].
        """
        yb = torch.einsum("pm,Mm->Mp", self.C_human, xb) + vb  # (M, p_human)
        yb = torch.clamp(yb, y_min, y_max)
        return yb

    def expression_module(
        self,
        yb: Tensor,  # (T, M, p_human)
        test_dict: Dict[int, List[str]],
        y_min: float,
        y_max: float,
    ) -> List[str]:
        """
        Map cognitive values to text expressions.

        Parameters
        ----------
        yb : Tensor, shape (T, M, p_human)
            Cognitive values (batched in time).
        test_dict : dict[int, list[str]]
            Mapping from percentage (0, 2, ..., 100) to a list of text templates.
        y_min : float
            Minimum value for y.
        y_max : float
            Maximum value for y.

        Returns
        -------
        text_list : list of str
            Flattened list of texts, length = T * M * p_human.
        """
        # Convert to percentage in {0, 2, 4, ..., 100}
        yb_moved = yb - y_min
        yb_percent = (yb_moved * 50.0 / (y_max - y_min)).to(int) * 2  # (T, M, p_human)
        text_list = [
            random.choice(test_dict[int(v)]) for v in yb_percent.flatten().tolist()
        ]
        return text_list


def simulate_plant_and_human(
    A: Tensor = None,
    B: Optional[Tensor] = None,
    control: Optional[Tensor] = None,
    Q: Tensor = None,
    mu_w: Optional[Tensor] = None,
    C_human: Tensor = None,
    R_human: Tensor = None,
    mu_v_human: Optional[Tensor] = None,
    M: int = None,
    T: int = None,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    init_x: Optional[float] = None,
    seed_system: int = 42,
) -> Tuple[Tensor, Tensor]:
    """
    Simulate the true plant and human sensor trajectories.

    Parameters
    ----------
    A, B, control, Q, mu_w, C_human, R_human, mu_v_human : Tensors
        System and noise matrices (already on the desired device).
    M : int
        Batch size (number of Monte Carlo runs).
    T : int
        Number of time steps.
    x_min : float
        Minimum value for state x.
    x_max : float
        Maximum value for state x.
    y_min : float
        Minimum value for cognitive output y.
    y_max : float
        Maximum value for cognitive output y.
    init_x : Optional[float]
        Initial state value for all entries. If None, sampled uniformly from [x_min, x_max].
    seed_system : int
        Random seed for system and sensor noise.

    Returns
    -------
    X_true : Tensor, shape (T, M, n)
        True state trajectories (on A.device).
    text_list : list of str
        Generated text from the human sensor.
    """
    if A is None or Q is None:
        raise ValueError("A and Q matrices must be provided.")

    device = A.device  # use the device of system matrices as the source of truth

    n = A.shape[0]
    p_human = C_human.shape[0]

    torch.manual_seed(seed_system)

    if mu_w is None:
        mu_w = torch.zeros(n, device=device)
    W = MultivariateNormal(mu_w.to(device), Q.to(device)).sample((T, M))  # (T, M, n)

    if mu_v_human is None:
        mu_v_human = torch.zeros(p_human, device=device)
    V_human = MultivariateNormal(mu_v_human.to(device), R_human.to(device)).sample(
        (T, M)
    )  # (T, M, p_human)

    plant = PlantSystem(A, B)
    sensor = HumanSensor(C_human)

    X_true = torch.zeros(T, M, n, device=device)
    Y_human = torch.zeros(T, M, p_human, device=device)

    # Initial state x_0
    if init_x is None:
        init_x = x_min + (x_max - x_min) * torch.rand(1).item()
    X_true[0, :, :] = torch.full((M, n), init_x, device=device)

    for t in range(1, T):
        X_true[t, :, :] = plant.get_next_state(
            X_true[t - 1, :, :],
            control,
            W[t - 1, :, :],
            x_min,
            x_max,
        )
        Y_human[t, :, :] = sensor.cognitive_module(
            X_true[t, :, :],
            V_human[t, :, :],
            y_min,
            y_max,
        )

    text_list = sensor.expression_module(
        Y_human,
        TEST_DICT,
        y_min,
        y_max,
    )

    return X_true, text_list
