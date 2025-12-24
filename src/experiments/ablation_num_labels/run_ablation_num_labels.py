# src/experiments/ablation_num_labels/run_ablation_num_labels.py
# cd lapf-ifacjsc2026
# python -m src.experiments.ablation_num_labels.run_ablation_num_labels

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import inference_mode

from ...lapf_project.filters.lapf_for_clamped_linear_gaussian import (
    ClampedLinearGaussianLAPFConfig,
)
from ...lapf_project.models.lapf_nn import load_classifier_model
from ..utils.ground_truth import simulate_plant_and_human
from ..utils.plotting import plot_hist
from ..utils.run_filters import run_lapf
from ..utils.stats import (
    compute_mean_and_covariance,
    compute_mean_and_std_of_mse_by_state,
    compute_overall_mean_and_std_of_mse,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # -------------------------------
    # 1. System parameters
    # -------------------------------
    A = torch.tensor(
        [
            [0.4, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.6, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    C_human = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )

    n = A.shape[0]
    p_human = C_human.shape[0]

    Q = torch.diag(torch.tensor([1.0, 0.1, 0.1, 0.1, 0.1], device=device))
    mu_w = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    var_human = 1.0
    R_human = torch.diag(torch.full((p_human,), var_human, device=device))

    x_min, x_max = 0.0, 5.0
    y_min, y_max = 0.0, 5.0
    init_x = 2.5

    # Experiment settings
    M = 1000  # batch size
    T = 100  # time steps
    num_particles = 1000

    print("1. System parameters set.")

    # -------------------------------
    # 2. Simulate true system
    # -------------------------------
    X_true, text_list = simulate_plant_and_human(
        A=A,
        Q=Q,
        mu_w=mu_w,
        C_human=C_human,
        R_human=R_human,
        M=M,
        T=T,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        init_x=init_x,
        seed_system=42,
    )

    print("2. True system simulated.")

    # -------------------------------
    # 3. Compute the probability distribution p(q | s) for LAPF and the predicted cognitive values for EDAPF
    # -------------------------------
    probs_list = []
    for i, num_labels in enumerate(range(2, 11)):  # 2 to 10 labels
        classifier_model = load_classifier_model(
            device=device,
            num_labels=num_labels,
        )

        with inference_mode():
            outputs = classifier_model(
                text_list
            )  # Tensor, shape (T*M*p_human, num_labels)

        probs = outputs.reshape(T, M, p_human, num_labels).to(device)
        # probs[t, m, p, :] = p(q | s) for human sensor at time t for batch m
        probs_list.append(probs)

        classifier_model = None  # free memory

        print(
            f"3-{i+1}. LAPF: Compute the probability distribution p(q | s) done for num_labels={num_labels}."
        )

    # -------------------------------
    # 4. Define and run LAPF and EDAPF
    # -------------------------------
    init_mu = X_true[0, 0, :].detach()  # (n,)
    init_P = torch.eye(n, device=device)  # (n, n)

    config = ClampedLinearGaussianLAPFConfig(
        num_particles=num_particles,
        resample_threshold=None,  # resample every step (original style)
        device=device,
        state_clamp_min=x_min,
        state_clamp_max=x_max,
        observation_clamp_min=y_min,
        observation_clamp_max=y_max,
    )

    historys = []

    for i, probs in enumerate(probs_list):
        particles_hist_LAPF, weights_hist_LAPF = run_lapf(
            config=config,
            T=T,
            M=M,
            A=A,
            Q=Q,
            mu_w=mu_w,
            state_prior_mean=init_mu,
            state_prior_cov=init_P,
            C_human=C_human,
            R_human=R_human,
            Y_phys=None,
            probs=probs,
        )

        historys.append((particles_hist_LAPF, weights_hist_LAPF))

        print(f"4-{i+1}. LAPF filtering done for num_labels={2 + i}.")

    # -------------------------------
    # 5. Compute mean and std of MSE by state
    # -------------------------------
    MSE_results = [
        {
            "method": f"m={i}",
            "color": plt.get_cmap("tab10").colors[i - 2],
        }
        for i in range(2, 11)
    ]

    for method_idx, (particles_hist, weights_hist) in enumerate(historys):
        M_hat = torch.zeros(T, M, n, device=device)
        P_hat = torch.zeros(T, M, n, n, device=device)

        for t in range(T):
            particles_t = particles_hist[t]  # (M, N, n)
            weights_t = weights_hist[t]  # (M, N)
            mean_t, cov_t = compute_mean_and_covariance(particles_t, weights_t)
            M_hat[t, :, :] = mean_t
            P_hat[t, :, :, :] = cov_t

        mean_MSE, std_MSE = compute_mean_and_std_of_mse_by_state(
            X_true=X_true,
            X_est=M_hat,
        )

        MSE_results[method_idx]["mean_MSE"] = mean_MSE
        MSE_results[method_idx]["std_MSE"] = std_MSE

        mean_MSE_overall, std_MSE_overall = compute_overall_mean_and_std_of_mse(
            X_true=X_true,
            X_est=M_hat,
        )
        print(
            f"{MSE_results[method_idx]['method']} Overall MSE: {mean_MSE_overall:.4f} ± {std_MSE_overall:.4f}"
        )

    print("5. Compute mean and std of MSE by state done.")

    # -------------------------------
    # 6. Draw histogram of MSE by state
    # -------------------------------
    fig = plt.figure(figsize=(8, 3))

    plot_hist(
        fig=fig,
        MSE_results=MSE_results,
        width=0.1,
    )

    plt.show()

    # Save figure next to checkpoints for reproducibility
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "ablation_num_labels"
    result_dir.mkdir(parents=True, exist_ok=True)

    fig_path = result_dir / "ablation_num_labels_mse_hist.png"
    fig.savefig(fig_path)
    print(f"Saved MSE histogram to: {fig_path}")


if __name__ == "__main__":
    main()
