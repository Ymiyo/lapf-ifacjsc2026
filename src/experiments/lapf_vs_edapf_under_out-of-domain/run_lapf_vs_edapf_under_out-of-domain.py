# src/experiments/lapf_vs_edapf_under_out-of-domain/run_lapf_vs_edapf_under_out-of-domain.py
# cd lapf-ifacjsc2026
# python -m src.experiments.lapf_vs_edapf_under_out-of-domain.run_lapf_vs_edapf_under_out-of-domain

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import inference_mode

from ...lapf_project.filters.lapf_for_clamped_linear_gaussian import (
    ClampedLinearGaussianLAPFConfig,
)
from ...lapf_project.models.edapf_nn import (
    compute_mse_on_val_set,
    load_prediction_model,
)
from ...lapf_project.models.lapf_nn import load_classifier_model
from ..utils.ground_truth import simulate_plant_and_human_under_out_of_domain
from ..utils.plotting import plot_hist
from ..utils.run_filters import run_edapf, run_lapf, run_no_observation
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

    out_of_domain_text = (
        "ミジ、ナランサー！"  # "There's barely any water out here!" in Okinawa dialect
    )

    # Experiment settings
    M = 1000  # batch size
    T = 100  # time steps
    num_particles = 1000

    print("1. System parameters set.")

    # -------------------------------
    # 2. Simulate true system
    # -------------------------------
    X_true, text_list = simulate_plant_and_human_under_out_of_domain(
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
        out_of_domain_threshold=0.2,
        out_of_domain_text=out_of_domain_text,
    )

    print("2. True system simulated.")

    # -------------------------------
    # 3. Compute the probability distribution p(q | s) for LAPF and the predicted cognitive values for EDAPF
    # -------------------------------
    num_labels = 5

    classifier_model = load_classifier_model(
        device=device,
        num_labels=num_labels,
    )

    with inference_mode():
        outputs = classifier_model(text_list)  # Tensor, shape (T*M*p_human, num_labels)

        out_of_domain_outputs = classifier_model([out_of_domain_text])
        print("Example classifier output for out-of-domain text:")
        print(out_of_domain_outputs)

    probs = outputs.reshape(T, M, p_human, num_labels).to(device)
    # probs[t, m, p, :] = p(q | s) for human sensor at time t for batch m

    print("3-1. LAPF: Compute the probability distribution p(q | s) done.")

    prediction_model = load_prediction_model(
        device=device,
        batch_size=16,
        num_epochs=100,
        lr=1e-5,
    )

    with inference_mode():
        outputs = prediction_model(text_list)  # Tensor, shape (T*M*p_human)

        out_of_domain_outputs = prediction_model([out_of_domain_text])
        print("Example prediction output for out-of-domain text:")
        print(out_of_domain_outputs)

    Y_human_tilde = (
        outputs.reshape(T, M, p_human).to(device) * y_max / 100
    )  # Tensor, shape (T, M, p_human)

    mse_on_val_set = compute_mse_on_val_set(
        model=prediction_model,
        device=device,
        batch_size=16,
    )
    r_tilde = mse_on_val_set * (y_max / 100) ** 2
    R_tilde = r_tilde * torch.eye(p_human, device=device)

    print(f"     r_tilde: {r_tilde:.4f}")
    print("3-2. EDAPF: Compute the predicted cognitive values done.")

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

    particles_hist_EDAPF, weights_hist_EDAPF = run_edapf(
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
        Y_human_tilde=Y_human_tilde,
        R_tilde=R_tilde,
    )

    historys.append((particles_hist_EDAPF, weights_hist_EDAPF))

    print("4-1. EDAPF filtering done.")

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

    print("4-2. LAPF filtering done.")

    # -------------------------------
    # 5. Compute mean and std of MSE by state
    # -------------------------------
    MSE_results = [
        {
            "method": "EDAPF",
            "color": "red",
        },
        {
            "method": "LAPF",
            "color": "blue",
        },
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
    fig = plt.figure(figsize=(5, 3))

    plot_hist(
        fig=fig,
        MSE_results=MSE_results,
        width=0.2,
    )

    plt.show()

    # Save figure next to checkpoints for reproducibility
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "lapf_vs_edapf_under_out-of-domain"
    result_dir.mkdir(parents=True, exist_ok=True)

    fig_path = result_dir / "lapf_vs_edapf_under_out-of-domain_mse_hist.png"
    fig.savefig(fig_path)
    print(f"Saved MSE histogram to: {fig_path}")


if __name__ == "__main__":
    main()
