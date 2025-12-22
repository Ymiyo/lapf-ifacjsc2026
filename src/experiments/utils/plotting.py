import matplotlib.pyplot as plt
import numpy as np


def plot_hist(
    fig: plt.Figure,
    MSE_results: list[dict],
    width: float = 0.2,
) -> None:
    ax = fig.add_subplot(1, 1, 1)

    x_dim = len(MSE_results[0]["mean_MSE"])
    x_indices = np.arange(1, x_dim + 1)

    method_num = len(MSE_results)
    mid = (method_num - 1) * width / 2

    for i, d in enumerate(MSE_results):
        label = d["method"]
        mean_MSE = d["mean_MSE"]
        std_MSE = d["std_MSE"]
        color = d["color"]

        ax.bar(
            x_indices + i * width - mid,
            mean_MSE,
            yerr=std_MSE,
            width=width,
            label=label,
            capsize=4,
            alpha=0.55,
            color=color,
        )

    ax.set_xlabel("Location")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.tight_layout()
