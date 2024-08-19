from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif"}  # require LaTeX and type1cm on Ubuntu
)


def config_plot():
    """Function to remove axis tickers and box around figure."""
    plt.box(False)
    plt.axis("off")
    plt.tight_layout()


def plot_point_clouds(
    fp: str,
    points: np.ndarray,
    seed_points: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Visualize points with their labels as different colors."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        points[:, 0], points[:, 1], s=5, c=colors if colors is not None else "aqua"
    )
    if seed_points is not None:
        ax.scatter(seed_points[:, 0], seed_points[:, 1], s=10, c="black")
    ax.invert_yaxis()
    config_plot()
    plt.savefig(fp, dpi=300)
    plt.close()


def plot_explained_variance(fp: str, expalined_variance_ratio: np.ndarray) -> None:
    cumulative_variance = expalined_variance_ratio.cumsum()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        np.arange(1, cumulative_variance.shape[0] + 1), cumulative_variance, linewidth=1
    )
    ax.set_ylim(top=1.0)
    ax.margins(x=0, y=0)
    ax.grid(linestyle="--")

    ax.set_title("Cumulative Relative Variance")
    ax.set_xlabel("\# of Principal Components")
    ax.set_ylabel("Cumulative Variance Ratio")

    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()
