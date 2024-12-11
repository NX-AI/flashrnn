import numpy as np
from matplotlib import pyplot as plt

from .ewma import ewma_vectorized


def compute_errors_per_batchhead(
    baseline: np.ndarray,  # (B, NH, S, ...)
    target: np.ndarray,  # (B, NH, S, ...)
) -> np.ndarray:  # (B * NH, S, F) F are flattened features
    # compute the difference in float64 to avoid numerical issues
    error = np.abs(baseline.astype(np.float64) - target.astype(np.float64))
    all_timesteps_np = error

    B, NH, S = error.shape[:3]
    # reshape to target shape
    # flatten B, NH
    all_timesteps_np = all_timesteps_np.reshape(-1, *all_timesteps_np.shape[2:])

    # flatten features
    all_timesteps_np = all_timesteps_np.reshape(B * NH, S, -1)

    return all_timesteps_np


def plot_error_statistics_over_time_single(
    errors: np.ndarray,  # shape: (num_timesteps, num_features)
    percentiles: list = [50, 90, 100],
    title: str = "",
    add_mean: bool = False,
    ema_alpha: float = 0.02,
    figsize=(10, 6),
):
    assert len(errors.shape) == 2, "errors must have shape (num_timesteps, num_features)"
    # title = f"{title}--ema{ema_alpha}"

    # compute percentiles
    percentiles_sequence_data = np.percentile(errors, percentiles, axis=-1)

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, p in enumerate(percentiles):
        ema_percentile_data = ewma_vectorized(percentiles_sequence_data[i], alpha=ema_alpha)

        ax.plot(ema_percentile_data, label=f"{percentiles[i]}th percentile")

    if add_mean:
        ema_mean_data = ewma_vectorized(np.mean(errors, axis=-1), alpha=ema_alpha)
        ax.plot(ema_mean_data, label="mean")

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Absolute Error")
    ax.legend()

    ax.grid(alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_error_statistics_over_time_per_batchhead(
    errors: np.ndarray,  # shape: (num_batchheads, num_timesteps, num_features)
    percentiles: list = [50, 90, 100],
    title: str = "",
    add_mean: bool = False,
    ema_alpha: float = 0.02,
    max_num_batchhead_plots: int = -1,  # -1 means all
    figsize=(10, 6),
):
    num_batchheads = errors.shape[0]
    if max_num_batchhead_plots > 0:
        max_num_batchhead_plots = min(num_batchheads, max_num_batchhead_plots)
    else:
        max_num_batchhead_plots = num_batchheads

    figs = []
    for i in range(max_num_batchhead_plots):
        # title_i = f"BH({i}):{title}"
        title_i = f"{title}"
        fig = plot_error_statistics_over_time_single(
            errors[i],
            percentiles,
            title=title_i,
            add_mean=add_mean,
            ema_alpha=ema_alpha,
            figsize=figsize,
        )
        figs.append(fig)

    return figs
