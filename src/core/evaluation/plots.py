"""Publication-ready effort curves from held-out selected-pipeline predictions."""
import numpy as np
import pandas as pd

from .metrics import metric_table, point_loss_matrix
from .persistence import write_frame, write_json


def save_effort_analysis(predictions, schema, settings, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = metric_table(predictions, schema)
    write_frame(output_dir / "metrics_by_repeat.csv", metrics)
    selected = predictions[predictions.evaluation == "selected"]
    matrix = point_loss_matrix(selected, schema)
    efforts = matrix.columns.to_numpy(int)
    losses = matrix.to_numpy()
    means = losses.mean(axis=0)
    rng = np.random.default_rng(settings.sampling_seed + 90000)
    bootstrap = np.stack([losses[rng.integers(0, len(losses), len(losses))].mean(axis=0) for _ in range(settings.bootstrap_repeats)]) if settings.bootstrap_repeats else None
    tail = (1 - settings.confidence_level) / 2
    lower, upper = np.quantile(bootstrap, [tail, 1 - tail], axis=0) if bootstrap is not None else (np.full(len(efforts), np.nan), np.full(len(efforts), np.nan))
    summary = metrics[metrics.evaluation == "selected"].groupby("n_recordings").mean(numeric_only=True).drop(columns="repeat").reset_index()
    summary["MAE"] = means
    summary["MAE_ci_low"] = lower
    summary["MAE_ci_high"] = upper
    write_frame(output_dir / "effort_summary.csv", summary)
    increments = pd.DataFrame({"from_count": efforts[:-1], "to_count": efforts[1:], "added_recordings": np.diff(efforts), "MAE_improvement": means[:-1] - means[1:]})
    if bootstrap is not None:
        delta = bootstrap[:, :-1] - bootstrap[:, 1:]
        increments["ci_low"], increments["ci_high"] = np.quantile(delta, [tail, 1-tail], axis=0)
    write_frame(output_dir / "incremental_improvement.csv", increments)
    plateau = None
    if settings.plateau_tolerance > 0 and bootstrap is not None:
        horizon = settings.plateau_window
        for i in range(len(efforts) - horizon):
            gains = bootstrap[:, i, None] - bootstrap[:, i + 1:i + horizon + 1]
            if np.all(np.quantile(gains, settings.confidence_level, axis=0) < settings.plateau_tolerance):
                plateau = int(efforts[i])
                break
    write_json(output_dir / "plateau.json", {
        "candidate_count": plateau, "tolerance_hfi": settings.plateau_tolerance,
        "forward_evaluated_counts": settings.plateau_window,
        "interpretation": "Exploratory conditional diagnostic over the next evaluated counts, not a guarantee of a permanent plateau.",
        "uncertainty": "Point-cluster bootstrap of fixed OOF predictions, conditional on fitted models and sampled submissions; no retraining uncertainty or simultaneous coverage guarantee.",
    })
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].plot(efforts, means, marker="o", label="Selected procedure")
    if bootstrap is not None:
        axes[0].fill_between(efforts, lower, upper, alpha=0.2, label=f"{settings.confidence_level:.0%} conditional interval")
    axes[0].set(xlabel="Aggregated Audio_Name samples", ylabel="Point-balanced MAE (HFI)", title="Recording effort on held-out Points")
    axes[0].legend()
    axes[1].plot(increments.to_count, increments.MAE_improvement, marker="o")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Recording count", ylabel="MAE gain from preceding evaluated count")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "effort_curve.png", dpi=200)
    fig.savefig(output_dir / "effort_curve.pdf")
    plt.close(fig)
