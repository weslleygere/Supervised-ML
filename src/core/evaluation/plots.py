import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


# =============================================================================
# MODEL COMPARISON
# =============================================================================


def plot_outer_fold_mae(
    outer_metrics: pd.DataFrame,
) -> go.Figure:
    """
    Compare models using MAE from the outer grouped CV folds.

    Each small point represents one outer fold. The diamond represents
    the mean MAE across folds and the horizontal error bar represents
    one standard deviation.

    Lower MAE indicates better performance.
    """
    stats = (
        outer_metrics
        .groupby("model", as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
        )
        .sort_values("MAE_mean")
        .reset_index(drop=True)
    )

    model_order = stats["model"].tolist()

    fig = go.Figure()

    for model_position, model_name in enumerate(model_order):
        fold_data = (
            outer_metrics[
                outer_metrics["model"] == model_name
            ]
            .sort_values("outer_fold")
        )

        n_folds = len(fold_data)

        offsets = np.linspace(
            -0.14,
            0.14,
            n_folds,
        )

        y_positions = (
            model_position + offsets
        )

        fig.add_trace(
            go.Scatter(
                x=fold_data["MAE"],
                y=y_positions,
                mode="markers",
                name="Outer folds",
                legendgroup="folds",
                showlegend=model_position == 0,
                customdata=fold_data[
                    ["outer_fold"]
                ].to_numpy(),
                hovertemplate=(
                    "Model: "
                    + model_name
                    + "<br>"
                    + "Outer fold: %{customdata[0]}"
                    + "<br>"
                    + "MAE: %{x:.4f}"
                    + "<extra></extra>"
                ),
            )
        )

        model_stats = stats[
            stats["model"] == model_name
        ].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=[model_stats["MAE_mean"]],
                y=[model_position],
                mode="markers",
                name="Mean ± SD",
                legendgroup="mean",
                showlegend=model_position == 0,
                marker={
                    "symbol": "diamond",
                    "size": 12,
                },
                error_x={
                    "type": "data",
                    "array": [
                        model_stats["MAE_std"]
                    ],
                    "visible": True,
                },
                hovertemplate=(
                    "Model: "
                    + model_name
                    + "<br>"
                    + "Mean MAE: %{x:.4f}"
                    + "<br>"
                    + "SD: "
                    + f"{model_stats['MAE_std']:.4f}"
                    + "<extra></extra>"
                ),
            )
        )

    feature_set = (
        outer_metrics["feature_set"].iloc[0]
        if "feature_set" in outer_metrics.columns
        else ""
    )

    title = "Outer-fold MAE"

    if feature_set:
        title += f" — {feature_set}"

    fig.update_layout(
        title=(
            f"{title}"
            "<br>"
            "<sup>"
            "Each point is one outer GroupKFold result; "
            "diamond = mean, error bar = ±1 SD"
            "</sup>"
        ),
        xaxis_title="MAE (lower is better)",
        yaxis={
            "title": "Model",
            "tickmode": "array",
            "tickvals": list(
                range(len(model_order))
            ),
            "ticktext": model_order,
            "autorange": "reversed",
        },
        template="plotly_white",
        hovermode="closest",
        height=max(
            450,
            90 * len(model_order),
        ),
    )

    return fig


# =============================================================================
# WINNING MODEL DIAGNOSTIC
# =============================================================================


def plot_oof_observed_vs_predicted(
    oof_predictions: pd.DataFrame,
    model_name: str,
    target_column: str,
    group_column: str,
    bag_column: str,
    inference_k: int,
    inference_repeats: int,
    seed: int,
) -> go.Figure:
    """
    Plot observed versus OOF-predicted HFI for the selected model.

    Individual 1-minute OOF predictions are converted into CapturePointId-level
    predictions using the same fixed-k inference strategy used during model
    evaluation.

    For each CapturePointId:
    - k recordings are randomly selected without replacement;
    - their predictions are averaged;
    - this is repeated inference_repeats times.

    The plotted prediction is the mean across repetitions and the vertical
    error bar is the standard deviation across repetitions.
    """
    model_predictions = (
        oof_predictions[
            oof_predictions["model"] == model_name
        ]
        .copy()
    )

    capture_summary = _capture_prediction_summary(
        predictions=model_predictions,
        target_column=target_column,
        group_column=group_column,
        bag_column=bag_column,
        inference_k=inference_k,
        inference_repeats=inference_repeats,
        seed=seed,
    )

    observed = capture_summary["observed"]
    predicted = capture_summary["predicted"]

    axis_min = float(
        min(
            observed.min(),
            predicted.min(),
        )
    )

    axis_max = float(
        max(
            observed.max(),
            predicted.max(),
        )
    )

    margin = 0.05 * (
        axis_max - axis_min
    )

    if margin == 0:
        margin = 0.1

    axis_min -= margin
    axis_max += margin

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=observed,
            y=predicted,
            mode="markers",
            error_y={
                "type": "data",
                "array": capture_summary[
                    "prediction_sd"
                ],
                "visible": True,
            },
            customdata=capture_summary[
                [
                    group_column,
                    bag_column,
                ]
            ].to_numpy(),
            hovertemplate=(
                f"{group_column}: "
                "%{customdata[0]}"
                "<br>"
                f"{bag_column}: "
                "%{customdata[1]}"
                "<br>"
                "Observed HFI: %{x:.4f}"
                "<br>"
                "Predicted HFI: %{y:.4f}"
                "<extra></extra>"
            ),
            name="CapturePointId",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                axis_min,
                axis_max,
            ],
            y=[
                axis_min,
                axis_max,
            ],
            mode="lines",
            name="Ideal prediction",
            line={
                "dash": "dash",
            },
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            f"{model_name} — observed vs OOF-predicted HFI"
            "<br>"
            "<sup>"
            f"k={inference_k} one-minute recordings per CapturePointId; "
            f"{inference_repeats} repeated samples"
            "</sup>"
        ),
        xaxis={
            "title": "Observed HFI",
            "range": [
                axis_min,
                axis_max,
            ],
        },
        yaxis={
            "title": "OOF-predicted HFI",
            "range": [
                axis_min,
                axis_max,
            ],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        template="plotly_white",
        hovermode="closest",
        height=650,
        width=700,
    )

    return fig


# =============================================================================
# INSTANCE-MIR AGGREGATION FOR DIAGNOSTIC PLOT
# =============================================================================


def _capture_prediction_summary(
    predictions: pd.DataFrame,
    target_column: str,
    group_column: str,
    bag_column: str,
    inference_k: int,
    inference_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """
    Summarize fixed-k OOF predictions at CapturePointId level.

    Returns one row per CapturePointId containing:
    - observed HFI;
    - mean predicted HFI across repeated fixed-k samples;
    - standard deviation of predicted HFI across repetitions.
    """
    capture_predictions: dict[
        tuple[object, object],
        list[float],
    ] = {}

    capture_observed: dict[
        tuple[object, object],
        float,
    ] = {}

    grouped = list(
        predictions.groupby(
            [
                group_column,
                bag_column,
            ],
            sort=False,
        )
    )

    for repeat in range(
        inference_repeats
    ):
        rng = np.random.default_rng(
            seed + repeat
        )

        for (
            point,
            capture,
        ), capture_df in grouped:

            selected_positions = rng.choice(
                len(capture_df),
                size=inference_k,
                replace=False,
            )

            selected = capture_df.iloc[
                selected_positions
            ]

            key = (
                point,
                capture,
            )

            capture_predictions.setdefault(
                key,
                [],
            ).append(
                float(
                    selected[
                        "prediction"
                    ].mean()
                )
            )

            capture_observed[key] = float(
                selected[
                    target_column
                ].mean()
            )

    rows = []

    for (
        point,
        capture,
    ), values in capture_predictions.items():

        rows.append(
            {
                group_column: point,
                bag_column: capture,
                "observed": (
                    capture_observed[
                        (
                            point,
                            capture,
                        )
                    ]
                ),
                "predicted": float(
                    np.mean(values)
                ),
                "prediction_sd": (
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if len(values) > 1
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SAVE EXPERIMENT FIGURES
# =============================================================================


def save_evaluation_plots(
    outer_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    output_dir: str | Path,
    target_column: str,
    group_column: str,
    bag_column: str,
    inference_k: int,
    inference_repeats: int,
    seed: int,
) -> None:
    """
    Generate and save the two figures used in the first instance-MIR experiment.
    """
    figures_dir = (
        Path(output_dir)
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_fig = (
        plot_outer_fold_mae(
            outer_metrics
        )
    )

    winner = (
        summary
        .sort_values(
            "OOF_MAE"
        )
        .iloc[0]["model"]
    )

    diagnostic_fig = (
        plot_oof_observed_vs_predicted(
            oof_predictions=oof_predictions,
            model_name=winner,
            target_column=target_column,
            group_column=group_column,
            bag_column=bag_column,
            inference_k=inference_k,
            inference_repeats=inference_repeats,
            seed=seed,
        )
    )

    _save_figure(
        comparison_fig,
        figures_dir
        / "outer_fold_mae",
    )

    _save_figure(
        diagnostic_fig,
        figures_dir
        / "winner_observed_vs_predicted",
    )

    logger.info(
        "Evaluation figures saved to %s",
        figures_dir,
    )


def _save_figure(
    figure: go.Figure,
    output_path: Path,
) -> None:
    """
    Save a Plotly figure as HTML and, when available, PNG.
    """
    figure.write_html(
        output_path.with_suffix(
            ".html"
        )
    )

    try:
        figure.write_image(
            output_path.with_suffix(
                ".png"
            ),
            scale=2,
        )

    except Exception as exc:
        logger.warning(
            "Could not save PNG for %s: %s",
            output_path.name,
            exc,
        )
