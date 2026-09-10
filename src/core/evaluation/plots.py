import logging
from pathlib import Path

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
    Plot outer-fold MAE for each model.

    Individual points represent outer folds.
    Diamonds represent mean MAE with ±1 standard deviation.
    """

    order = (
        outer_metrics
        .groupby("model")["MAE"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    fig = go.Figure()

    for model in order:
        data = outer_metrics[
            outer_metrics["model"] == model
        ]

        fig.add_trace(
            go.Scatter(
                x=[model] * len(data),
                y=data["MAE"],
                mode="markers",
                name=model,
                showlegend=False,
                marker={
                    "size": 8,
                    "opacity": 0.65,
                },
                hovertemplate=(
                    f"{model}"
                    "<br>Outer fold: %{customdata}"
                    "<br>MAE: %{y:.3f}"
                    "<extra></extra>"
                ),
                customdata=data[
                    "outer_fold"
                ],
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[model],
                y=[data["MAE"].mean()],
                mode="markers",
                showlegend=False,
                marker={
                    "size": 12,
                    "symbol": "diamond",
                },
                error_y={
                    "type": "data",
                    "array": [
                        data["MAE"].std()
                    ],
                    "visible": True,
                },
                hovertemplate=(
                    f"{model}"
                    "<br>Mean MAE: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            "Model comparison"
            "<br>"
            "<sup>"
            "Outer GroupKFold performance "
            "on unseen Points"
            "</sup>"
        ),
        xaxis_title="Model",
        yaxis_title="MAE (HFI units)",
        template="plotly_white",
        height=550,
    )

    return fig


# =============================================================================
# OBSERVED VS PREDICTED
# =============================================================================


def plot_oof_observed_vs_predicted(
    summary: pd.DataFrame,
    oof_predictions: pd.DataFrame,
) -> go.Figure:
    """
    Plot pooled outer OOF predictions for the best model.

    The winner is selected using the lowest OOF MAE.
    """

    winner = summary.iloc[0]

    model_name = winner["model"]

    data = oof_predictions[
        oof_predictions["model"] == model_name
    ].copy()

    observed = data["meanHFI"]
    predicted = data["prediction"]

    lower = min(
        observed.min(),
        predicted.min(),
    )

    upper = max(
        observed.max(),
        predicted.max(),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=observed,
            y=predicted,
            mode="markers",
            marker={
                "size": 9,
                "opacity": 0.75,
            },
            customdata=data[
                [
                    "Point",
                    "CapturePointId",
                ]
            ],
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>"
                "CapturePointId: %{customdata[1]}"
                "<br>"
                "Observed HFI: %{x:.3f}"
                "<br>"
                "Predicted HFI: %{y:.3f}"
                "<extra></extra>"
            ),
            name="CapturePointId",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line={
                "dash": "dash",
            },
            name="Perfect prediction",
        )
    )

    fig.update_layout(
        title=(
            f"{model_name} — observed vs predicted HFI"
            "<br>"
            "<sup>"
            f"OOF MAE = {winner['OOF_MAE']:.3f} | "
            f"RMSE = {winner['OOF_RMSE']:.3f} | "
            f"R² = {winner['OOF_R2']:.3f}"
            "</sup>"
        ),
        xaxis_title="Observed HFI",
        yaxis_title="Predicted HFI",
        template="plotly_white",
        height=600,
        width=700,
    )

    fig.update_xaxes(
        range=[lower, upper]
    )

    fig.update_yaxes(
        range=[lower, upper],
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# =============================================================================
# SAVE FIGURES
# =============================================================================


def save_evaluation_plots(
    outer_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Create and save the main evaluation figures.
    """

    figures_dir = (
        Path(output_dir)
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures = {
        "outer_fold_mae": (
            plot_outer_fold_mae(
                outer_metrics
            )
        ),
        "winner_observed_vs_predicted": (
            plot_oof_observed_vs_predicted(
                summary,
                oof_predictions,
            )
        ),
    }

    for name, fig in figures.items():
        _save_figure(
            fig,
            figures_dir / name,
        )


# =============================================================================
# FIGURE OUTPUT
# =============================================================================


def _save_figure(
    fig: go.Figure,
    path: Path,
) -> None:
    """
    Save a Plotly figure as HTML and, when available, PNG.
    """

    fig.write_html(
        path.with_suffix(".html"),
        include_plotlyjs="directory",
    )

    try:
        fig.write_image(
            path.with_suffix(".png"),
            scale=2,
        )

    except Exception as exc:
        logger.warning(
            "Could not save PNG '%s': %s",
            path,
            exc,
        )
