from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API
# =============================================================================


def save_evaluation_plots(output_dir: str | Path) -> None:
    """Create and save the main nested-CV diagnostic figures."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outer = pd.read_csv(output_dir / "outer_fold_results.csv")
    inner = pd.read_csv(output_dir / "inner_candidate_results.csv")
    oof = pd.read_csv(output_dir / "oof_predictions.csv")
    repeats = pd.read_csv(output_dir / "repeat_metrics.csv")
    final_candidates = pd.read_csv(
        output_dir / "final_candidate_results.csv"
    )

    with open(
        output_dir / "final_selection.json",
        "r",
        encoding="utf-8",
    ) as file:
        final_selection = json.load(file)

    figures = [
        (
            "01_performance_vs_baselines",
            plot_performance_vs_baselines,
            (repeats,),
        ),
        (
            "02_observed_vs_predicted",
            plot_observed_vs_predicted,
            (oof,),
        ),
        (
            "03_residuals_vs_hfi",
            plot_residuals_vs_hfi,
            (oof,),
        ),
        (
            "04_outer_cv_stability",
            plot_outer_cv_stability,
            (outer,),
        ),
        (
            "05_inner_model_competition",
            plot_inner_model_competition,
            (inner,),
        ),
        (
            "06_selection_frequency",
            plot_selection_frequency,
            (outer,),
        ),
        (
            "07_inner_vs_outer_mae",
            plot_inner_vs_outer_mae,
            (outer,),
        ),
        (
            "08_final_model_selection",
            plot_final_model_selection,
            (
                final_candidates,
                final_selection,
            ),
        ),
    ]

    for name, builder, args in figures:
        try:
            _save_figure(
                builder(*args),
                figures_dir / name,
            )
        except Exception as exc:
            logger.warning(
                "Could not create figure '%s': %s",
                name,
                exc,
                exc_info=True,
            )


# =============================================================================
# 1. PERFORMANCE VS BASELINES
# =============================================================================


def plot_performance_vs_baselines(
    repeat_metrics: pd.DataFrame,
) -> go.Figure:
    """Compare the selected pipeline with mean and median baselines."""
    data = _repeat_mae_long(repeat_metrics)

    order = [
        "Selected pipeline",
        "Mean baseline",
        "Median baseline",
    ]

    fig = go.Figure()

    for method in order:
        subset = data[data["method"] == method]

        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=[method] * len(subset),
                y=subset["mae"],
                mode="markers",
                marker={"size": 9, "opacity": 0.7},
                customdata=subset[["repeat"]].to_numpy(),
                hovertemplate=(
                    f"{method}"
                    "<br>Outer repeat: %{customdata[0]}"
                    "<br>MAE: %{y:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        sd = (
            float(subset["mae"].std(ddof=1))
            if len(subset) > 1
            else 0.0
        )

        fig.add_trace(
            go.Scatter(
                x=[method],
                y=[subset["mae"].mean()],
                mode="markers",
                marker={"size": 13, "symbol": "diamond"},
                error_y={
                    "type": "data",
                    "array": [sd],
                    "visible": True,
                },
                hovertemplate=(
                    f"{method}"
                    "<br>Mean MAE: %{y:.3f}"
                    f"<br>SD: {sd:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=(
            "Predictive performance versus simple baselines"
            "<br><sup>"
            "Each point is one complete outer-CV repeat; "
            "diamonds show mean ± 1 SD. Lower MAE is better."
            "</sup>"
        ),
        xaxis_title="Prediction strategy",
        yaxis_title="MAE (HFI units)",
        template="plotly_white",
        height=560,
    )

    return fig


# =============================================================================
# 2. OBSERVED VS PREDICTED
# =============================================================================


def plot_observed_vs_predicted(
    oof_predictions: pd.DataFrame,
) -> go.Figure:
    """Plot observed HFI against repeated outer-CV predictions."""
    data = _average_oof_predictions(oof_predictions)
    lower, upper = _shared_limits(
        data["observed"],
        data["prediction_mean"],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["observed"],
            y=data["prediction_mean"],
            mode="markers",
            marker={"size": 9, "opacity": 0.75},
            customdata=data[
                ["group", "bag", "prediction_sd", "n_predictions"]
            ].to_numpy(),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Mean predicted HFI: %{y:.3f}"
                "<br>Prediction SD across repeats: %{customdata[2]:.3f}"
                "<br>Outer predictions: %{customdata[3]}"
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
            line={"dash": "dash"},
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            "Observed versus predicted HFI"
            "<br><sup>"
            "Each marker is one CapturePointId. For visualization only, "
            "predictions are averaged across outer repeats."
            "</sup>"
        ),
        xaxis_title="Observed HFI",
        yaxis_title="Predicted HFI",
        template="plotly_white",
        height=650,
        width=760,
    )

    fig.update_xaxes(range=[lower, upper])
    fig.update_yaxes(
        range=[lower, upper],
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# =============================================================================
# 3. RESIDUALS
# =============================================================================


def plot_residuals_vs_hfi(
    oof_predictions: pd.DataFrame,
) -> go.Figure:
    """Plot prediction residuals across the observed HFI gradient."""
    data = _average_oof_predictions(oof_predictions)
    data["residual"] = (
        data["prediction_mean"]
        - data["observed"]
    )

    fig = go.Figure(
        go.Scatter(
            x=data["observed"],
            y=data["residual"],
            mode="markers",
            marker={"size": 9, "opacity": 0.75},
            customdata=data[
                ["group", "bag", "prediction_mean", "prediction_sd"]
            ].to_numpy(),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Mean predicted HFI: %{customdata[2]:.3f}"
                "<br>Residual: %{y:.3f}"
                "<br>Prediction SD across repeats: %{customdata[3]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dash")

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.98,
        text="Positive = overprediction",
        showarrow=False,
        xanchor="right",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.02,
        text="Negative = underprediction",
        showarrow=False,
        xanchor="right",
    )

    fig.update_layout(
        title=(
            "Prediction residuals across the HFI gradient"
            "<br><sup>"
            "Residual = predicted − observed. Patterns around zero reveal "
            "bias that can be hidden by the overall MAE."
            "</sup>"
        ),
        xaxis_title="Observed HFI",
        yaxis_title="Residual (HFI units)",
        template="plotly_white",
        height=590,
    )

    return fig


# =============================================================================
# 4. OUTER-CV STABILITY
# =============================================================================


def plot_outer_cv_stability(
    outer_results: pd.DataFrame,
) -> go.Figure:
    """Show outer-test MAE across repeats and folds."""
    repeat_col = _column(
        outer_results,
        "outer_repeat",
        "repeat",
    )
    fold_col = _column(
        outer_results,
        "outer_fold",
        "fold",
    )
    mae_col = _column(
        outer_results,
        "mae",
        "MAE",
        "outer_mae",
        "test_mae",
    )

    pivot = (
        outer_results
        .pivot_table(
            index=repeat_col,
            columns=fold_col,
            values=mae_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[f"Fold {value}" for value in pivot.columns],
            y=[f"Repeat {value}" for value in pivot.index],
            texttemplate="%{z:.2f}",
            hovertemplate=(
                "%{y}"
                "<br>%{x}"
                "<br>Outer MAE: %{z:.3f}"
                "<extra></extra>"
            ),
            colorbar={"title": "MAE"},
        )
    )

    fig.update_layout(
        title=(
            "Outer-CV stability"
            "<br><sup>"
            "Each cell evaluates Points never used for model selection "
            "or fitting. Lower MAE is better."
            "</sup>"
        ),
        xaxis_title="Outer fold",
        yaxis_title="Outer repeat",
        template="plotly_white",
        height=max(
            420,
            90 + 75 * len(pivot.index),
        ),
    )

    return fig


# =============================================================================
# 5. INNER MODEL COMPETITION
# =============================================================================


def plot_inner_model_competition(
    inner_results: pd.DataFrame,
) -> go.Figure:
    """Compare each family's best inner-CV MAE in every outer split."""
    model_col = _column(
        inner_results,
        "model",
        "model_family",
    )
    repeat_col = _column(
        inner_results,
        "outer_repeat",
        "repeat",
    )
    fold_col = _column(
        inner_results,
        "outer_fold",
        "fold",
    )
    mae_col = _column(
        inner_results,
        "inner_MAE",
        "inner_mae",
        "best_inner_mae",
        "mae",
        "MAE",
    )

    data = inner_results.copy()
    data["_split"] = (
        "R"
        + data[repeat_col].astype(str)
        + "–F"
        + data[fold_col].astype(str)
    )

    split_order = (
        data[
            [repeat_col, fold_col, "_split"]
        ]
        .drop_duplicates()
        .sort_values(
            [repeat_col, fold_col]
        )["_split"]
        .tolist()
    )

    model_order = (
        data.groupby(model_col)[mae_col]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    pivot = (
        data.pivot_table(
            index=model_col,
            columns="_split",
            values=mae_col,
            aggfunc="mean",
        )
        .reindex(
            index=model_order,
            columns=split_order,
        )
    )

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=pivot.columns.tolist(),
            y=[
                _pretty_label(value)
                for value in pivot.index
            ],
            texttemplate="%{z:.2f}",
            hovertemplate=(
                "Model: %{y}"
                "<br>Outer split: %{x}"
                "<br>Best inner MAE: %{z:.3f}"
                "<extra></extra>"
            ),
            colorbar={"title": "Inner MAE"},
        )
    )

    fig.update_layout(
        title=(
            "Model-family competition inside nested CV"
            "<br><sup>"
            "Each cell is a family's best inner-CV MAE for one outer "
            "training set. These values drive family selection."
            "</sup>"
        ),
        xaxis_title="Outer split",
        yaxis_title="Model family",
        template="plotly_white",
        height=max(
            520,
            130 + 55 * len(pivot.index),
        ),
    )

    return fig


# =============================================================================
# 6. SELECTION FREQUENCY
# =============================================================================


def plot_selection_frequency(
    outer_results: pd.DataFrame,
) -> go.Figure:
    """Show how often each pipeline component is selected."""
    components = [
        (
            "Model family",
            _column(
                outer_results,
                "model",
                "selected_model",
                "model_family",
            ),
        ),
        (
            "Feature representation",
            _column(
                outer_results,
                "feature_set",
            ),
        ),
        (
            "Temporal aggregation",
            _column(
                outer_results,
                "aggregation",
            ),
        ),
        (
            "Dimensionality reduction",
            _column(
                outer_results,
                "reduction",
            ),
        ),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            title
            for title, _ in components
        ],
        horizontal_spacing=0.18,
        vertical_spacing=0.20,
    )

    total = len(outer_results)

    for index, (_, column) in enumerate(components):
        row = index // 2 + 1
        col = index % 2 + 1

        counts = (
            outer_results[column]
            .fillna("None")
            .astype(str)
            .value_counts()
            .sort_values()
        )
        percentages = 100.0 * counts / total

        fig.add_trace(
            go.Bar(
                x=counts.to_numpy(),
                y=[
                    _pretty_label(value)
                    for value in counts.index
                ],
                orientation="h",
                text=[
                    f"{count} ({percent:.0f}%)"
                    for count, percent
                    in zip(counts, percentages)
                ],
                textposition="outside",
                customdata=percentages.to_numpy(),
                hovertemplate=(
                    "%{y}"
                    "<br>Selected: %{x} folds"
                    "<br>Frequency: %{customdata:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="Number of outer folds",
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            "Stability of the selected pipeline structure"
            "<br><sup>"
            "Selection frequency is a stability diagnostic, not an "
            "outer-test performance ranking."
            "</sup>"
        ),
        template="plotly_white",
        height=850,
        margin={
            "l": 110,
            "r": 70,
            "t": 115,
            "b": 70,
        },
    )

    return fig


# =============================================================================
# 7. INNER VS OUTER ERROR
# =============================================================================


def plot_inner_vs_outer_mae(
    outer_results: pd.DataFrame,
) -> go.Figure:
    """Compare selected inner-CV MAE with subsequent outer-test MAE."""
    inner_col = _column(
        outer_results,
        "selected_inner_MAE",
        "selected_inner_mae",
        "inner_MAE",
        "inner_mae",
        "best_inner_mae",
    )
    outer_col = _column(
        outer_results,
        "mae",
        "MAE",
        "outer_mae",
        "test_mae",
    )
    repeat_col = _column(
        outer_results,
        "outer_repeat",
        "repeat",
    )
    fold_col = _column(
        outer_results,
        "outer_fold",
        "fold",
    )
    model_col = _column(
        outer_results,
        "model",
        "selected_model",
        "model_family",
    )

    feature_col = _optional_column(
        outer_results,
        "feature_set",
    )
    aggregation_col = _optional_column(
        outer_results,
        "aggregation",
    )

    inner = outer_results[inner_col].to_numpy(dtype=float)
    outer = outer_results[outer_col].to_numpy(dtype=float)
    lower, upper = _shared_limits(inner, outer)

    customdata = np.column_stack(
        [
            outer_results[repeat_col],
            outer_results[fold_col],
            outer_results[model_col].map(_pretty_label),
            (
                outer_results[feature_col].astype(str)
                if feature_col
                else np.repeat("", len(outer_results))
            ),
            (
                outer_results[aggregation_col].astype(str)
                if aggregation_col
                else np.repeat("", len(outer_results))
            ),
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=inner,
            y=outer,
            mode="markers",
            marker={"size": 10, "opacity": 0.8},
            customdata=customdata,
            hovertemplate=(
                "Repeat: %{customdata[0]}"
                "<br>Fold: %{customdata[1]}"
                "<br>Selected model: %{customdata[2]}"
                "<br>Feature set: %{customdata[3]}"
                "<br>Aggregation: %{customdata[4]}"
                "<br>Selected inner MAE: %{x:.3f}"
                "<br>Outer-test MAE: %{y:.3f}"
                "<extra></extra>"
            ),
            name="Outer fold",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line={"dash": "dash"},
            name="Inner = outer",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            "Inner-CV error versus outer-test error"
            "<br><sup>"
            "Points above the diagonal performed worse on unseen outer "
            "Points than estimated during inner selection."
            "</sup>"
        ),
        xaxis_title="Selected inner-CV MAE",
        yaxis_title="Outer-test MAE",
        template="plotly_white",
        height=620,
        width=720,
    )

    fig.update_xaxes(range=[lower, upper])
    fig.update_yaxes(
        range=[lower, upper],
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# =============================================================================
# 8. FINAL MODEL SELECTION
# =============================================================================


def plot_final_model_selection(
    final_candidates: pd.DataFrame,
    final_selection: dict,
) -> go.Figure:
    """Show the family competition used for final all-data selection."""
    model_col = _column(
        final_candidates,
        "model",
        "model_family",
    )
    mae_col = _column(
        final_candidates,
        "inner_MAE",
        "inner_mae",
        "best_inner_mae",
        "mae",
        "MAE",
    )

    data = (
        final_candidates[
            [model_col, mae_col]
        ]
        .copy()
        .sort_values(mae_col)
    )

    configuration = final_selection.get(
        "configuration",
        final_selection,
    )

    selected_model = _enum_to_string(
        configuration.get("model")
        or configuration.get("model_family")
        or configuration.get("selected_model")
    )

    fig = go.Figure(
        go.Bar(
            x=data[mae_col],
            y=[
                _pretty_label(value)
                for value in data[model_col]
            ],
            orientation="h",
            text=[
                f"{value:.3f}"
                for value in data[mae_col]
            ],
            textposition="outside",
            hovertemplate=(
                "Model family: %{y}"
                "<br>Best inner MAE: %{x:.3f}"
                "<extra></extra>"
            ),
            name="Best inner MAE",
        )
    )

    if selected_model is not None:
        normalized = data[model_col].map(
            _enum_to_string
        )
        selected = data[
            normalized == selected_model
        ]

        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=[
                        float(
                            selected.iloc[0][mae_col]
                        )
                    ],
                    y=[
                        _pretty_label(
                            selected.iloc[0][model_col]
                        )
                    ],
                    mode="markers",
                    marker={
                        "size": 16,
                        "symbol": "star",
                    },
                    name="Final selection",
                )
            )

    configuration_text = _final_configuration_text(
        configuration
    )

    fig.update_layout(
        title=(
            "Final model-family selection on all development Points"
            "<br><sup>"
            "Lower inner-CV MAE is better. This is a selection diagnostic, "
            "not an independent performance estimate."
            + (
                f"<br>{configuration_text}"
                if configuration_text
                else ""
            )
            + "</sup>"
        ),
        xaxis_title="Best inner-CV MAE",
        yaxis_title="Model family",
        template="plotly_white",
        height=max(
            520,
            150 + 55 * len(data),
        ),
        margin={
            "l": 140,
            "r": 80,
            "t": 135,
            "b": 70,
        },
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    return fig


# =============================================================================
# DATA PREPARATION
# =============================================================================


def _average_oof_predictions(
    oof_predictions: pd.DataFrame,
) -> pd.DataFrame:
    group_col = _column(
        oof_predictions,
        "Point",
        "point",
        "group",
    )
    bag_col = _column(
        oof_predictions,
        "CapturePointId",
        "capture_point_id",
        "bag",
    )
    observed_col = _column(
        oof_predictions,
        "observed",
        "y_true",
        "target",
        "meanHFI",
    )
    prediction_col = _column(
        oof_predictions,
        "prediction",
        "y_pred",
        "predicted",
    )

    data = (
        oof_predictions
        .groupby(
            [
                group_col,
                bag_col,
                observed_col,
            ],
            as_index=False,
            dropna=False,
        )[prediction_col]
        .agg(
            ["mean", "std", "count"]
        )
        .reset_index()
        .rename(
            columns={
                group_col: "group",
                bag_col: "bag",
                observed_col: "observed",
                "mean": "prediction_mean",
                "std": "prediction_sd",
                "count": "n_predictions",
            }
        )
    )

    data["prediction_sd"] = (
        data["prediction_sd"]
        .fillna(0.0)
    )

    return data


def _repeat_mae_long(
    repeat_metrics: pd.DataFrame,
) -> pd.DataFrame:
    repeat_col = _column(
        repeat_metrics,
        "outer_repeat",
        "repeat",
    )
    method_col = _optional_column(
        repeat_metrics,
        "method",
        "predictor",
        "strategy",
        "model",
    )

    if method_col is not None:
        mae_col = _column(
            repeat_metrics,
            "mae",
            "MAE",
        )

        data = repeat_metrics[
            [
                repeat_col,
                method_col,
                mae_col,
            ]
        ].copy()

        data.columns = [
            "repeat",
            "method",
            "mae",
        ]

        data["method"] = data[
            "method"
        ].map(
            _canonical_method
        )

        return data

    selected_col = _column(
        repeat_metrics,
        "mae",
        "MAE",
        "selected_mae",
        "pipeline_mae",
        "outer_mae",
    )
    mean_col = _column(
        repeat_metrics,
        "mean_baseline_mae",
        "baseline_mean_mae",
        "mean_baseline_MAE",
        "baseline_mean_MAE",
    )
    median_col = _column(
        repeat_metrics,
        "median_baseline_mae",
        "baseline_median_mae",
        "median_baseline_MAE",
        "baseline_median_MAE",
    )

    rows = []

    for _, row in repeat_metrics.iterrows():
        repeat = row[repeat_col]

        rows.extend(
            [
                {
                    "repeat": repeat,
                    "method": "Selected pipeline",
                    "mae": row[selected_col],
                },
                {
                    "repeat": repeat,
                    "method": "Mean baseline",
                    "mae": row[mean_col],
                },
                {
                    "repeat": repeat,
                    "method": "Median baseline",
                    "mae": row[median_col],
                },
            ]
        )

    return pd.DataFrame(rows)


# =============================================================================
# OUTPUT
# =============================================================================


def _save_figure(
    fig: go.Figure,
    path: Path,
) -> None:
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


# =============================================================================
# HELPERS
# =============================================================================


def _column(
    df: pd.DataFrame,
    *candidates: str,
) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        "None of the expected columns were found: "
        + ", ".join(candidates)
    )


def _optional_column(
    df: pd.DataFrame,
    *candidates: str,
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def _canonical_method(
    value,
) -> str:
    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if (
        "median" in text
        and "baseline" in text
    ):
        return "Median baseline"

    if (
        "mean" in text
        and "baseline" in text
    ):
        return "Mean baseline"

    return "Selected pipeline"


def _pretty_label(
    value,
) -> str:
    text = _enum_to_string(value)

    if text is None:
        return "None"

    return (
        text
        .replace("_", " ")
        .title()
    )


def _enum_to_string(
    value,
) -> str | None:
    if value is None:
        return None

    text = str(value)

    if "." in text:
        text = text.split(".")[-1]

    return text


def _shared_limits(
    first,
    second,
) -> tuple[float, float]:
    values = np.concatenate(
        [
            np.asarray(
                first,
                dtype=float,
            ),
            np.asarray(
                second,
                dtype=float,
            ),
        ]
    )

    lower = float(
        np.nanmin(values)
    )
    upper = float(
        np.nanmax(values)
    )

    padding = (
        0.05 * (upper - lower)
        if upper > lower
        else 1.0
    )

    return (
        lower - padding,
        upper + padding,
    )


def _final_configuration_text(
    selection: dict,
) -> str:
    parts = []

    for key, label in [
        ("feature_set", "features"),
        ("aggregation", "aggregation"),
        ("reduction", "reduction"),
    ]:
        value = selection.get(key)

        if value is not None:
            parts.append(
                f"{label}: {value}"
            )

    return " | ".join(parts)
