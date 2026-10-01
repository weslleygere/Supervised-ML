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
# VISUAL CONSTANTS
# =============================================================================


MODEL_COLOR = "#1f77b4"
MEAN_BASELINE_COLOR = "#d95f02"
MEDIAN_BASELINE_COLOR = "#7570b3"

REFERENCE_COLOR = "#4d4d4d"
DEVELOPMENT_COLOR = "#1f77b4"
TEST_COLOR = "#d95f02"


# =============================================================================
# PUBLIC API
# =============================================================================


def save_evaluation_plots(
    output_dir: str | Path,
) -> None:
    """
    Create report- and publication-oriented figures from one completed
    experiment.

    Main figures
    ------------
    Figure 1
        Observed versus predicted HFI on the independent final test set.

    Figure 2
        Point-level MAE gain of the selected model relative to the reference
        baselines.

    Figure 3
        Repeated-CV MAE for the selected model and both baselines.

    Supplementary figures
    ---------------------
    Figure S1
        Final-test residuals across the observed HFI gradient.

    Figure S2
        Mean OOF prediction and prediction variability across repeated CV.

    Figure S3
        Point-balanced empirical HFI distributions in development and test.

    Figure S4
        Optuna optimization history.

    Figure S5
        Highest-performing complete pipeline candidates.

    Figure S6
        Model dimensionality versus repeated-CV MAE.

    Figure S7
        Adaptive search-space sampling coverage.

    Notes
    -----
    Figures based on Optuna trial distributions are descriptive diagnostics
    of the adaptive search and are not formal comparisons of pipeline
    components.
    """

    output_dir = Path(
        output_dir
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    main_dir = (
        figures_dir
        / "main"
    )

    supplementary_dir = (
        figures_dir
        / "supplementary"
    )

    main_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    supplementary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD RAW EXPERIMENT OUTPUTS
    # =========================================================================

    trials = _read_csv(
        output_dir
        / "model_selection_trials.csv"
    )

    final_test = _read_csv(
        output_dir
        / "final_test_predictions.csv"
    )

    selected_pipeline = _read_json(
        output_dir
        / "selected_pipeline.json"
    )

    split_manifest = _read_csv(
        output_dir
        / "reproducibility"
        / "data_split.csv"
    )

    # =========================================================================
    # LOAD ANALYSIS TABLES
    # =========================================================================

    analysis_dir = (
        output_dir
        / "analysis"
    )

    point_errors = _read_csv(
        analysis_dir
        / "final_test_point_errors.csv"
    )

    cv_repeat_metrics = _read_csv(
        analysis_dir
        / "selected_cv_repeat_metrics.csv"
    )

    cv_stability = _read_csv(
        analysis_dir
        / "selected_cv_prediction_stability.csv"
    )

    top_candidates = _read_csv(
        analysis_dir
        / "top_pipeline_candidates.csv"
    )

    search_components = _read_csv(
        analysis_dir
        / "search_component_descriptive.csv"
    )

    # =========================================================================
    # MAIN FIGURES
    # =========================================================================

    main_figures = [
        (
            "figure_01_observed_vs_predicted",
            plot_observed_vs_predicted(
                final_test=final_test,
            ),
        ),
        (
            "figure_02_pointwise_baseline_gain",
            plot_pointwise_baseline_gain(
                point_errors=point_errors,
            ),
        ),
        (
            "figure_03_cv_repeat_stability",
            plot_cv_repeat_stability(
                cv_repeat_metrics=(
                    cv_repeat_metrics
                ),
            ),
        ),
    ]

    # =========================================================================
    # SUPPLEMENTARY FIGURES
    # =========================================================================

    supplementary_figures = [
        (
            "figure_s01_residuals",
            plot_residuals(
                final_test=final_test,
            ),
        ),
        (
            "figure_s02_cv_prediction_stability",
            plot_cv_prediction_stability(
                stability=cv_stability,
            ),
        ),
        (
            "figure_s03_hfi_partition_ecdf",
            plot_hfi_partition_ecdf(
                split_manifest=(
                    split_manifest
                ),
            ),
        ),
        (
            "figure_s04_optuna_history",
            plot_optuna_history(
                trials=trials,
                selected_pipeline=(
                    selected_pipeline
                ),
            ),
        ),
        (
            "figure_s05_top_candidates",
            plot_top_candidates(
                candidates=(
                    top_candidates
                ),
            ),
        ),
        (
            "figure_s06_dimensionality",
            plot_dimensionality(
                trials=trials,
                selected_pipeline=(
                    selected_pipeline
                ),
            ),
        ),
        (
            "figure_s07_search_coverage",
            plot_search_coverage(
                search_components=(
                    search_components
                ),
            ),
        ),
    ]

    # =========================================================================
    # SAVE FIGURES
    # =========================================================================

    for name, figure in (
        main_figures
    ):

        _save_figure(
            figure,
            main_dir
            / name,
        )

        logger.info(
            "Saved main figure: %s",
            name,
        )

    for name, figure in (
        supplementary_figures
    ):

        _save_figure(
            figure,
            supplementary_dir
            / name,
        )

        logger.info(
            "Saved supplementary figure: %s",
            name,
        )

    # =========================================================================
    # FIGURE MANIFEST
    # =========================================================================

    manifest = (
        _build_figure_manifest()
    )

    manifest.to_csv(
        figures_dir
        / "figure_manifest.csv",
        index=False,
    )

    logger.info(
        "Figure manifest saved to %s",
        figures_dir
        / "figure_manifest.csv",
    )


# =============================================================================
# MAIN FIGURE 1 — OBSERVED VS PREDICTED
# =============================================================================


def plot_observed_vs_predicted(
    final_test: pd.DataFrame,
) -> go.Figure:
    """
    Observed versus predicted HFI on the independent final test set.

    Each marker represents one CapturePointId.

    Different CapturePointIds belonging to the same Point are retained
    separately because they may have different HFI targets.
    """

    data = (
        _prepare_test_predictions(
            final_test
        )
    )

    lower, upper = (
        _shared_limits(
            data[
                "observed_HFI"
            ],
            data[
                "predicted_HFI"
            ],
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[
                "observed_HFI"
            ],
            y=data[
                "predicted_HFI"
            ],
            mode="markers",
            marker={
                "size":
                    9,

                "opacity":
                    0.78,

                "color":
                    MODEL_COLOR,
            },
            customdata=np.column_stack(
                [
                    data[
                        "Point"
                    ],
                    data[
                        "CapturePointId"
                    ],
                    data[
                        "absolute_error"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Predicted HFI: %{y:.3f}"
                "<br>Absolute error: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                lower,
                upper,
            ],
            y=[
                lower,
                upper,
            ],
            mode="lines",
            line={
                "width":
                    1.4,

                "dash":
                    "dash",

                "color":
                    REFERENCE_COLOR,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        title_text="Observed HFI",
        range=[
            lower,
            upper,
        ],
    )

    fig.update_yaxes(
        title_text="Predicted HFI",
        range=[
            lower,
            upper,
        ],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        width=650,
        height=650,
        margin={
            "l":
                90,

            "r":
                35,

            "t":
                30,

            "b":
                80,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# MAIN FIGURE 2 — POINT-LEVEL BASELINE GAIN
# =============================================================================


def plot_pointwise_baseline_gain(
    point_errors: pd.DataFrame,
) -> go.Figure:
    """
    Show Point-level MAE gain of the selected model relative to both
    reference baselines.

    Gain is defined as:

        baseline MAE - model MAE

    Positive values therefore indicate smaller error for the selected model.

    Errors have already been calculated at CapturePointId level before being
    averaged within Point.
    """

    required = {
        "Point",
        "MAE_gain_vs_mean_baseline",
        "MAE_gain_vs_median_baseline",
    }

    _require_columns(
        point_errors,
        required,
    )

    data = (
        point_errors.copy()
    )

    data = (
        data.sort_values(
            "MAE_gain_vs_mean_baseline",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[
                "MAE_gain_vs_mean_baseline"
            ],
            y=data[
                "Point"
            ],
            mode="markers",
            marker={
                "size":
                    9,

                "color":
                    MEAN_BASELINE_COLOR,
            },
            name="vs. mean baseline",
            customdata=np.column_stack(
                [
                    data[
                        "model_MAE"
                    ],
                    data[
                        "mean_baseline_MAE"
                    ],
                    data[
                        "n_capture_points"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{y}"
                "<br>MAE gain: %{x:.3f}"
                "<br>Model MAE: %{customdata[0]:.3f}"
                "<br>Mean baseline MAE: %{customdata[1]:.3f}"
                "<br>CapturePointIds: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data[
                "MAE_gain_vs_median_baseline"
            ],
            y=data[
                "Point"
            ],
            mode="markers",
            marker={
                "size":
                    9,

                "symbol":
                    "diamond",

                "color":
                    MEDIAN_BASELINE_COLOR,
            },
            name="vs. median baseline",
            customdata=np.column_stack(
                [
                    data[
                        "model_MAE"
                    ],
                    data[
                        "median_baseline_MAE"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{y}"
                "<br>MAE gain: %{x:.3f}"
                "<br>Model MAE: %{customdata[0]:.3f}"
                "<br>Median baseline MAE: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line_width=1.3,
        line_dash="dash",
        line_color=(
            REFERENCE_COLOR
        ),
    )

    fig.update_xaxes(
        title_text=(
            "MAE gain relative to baseline "
            "(baseline MAE − model MAE)"
        ),
    )

    fig.update_yaxes(
        title_text="",
        autorange="reversed",
    )

    fig.update_layout(
        width=820,
        height=max(
            500,
            130
            + 30
            * len(
                data
            ),
        ),
        legend={
            "orientation":
                "h",

            "x":
                0,

            "y":
                1.06,
        },
        margin={
            "l":
                100,

            "r":
                40,

            "t":
                65,

            "b":
                85,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# MAIN FIGURE 3 — CV REPEAT STABILITY
# =============================================================================


def plot_cv_repeat_stability(
    cv_repeat_metrics: pd.DataFrame,
) -> go.Figure:
    """
    Compare Point-balanced MAE across complete repeated-CV repetitions.

    This figure shows whether the selected model's advantage over the
    baselines is consistent across different grouped partitions.
    """

    required = {
        "repeat",
        "method",
        "MAE",
    }

    _require_columns(
        cv_repeat_metrics,
        required,
    )

    specifications = [
        (
            "selected_pipeline",
            "Selected pipeline",
            MODEL_COLOR,
            "circle",
        ),
        (
            "mean_baseline",
            "Mean baseline",
            MEAN_BASELINE_COLOR,
            "square",
        ),
        (
            "median_baseline",
            "Median baseline",
            MEDIAN_BASELINE_COLOR,
            "diamond",
        ),
    ]

    fig = go.Figure()

    for (
        method,
        label,
        color,
        symbol,
    ) in specifications:

        data = (
            cv_repeat_metrics[
                cv_repeat_metrics[
                    "method"
                ]
                == method
            ]
            .sort_values(
                "repeat"
            )
        )

        if data.empty:

            continue

        fig.add_trace(
            go.Scatter(
                x=data[
                    "repeat"
                ],
                y=data[
                    "MAE"
                ],
                mode="lines+markers",
                line={
                    "width":
                        1.8,

                    "color":
                        color,
                },
                marker={
                    "size":
                        9,

                    "symbol":
                        symbol,

                    "color":
                        color,
                },
                name=label,
                customdata=np.column_stack(
                    [
                        data[
                            "RMSE"
                        ],
                        data[
                            "R2"
                        ],
                    ]
                ),
                hovertemplate=(
                    "Repeat %{x}"
                    "<br>MAE: %{y:.3f}"
                    "<br>RMSE: %{customdata[0]:.3f}"
                    "<br>R²: %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    repeats = sorted(
        cv_repeat_metrics[
            "repeat"
        ]
        .dropna()
        .unique()
    )

    fig.update_xaxes(
        title_text="Cross-validation repeat",
        tickmode="array",
        tickvals=repeats,
    )

    fig.update_yaxes(
        title_text="Point-balanced MAE",
    )

    fig.update_layout(
        width=760,
        height=500,
        legend={
            "orientation":
                "h",

            "x":
                0,

            "y":
                1.08,
        },
        margin={
            "l":
                90,

            "r":
                35,

            "t":
                65,

            "b":
                75,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S1 — FINAL-TEST RESIDUALS
# =============================================================================


def plot_residuals(
    final_test: pd.DataFrame,
) -> go.Figure:
    """
    Show final-test residuals across the observed HFI gradient.

    Residual is defined as:

        observed - predicted

    Positive residuals indicate underprediction.
    Negative residuals indicate overprediction.
    """

    data = (
        _prepare_test_predictions(
            final_test
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[
                "observed_HFI"
            ],
            y=data[
                "residual"
            ],
            mode="markers",
            marker={
                "size":
                    9,

                "opacity":
                    0.78,

                "color":
                    MODEL_COLOR,
            },
            customdata=np.column_stack(
                [
                    data[
                        "Point"
                    ],
                    data[
                        "CapturePointId"
                    ],
                    data[
                        "absolute_error"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Residual: %{y:.3f}"
                "<br>Absolute error: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1.3,
        line_color=(
            REFERENCE_COLOR
        ),
    )

    fig.update_xaxes(
        title_text="Observed HFI",
    )

    fig.update_yaxes(
        title_text="Residual (observed − predicted)",
    )

    fig.update_layout(
        width=700,
        height=500,
        margin={
            "l":
                90,

            "r":
                35,

            "t":
                30,

            "b":
                75,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S2 — CV PREDICTION STABILITY
# =============================================================================


def plot_cv_prediction_stability(
    stability: pd.DataFrame,
) -> go.Figure:
    """
    Show the mean OOF prediction for each CapturePointId across repeated CV,
    together with the between-repeat SD of its predictions.

    This is a stability diagnostic rather than an independent performance
    estimate.
    """

    required = {
        "Point",
        "CapturePointId",
        "observed_HFI",
        "predicted_HFI_mean",
        "predicted_HFI_sd",
    }

    _require_columns(
        stability,
        required,
    )

    data = (
        stability.copy()
    )

    lower, upper = (
        _shared_limits(
            data[
                "observed_HFI"
            ],
            data[
                "predicted_HFI_mean"
            ],
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[
                "observed_HFI"
            ],
            y=data[
                "predicted_HFI_mean"
            ],
            mode="markers",
            marker={
                "size":
                    8,

                "opacity":
                    0.75,

                "color":
                    MODEL_COLOR,
            },
            error_y={
                "type":
                    "data",

                "array":
                    data[
                        "predicted_HFI_sd"
                    ]
                    .fillna(
                        0.0
                    ),

                "visible":
                    True,

                "thickness":
                    1,

                "width":
                    3,
            },
            customdata=np.column_stack(
                [
                    data[
                        "Point"
                    ],
                    data[
                        "CapturePointId"
                    ],
                    data[
                        "n_repeats"
                    ],
                    data[
                        "prediction_range"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Mean OOF prediction: %{y:.3f}"
                "<br>Repeats: %{customdata[2]}"
                "<br>Prediction range: %{customdata[3]:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                lower,
                upper,
            ],
            y=[
                lower,
                upper,
            ],
            mode="lines",
            line={
                "width":
                    1.3,

                "dash":
                    "dash",

                "color":
                    REFERENCE_COLOR,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        title_text="Observed HFI",
        range=[
            lower,
            upper,
        ],
    )

    fig.update_yaxes(
        title_text="Mean repeated-CV OOF prediction",
        range=[
            lower,
            upper,
        ],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        width=650,
        height=650,
        margin={
            "l":
                95,

            "r":
                40,

            "t":
                30,

            "b":
                80,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S3 — HFI PARTITION DISTRIBUTION
# =============================================================================


def plot_hfi_partition_ecdf(
    split_manifest: pd.DataFrame,
) -> go.Figure:
    """
    Compare HFI distributions in development and final-test partitions using
    Point-balanced empirical cumulative distributions.

    Each Point receives equal total weight even when it contains multiple
    CapturePointIds.
    """

    required = {
        "Point",
        "CapturePointId",
        "meanHFI",
        "split",
    }

    _require_columns(
        split_manifest,
        required,
    )

    specifications = [
        (
            "development",
            "Development",
            DEVELOPMENT_COLOR,
        ),
        (
            "test",
            "Final test",
            TEST_COLOR,
        ),
    ]

    fig = go.Figure()

    for (
        split_name,
        label,
        color,
    ) in specifications:

        data = (
            split_manifest[
                split_manifest[
                    "split"
                ]
                == split_name
            ]
            .copy()
        )

        if data.empty:

            continue

        ecdf = (
            _point_balanced_ecdf(
                data
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ecdf[
                    "meanHFI"
                ],
                y=ecdf[
                    "cumulative_weight"
                ],
                mode="lines",
                line={
                    "width":
                        2.2,

                    "shape":
                        "hv",

                    "color":
                        color,
                },
                name=label,
                hovertemplate=(
                    "HFI: %{x:.3f}"
                    "<br>Cumulative weighted fraction: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(
        title_text="HFI",
    )

    fig.update_yaxes(
        title_text="Point-balanced cumulative fraction",
        range=[
            0,
            1.02,
        ],
    )

    fig.update_layout(
        width=720,
        height=500,
        legend={
            "orientation":
                "h",

            "x":
                0,

            "y":
                1.08,
        },
        margin={
            "l":
                95,

            "r":
                35,

            "t":
                65,

            "b":
                75,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S4 — OPTUNA HISTORY
# =============================================================================


def plot_optuna_history(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> go.Figure:
    """
    Show the adaptive optimization history of the CASH search.

    Individual markers represent completed trials.

    The line represents the cumulative best CV MAE found up to each trial.
    """

    data = (
        _completed_trials(
            trials
        )
        .sort_values(
            "trial"
        )
        .reset_index(
            drop=True
        )
    )

    if data.empty:

        raise ValueError(
            "No completed Optuna trials available."
        )

    data[
        "best_so_far"
    ] = (
        data[
            "CV_MAE"
        ]
        .cummin()
    )

    selected_trial = (
        _selected_trial_number(
            selected_pipeline
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[
                "trial"
            ],
            y=data[
                "CV_MAE"
            ],
            mode="markers",
            marker={
                "size":
                    6,

                "opacity":
                    0.40,

                "color":
                    "#8c8c8c",
            },
            customdata=np.column_stack(
                [
                    data[
                        "model"
                    ],
                    data[
                        "feature_set"
                    ],
                    data[
                        "aggregation"
                    ],
                    data[
                        "reduction"
                    ],
                ]
            ),
            hovertemplate=(
                "Trial %{x}"
                "<br>CV MAE: %{y:.3f}"
                "<br>Model: %{customdata[0]}"
                "<br>Features: %{customdata[1]}"
                "<br>Aggregation: %{customdata[2]}"
                "<br>Reduction: %{customdata[3]}"
                "<extra></extra>"
            ),
            name="Completed trials",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data[
                "trial"
            ],
            y=data[
                "best_so_far"
            ],
            mode="lines",
            line={
                "width":
                    2.2,

                "color":
                    MODEL_COLOR,
            },
            name="Best so far",
            hovertemplate=(
                "Trial %{x}"
                "<br>Best CV MAE: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    selected = data[
        data[
            "trial"
        ]
        == selected_trial
    ]

    if not selected.empty:

        fig.add_trace(
            go.Scatter(
                x=selected[
                    "trial"
                ],
                y=selected[
                    "CV_MAE"
                ],
                mode="markers",
                marker={
                    "size":
                        14,

                    "symbol":
                        "star",

                    "color":
                        MODEL_COLOR,
                },
                name="Selected",
                hovertemplate=(
                    "Selected trial %{x}"
                    "<br>CV MAE: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(
        title_text="Optuna trial",
    )

    fig.update_yaxes(
        title_text="Repeated-CV MAE",
    )

    fig.update_layout(
        width=780,
        height=500,
        legend={
            "orientation":
                "h",

            "x":
                0,

            "y":
                1.08,
        },
        margin={
            "l":
                90,

            "r":
                35,

            "t":
                65,

            "b":
                75,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S5 — TOP CANDIDATES
# =============================================================================


def plot_top_candidates(
    candidates: pd.DataFrame,
    top_n: int = 12,
) -> go.Figure:
    """
    Show the strongest observed complete pipeline candidates.

    Horizontal error bars represent ±1 SD across complete CV repetitions.

    The figure describes the best configurations observed during the adaptive
    search and must not be interpreted as a balanced component comparison.
    """

    required = {
        "trial",
        "selected",
        "model",
        "feature_set",
        "aggregation",
        "reduction",
        "CV_MAE",
        "CV_MAE_std",
    }

    _require_columns(
        candidates,
        required,
    )

    data = (
        candidates.head(
            top_n
        )
        .copy()
    )

    data[
        "label"
    ] = (
        data.apply(
            _candidate_label,
            axis=1,
        )
    )

    data = (
        data.sort_values(
            "CV_MAE",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    fig = go.Figure()

    non_selected = (
        data[
            ~data[
                "selected"
            ]
            .astype(
                bool
            )
        ]
    )

    if not non_selected.empty:

        fig.add_trace(
            go.Scatter(
                x=non_selected[
                    "CV_MAE"
                ],
                y=non_selected[
                    "label"
                ],
                mode="markers",
                marker={
                    "size":
                        8,

                    "color":
                        "#7f7f7f",
                },
                error_x={
                    "type":
                        "data",

                    "array":
                        non_selected[
                            "CV_MAE_std"
                        ]
                        .fillna(
                            0.0
                        ),

                    "visible":
                        True,

                    "thickness":
                        1,
                },
                customdata=np.column_stack(
                    [
                        non_selected[
                            "trial"
                        ],
                        non_selected[
                            "model_feature_count"
                        ],
                    ]
                ),
                hovertemplate=(
                    "Trial %{customdata[0]}"
                    "<br>CV MAE: %{x:.3f}"
                    "<br>Model features: %{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    selected = (
        data[
            data[
                "selected"
            ]
            .astype(
                bool
            )
        ]
    )

    if not selected.empty:

        fig.add_trace(
            go.Scatter(
                x=selected[
                    "CV_MAE"
                ],
                y=selected[
                    "label"
                ],
                mode="markers",
                marker={
                    "size":
                        14,

                    "symbol":
                        "star",

                    "color":
                        MODEL_COLOR,
                },
                error_x={
                    "type":
                        "data",

                    "array":
                        selected[
                            "CV_MAE_std"
                        ]
                        .fillna(
                            0.0
                        ),

                    "visible":
                        True,

                    "thickness":
                        1,
                },
                customdata=selected[
                    "trial"
                ],
                hovertemplate=(
                    "Selected trial %{customdata}"
                    "<br>CV MAE: %{x:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(
        title_text="Repeated-CV MAE",
    )

    fig.update_yaxes(
        title_text="",
    )

    fig.update_layout(
        width=980,
        height=max(
            520,
            120
            + 42
            * len(
                data
            ),
        ),
        margin={
            "l":
                350,

            "r":
                45,

            "t":
                30,

            "b":
                80,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S6 — DIMENSIONALITY
# =============================================================================


def plot_dimensionality(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> go.Figure:
    """
    Show model dimensionality versus repeated-CV MAE.

    This is a descriptive diagnostic of the adaptive Optuna search.
    """

    data = (
        _completed_trials(
            trials
        )
    )

    _require_columns(
        data,
        {
            "model_feature_count",
            "reduction",
        },
    )

    data = data[
        pd.notna(
            data[
                "model_feature_count"
            ]
        )
    ].copy()

    data = data[
        data[
            "model_feature_count"
        ]
        > 0
    ]

    selected_trial = (
        _selected_trial_number(
            selected_pipeline
        )
    )

    reduction_specs = {
        "none":
            (
                "No reduction",
                "#7f7f7f",
                "circle",
            ),

        "pca":
            (
                "PCA",
                "#2ca02c",
                "square",
            ),

        "supervised_selection":
            (
                "Supervised selection",
                "#9467bd",
                "diamond",
            ),
    }

    fig = go.Figure()

    for reduction, (
        label,
        color,
        symbol,
    ) in reduction_specs.items():

        subset = data[
            data[
                "reduction"
            ]
            == reduction
        ]

        if subset.empty:

            continue

        fig.add_trace(
            go.Scatter(
                x=subset[
                    "model_feature_count"
                ],
                y=subset[
                    "CV_MAE"
                ],
                mode="markers",
                marker={
                    "size":
                        7,

                    "opacity":
                        0.50,

                    "color":
                        color,

                    "symbol":
                        symbol,
                },
                name=label,
                customdata=np.column_stack(
                    [
                        subset[
                            "trial"
                        ],
                        subset[
                            "model"
                        ],
                        subset[
                            "feature_set"
                        ],
                        subset[
                            "aggregation"
                        ],
                    ]
                ),
                hovertemplate=(
                    "Trial %{customdata[0]}"
                    "<br>Model: %{customdata[1]}"
                    "<br>Features: %{customdata[2]}"
                    "<br>Aggregation: %{customdata[3]}"
                    "<br>Model features: %{x}"
                    "<br>CV MAE: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    selected = data[
        data[
            "trial"
        ]
        == selected_trial
    ]

    if not selected.empty:

        fig.add_trace(
            go.Scatter(
                x=selected[
                    "model_feature_count"
                ],
                y=selected[
                    "CV_MAE"
                ],
                mode="markers",
                marker={
                    "size":
                        15,

                    "symbol":
                        "star",

                    "color":
                        MODEL_COLOR,

                    "line":
                        {
                            "width":
                                1,

                            "color":
                                "black",
                        },
                },
                name="Selected",
                hovertemplate=(
                    "Selected"
                    "<br>Features: %{x}"
                    "<br>CV MAE: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(
        title_text="Number of model features",
        type="log",
    )

    fig.update_yaxes(
        title_text="Repeated-CV MAE",
    )

    fig.update_layout(
        width=760,
        height=520,
        legend={
            "orientation":
                "h",

            "x":
                0,

            "y":
                1.10,
        },
        margin={
            "l":
                90,

            "r":
                35,

            "t":
                75,

            "b":
                80,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S7 — SEARCH COVERAGE
# =============================================================================


def plot_search_coverage(
    search_components: pd.DataFrame,
) -> go.Figure:
    """
    Show how the adaptive Optuna search allocated completed trials across
    pipeline components.

    This figure visualizes search coverage, not component superiority.
    """

    required = {
        "component",
        "level",
        "n_trials",
        "sampling_fraction",
    }

    _require_columns(
        search_components,
        required,
    )

    specifications = [
        (
            "model",
            "Model family",
            1,
            1,
        ),
        (
            "feature_set",
            "Feature representation",
            1,
            2,
        ),
        (
            "aggregation",
            "Acoustic aggregation",
            2,
            1,
        ),
        (
            "reduction",
            "Reduction method",
            2,
            2,
        ),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            item[
                1
            ]
            for item in specifications
        ],
        horizontal_spacing=0.16,
        vertical_spacing=0.22,
    )

    for (
        component,
        _,
        row,
        column,
    ) in specifications:

        data = (
            search_components[
                search_components[
                    "component"
                ]
                == component
            ]
            .copy()
        )

        if data.empty:

            continue

        data = (
            data.sort_values(
                "n_trials",
                ascending=False,
            )
        )

        labels = (
            data[
                "level"
            ]
            .map(
                lambda value:
                    _component_level_label(
                        component,
                        value,
                    )
            )
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=data[
                    "n_trials"
                ],
                marker={
                    "color":
                        MODEL_COLOR,
                },
                customdata=data[
                    "sampling_fraction"
                ],
                hovertemplate=(
                    "%{x}"
                    "<br>Completed trials: %{y}"
                    "<br>Sampling fraction: %{customdata:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )

    fig.update_yaxes(
        title_text="Completed trials",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Completed trials",
        row=2,
        col=1,
    )

    fig.update_xaxes(
        tickangle=-25,
    )

    fig.update_layout(
        width=1000,
        height=720,
        margin={
            "l":
                85,

            "r":
                35,

            "t":
                70,

            "b":
                110,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# TEST DATA PREPARATION
# =============================================================================


def _prepare_test_predictions(
    final_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare CapturePointId-level independent-test predictions.
    """

    required = {
        "Point",
        "CapturePointId",
        "observed_HFI",
        "predicted_HFI",
    }

    _require_columns(
        final_test,
        required,
    )

    data = (
        final_test[
            [
                "Point",
                "CapturePointId",
                "observed_HFI",
                "predicted_HFI",
            ]
        ]
        .copy()
    )

    data[
        "residual"
    ] = (
        data[
            "observed_HFI"
        ]
        - data[
            "predicted_HFI"
        ]
    )

    data[
        "absolute_error"
    ] = (
        data[
            "residual"
        ]
        .abs()
    )

    return (
        data.sort_values(
            [
                "Point",
                "CapturePointId",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# POINT-BALANCED ECDF
# =============================================================================


def _point_balanced_ecdf(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a weighted ECDF in which every physical Point contributes equal
    total weight.

    Multiple CapturePointIds belonging to the same Point divide that Point's
    weight among themselves.
    """

    result = (
        data[
            [
                "Point",
                "CapturePointId",
                "meanHFI",
            ]
        ]
        .copy()
    )

    n_captures = (
        result.groupby(
            "Point"
        )[
            "CapturePointId"
        ]
        .transform(
            "nunique"
        )
        .to_numpy(
            dtype=float
        )
    )

    result[
        "weight"
    ] = (
        1.0
        / n_captures
    )

    result = (
        result.sort_values(
            "meanHFI"
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "cumulative_weight"
    ] = (
        result[
            "weight"
        ]
        .cumsum()
        / result[
            "weight"
        ]
        .sum()
    )

    return result


# =============================================================================
# TRIAL DATA PREPARATION
# =============================================================================


def _completed_trials(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return successfully completed trials with finite CV MAE.
    """

    _require_columns(
        trials,
        {
            "trial",
            "state",
            "CV_MAE",
        },
    )

    data = (
        trials.copy()
    )

    data = data[
        data[
            "state"
        ]
        .astype(
            str
        )
        .str.upper()
        == "COMPLETE"
    ]

    data = data[
        pd.notna(
            data[
                "CV_MAE"
            ]
        )
    ].copy()

    finite = np.isfinite(
        data[
            "CV_MAE"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return (
        data.loc[
            finite
        ]
        .copy()
    )


# =============================================================================
# CANDIDATE LABEL
# =============================================================================


def _candidate_label(
    row: pd.Series,
) -> str:
    """
    Compact report label for one complete candidate pipeline.
    """

    model = (
        _short_model_name(
            row.get(
                "model"
            )
        )
    )

    feature_set = (
        _feature_label(
            row.get(
                "feature_set"
            )
        )
    )

    aggregation = (
        _aggregation_label(
            row.get(
                "aggregation"
            )
        )
    )

    reduction = (
        _reduction_label(
            row.get(
                "reduction"
            )
        )
    )

    return (
        f"{model} | "
        f"{feature_set} | "
        f"{aggregation} | "
        f"{reduction}"
    )


# =============================================================================
# SELECTED PIPELINE
# =============================================================================


def _selected_trial_number(
    selected_pipeline: dict,
) -> int:
    """
    Extract the selected Optuna trial number.
    """

    if (
        "selected_trial"
        not in selected_pipeline
    ):

        raise KeyError(
            "selected_pipeline.json does not contain "
            "'selected_trial'."
        )

    return int(
        selected_pipeline[
            "selected_trial"
        ]
    )


# =============================================================================
# FIGURE MANIFEST
# =============================================================================


def _build_figure_manifest() -> pd.DataFrame:
    """
    Document the scientific question, source and interpretation of each
    generated figure.
    """

    rows = [
        {
            "figure":
                "figure_01_observed_vs_predicted",

            "section":
                "main",

            "scientific_question":
                (
                    "How closely do independent final-test predictions "
                    "match observed HFI values?"
                ),

            "data_level":
                "CapturePointId",

            "interpretation":
                "primary_independent_test_result",
        },

        {
            "figure":
                "figure_02_pointwise_baseline_gain",

            "section":
                "main",

            "scientific_question":
                (
                    "At which independent Points does the selected pipeline "
                    "improve on simple reference predictions?"
                ),

            "data_level":
                "Point after CapturePointId-level error calculation",

            "interpretation":
                "independent_test_baseline_comparison",
        },

        {
            "figure":
                "figure_03_cv_repeat_stability",

            "section":
                "main",

            "scientific_question":
                (
                    "Is model-selection performance stable across repeated "
                    "grouped CV partitions?"
                ),

            "data_level":
                "complete CV repeat",

            "interpretation":
                "model_selection_stability",
        },

        {
            "figure":
                "figure_s01_residuals",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "Does prediction bias change across the observed HFI "
                    "gradient?"
                ),

            "data_level":
                "CapturePointId",

            "interpretation":
                "independent_test_diagnostic",
        },

        {
            "figure":
                "figure_s02_cv_prediction_stability",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "How much do OOF predictions for individual "
                    "CapturePointIds vary across CV repeats?"
                ),

            "data_level":
                "CapturePointId across repeated CV",

            "interpretation":
                "stability_diagnostic",
        },

        {
            "figure":
                "figure_s03_hfi_partition_ecdf",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "How similar are the Point-balanced HFI distributions "
                    "of development and final-test partitions?"
                ),

            "data_level":
                "Point-balanced CapturePointId distribution",

            "interpretation":
                "experimental_design_diagnostic",
        },

        {
            "figure":
                "figure_s04_optuna_history",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "How did the best observed CV MAE evolve during the "
                    "adaptive search?"
                ),

            "data_level":
                "Optuna trial",

            "interpretation":
                "adaptive_search_diagnostic",
        },

        {
            "figure":
                "figure_s05_top_candidates",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "Which complete pipeline configurations achieved the "
                    "lowest observed CV MAE?"
                ),

            "data_level":
                "Optuna trial",

            "interpretation":
                "adaptive_search_descriptive_not_formal_comparison",
        },

        {
            "figure":
                "figure_s06_dimensionality",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "How does model dimensionality relate descriptively to "
                    "observed CV performance?"
                ),

            "data_level":
                "Optuna trial",

            "interpretation":
                "adaptive_search_descriptive_not_formal_comparison",
        },

        {
            "figure":
                "figure_s07_search_coverage",

            "section":
                "supplementary",

            "scientific_question":
                (
                    "How was the adaptive search budget distributed across "
                    "pipeline components?"
                ),

            "data_level":
                "Optuna completed trials",

            "interpretation":
                "adaptive_search_coverage",
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# OUTPUT
# =============================================================================


def _save_figure(
    fig: go.Figure,
    path: Path,
) -> None:
    """
    Save one high-resolution PNG figure.
    """

    fig.write_image(
        path.with_suffix(
            ".png"
        ),
        scale=3,
    )


# =============================================================================
# PUBLICATION STYLE
# =============================================================================


def _publication_style(
    fig: go.Figure,
) -> None:
    """
    Apply a restrained manuscript-oriented visual style.

    Figures deliberately contain no overall internal title. Scientific
    interpretation belongs in the report or manuscript caption.
    """

    fig.update_layout(
        template="plotly_white",
        font={
            "family":
                "Arial",

            "size":
                15,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1.2,
        linecolor="black",
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
        title_standoff=12,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1.2,
        linecolor="black",
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
        title_standoff=12,
    )


# =============================================================================
# LABEL HELPERS
# =============================================================================


def _short_model_name(
    value,
) -> str:
    """
    Compact model-family labels for figures.
    """

    mapping = {
        "RIDGE_REGRESSION":
            "Ridge",

        "ELASTIC_NET":
            "Elastic Net",

        "BAYESIAN_RIDGE":
            "Bayesian Ridge",

        "SVR":
            "SVR",

        "KERNEL_RIDGE":
            "Kernel Ridge",

        "RANDOM_FOREST":
            "Random Forest",

        "EXTRA_TREES":
            "Extra Trees",

        "GRADIENT_BOOSTING":
            "Gradient Boosting",

        "XGBOOST":
            "XGBoost",

        "LIGHTGBM":
            "LightGBM",

        "CATBOOST":
            "CatBoost",
    }

    text = str(
        value
    )

    return mapping.get(
        text,
        text,
    )


def _feature_label(
    value,
) -> str:
    """
    Compact acoustic-feature representation label.
    """

    mapping = {
        "indices":
            "Indices",

        "embeddings":
            "Embeddings",

        "both":
            "Both",
    }

    text = str(
        value
    )

    return mapping.get(
        text,
        text,
    )


def _aggregation_label(
    value,
) -> str:
    """
    Compact acoustic aggregation label.
    """

    mapping = {
        "mean":
            "Mean",

        "mean_std":
            "Mean+SD",

        "hierarchical":
            "Hierarchical",

        "robust_daily":
            "Robust daily",

        "dawn_profile":
            "Dawn profile",

        "dawn_trend":
            "Dawn trend",
    }

    text = str(
        value
    )

    return mapping.get(
        text,
        text,
    )


def _reduction_label(
    value,
) -> str:
    """
    Compact dimensionality-reduction label.
    """

    mapping = {
        "none":
            "No reduction",

        "pca":
            "PCA",

        "supervised_selection":
            "Supervised selection",
    }

    text = str(
        value
    )

    return mapping.get(
        text,
        text,
    )


def _component_level_label(
    component: str,
    value,
) -> str:
    """
    Format adaptive-search component levels for the coverage figure.
    """

    if component == "model":

        return _short_model_name(
            value
        )

    if component == "feature_set":

        return _feature_label(
            value
        )

    if component == "aggregation":

        return _aggregation_label(
            value
        )

    if component == "reduction":

        return _reduction_label(
            value
        )

    return str(
        value
    )


# =============================================================================
# FILE HELPERS
# =============================================================================


def _read_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Read a required CSV result.
    """

    if not path.is_file():

        raise FileNotFoundError(
            f"Required output not found: {path}"
        )

    return pd.read_csv(
        path
    )


def _read_json(
    path: Path,
) -> dict:
    """
    Read a required JSON result.
    """

    if not path.is_file():

        raise FileNotFoundError(
            f"Required output not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def _require_columns(
    df: pd.DataFrame,
    columns: set[str],
) -> None:
    """
    Validate required columns.
    """

    missing = (
        columns
        - set(
            df.columns
        )
    )

    if missing:

        raise KeyError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )


def _shared_limits(
    first,
    second,
) -> tuple[
    float,
    float,
]:
    """
    Create common limits for observed-versus-predicted plots.
    """

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

    values = values[
        np.isfinite(
            values
        )
    ]

    if not len(
        values
    ):

        raise ValueError(
            "Cannot calculate plot limits from empty data."
        )

    lower = float(
        np.min(
            values
        )
    )

    upper = float(
        np.max(
            values
        )
    )

    if upper > lower:

        padding = (
            0.06
            * (
                upper
                - lower
            )
        )

    else:

        padding = 1.0

    return (
        lower
        - padding,
        upper
        + padding,
    )