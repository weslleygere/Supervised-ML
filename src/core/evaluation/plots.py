from __future__ import annotations

import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API
# =============================================================================


def save_evaluation_plots(
    output_dir: str | Path,
) -> None:
    """
    Create publication-oriented figures from a completed experiment.

    Main article
    ------------
    Figure 1
        Observed versus predicted HFI on the independent final test set.

    Supplementary material
    ----------------------
    Figure S1
        Optuna optimization history.

    Figure S2
        Best candidate pipelines during model selection.

    Figure S3
        Residuals across the observed HFI gradient.

    Figure S4
        Model dimensionality versus repeated-CV MAE.

    Only PNG files are generated.
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
    # LOAD OUTPUTS
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

    # =========================================================================
    # MAIN ARTICLE
    # =========================================================================

    main_figures = [
        (
            "figure_01_observed_vs_predicted",
            plot_observed_vs_predicted(
                final_test=final_test,
            ),
        ),
    ]

    # =========================================================================
    # SUPPLEMENTARY MATERIAL
    # =========================================================================

    supplementary_figures = [
        (
            "figure_s01_optuna_history",
            plot_optuna_history(
                trials=trials,
                selected_pipeline=selected_pipeline,
            ),
        ),
        (
            "figure_s02_top_candidates",
            plot_top_candidates(
                trials=trials,
                selected_pipeline=selected_pipeline,
            ),
        ),
        (
            "figure_s03_residuals",
            plot_residuals(
                final_test=final_test,
            ),
        ),
        (
            "figure_s04_dimensionality",
            plot_dimensionality(
                trials=trials,
                selected_pipeline=selected_pipeline,
            ),
        ),
    ]

    # =========================================================================
    # SAVE
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


# =============================================================================
# MAIN FIGURE 1 — OBSERVED VS PREDICTED
# =============================================================================


def plot_observed_vs_predicted(
    final_test: pd.DataFrame,
) -> go.Figure:
    """
    Observed versus predicted HFI on the independent final test set.

    Each marker represents one CapturePointId.

    Point remains the independent grouping unit used for splitting,
    cross-validation and weighting.
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

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

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
                    0.75,
            },
            customdata=np.column_stack(
                [
                    data[
                        "Point"
                    ],
                    data[
                        "CapturePointId"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Predicted HFI: %{y:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # =========================================================================
    # 1:1 LINE
    # =========================================================================

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
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # =========================================================================
    # AXES
    # =========================================================================

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
# SUPPLEMENTARY FIGURE S1 — OPTUNA HISTORY
# =============================================================================


def plot_optuna_history(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> go.Figure:
    """
    Show the optimization history of the CASH search.

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

    # =========================================================================
    # INDIVIDUAL TRIALS
    # =========================================================================

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
                    0.45,
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
                "<br>%{customdata[0]}"
                "<br>%{customdata[1]}"
                "<br>%{customdata[2]}"
                "<br>%{customdata[3]}"
                "<extra></extra>"
            ),
            name="Trials",
        )
    )

    # =========================================================================
    # BEST SO FAR
    # =========================================================================

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
                    2,
            },
            name="Best so far",
            hovertemplate=(
                "Trial %{x}"
                "<br>Best CV MAE: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # =========================================================================
    # SELECTED PIPELINE
    # =========================================================================

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
                        13,

                    "symbol":
                        "star",
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
        title_text="CV MAE",
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
                85,

            "r":
                35,

            "t":
                60,

            "b":
                75,
        },
    )

    _publication_style(
        fig
    )

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURE S2 — TOP CANDIDATES
# =============================================================================


def plot_top_candidates(
    trials: pd.DataFrame,
    selected_pipeline: dict,
    top_n: int = 10,
) -> go.Figure:
    """
    Compare the best complete pipeline candidates.

    Points show mean repeated-CV MAE.

    Horizontal error bars show ±1 SD across CV repeats.
    """

    data = (
        _completed_trials(
            trials
        )
    )

    top_n = min(
        top_n,
        len(
            data
        ),
    )

    data = (
        data.nsmallest(
            top_n,
            "CV_MAE",
        )
        .copy()
    )

    selected_trial = (
        _selected_trial_number(
            selected_pipeline
        )
    )

    data[
        "label"
    ] = (
        data.apply(
            _candidate_label,
            axis=1,
        )
    )

    # Best candidate at top.
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

    # =========================================================================
    # ALL TOP CANDIDATES
    # =========================================================================

    fig.add_trace(
        go.Scatter(
            x=data[
                "CV_MAE"
            ],
            y=data[
                "label"
            ],
            mode="markers",
            marker={
                "size":
                    8,
            },
            error_x={
                "type":
                    "data",

                "array":
                    data[
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
            customdata=data[
                "trial"
            ],
            hovertemplate=(
                "Trial %{customdata}"
                "<br>CV MAE: %{x:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # =========================================================================
    # SELECTED PIPELINE
    # =========================================================================

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
                },
                hovertemplate=(
                    "Selected"
                    "<br>CV MAE: %{x:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(
        title_text="CV MAE",
    )

    fig.update_yaxes(
        title_text="",
    )

    fig.update_layout(
        width=900,
        height=max(
            480,
            95
            + 42
            * top_n,
        ),
        margin={
            "l":
                300,

            "r":
                40,

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
# SUPPLEMENTARY FIGURE S3 — RESIDUALS
# =============================================================================


def plot_residuals(
    final_test: pd.DataFrame,
) -> go.Figure:
    """
    Show residuals across the observed HFI gradient.

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
                    0.75,
            },
            customdata=np.column_stack(
                [
                    data[
                        "Point"
                    ],
                    data[
                        "CapturePointId"
                    ],
                ]
            ),
            hovertemplate=(
                "Point: %{customdata[0]}"
                "<br>CapturePointId: %{customdata[1]}"
                "<br>Observed HFI: %{x:.3f}"
                "<br>Residual: %{y:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1.2,
    )

    fig.update_xaxes(
        title_text="Observed HFI",
    )

    fig.update_yaxes(
        title_text="Residual",
    )

    fig.update_layout(
        width=700,
        height=500,
        margin={
            "l":
                85,

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
# SUPPLEMENTARY FIGURE S4 — DIMENSIONALITY
# =============================================================================


def plot_dimensionality(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> go.Figure:
    """
    Show model dimensionality versus CV performance.

    This figure is descriptive only because Optuna samples candidate
    configurations adaptively.
    """

    data = (
        _completed_trials(
            trials
        )
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

    fig = go.Figure()

    # =========================================================================
    # CANDIDATES
    # =========================================================================

    fig.add_trace(
        go.Scatter(
            x=data[
                "model_feature_count"
            ],
            y=data[
                "CV_MAE"
            ],
            mode="markers",
            marker={
                "size":
                    7,

                "opacity":
                    0.5,
            },
            customdata=np.column_stack(
                [
                    data[
                        "trial"
                    ],
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
                "Trial %{customdata[0]}"
                "<br>%{customdata[1]}"
                "<br>%{customdata[2]}"
                "<br>%{customdata[3]}"
                "<br>%{customdata[4]}"
                "<br>Features: %{x}"
                "<br>CV MAE: %{y:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # =========================================================================
    # SELECTED PIPELINE
    # =========================================================================

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
                        14,

                    "symbol":
                        "star",
                },
                showlegend=False,
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
        title_text="CV MAE",
    )

    fig.update_layout(
        width=700,
        height=500,
        margin={
            "l":
                85,

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
# TEST DATA PREPARATION
# =============================================================================


def _prepare_test_predictions(
    final_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare CapturePointId-level independent-test predictions.

    Different CapturePointIds belonging to one Point may have different
    observed HFI values, so they are retained as separate observations.
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


def _candidate_label(
    row: pd.Series,
) -> str:
    """
    Compact label for one complete candidate pipeline.
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

    No internal title is added.

    The manuscript caption should contain the interpretation and
    methodological explanation.
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
    Compact model-family names for figures.
    """

    mapping = {
        "RIDGE_REGRESSION":
            "Ridge",

        "ELASTIC_NET":
            "Elastic Net",

        "SVR":
            "SVR",

        "RANDOM_FOREST":
            "Random Forest",

        "EXTRA_TREES":
            "Extra Trees",

        "GRADIENT_BOOSTING":
            "Gradient Boosting",

        "XGBOOST":
            "XGBoost",
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
    Compact acoustic-representation label.
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
    Compact aggregation label.
    """

    mapping = {
        "mean":
            "Mean",

        "mean_std":
            "Mean+SD",

        "hierarchical":
            "Hierarchical",
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
            "No PCA",

        "pca":
            "PCA",
    }

    text = str(
        value
    )

    return mapping.get(
        text,
        text,
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
    Create common x/y limits for observed-versus-predicted plots.
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