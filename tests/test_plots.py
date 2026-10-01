import json

import numpy as np
import pandas as pd
import pytest

from src.core.evaluation import plots

from src.core.evaluation.plots import (
    _completed_trials,
    _point_balanced_ecdf,
    plot_cv_prediction_stability,
    plot_cv_repeat_stability,
    plot_dimensionality,
    plot_hfi_partition_ecdf,
    plot_observed_vs_predicted,
    plot_optuna_history,
    plot_pointwise_baseline_gain,
    plot_residuals,
    plot_search_coverage,
    plot_top_candidates,
    save_evaluation_plots,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def final_test() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P1",
                "P2",
                "P3",
            ],

            "CapturePointId": [
                "B1",
                "B2",
                "B3",
                "B4",
            ],

            "observed_HFI": [
                0.0,
                10.0,
                20.0,
                40.0,
            ],

            "predicted_HFI": [
                2.0,
                8.0,
                25.0,
                35.0,
            ],
        }
    )


@pytest.fixture
def point_errors() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P2",
                "P3",
            ],

            "n_capture_points": [
                2,
                1,
                1,
            ],

            "model_MAE": [
                2.0,
                5.0,
                5.0,
            ],

            "mean_baseline_MAE": [
                10.0,
                3.0,
                15.0,
            ],

            "median_baseline_MAE": [
                8.0,
                4.0,
                12.0,
            ],

            "MAE_gain_vs_mean_baseline": [
                8.0,
                -2.0,
                10.0,
            ],

            "MAE_gain_vs_median_baseline": [
                6.0,
                -1.0,
                7.0,
            ],
        }
    )


@pytest.fixture
def cv_repeat_metrics() -> pd.DataFrame:

    rows = []

    values = {
        "selected_pipeline": [
            4.0,
            5.0,
            4.5,
        ],

        "mean_baseline": [
            8.0,
            8.5,
            8.2,
        ],

        "median_baseline": [
            9.0,
            9.5,
            9.2,
        ],
    }

    for method, maes in (
        values.items()
    ):

        for repeat, mae in enumerate(
            maes,
            start=1,
        ):

            rows.append(
                {
                    "repeat":
                        repeat,

                    "method":
                        method,

                    "MAE":
                        mae,

                    "RMSE":
                        mae
                        + 1.0,

                    "R2":
                        0.5,
                }
            )

    return pd.DataFrame(
        rows
    )


@pytest.fixture
def cv_stability() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P2",
                "P3",
            ],

            "CapturePointId": [
                "B1",
                "B2",
                "B3",
            ],

            "observed_HFI": [
                10.0,
                20.0,
                30.0,
            ],

            "n_repeats": [
                5,
                5,
                5,
            ],

            "predicted_HFI_mean": [
                11.0,
                18.0,
                32.0,
            ],

            "predicted_HFI_sd": [
                1.0,
                2.0,
                3.0,
            ],

            "prediction_range": [
                3.0,
                6.0,
                9.0,
            ],

            "absolute_error_mean": [
                1.0,
                2.0,
                2.0,
            ],

            "absolute_error_sd": [
                0.5,
                1.0,
                1.5,
            ],

            "residual_mean": [
                -1.0,
                2.0,
                -2.0,
            ],

            "residual_sd": [
                0.5,
                1.0,
                1.5,
            ],
        }
    )


@pytest.fixture
def split_manifest() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P1",
                "P2",
                "P3",
                "P4",
            ],

            "CapturePointId": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
            ],

            "meanHFI": [
                0.0,
                10.0,
                40.0,
                20.0,
                60.0,
            ],

            "split": [
                "development",
                "development",
                "development",
                "test",
                "test",
            ],
        }
    )


@pytest.fixture
def trials() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "trial":
                    0,

                "state":
                    "COMPLETE",

                "model":
                    "SVR",

                "feature_set":
                    "embeddings",

                "aggregation":
                    "mean",

                "reduction":
                    "pca",

                "CV_MAE":
                    5.0,

                "CV_MAE_std":
                    0.5,

                "model_feature_count":
                    16,
            },

            {
                "trial":
                    1,

                "state":
                    "COMPLETE",

                "model":
                    "RIDGE_REGRESSION",

                "feature_set":
                    "both",

                "aggregation":
                    "dawn_profile",

                "reduction":
                    "supervised_selection",

                "CV_MAE":
                    3.0,

                "CV_MAE_std":
                    0.3,

                "model_feature_count":
                    40,
            },

            {
                "trial":
                    2,

                "state":
                    "COMPLETE",

                "model":
                    "RANDOM_FOREST",

                "feature_set":
                    "indices",

                "aggregation":
                    "hierarchical",

                "reduction":
                    "none",

                "CV_MAE":
                    4.0,

                "CV_MAE_std":
                    0.4,

                "model_feature_count":
                    180,
            },

            {
                "trial":
                    3,

                "state":
                    "FAIL",

                "model":
                    "ELASTIC_NET",

                "feature_set":
                    "indices",

                "aggregation":
                    "mean",

                "reduction":
                    "none",

                "CV_MAE":
                    np.nan,

                "CV_MAE_std":
                    np.nan,

                "model_feature_count":
                    60,
            },

            {
                "trial":
                    4,

                "state":
                    "PRUNED",

                "model":
                    "CATBOOST",

                "feature_set":
                    "both",

                "aggregation":
                    "dawn_trend",

                "reduction":
                    "pca",

                "CV_MAE":
                    np.nan,

                "CV_MAE_std":
                    np.nan,

                "model_feature_count":
                    40,
            },
        ]
    )


@pytest.fixture
def selected_pipeline() -> dict:

    return {
        "selected_trial":
            1,

        "pipeline": {
            "model":
                "RIDGE_REGRESSION",

            "feature_set":
                "both",

            "aggregation":
                "dawn_profile",

            "reduction":
                "supervised_selection",

            "model_params": {
                "alpha":
                    1.0,
            },
        },
    }


@pytest.fixture
def top_candidates() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "candidate_rank": [
                1,
                2,
                3,
            ],

            "trial": [
                1,
                2,
                0,
            ],

            "selected": [
                True,
                False,
                False,
            ],

            "model": [
                "RIDGE_REGRESSION",
                "RANDOM_FOREST",
                "SVR",
            ],

            "feature_set": [
                "both",
                "indices",
                "embeddings",
            ],

            "aggregation": [
                "dawn_profile",
                "hierarchical",
                "mean",
            ],

            "reduction": [
                "supervised_selection",
                "none",
                "pca",
            ],

            "model_feature_count": [
                40,
                180,
                16,
            ],

            "CV_MAE": [
                3.0,
                4.0,
                5.0,
            ],

            "CV_MAE_std": [
                0.3,
                0.4,
                0.5,
            ],
        }
    )


@pytest.fixture
def search_components() -> pd.DataFrame:

    rows = []

    groups = {
        "model": {
            "RIDGE_REGRESSION":
                4,

            "SVR":
                3,
        },

        "feature_set": {
            "indices":
                3,

            "both":
                4,
        },

        "aggregation": {
            "mean":
                3,

            "dawn_profile":
                4,
        },

        "reduction": {
            "none":
                2,

            "supervised_selection":
                5,
        },
    }

    for component, levels in (
        groups.items()
    ):

        total = sum(
            levels.values()
        )

        for level, count in (
            levels.items()
        ):

            rows.append(
                {
                    "component":
                        component,

                    "level":
                        level,

                    "n_trials":
                        count,

                    "sampling_fraction":
                        count
                        / total,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN FIGURE 1
# =============================================================================


def test_observed_vs_predicted_keeps_capture_points_separate(
    final_test: pd.DataFrame,
) -> None:
    """
    The final-test scatter must contain one marker per CapturePointId rather
    than collapsing observations belonging to the same Point.
    """

    figure = (
        plot_observed_vs_predicted(
            final_test
        )
    )

    points = figure.data[
        0
    ]

    np.testing.assert_allclose(
        points.x,
        final_test[
            "observed_HFI"
        ].to_numpy(),
    )

    np.testing.assert_allclose(
        points.y,
        final_test[
            "predicted_HFI"
        ].to_numpy(),
    )

    assert len(
        points.x
    ) == len(
        final_test
    )

    assert len(
        figure.data
    ) == 2

    identity = figure.data[
        1
    ]

    np.testing.assert_allclose(
        identity.x,
        identity.y,
    )


# =============================================================================
# MAIN FIGURE 2
# =============================================================================


def test_pointwise_gain_uses_point_level_errors(
    point_errors: pd.DataFrame,
) -> None:
    """
    The baseline-gain figure must use the already aggregated independent
    Point-level error table.
    """

    figure = (
        plot_pointwise_baseline_gain(
            point_errors
        )
    )

    assert len(
        figure.data
    ) == 2

    mean_trace = figure.data[
        0
    ]

    median_trace = figure.data[
        1
    ]

    expected = (
        point_errors
        .sort_values(
            "MAE_gain_vs_mean_baseline"
        )
    )

    np.testing.assert_allclose(
        mean_trace.x,
        expected[
            "MAE_gain_vs_mean_baseline"
        ].to_numpy(),
    )

    np.testing.assert_allclose(
        median_trace.x,
        expected[
            "MAE_gain_vs_median_baseline"
        ].to_numpy(),
    )

    assert list(
        mean_trace.y
    ) == expected[
        "Point"
    ].tolist()


# =============================================================================
# MAIN FIGURE 3
# =============================================================================


def test_cv_repeat_stability_contains_all_methods_and_repeats(
    cv_repeat_metrics: pd.DataFrame,
) -> None:
    """
    The stability figure must compare the same complete CV repetitions for
    the selected pipeline and both reference baselines.
    """

    figure = (
        plot_cv_repeat_stability(
            cv_repeat_metrics
        )
    )

    assert len(
        figure.data
    ) == 3

    names = {
        trace.name
        for trace
        in figure.data
    }

    assert names == {
        "Selected pipeline",
        "Mean baseline",
        "Median baseline",
    }

    for trace in figure.data:

        assert list(
            trace.x
        ) == [
            1,
            2,
            3,
        ]


# =============================================================================
# RESIDUAL DIAGNOSTIC
# =============================================================================


def test_residual_plot_uses_observed_minus_predicted(
    final_test: pd.DataFrame,
) -> None:
    """
    Residual sign must follow the project convention:

        observed - predicted
    """

    figure = (
        plot_residuals(
            final_test
        )
    )

    expected = (
        final_test[
            "observed_HFI"
        ]
        - final_test[
            "predicted_HFI"
        ]
    )

    np.testing.assert_allclose(
        figure.data[
            0
        ].y,
        expected.to_numpy(),
    )


# =============================================================================
# CV PREDICTION STABILITY
# =============================================================================


def test_cv_prediction_stability_uses_between_repeat_sd(
    cv_stability: pd.DataFrame,
) -> None:
    """
    Error bars must represent prediction SD across repeated CV rather than
    observational error or residual magnitude.
    """

    figure = (
        plot_cv_prediction_stability(
            cv_stability
        )
    )

    prediction_trace = (
        figure.data[
            0
        ]
    )

    np.testing.assert_allclose(
        prediction_trace.y,
        cv_stability[
            "predicted_HFI_mean"
        ].to_numpy(),
    )

    np.testing.assert_allclose(
        prediction_trace.error_y.array,
        cv_stability[
            "predicted_HFI_sd"
        ].to_numpy(),
    )


# =============================================================================
# POINT-BALANCED ECDF
# =============================================================================


def test_point_balanced_ecdf_gives_each_point_equal_total_weight() -> None:
    """
    Two CapturePointIds under P1 must jointly contribute the same total ECDF
    weight as the single CapturePointId under P2.
    """

    data = pd.DataFrame(
        {
            "Point": [
                "P1",
                "P1",
                "P2",
            ],

            "CapturePointId": [
                "B1",
                "B2",
                "B3",
            ],

            "meanHFI": [
                0.0,
                10.0,
                40.0,
            ],
        }
    )

    result = (
        _point_balanced_ecdf(
            data
        )
    )

    np.testing.assert_allclose(
        result[
            "weight"
        ].to_numpy(),
        np.array(
            [
                0.5,
                0.5,
                1.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        result[
            "cumulative_weight"
        ].to_numpy(),
        np.array(
            [
                0.25,
                0.50,
                1.00,
            ]
        ),
    )


def test_partition_ecdf_contains_development_and_test(
    split_manifest: pd.DataFrame,
) -> None:

    figure = (
        plot_hfi_partition_ecdf(
            split_manifest
        )
    )

    assert len(
        figure.data
    ) == 2

    names = {
        trace.name
        for trace
        in figure.data
    }

    assert names == {
        "Development",
        "Final test",
    }

    for trace in figure.data:

        assert trace.y[
            -1
        ] == pytest.approx(
            1.0
        )


# =============================================================================
# OPTUNA FILTERING AND HISTORY
# =============================================================================


def test_completed_trials_excludes_failed_pruned_and_nonfinite(
    trials: pd.DataFrame,
) -> None:

    data = (
        _completed_trials(
            trials
        )
    )

    assert data[
        "trial"
    ].tolist() == [
        0,
        1,
        2,
    ]

    assert set(
        data[
            "state"
        ]
    ) == {
        "COMPLETE"
    }


def test_optuna_history_tracks_cumulative_best(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> None:

    figure = (
        plot_optuna_history(
            trials=trials,
            selected_pipeline=(
                selected_pipeline
            ),
        )
    )

    best_trace = next(
        trace
        for trace in figure.data
        if trace.name
        == "Best so far"
    )

    np.testing.assert_allclose(
        best_trace.y,
        np.array(
            [
                5.0,
                3.0,
                3.0,
            ]
        ),
    )


# =============================================================================
# TOP CANDIDATES
# =============================================================================


def test_top_candidates_highlights_selected_pipeline(
    top_candidates: pd.DataFrame,
) -> None:

    figure = (
        plot_top_candidates(
            candidates=(
                top_candidates
            ),
            top_n=3,
        )
    )

    selected_traces = [
        trace
        for trace in figure.data
        if (
            getattr(
                trace.marker,
                "symbol",
                None,
            )
            == "star"
        )
    ]

    assert len(
        selected_traces
    ) == 1

    selected = (
        selected_traces[
            0
        ]
    )

    assert selected.x[
        0
    ] == pytest.approx(
        3.0
    )


# =============================================================================
# DIMENSIONALITY
# =============================================================================


def test_dimensionality_figure_contains_reduction_groups_and_selected_trial(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> None:

    figure = (
        plot_dimensionality(
            trials=trials,
            selected_pipeline=(
                selected_pipeline
            ),
        )
    )

    names = {
        trace.name
        for trace
        in figure.data
    }

    assert {
        "No reduction",
        "PCA",
        "Supervised selection",
        "Selected",
    }.issubset(
        names
    )

    assert (
        figure.layout.xaxis.type
        == "log"
    )


# =============================================================================
# SEARCH COVERAGE
# =============================================================================


def test_search_coverage_visualizes_sampling_not_performance(
    search_components: pd.DataFrame,
) -> None:
    """
    Search coverage must use number of sampled completed trials rather than
    CV performance as the bar height.
    """

    figure = (
        plot_search_coverage(
            search_components
        )
    )

    assert len(
        figure.data
    ) == 4

    observed_counts = []

    for trace in figure.data:

        observed_counts.extend(
            list(
                trace.y
            )
        )

    expected_counts = (
        search_components[
            "n_trials"
        ]
        .tolist()
    )

    assert sorted(
        observed_counts
    ) == sorted(
        expected_counts
    )


# =============================================================================
# END-TO-END FIGURE GENERATION
# =============================================================================


def test_save_evaluation_plots_builds_all_figures_and_manifest(
    tmp_path,
    monkeypatch,
    final_test: pd.DataFrame,
    point_errors: pd.DataFrame,
    cv_repeat_metrics: pd.DataFrame,
    cv_stability: pd.DataFrame,
    split_manifest: pd.DataFrame,
    trials: pd.DataFrame,
    selected_pipeline: dict,
    top_candidates: pd.DataFrame,
    search_components: pd.DataFrame,
) -> None:
    """
    The plotting pipeline must construct all main and supplementary figures
    from the saved experiment outputs.

    Image rendering itself is mocked so the test does not depend on Kaleido.
    """

    analysis_dir = (
        tmp_path
        / "analysis"
    )

    reproducibility_dir = (
        tmp_path
        / "reproducibility"
    )

    analysis_dir.mkdir(
        parents=True
    )

    reproducibility_dir.mkdir(
        parents=True
    )

    trials.to_csv(
        tmp_path
        / "model_selection_trials.csv",
        index=False,
    )

    final_test.to_csv(
        tmp_path
        / "final_test_predictions.csv",
        index=False,
    )

    split_manifest.to_csv(
        reproducibility_dir
        / "data_split.csv",
        index=False,
    )

    point_errors.to_csv(
        analysis_dir
        / "final_test_point_errors.csv",
        index=False,
    )

    cv_repeat_metrics.to_csv(
        analysis_dir
        / "selected_cv_repeat_metrics.csv",
        index=False,
    )

    cv_stability.to_csv(
        analysis_dir
        / "selected_cv_prediction_stability.csv",
        index=False,
    )

    top_candidates.to_csv(
        analysis_dir
        / "top_pipeline_candidates.csv",
        index=False,
    )

    search_components.to_csv(
        analysis_dir
        / "search_component_descriptive.csv",
        index=False,
    )

    with open(
        tmp_path
        / "selected_pipeline.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            selected_pipeline,
            file,
        )

    saved_paths = []

    def fake_save_figure(
        figure,
        path,
    ):

        saved_paths.append(
            path
        )

    monkeypatch.setattr(
        plots,
        "_save_figure",
        fake_save_figure,
    )

    save_evaluation_plots(
        tmp_path
    )

    assert len(
        saved_paths
    ) == 10

    names = {
        path.name
        for path in saved_paths
    }

    assert names == {
        "figure_01_observed_vs_predicted",
        "figure_02_pointwise_baseline_gain",
        "figure_03_cv_repeat_stability",
        "figure_s01_residuals",
        "figure_s02_cv_prediction_stability",
        "figure_s03_hfi_partition_ecdf",
        "figure_s04_optuna_history",
        "figure_s05_top_candidates",
        "figure_s06_dimensionality",
        "figure_s07_search_coverage",
    }

    manifest_path = (
        tmp_path
        / "figures"
        / "figure_manifest.csv"
    )

    assert manifest_path.is_file()

    manifest = pd.read_csv(
        manifest_path
    )

    assert len(
        manifest
    ) == 10

    assert (
        manifest[
            "figure"
        ].nunique()
        == 10
    )

    assert set(
        manifest[
            "section"
        ]
    ) == {
        "main",
        "supplementary",
    }

    adaptive = manifest[
        manifest[
            "figure"
        ]
        .isin(
            [
                "figure_s05_top_candidates",
                "figure_s06_dimensionality",
            ]
        )
    ]

    assert all(
        adaptive[
            "interpretation"
        ]
        == (
            "adaptive_search_descriptive_not_formal_comparison"
        )
    )