import json

import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.evaluation.result_analysis import (
    build_cv_prediction_stability,
    build_dataset_partition_summary,
    build_final_test_baseline_comparison,
    build_final_test_point_errors,
    build_search_component_descriptive,
    build_search_status_summary,
    build_selected_cv_repeat_metrics,
    build_selected_pipeline_summary,
    build_top_pipeline_candidates,
    save_result_analysis,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def schema() -> Schema:
    return Schema(
        target="meanHFI",
        group="Point",
        bag="CapturePointId",
        audio="Audio_Name",
        datetime="Datetime",
        index_prefixes=("ACI",),
        embedding="Embedding",
    )


@pytest.fixture
def cv_predictions() -> pd.DataFrame:
    """
    Two complete CV repetitions.

    P1 contains two CapturePointIds, while P2 and P3 contain one each.
    Metrics must therefore be Point-balanced rather than row-balanced.
    """

    rows = []

    observed = np.array(
        [0.0, 10.0, 20.0, 40.0]
    )

    points = [
        "P1",
        "P1",
        "P2",
        "P3",
    ]

    bags = [
        "B1",
        "B2",
        "B3",
        "B4",
    ]

    predictions = {
        1:
            np.array(
                [0.0, 10.0, 20.0, 30.0]
            ),

        2:
            np.array(
                [10.0, 20.0, 20.0, 40.0]
            ),
    }

    for repeat, predicted in predictions.items():

        for i in range(
            len(
                observed
            )
        ):

            residual = (
                observed[i]
                - predicted[i]
            )

            mean_prediction = 20.0

            mean_residual = (
                observed[i]
                - mean_prediction
            )

            median_prediction = 15.0

            median_residual = (
                observed[i]
                - median_prediction
            )

            rows.append(
                {
                    "repeat":
                        repeat,

                    "fold":
                        1
                        if i < 2
                        else 2,

                    "Point":
                        points[i],

                    "CapturePointId":
                        bags[i],

                    "observed_HFI":
                        observed[i],

                    "predicted_HFI":
                        predicted[i],

                    "mean_baseline_prediction":
                        mean_prediction,

                    "median_baseline_prediction":
                        median_prediction,

                    "residual":
                        residual,

                    "absolute_error":
                        abs(
                            residual
                        ),

                    "squared_error":
                        residual
                        ** 2,

                    "mean_baseline_residual":
                        mean_residual,

                    "mean_baseline_absolute_error":
                        abs(
                            mean_residual
                        ),

                    "mean_baseline_squared_error":
                        mean_residual
                        ** 2,

                    "median_baseline_residual":
                        median_residual,

                    "median_baseline_absolute_error":
                        abs(
                            median_residual
                        ),

                    "median_baseline_squared_error":
                        median_residual
                        ** 2,
                }
            )

    return pd.DataFrame(
        rows
    )


@pytest.fixture
def final_test() -> pd.DataFrame:
    """
    Independent-test predictions with multiple CapturePointIds under P1.
    """

    df = pd.DataFrame(
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
                0.0,
                20.0,
                18.0,
                35.0,
            ],

            "mean_baseline_prediction": [
                20.0,
                20.0,
                20.0,
                20.0,
            ],

            "median_baseline_prediction": [
                15.0,
                15.0,
                15.0,
                15.0,
            ],
        }
    )

    df[
        "residual"
    ] = (
        df[
            "observed_HFI"
        ]
        - df[
            "predicted_HFI"
        ]
    )

    df[
        "absolute_error"
    ] = (
        df[
            "residual"
        ].abs()
    )

    df[
        "squared_error"
    ] = (
        df[
            "residual"
        ]
        ** 2
    )

    for baseline in (
        "mean",
        "median",
    ):

        residual = (
            df[
                "observed_HFI"
            ]
            - df[
                f"{baseline}_baseline_prediction"
            ]
        )

        df[
            f"{baseline}_baseline_residual"
        ] = residual

        df[
            f"{baseline}_baseline_absolute_error"
        ] = (
            residual.abs()
        )

        df[
            f"{baseline}_baseline_squared_error"
        ] = (
            residual
            ** 2
        )

    return df


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
def performance_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage":
                    "development_cv",

                "method":
                    "selected_pipeline",

                "MAE":
                    3.0,

                "MAE_SD":
                    0.3,

                "RMSE":
                    4.0,

                "RMSE_SD":
                    0.4,

                "R2":
                    0.70,

                "R2_SD":
                    0.05,

                "n_points":
                    50,

                "n_capture_points":
                    90,

                "n_repeats":
                    5,
            },

            {
                "stage":
                    "development_cv",

                "method":
                    "mean_baseline",

                "MAE":
                    5.0,

                "MAE_SD":
                    0.2,

                "RMSE":
                    6.0,

                "RMSE_SD":
                    0.3,

                "R2":
                    0.0,

                "R2_SD":
                    0.0,

                "n_points":
                    50,

                "n_capture_points":
                    90,

                "n_repeats":
                    5,
            },

            {
                "stage":
                    "development_cv",

                "method":
                    "median_baseline",

                "MAE":
                    6.0,

                "MAE_SD":
                    0.2,

                "RMSE":
                    7.0,

                "RMSE_SD":
                    0.3,

                "R2":
                    -0.1,

                "R2_SD":
                    0.0,

                "n_points":
                    50,

                "n_capture_points":
                    90,

                "n_repeats":
                    5,
            },

            {
                "stage":
                    "final_test",

                "method":
                    "selected_pipeline",

                "MAE":
                    4.0,

                "MAE_SD":
                    np.nan,

                "RMSE":
                    5.0,

                "RMSE_SD":
                    np.nan,

                "R2":
                    0.60,

                "R2_SD":
                    np.nan,

                "n_points":
                    13,

                "n_capture_points":
                    28,

                "n_repeats":
                    1,
            },

            {
                "stage":
                    "final_test",

                "method":
                    "mean_baseline",

                "MAE":
                    6.0,

                "MAE_SD":
                    np.nan,

                "RMSE":
                    7.0,

                "RMSE_SD":
                    np.nan,

                "R2":
                    0.0,

                "R2_SD":
                    np.nan,

                "n_points":
                    13,

                "n_capture_points":
                    28,

                "n_repeats":
                    1,
            },

            {
                "stage":
                    "final_test",

                "method":
                    "median_baseline",

                "MAE":
                    7.0,

                "MAE_SD":
                    np.nan,

                "RMSE":
                    8.0,

                "RMSE_SD":
                    np.nan,

                "R2":
                    -0.1,

                "R2_SD":
                    np.nan,

                "n_points":
                    13,

                "n_capture_points":
                    28,

                "n_repeats":
                    1,
            },
        ]
    )


@pytest.fixture
def selected_pipeline() -> dict:
    return {
        "selected_trial":
            1,

        "selection_objective": {
            "metric":
                "point_balanced_MAE",

            "CV_MAE":
                3.0,

            "CV_MAE_std":
                0.3,
        },

        "pipeline": {
            "model":
                "RIDGE_REGRESSION",

            "feature_set":
                "both",

            "aggregation":
                "dawn_profile",

            "reduction":
                "supervised_selection",

            "pca_indices_components":
                None,

            "pca_embeddings_components":
                None,

            "selection_indices_features":
                8,

            "selection_embeddings_features":
                32,

            "model_params": {
                "alpha":
                    1.0,
            },
        },

        "raw_feature_count":
            2288,

        "model_feature_count":
            40,

        "final_fit_time":
            0.25,

        "final_prediction_time":
            0.01,
    }


@pytest.fixture
def trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial":
                    0,

                "state":
                    "COMPLETE",

                "objective":
                    4.0,

                "model":
                    "SVR",

                "feature_set":
                    "embeddings",

                "aggregation":
                    "mean",

                "reduction":
                    "pca",

                "pca_indices_components":
                    np.nan,

                "pca_embeddings_components":
                    16,

                "selection_indices_features":
                    np.nan,

                "selection_embeddings_features":
                    np.nan,

                "model_params":
                    '{"C": 10.0}',

                "CV_MAE":
                    4.0,

                "CV_MAE_std":
                    0.4,

                "CV_RMSE":
                    5.0,

                "CV_RMSE_std":
                    0.4,

                "CV_R2":
                    0.5,

                "CV_R2_std":
                    0.05,

                "raw_feature_count":
                    512,

                "model_feature_count":
                    16,

                "feature_to_point_ratio":
                    0.4,

                "mean_fit_time":
                    0.10,

                "duration_seconds":
                    2.0,
            },

            {
                "trial":
                    1,

                "state":
                    "COMPLETE",

                "objective":
                    3.0,

                "model":
                    "RIDGE_REGRESSION",

                "feature_set":
                    "both",

                "aggregation":
                    "dawn_profile",

                "reduction":
                    "supervised_selection",

                "pca_indices_components":
                    np.nan,

                "pca_embeddings_components":
                    np.nan,

                "selection_indices_features":
                    8,

                "selection_embeddings_features":
                    32,

                "model_params":
                    '{"alpha": 1.0}',

                "CV_MAE":
                    3.0,

                "CV_MAE_std":
                    0.3,

                "CV_RMSE":
                    4.0,

                "CV_RMSE_std":
                    0.3,

                "CV_R2":
                    0.7,

                "CV_R2_std":
                    0.04,

                "raw_feature_count":
                    2288,

                "model_feature_count":
                    40,

                "feature_to_point_ratio":
                    1.0,

                "mean_fit_time":
                    0.05,

                "duration_seconds":
                    1.0,
            },

            {
                "trial":
                    2,

                "state":
                    "COMPLETE",

                "objective":
                    5.0,

                "model":
                    "SVR",

                "feature_set":
                    "indices",

                "aggregation":
                    "hierarchical",

                "reduction":
                    "none",

                "pca_indices_components":
                    np.nan,

                "pca_embeddings_components":
                    np.nan,

                "selection_indices_features":
                    np.nan,

                "selection_embeddings_features":
                    np.nan,

                "model_params":
                    '{"C": 1.0}',

                "CV_MAE":
                    5.0,

                "CV_MAE_std":
                    0.5,

                "CV_RMSE":
                    6.0,

                "CV_RMSE_std":
                    0.5,

                "CV_R2":
                    0.3,

                "CV_R2_std":
                    0.06,

                "raw_feature_count":
                    180,

                "model_feature_count":
                    180,

                "feature_to_point_ratio":
                    4.5,

                "mean_fit_time":
                    0.08,

                "duration_seconds":
                    1.5,
            },

            {
                "trial":
                    3,

                "state":
                    "FAIL",

                "objective":
                    np.nan,

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
            },

            {
                "trial":
                    4,

                "state":
                    "PRUNED",

                "objective":
                    np.nan,

                "model":
                    np.nan,

                "feature_set":
                    np.nan,

                "aggregation":
                    np.nan,

                "reduction":
                    np.nan,

                "CV_MAE":
                    np.nan,
            },
        ]
    )


# =============================================================================
# CV REPEAT METRICS
# =============================================================================


def test_cv_repeat_metrics_are_point_balanced(
    cv_predictions: pd.DataFrame,
    schema: Schema,
) -> None:

    result = build_selected_cv_repeat_metrics(
        predictions=cv_predictions,
        schema=schema,
    )

    selected_repeat_1 = result[
        (
            result[
                "repeat"
            ]
            == 1
        )
        & (
            result[
                "method"
            ]
            == "selected_pipeline"
        )
    ].iloc[
        0
    ]

    # P1 errors: 0, 0
    # P2 error: 0
    # P3 error: 10
    #
    # Point-balanced MAE:
    #
    # (0 + 0 + 10) / 3 = 3.333...

    assert selected_repeat_1[
        "MAE"
    ] == pytest.approx(
        10.0
        / 3.0
    )

    assert selected_repeat_1[
        "n_points"
    ] == 3

    assert selected_repeat_1[
        "n_capture_points"
    ] == 4

    assert set(
        result[
            "method"
        ]
    ) == {
        "selected_pipeline",
        "mean_baseline",
        "median_baseline",
    }


# =============================================================================
# CV PREDICTION STABILITY
# =============================================================================


def test_cv_prediction_stability_tracks_between_repeat_variation(
    cv_predictions: pd.DataFrame,
    schema: Schema,
) -> None:

    result = build_cv_prediction_stability(
        predictions=cv_predictions,
        schema=schema,
    )

    b1 = result[
        result[
            "CapturePointId"
        ]
        == "B1"
    ].iloc[
        0
    ]

    assert b1[
        "n_repeats"
    ] == 2

    assert b1[
        "predicted_HFI_mean"
    ] == pytest.approx(
        5.0
    )

    assert b1[
        "predicted_HFI_sd"
    ] == pytest.approx(
        np.std(
            [
                0.0,
                10.0,
            ],
            ddof=1,
        )
    )

    assert b1[
        "prediction_range"
    ] == pytest.approx(
        10.0
    )

    assert b1[
        "residual_mean"
    ] == pytest.approx(
        -5.0
    )


# =============================================================================
# FINAL TEST — CAPTURE LEVEL
# =============================================================================


def test_capture_level_baseline_gain_has_correct_direction(
    final_test: pd.DataFrame,
    schema: Schema,
) -> None:

    result = build_final_test_baseline_comparison(
        final_test=final_test,
        schema=schema,
    )

    b1 = result[
        result[
            "CapturePointId"
        ]
        == "B1"
    ].iloc[
        0
    ]

    # Model error = 0
    # Mean baseline error = 20
    #
    # gain = baseline error - model error = +20

    assert b1[
        "absolute_error_gain_vs_mean_baseline"
    ] == pytest.approx(
        20.0
    )

    assert bool(
        b1[
            "model_better_than_mean_baseline"
        ]
    )

    b3 = result[
        result[
            "CapturePointId"
        ]
        == "B3"
    ].iloc[
        0
    ]

    # Model error = 2
    # Mean baseline error = 0

    assert b3[
        "absolute_error_gain_vs_mean_baseline"
    ] == pytest.approx(
        -2.0
    )

    assert not bool(
        b3[
            "model_better_than_mean_baseline"
        ]
    )


# =============================================================================
# FINAL TEST — POINT LEVEL
# =============================================================================


def test_point_level_errors_are_computed_after_capture_errors(
    final_test: pd.DataFrame,
    schema: Schema,
) -> None:

    result = build_final_test_point_errors(
        final_test=final_test,
        schema=schema,
    )

    p1 = result[
        result[
            "Point"
        ]
        == "P1"
    ].iloc[
        0
    ]

    # Capture-level model absolute errors for P1:
    #
    # B1 = 0
    # B2 = 10
    #
    # Point MAE = 5

    assert p1[
        "model_MAE"
    ] == pytest.approx(
        5.0
    )

    # Mean baseline errors:
    #
    # B1 = 20
    # B2 = 10
    #
    # Point MAE = 15

    assert p1[
        "mean_baseline_MAE"
    ] == pytest.approx(
        15.0
    )

    assert p1[
        "MAE_gain_vs_mean_baseline"
    ] == pytest.approx(
        10.0
    )

    assert p1[
        "n_capture_points"
    ] == 2

    # Observed/predicted HFI must not be collapsed across captures.
    assert (
        "observed_HFI"
        not in result.columns
    )

    assert (
        "predicted_HFI"
        not in result.columns
    )


# =============================================================================
# DATASET PARTITION
# =============================================================================


def test_dataset_partition_summary_reports_point_balanced_hfi(
    split_manifest: pd.DataFrame,
    schema: Schema,
) -> None:

    result = build_dataset_partition_summary(
        split_manifest=split_manifest,
        schema=schema,
    )

    development = result[
        result[
            "split"
        ]
        == "development"
    ].iloc[
        0
    ]

    assert development[
        "n_points"
    ] == 2

    assert development[
        "n_capture_points"
    ] == 3

    assert development[
        "HFI_mean_capture"
    ] == pytest.approx(
        (
            0.0
            + 10.0
            + 40.0
        )
        / 3.0
    )

    # P1 contributes its two CapturePointIds jointly:
    #
    # P1 mean contribution = 5
    # P2 contribution = 40
    #
    # Point-balanced mean = (5 + 40) / 2 = 22.5

    assert development[
        "HFI_mean_point_balanced"
    ] == pytest.approx(
        22.5
    )

    assert development[
        "HFI_mean_point_balanced"
    ] != pytest.approx(
        development[
            "HFI_mean_capture"
        ]
    )


# =============================================================================
# SEARCH STATUS
# =============================================================================


def test_search_status_summary_counts_all_trial_states(
    trials: pd.DataFrame,
) -> None:

    result = build_search_status_summary(
        trials
    )

    counts = result.set_index(
        "state"
    )[
        "n_trials"
    ].to_dict()

    assert counts[
        "COMPLETE"
    ] == 3

    assert counts[
        "FAIL"
    ] == 1

    assert counts[
        "PRUNED"
    ] == 1

    assert result[
        "fraction"
    ].sum() == pytest.approx(
        1.0
    )


# =============================================================================
# TOP CANDIDATES
# =============================================================================


def test_top_pipeline_candidates_are_ranked_by_cv_mae(
    trials: pd.DataFrame,
    selected_pipeline: dict,
) -> None:

    result = build_top_pipeline_candidates(
        trials=trials,
        selected_pipeline=selected_pipeline,
        top_n=3,
    )

    assert result[
        "trial"
    ].tolist() == [
        1,
        0,
        2,
    ]

    assert result[
        "candidate_rank"
    ].tolist() == [
        1,
        2,
        3,
    ]

    selected = result[
        result[
            "selected"
        ]
    ].iloc[
        0
    ]

    assert selected[
        "trial"
    ] == 1

    assert selected[
        "delta_CV_MAE_from_selected"
    ] == pytest.approx(
        0.0
    )

    second = result.iloc[
        1
    ]

    assert second[
        "delta_CV_MAE_from_selected"
    ] == pytest.approx(
        1.0
    )


# =============================================================================
# ADAPTIVE SEARCH DESCRIPTIVE SUMMARY
# =============================================================================


def test_search_component_summary_is_explicitly_descriptive(
    trials: pd.DataFrame,
) -> None:

    result = build_search_component_descriptive(
        trials
    )

    assert set(
        result[
            "component"
        ]
    ) == {
        "model",
        "feature_set",
        "aggregation",
        "reduction",
    }

    assert set(
        result[
            "interpretation"
        ]
    ) == {
        "descriptive_adaptive_search"
    }

    svr = result[
        (
            result[
                "component"
            ]
            == "model"
        )
        & (
            result[
                "level"
            ]
            == "SVR"
        )
    ].iloc[
        0
    ]

    assert svr[
        "n_trials"
    ] == 2

    assert svr[
        "CV_MAE_median"
    ] == pytest.approx(
        4.5
    )

    assert svr[
        "best_observed_CV_MAE"
    ] == pytest.approx(
        4.0
    )

    assert svr[
        "sampling_fraction"
    ] == pytest.approx(
        2.0
        / 3.0
    )


# =============================================================================
# SELECTED PIPELINE SUMMARY
# =============================================================================


def test_selected_pipeline_summary_combines_cv_test_and_baselines(
    selected_pipeline: dict,
    performance_summary: pd.DataFrame,
) -> None:

    result = build_selected_pipeline_summary(
        selected_pipeline=selected_pipeline,
        performance=performance_summary,
    )

    assert len(
        result
    ) == 1

    row = result.iloc[
        0
    ]

    assert row[
        "selected_trial"
    ] == 1

    assert row[
        "model"
    ] == "RIDGE_REGRESSION"

    assert row[
        "aggregation"
    ] == "dawn_profile"

    assert row[
        "reduction"
    ] == "supervised_selection"

    assert row[
        "model_feature_count"
    ] == 40

    assert row[
        "CV_MAE"
    ] == pytest.approx(
        3.0
    )

    assert row[
        "test_MAE"
    ] == pytest.approx(
        4.0
    )

    assert row[
        "test_MAE_gain_vs_mean_baseline"
    ] == pytest.approx(
        2.0
    )

    assert row[
        "test_MAE_gain_vs_median_baseline"
    ] == pytest.approx(
        3.0
    )


# =============================================================================
# END-TO-END ANALYSIS OUTPUT
# =============================================================================


def test_save_result_analysis_creates_report_oriented_outputs(
    tmp_path,
    schema: Schema,
    cv_predictions: pd.DataFrame,
    final_test: pd.DataFrame,
    split_manifest: pd.DataFrame,
    performance_summary: pd.DataFrame,
    selected_pipeline: dict,
    trials: pd.DataFrame,
) -> None:

    reproducibility_dir = (
        tmp_path
        / "reproducibility"
    )

    reproducibility_dir.mkdir(
        parents=True
    )

    performance_summary.to_csv(
        tmp_path
        / "performance_summary.csv",
        index=False,
    )

    trials.to_csv(
        tmp_path
        / "model_selection_trials.csv",
        index=False,
    )

    cv_predictions.to_csv(
        tmp_path
        / "selected_cv_predictions.csv",
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

    results = save_result_analysis(
        output_dir=tmp_path,
        schema=schema,
        top_n_candidates=3,
    )

    expected = {
        "selected_pipeline_summary",
        "selected_cv_repeat_metrics",
        "selected_cv_prediction_stability",
        "final_test_baseline_comparison",
        "final_test_point_errors",
        "dataset_partition_summary",
        "search_status_summary",
        "top_pipeline_candidates",
        "search_component_descriptive",
        "analysis_manifest",
    }

    assert set(
        results
    ) == expected

    analysis_dir = (
        tmp_path
        / "analysis"
    )

    assert analysis_dir.is_dir()

    for name in expected:

        path = (
            analysis_dir
            / f"{name}.csv"
        )

        assert path.is_file()

        saved = pd.read_csv(
            path
        )

        assert not saved.empty