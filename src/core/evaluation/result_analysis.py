import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.core.data.schema import Schema


logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API
# =============================================================================


def save_result_analysis(
    output_dir: str | Path,
    schema: Schema,
    top_n_candidates: int = 20,
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Build report-oriented scientific tables from a completed experiment.

    The analysis separates:

        experimental performance
        prediction stability
        final-test errors
        baseline comparisons
        dataset partition characteristics
        Optuna search diagnostics

    Search-space component summaries are descriptive only because Optuna
    samples the search space adaptively.
    """

    output_dir = Path(
        output_dir
    )

    analysis_dir = (
        output_dir
        / "analysis"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD EXPERIMENT OUTPUTS
    # =========================================================================

    performance = pd.read_csv(
        output_dir
        / "performance_summary.csv"
    )

    trials = pd.read_csv(
        output_dir
        / "model_selection_trials.csv"
    )

    cv_predictions = pd.read_csv(
        output_dir
        / "selected_cv_predictions.csv"
    )

    final_test = pd.read_csv(
        output_dir
        / "final_test_predictions.csv"
    )

    split_manifest = pd.read_csv(
        output_dir
        / "reproducibility"
        / "data_split.csv"
    )

    with open(
        output_dir
        / "selected_pipeline.json",
        "r",
        encoding="utf-8",
    ) as file:

        selected_pipeline = json.load(
            file
        )

    # =========================================================================
    # BUILD ANALYSIS TABLES
    # =========================================================================

    results = {
        "selected_pipeline_summary":
            build_selected_pipeline_summary(
                selected_pipeline=(
                    selected_pipeline
                ),
                performance=(
                    performance
                ),
            ),

        "selected_cv_repeat_metrics":
            build_selected_cv_repeat_metrics(
                predictions=(
                    cv_predictions
                ),
                schema=(
                    schema
                ),
            ),

        "selected_cv_prediction_stability":
            build_cv_prediction_stability(
                predictions=(
                    cv_predictions
                ),
                schema=(
                    schema
                ),
            ),

        "final_test_baseline_comparison":
            build_final_test_baseline_comparison(
                final_test=(
                    final_test
                ),
                schema=(
                    schema
                ),
            ),

        "final_test_point_errors":
            build_final_test_point_errors(
                final_test=(
                    final_test
                ),
                schema=(
                    schema
                ),
            ),

        "dataset_partition_summary":
            build_dataset_partition_summary(
                split_manifest=(
                    split_manifest
                ),
                schema=(
                    schema
                ),
            ),

        "search_status_summary":
            build_search_status_summary(
                trials
            ),

        "top_pipeline_candidates":
            build_top_pipeline_candidates(
                trials=(
                    trials
                ),
                selected_pipeline=(
                    selected_pipeline
                ),
                top_n=(
                    top_n_candidates
                ),
            ),

        "search_component_descriptive":
            build_search_component_descriptive(
                trials
            ),
    }

    results[
        "analysis_manifest"
    ] = (
        build_analysis_manifest()
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    for name, dataframe in (
        results.items()
    ):

        dataframe.to_csv(
            analysis_dir
            / f"{name}.csv",
            index=False,
        )

    logger.info(
        "Scientific analysis tables saved to %s",
        analysis_dir,
    )

    return results


# =============================================================================
# SELECTED PIPELINE SUMMARY
# =============================================================================


def build_selected_pipeline_summary(
    selected_pipeline: dict,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one report-ready row describing the selected pipeline and its
    performance during model selection and independent testing.
    """

    pipeline = (
        selected_pipeline[
            "pipeline"
        ]
    )

    cv_model = _performance_row(
        performance=performance,
        stage="development_cv",
        method="selected_pipeline",
    )

    cv_mean = _performance_row(
        performance=performance,
        stage="development_cv",
        method="mean_baseline",
    )

    cv_median = _performance_row(
        performance=performance,
        stage="development_cv",
        method="median_baseline",
    )

    test_model = _performance_row(
        performance=performance,
        stage="final_test",
        method="selected_pipeline",
    )

    test_mean = _performance_row(
        performance=performance,
        stage="final_test",
        method="mean_baseline",
    )

    test_median = _performance_row(
        performance=performance,
        stage="final_test",
        method="median_baseline",
    )

    row = {
        "selected_trial":
            selected_pipeline[
                "selected_trial"
            ],

        "model":
            pipeline[
                "model"
            ],

        "feature_set":
            pipeline[
                "feature_set"
            ],

        "aggregation":
            pipeline[
                "aggregation"
            ],

        "reduction":
            pipeline[
                "reduction"
            ],

        "pca_indices_components":
            pipeline.get(
                "pca_indices_components"
            ),

        "pca_embeddings_components":
            pipeline.get(
                "pca_embeddings_components"
            ),

        "selection_indices_features":
            pipeline.get(
                "selection_indices_features"
            ),

        "selection_embeddings_features":
            pipeline.get(
                "selection_embeddings_features"
            ),

        "raw_feature_count":
            selected_pipeline.get(
                "raw_feature_count"
            ),

        "model_feature_count":
            selected_pipeline.get(
                "model_feature_count"
            ),

        "model_params":
            json.dumps(
                pipeline[
                    "model_params"
                ],
                sort_keys=True,
            ),

        # =====================================================================
        # DEVELOPMENT CV
        # =====================================================================

        "CV_MAE":
            cv_model[
                "MAE"
            ],

        "CV_MAE_SD":
            cv_model[
                "MAE_SD"
            ],

        "CV_RMSE":
            cv_model[
                "RMSE"
            ],

        "CV_RMSE_SD":
            cv_model[
                "RMSE_SD"
            ],

        "CV_R2":
            cv_model[
                "R2"
            ],

        "CV_R2_SD":
            cv_model[
                "R2_SD"
            ],

        "CV_mean_baseline_MAE":
            cv_mean[
                "MAE"
            ],

        "CV_median_baseline_MAE":
            cv_median[
                "MAE"
            ],

        # =====================================================================
        # FINAL TEST
        # =====================================================================

        "test_MAE":
            test_model[
                "MAE"
            ],

        "test_RMSE":
            test_model[
                "RMSE"
            ],

        "test_R2":
            test_model[
                "R2"
            ],

        "test_mean_baseline_MAE":
            test_mean[
                "MAE"
            ],

        "test_median_baseline_MAE":
            test_median[
                "MAE"
            ],

        "test_MAE_gain_vs_mean_baseline":
            (
                test_mean[
                    "MAE"
                ]
                - test_model[
                    "MAE"
                ]
            ),

        "test_MAE_gain_vs_median_baseline":
            (
                test_median[
                    "MAE"
                ]
                - test_model[
                    "MAE"
                ]
            ),

        "final_fit_time":
            selected_pipeline.get(
                "final_fit_time"
            ),

        "final_prediction_time":
            selected_pipeline.get(
                "final_prediction_time"
            ),
    }

    return pd.DataFrame(
        [
            row
        ]
    )


# =============================================================================
# SELECTED PIPELINE — CV REPEAT METRICS
# =============================================================================


def build_selected_cv_repeat_metrics(
    predictions: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """
    Recalculate Point-balanced metrics independently for each complete CV
    repetition.

    This provides the primary stability table for the selected pipeline and
    both training-only baselines.
    """

    methods = {
        "selected_pipeline":
            "predicted_HFI",

        "mean_baseline":
            "mean_baseline_prediction",

        "median_baseline":
            "median_baseline_prediction",
    }

    rows = []

    for repeat, data in (
        predictions.groupby(
            "repeat",
            sort=True,
        )
    ):

        for method, column in (
            methods.items()
        ):

            metrics = _regression_metrics(
                data=data,
                prediction_column=column,
                schema=schema,
            )

            rows.append(
                {
                    "repeat":
                        int(
                            repeat
                        ),

                    "method":
                        method,

                    "MAE":
                        metrics[
                            "MAE"
                        ],

                    "RMSE":
                        metrics[
                            "RMSE"
                        ],

                    "R2":
                        metrics[
                            "R2"
                        ],

                    "n_points":
                        data[
                            schema.group
                        ].nunique(),

                    "n_capture_points":
                        data[
                            schema.bag
                        ].nunique(),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CV PREDICTION STABILITY
# =============================================================================


def build_cv_prediction_stability(
    predictions: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """
    Quantify how much the selected model's prediction for each CapturePointId
    changes across repeated CV partitions.

    This is a model-stability diagnostic, not an independent performance
    estimate.
    """

    grouped = (
        predictions.groupby(
            [
                schema.group,
                schema.bag,
                "observed_HFI",
            ],
            as_index=False,
        )
        .agg(
            n_repeats=(
                "repeat",
                "nunique",
            ),

            predicted_HFI_mean=(
                "predicted_HFI",
                "mean",
            ),

            predicted_HFI_sd=(
                "predicted_HFI",
                _sample_sd,
            ),

            predicted_HFI_min=(
                "predicted_HFI",
                "min",
            ),

            predicted_HFI_max=(
                "predicted_HFI",
                "max",
            ),

            absolute_error_mean=(
                "absolute_error",
                "mean",
            ),

            absolute_error_sd=(
                "absolute_error",
                _sample_sd,
            ),

            residual_mean=(
                "residual",
                "mean",
            ),

            residual_sd=(
                "residual",
                _sample_sd,
            ),
        )
    )

    grouped[
        "prediction_range"
    ] = (
        grouped[
            "predicted_HFI_max"
        ]
        - grouped[
            "predicted_HFI_min"
        ]
    )

    return (
        grouped.sort_values(
            [
                schema.group,
                schema.bag,
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# FINAL TEST — CAPTURE-LEVEL BASELINE COMPARISON
# =============================================================================


def build_final_test_baseline_comparison(
    final_test: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """
    Build paired CapturePointId-level comparisons between the selected model
    and the two baselines.

    Positive gain means the selected model produced a smaller error.
    """

    columns = [
        schema.group,
        schema.bag,
        "observed_HFI",
        "predicted_HFI",
        "mean_baseline_prediction",
        "median_baseline_prediction",
        "residual",
        "absolute_error",
        "squared_error",
        "mean_baseline_residual",
        "mean_baseline_absolute_error",
        "mean_baseline_squared_error",
        "median_baseline_residual",
        "median_baseline_absolute_error",
        "median_baseline_squared_error",
    ]

    result = (
        final_test[
            columns
        ]
        .copy()
    )

    result[
        "absolute_error_gain_vs_mean_baseline"
    ] = (
        result[
            "mean_baseline_absolute_error"
        ]
        - result[
            "absolute_error"
        ]
    )

    result[
        "absolute_error_gain_vs_median_baseline"
    ] = (
        result[
            "median_baseline_absolute_error"
        ]
        - result[
            "absolute_error"
        ]
    )

    result[
        "squared_error_gain_vs_mean_baseline"
    ] = (
        result[
            "mean_baseline_squared_error"
        ]
        - result[
            "squared_error"
        ]
    )

    result[
        "squared_error_gain_vs_median_baseline"
    ] = (
        result[
            "median_baseline_squared_error"
        ]
        - result[
            "squared_error"
        ]
    )

    result[
        "model_better_than_mean_baseline"
    ] = (
        result[
            "absolute_error"
        ]
        < result[
            "mean_baseline_absolute_error"
        ]
    )

    result[
        "model_better_than_median_baseline"
    ] = (
        result[
            "absolute_error"
        ]
        < result[
            "median_baseline_absolute_error"
        ]
    )

    return (
        result.sort_values(
            [
                schema.group,
                schema.bag,
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# FINAL TEST — POINT-LEVEL ERRORS
# =============================================================================


def build_final_test_point_errors(
    final_test: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """
    Summarize prediction errors at the independent Point level.

    Errors are first calculated for individual CapturePointIds and only then
    averaged within Point.

    Observed and predicted HFI values are not collapsed across
    CapturePointIds because different CapturePointIds belonging to the same
    Point may have different targets.
    """

    result = (
        final_test.groupby(
            schema.group,
            as_index=False,
        )
        .agg(
            n_capture_points=(
                schema.bag,
                "nunique",
            ),

            model_mean_residual=(
                "residual",
                "mean",
            ),

            model_MAE=(
                "absolute_error",
                "mean",
            ),

            model_MSE=(
                "squared_error",
                "mean",
            ),

            mean_baseline_mean_residual=(
                "mean_baseline_residual",
                "mean",
            ),

            mean_baseline_MAE=(
                "mean_baseline_absolute_error",
                "mean",
            ),

            mean_baseline_MSE=(
                "mean_baseline_squared_error",
                "mean",
            ),

            median_baseline_mean_residual=(
                "median_baseline_residual",
                "mean",
            ),

            median_baseline_MAE=(
                "median_baseline_absolute_error",
                "mean",
            ),

            median_baseline_MSE=(
                "median_baseline_squared_error",
                "mean",
            ),
        )
    )

    result[
        "model_RMSE"
    ] = np.sqrt(
        result[
            "model_MSE"
        ]
    )

    result[
        "mean_baseline_RMSE"
    ] = np.sqrt(
        result[
            "mean_baseline_MSE"
        ]
    )

    result[
        "median_baseline_RMSE"
    ] = np.sqrt(
        result[
            "median_baseline_MSE"
        ]
    )

    result[
        "MAE_gain_vs_mean_baseline"
    ] = (
        result[
            "mean_baseline_MAE"
        ]
        - result[
            "model_MAE"
        ]
    )

    result[
        "MAE_gain_vs_median_baseline"
    ] = (
        result[
            "median_baseline_MAE"
        ]
        - result[
            "model_MAE"
        ]
    )

    return (
        result.sort_values(
            schema.group
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# DATASET PARTITION SUMMARY
# =============================================================================


def build_dataset_partition_summary(
    split_manifest: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """
    Describe the development/test partition and its HFI distribution.

    Both ordinary CapturePointId-level summaries and Point-balanced moments
    are reported.
    """

    frames = {
        "all":
            split_manifest,

        "development":
            split_manifest[
                split_manifest[
                    "split"
                ]
                == "development"
            ],

        "test":
            split_manifest[
                split_manifest[
                    "split"
                ]
                == "test"
            ],
    }

    rows = []

    for split_name, data in (
        frames.items()
    ):

        target = (
            data[
                schema.target
            ]
            .to_numpy(
                dtype=float
            )
        )

        weights = (
            _point_weights(
                data=data,
                schema=schema,
            )
        )

        weighted_mean = float(
            np.average(
                target,
                weights=weights,
            )
        )

        weighted_variance = float(
            np.average(
                (
                    target
                    - weighted_mean
                )
                ** 2,
                weights=weights,
            )
        )

        rows.append(
            {
                "split":
                    split_name,

                "n_points":
                    data[
                        schema.group
                    ].nunique(),

                "n_capture_points":
                    data[
                        schema.bag
                    ].nunique(),

                "HFI_mean_capture":
                    float(
                        np.mean(
                            target
                        )
                    ),

                "HFI_sd_capture":
                    float(
                        np.std(
                            target,
                            ddof=1,
                        )
                    )
                    if len(
                        target
                    ) > 1
                    else 0.0,

                "HFI_median_capture":
                    float(
                        np.median(
                            target
                        )
                    ),

                "HFI_q25_capture":
                    float(
                        np.quantile(
                            target,
                            0.25,
                        )
                    ),

                "HFI_q75_capture":
                    float(
                        np.quantile(
                            target,
                            0.75,
                        )
                    ),

                "HFI_min":
                    float(
                        np.min(
                            target
                        )
                    ),

                "HFI_max":
                    float(
                        np.max(
                            target
                        )
                    ),

                "HFI_mean_point_balanced":
                    weighted_mean,

                "HFI_sd_point_balanced":
                    float(
                        np.sqrt(
                            weighted_variance
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# OPTUNA SEARCH STATUS
# =============================================================================


def build_search_status_summary(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize completed, failed and pruned Optuna trials.
    """

    counts = (
        trials[
            "state"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "state"
        )
        .reset_index(
            name="n_trials"
        )
    )

    counts[
        "fraction"
    ] = (
        counts[
            "n_trials"
        ]
        / counts[
            "n_trials"
        ].sum()
    )

    return counts


# =============================================================================
# TOP PIPELINE CANDIDATES
# =============================================================================


def build_top_pipeline_candidates(
    trials: pd.DataFrame,
    selected_pipeline: dict,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Create a compact table of the lowest-CV-MAE complete pipeline
    configurations.

    This describes the best observed candidates from the adaptive search.
    It is not a formal comparison between model families or representations.
    """

    completed = (
        trials[
            (
                trials[
                    "state"
                ]
                == "COMPLETE"
            )
            & pd.notna(
                trials[
                    "CV_MAE"
                ]
            )
        ]
        .copy()
    )

    completed = (
        completed.sort_values(
            [
                "CV_MAE",
                "trial",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    completed[
        "candidate_rank"
    ] = (
        np.arange(
            len(
                completed
            )
        )
        + 1
    )

    selected_trial = int(
        selected_pipeline[
            "selected_trial"
        ]
    )

    selected_mae = float(
        completed.loc[
            completed[
                "trial"
            ]
            == selected_trial,
            "CV_MAE",
        ]
        .iloc[
            0
        ]
    )

    completed[
        "selected"
    ] = (
        completed[
            "trial"
        ]
        == selected_trial
    )

    completed[
        "delta_CV_MAE_from_selected"
    ] = (
        completed[
            "CV_MAE"
        ]
        - selected_mae
    )

    columns = [
        "candidate_rank",
        "trial",
        "selected",
        "model",
        "feature_set",
        "aggregation",
        "reduction",
        "pca_indices_components",
        "pca_embeddings_components",
        "selection_indices_features",
        "selection_embeddings_features",
        "raw_feature_count",
        "model_feature_count",
        "feature_to_point_ratio",
        "CV_MAE",
        "CV_MAE_std",
        "delta_CV_MAE_from_selected",
        "CV_RMSE",
        "CV_RMSE_std",
        "CV_R2",
        "CV_R2_std",
        "mean_fit_time",
        "duration_seconds",
        "model_params",
    ]

    columns = [
        column
        for column in columns
        if column in completed.columns
    ]

    return (
        completed[
            columns
        ]
        .head(
            top_n
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# ADAPTIVE SEARCH — DESCRIPTIVE COMPONENT SUMMARY
# =============================================================================


def build_search_component_descriptive(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    Describe CV results observed for each major pipeline component.

    IMPORTANT
    ---------
    Optuna TPE samples configurations adaptively.

    These summaries therefore describe the realized search history and must
    not be interpreted as unbiased evidence that one model, aggregation,
    feature set or reduction strategy is superior to another.
    """

    completed = (
        trials[
            (
                trials[
                    "state"
                ]
                == "COMPLETE"
            )
            & pd.notna(
                trials[
                    "CV_MAE"
                ]
            )
        ]
        .copy()
    )

    components = [
        "model",
        "feature_set",
        "aggregation",
        "reduction",
    ]

    rows = []

    n_completed = len(
        completed
    )

    for component in (
        components
    ):

        for level, data in (
            completed.groupby(
                component,
                dropna=False,
            )
        ):

            values = (
                data[
                    "CV_MAE"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            rows.append(
                {
                    "component":
                        component,

                    "level":
                        level,

                    "n_trials":
                        len(
                            data
                        ),

                    "sampling_fraction":
                        (
                            len(
                                data
                            )
                            / n_completed
                        ),

                    "CV_MAE_mean":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "CV_MAE_sd":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        )
                        if len(
                            values
                        ) > 1
                        else 0.0,

                    "CV_MAE_median":
                        float(
                            np.median(
                                values
                            )
                        ),

                    "CV_MAE_q25":
                        float(
                            np.quantile(
                                values,
                                0.25,
                            )
                        ),

                    "CV_MAE_q75":
                        float(
                            np.quantile(
                                values,
                                0.75,
                            )
                        ),

                    "best_observed_CV_MAE":
                        float(
                            np.min(
                                values
                            )
                        ),

                    "interpretation":
                        "descriptive_adaptive_search",
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ANALYSIS MANIFEST
# =============================================================================


def build_analysis_manifest() -> pd.DataFrame:
    """
    Document the scientific role and interpretation of every generated table.
    """

    rows = [
        {
            "file":
                "selected_pipeline_summary.csv",

            "purpose":
                (
                    "Report-ready summary of the selected pipeline, "
                    "development CV performance, final-test performance "
                    "and baseline comparisons."
                ),

            "interpretation":
                "primary_result",
        },

        {
            "file":
                "selected_cv_repeat_metrics.csv",

            "purpose":
                (
                    "Point-balanced MAE, RMSE and R2 for each complete "
                    "CV repetition and each method."
                ),

            "interpretation":
                "model_selection_stability",
        },

        {
            "file":
                "selected_cv_prediction_stability.csv",

            "purpose":
                (
                    "Variation in the selected model prediction for each "
                    "CapturePointId across repeated CV partitions."
                ),

            "interpretation":
                "stability_diagnostic",
        },

        {
            "file":
                "final_test_baseline_comparison.csv",

            "purpose":
                (
                    "Paired CapturePointId-level errors and gains relative "
                    "to mean and median baselines."
                ),

            "interpretation":
                "independent_test_diagnostic",
        },

        {
            "file":
                "final_test_point_errors.csv",

            "purpose":
                (
                    "Error summaries at the independent Point level after "
                    "first calculating error for each CapturePointId."
                ),

            "interpretation":
                "independent_test_primary_unit",
        },

        {
            "file":
                "dataset_partition_summary.csv",

            "purpose":
                (
                    "Development/test sample sizes and HFI distribution "
                    "including Point-balanced moments."
                ),

            "interpretation":
                "experimental_design_diagnostic",
        },

        {
            "file":
                "search_status_summary.csv",

            "purpose":
                (
                    "Numbers and fractions of complete, failed and pruned "
                    "Optuna trials."
                ),

            "interpretation":
                "optimization_diagnostic",
        },

        {
            "file":
                "top_pipeline_candidates.csv",

            "purpose":
                (
                    "Highest-performing observed complete configurations "
                    "ranked by repeated-CV MAE."
                ),

            "interpretation":
                "adaptive_search_descriptive",
        },

        {
            "file":
                "search_component_descriptive.csv",

            "purpose":
                (
                    "Distribution of observed trial performance by model, "
                    "feature set, aggregation and reduction."
                ),

            "interpretation":
                (
                    "descriptive_only_not_formal_component_comparison"
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# INTERNAL METRICS
# =============================================================================


def _regression_metrics(
    data: pd.DataFrame,
    prediction_column: str,
    schema: Schema,
) -> dict[
    str,
    float,
]:
    """
    Calculate Point-balanced regression metrics using the same weighting
    convention as ModelEvaluator.
    """

    observed = (
        data[
            "observed_HFI"
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted = (
        data[
            prediction_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    weights = (
        _point_weights(
            data=data,
            schema=schema,
        )
    )

    return {
        "MAE":
            float(
                mean_absolute_error(
                    observed,
                    predicted,
                    sample_weight=weights,
                )
            ),

        "RMSE":
            float(
                root_mean_squared_error(
                    observed,
                    predicted,
                    sample_weight=weights,
                )
            ),

        "R2":
            float(
                r2_score(
                    observed,
                    predicted,
                    sample_weight=weights,
                )
            ),
    }


# =============================================================================
# POINT BALANCING
# =============================================================================


def _point_weights(
    data: pd.DataFrame,
    schema: Schema,
) -> np.ndarray:
    """
    Give every physical Point the same total weight.
    """

    n_captures = (
        data.groupby(
            schema.group
        )[
            schema.bag
        ]
        .transform(
            "nunique"
        )
        .to_numpy(
            dtype=float
        )
    )

    weights = (
        1.0
        / n_captures
    )

    return (
        weights
        / weights.mean()
    )


# =============================================================================
# HELPERS
# =============================================================================


def _performance_row(
    performance: pd.DataFrame,
    stage: str,
    method: str,
) -> pd.Series:
    """
    Select one row from performance_summary.csv.
    """

    return (
        performance[
            (
                performance[
                    "stage"
                ]
                == stage
            )
            & (
                performance[
                    "method"
                ]
                == method
            )
        ]
        .iloc[
            0
        ]
    )


def _sample_sd(
    values: pd.Series,
) -> float:
    """
    Sample SD with a defined value of zero when only one observation exists.
    """

    values = (
        pd.Series(
            values
        )
        .dropna()
    )

    if len(
        values
    ) <= 1:

        return 0.0

    return float(
        values.std(
            ddof=1
        )
    )