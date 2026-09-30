import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.evaluation.evaluator import (
    ModelEvaluator,
    RegressionMetrics,
)
from src.core.models.factory import RegressionModels


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
def reference_data() -> pd.DataFrame:
    """
    Synthetic CapturePointId-level dataset.

    Eight independent Points are represented.

    Some Points contain more than one CapturePointId so tests can verify
    Point-level grouping and Point-balanced weighting.
    """

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P1",
                "P2",
                "P3",
                "P4",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
            ],

            "CapturePointId": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B9",
                "B10",
            ],

            "meanHFI": [
                10.0,
                10.0,
                20.0,
                30.0,
                40.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ],

            "ACI_mean": [
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                4.5,
                5.0,
                6.0,
                7.0,
                8.0,
            ],

            "Embedding": [
                np.array(
                    [1.0, 2.0]
                ),
                np.array(
                    [1.5, 2.5]
                ),
                np.array(
                    [2.0, 3.0]
                ),
                np.array(
                    [3.0, 4.0]
                ),
                np.array(
                    [4.0, 5.0]
                ),
                np.array(
                    [4.5, 5.5]
                ),
                np.array(
                    [5.0, 6.0]
                ),
                np.array(
                    [6.0, 7.0]
                ),
                np.array(
                    [7.0, 8.0]
                ),
                np.array(
                    [8.0, 9.0]
                ),
            ],
        }
    )


@pytest.fixture
def signatures(
    reference_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Minimal aligned acoustic representation.

    These tests target model-selection logic rather than aggregation logic,
    which is tested separately in test_presplit.py.
    """

    return {
        "mean":
            reference_data.copy(),
    }


@pytest.fixture
def evaluator(
    schema: Schema,
    tmp_path,
) -> ModelEvaluator:

    return ModelEvaluator(
        schema=schema,

        output_dir=str(
            tmp_path
        ),

        models=[
            RegressionModels.RIDGE_REGRESSION,
            RegressionModels.SVR,
        ],

        random_state=42,

        feature_sets=(
            "indices",
        ),

        aggregations=(
            "mean",
        ),

        reductions=(
            "none",
        ),

        pca_indices_candidates=(
            2,
            4,
        ),

        pca_embeddings_candidates=(
            2,
            4,
        ),

        test_size=0.25,

        cv_splits=2,

        cv_repeats=2,

        optuna_trials=3,
    )


# =============================================================================
# DEVELOPMENT / FINAL TEST SPLIT
# =============================================================================


def test_development_test_split_has_no_point_overlap(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    A physical Point must belong entirely to either development or final
    test data.
    """

    (
        development_bags,
        test_bags,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    development = evaluator._select_bags(
        reference_data,
        development_bags,
    )

    test = evaluator._select_bags(
        reference_data,
        test_bags,
    )

    development_points = set(
        development[
            "Point"
        ]
    )

    test_points = set(
        test[
            "Point"
        ]
    )

    assert development_points.isdisjoint(
        test_points
    )

    assert (
        development_points
        | test_points
    ) == set(
        reference_data[
            "Point"
        ]
    )


def test_development_test_split_keeps_all_captures_of_point_together(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    Multiple CapturePointIds belonging to the same Point must never be split
    between development and final test sets.
    """

    (
        _,
        _,
        manifest,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    split_counts = (
        manifest.groupby(
            "Point"
        )[
            "split"
        ]
        .nunique()
    )

    assert (
        split_counts
        == 1
    ).all()


def test_test_size_is_applied_to_independent_points(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    TEST_SIZE is interpreted using independent Points rather than
    CapturePointIds.

    Eight Points with TEST_SIZE=0.25 should reserve two Points.
    """

    (
        _,
        test_bags,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    test = evaluator._select_bags(
        reference_data,
        test_bags,
    )

    assert (
        test[
            "Point"
        ]
        .nunique()
        == 2
    )


def test_development_test_split_is_reproducible(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    A fixed RANDOM_STATE must produce exactly the same holdout split.
    """

    first = (
        evaluator
        ._make_development_test_split(
            reference_data
        )
    )

    second = (
        evaluator
        ._make_development_test_split(
            reference_data
        )
    )

    assert first[
        0
    ] == second[
        0
    ]

    assert first[
        1
    ] == second[
        1
    ]

    pd.testing.assert_frame_equal(
        first[
            2
        ],
        second[
            2
        ],
    )


# =============================================================================
# REPEATED GROUPED CROSS-VALIDATION
# =============================================================================


def test_cv_has_no_point_overlap(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    No physical Point may occur simultaneously in CV training and
    validation data.
    """

    (
        development_bags,
        _,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    development = evaluator._select_bags(
        reference_data,
        development_bags,
    )

    splits = (
        evaluator
        ._make_repeated_cv_splits(
            development
        )
    )

    bag_to_point = (
        development
        .set_index(
            "CapturePointId"
        )[
            "Point"
        ]
        .to_dict()
    )

    for split in splits:

        train_points = {
            bag_to_point[
                bag
            ]
            for bag in split.train_bags
        }

        validation_points = {
            bag_to_point[
                bag
            ]
            for bag in split.validation_bags
        }

        assert train_points.isdisjoint(
            validation_points
        )


def test_final_test_bags_never_enter_cv(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    Final-test CapturePointIds must never enter model-selection CV.
    """

    (
        development_bags,
        test_bags,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    development = evaluator._select_bags(
        reference_data,
        development_bags,
    )

    splits = (
        evaluator
        ._make_repeated_cv_splits(
            development
        )
    )

    test_bag_set = set(
        test_bags
    )

    for split in splits:

        cv_bags = (
            set(
                split.train_bags
            )
            | set(
                split.validation_bags
            )
        )

        assert cv_bags.isdisjoint(
            test_bag_set
        )


def test_expected_number_of_cv_splits_is_created(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    Total CV partitions must equal:

        CV_REPEATS × CV_SPLITS
    """

    (
        development_bags,
        _,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    development = evaluator._select_bags(
        reference_data,
        development_bags,
    )

    splits = (
        evaluator
        ._make_repeated_cv_splits(
            development
        )
    )

    assert len(
        splits
    ) == (
        evaluator.cv_repeats
        * evaluator.cv_splits
    )


def test_each_repeat_produces_complete_oof_coverage(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    Within every CV repetition, each development CapturePointId must appear
    exactly once in validation.

    Therefore the pooled validation predictions form one complete OOF
    prediction set.
    """

    (
        development_bags,
        _,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    development = evaluator._select_bags(
        reference_data,
        development_bags,
    )

    splits = (
        evaluator
        ._make_repeated_cv_splits(
            development
        )
    )

    expected = sorted(
        development[
            "CapturePointId"
        ].tolist()
    )

    for repeat in range(
        1,
        evaluator.cv_repeats + 1,
    ):

        validation_bags = []

        for split in splits:

            if (
                split.repeat
                == repeat
            ):

                validation_bags.extend(
                    split.validation_bags
                )

        assert sorted(
            validation_bags
        ) == expected

        assert len(
            validation_bags
        ) == len(
            set(
                validation_bags
            )
        )


def test_repeated_cv_splits_are_reproducible(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    The complete set of repeated CV partitions must be reproducible.
    """

    (
        development_bags,
        _,
        _,
    ) = evaluator._make_development_test_split(
        reference_data
    )

    development = evaluator._select_bags(
        reference_data,
        development_bags,
    )

    first = (
        evaluator
        ._make_repeated_cv_splits(
            development
        )
    )

    second = (
        evaluator
        ._make_repeated_cv_splits(
            development
        )
    )

    assert first == second


# =============================================================================
# SAME CV SPLITS FOR EVERY OPTUNA TRIAL
# =============================================================================


def test_all_trials_receive_exactly_the_same_cv_splits(
    evaluator: ModelEvaluator,
    signatures: dict[str, pd.DataFrame],
    monkeypatch,
) -> None:
    """
    Candidate pipelines must be compared on exactly the same precomputed
    train/validation partitions.
    """

    received_split_ids = []
    received_split_structures = []

    def fake_evaluate_configuration(
        signatures,
        configuration,
        cv_splits,
        trial_number,
    ):

        received_split_ids.append(
            id(
                cv_splits
            )
        )

        received_split_structures.append(
            tuple(
                (
                    split.repeat,
                    split.fold,
                    split.train_bags,
                    split.validation_bags,
                )
                for split in cv_splits
            )
        )

        score = float(
            trial_number
            + 1
        )

        repeat_metrics = [
            {
                "trial":
                    trial_number,

                "repeat":
                    repeat,

                "MAE":
                    score,

                "RMSE":
                    score,

                "R2":
                    0.0,
            }
            for repeat in range(
                1,
                evaluator.cv_repeats + 1,
            )
        ]

        oof = pd.DataFrame(
            {
                "Point": [
                    "P1",
                ],

                "CapturePointId": [
                    "B1",
                ],

                "meanHFI": [
                    10.0,
                ],

                "prediction": [
                    10.0,
                ],

                "trial": [
                    trial_number,
                ],

                "repeat": [
                    1,
                ],

                "fold": [
                    1,
                ],
            }
        )

        return {
            "summary": {
                "CV_MAE":
                    score,

                "CV_MAE_std":
                    0.0,

                "CV_RMSE":
                    score,

                "CV_RMSE_std":
                    0.0,

                "CV_R2":
                    0.0,

                "CV_R2_std":
                    0.0,

                "mean_fit_time":
                    0.0,

                "mean_prediction_time":
                    0.0,
            },

            "repeat_metrics":
                repeat_metrics,

            "oof_predictions":
                oof,
        }

    monkeypatch.setattr(
        evaluator,
        "_evaluate_configuration",
        fake_evaluate_configuration,
    )

    def fake_fit_final_pipeline(
        signatures,
        development_bags,
        test_bags,
        configuration,
    ):

        df = evaluator._select_bags(
            signatures[
                configuration[
                    "aggregation"
                ]
            ],
            test_bags,
        )

        predictions = df[
            [
                "Point",
                "CapturePointId",
                "meanHFI",
            ]
        ].copy()

        predictions[
            "prediction"
        ] = predictions[
            "meanHFI"
        ]

        return {
            "predictions":
                predictions,

            "metrics":
                RegressionMetrics(
                    mae=0.0,
                    rmse=0.0,
                    r2=1.0,
                ),

            "fit_time":
                0.0,

            "prediction_time":
                0.0,

            "n_model_features":
                1,
        }

    monkeypatch.setattr(
        evaluator,
        "_fit_final_pipeline",
        fake_fit_final_pipeline,
    )

    monkeypatch.setattr(
        evaluator,
        "_save_results",
        lambda **kwargs: None,
    )

    evaluator.evaluate(
        signatures
    )

    assert len(
        received_split_ids
    ) == evaluator.optuna_trials

    assert len(
        set(
            received_split_ids
        )
    ) == 1

    first_structure = (
        received_split_structures[
            0
        ]
    )

    assert all(
        structure
        == first_structure
        for structure in (
            received_split_structures
        )
    )


# =============================================================================
# PIPELINE SELECTION OBJECTIVE
# =============================================================================


def test_pipeline_is_selected_by_cv_mae_only(
    evaluator: ModelEvaluator,
    signatures: dict[str, pd.DataFrame],
    monkeypatch,
) -> None:
    """
    Pipeline selection must depend exclusively on development CV MAE.

    Final-test evaluation occurs only after Optuna has selected a trial.
    """

    scores = {
        0:
            3.0,

        1:
            1.0,

        2:
            2.0,
    }

    events = []

    def fake_evaluate_configuration(
        signatures,
        configuration,
        cv_splits,
        trial_number,
    ):

        events.append(
            f"trial_{trial_number}"
        )

        score = scores[
            trial_number
        ]

        repeat_metrics = [
            {
                "trial":
                    trial_number,

                "repeat":
                    repeat,

                "MAE":
                    score,

                "RMSE":
                    score,

                "R2":
                    0.0,
            }
            for repeat in range(
                1,
                evaluator.cv_repeats + 1,
            )
        ]

        oof = pd.DataFrame(
            {
                "Point": [
                    "P1",
                ],

                "CapturePointId": [
                    "B1",
                ],

                "meanHFI": [
                    10.0,
                ],

                "prediction": [
                    10.0,
                ],

                "trial": [
                    trial_number,
                ],

                "repeat": [
                    1,
                ],

                "fold": [
                    1,
                ],
            }
        )

        return {
            "summary": {
                "CV_MAE":
                    score,

                "CV_MAE_std":
                    0.0,

                "CV_RMSE":
                    score,

                "CV_RMSE_std":
                    0.0,

                "CV_R2":
                    0.0,

                "CV_R2_std":
                    0.0,

                "mean_fit_time":
                    0.0,

                "mean_prediction_time":
                    0.0,
            },

            "repeat_metrics":
                repeat_metrics,

            "oof_predictions":
                oof,
        }

    monkeypatch.setattr(
        evaluator,
        "_evaluate_configuration",
        fake_evaluate_configuration,
    )

    final_configurations = []

    def fake_fit_final_pipeline(
        signatures,
        development_bags,
        test_bags,
        configuration,
    ):

        events.append(
            "final_test"
        )

        final_configurations.append(
            configuration
        )

        df = evaluator._select_bags(
            signatures[
                configuration[
                    "aggregation"
                ]
            ],
            test_bags,
        )

        predictions = df[
            [
                "Point",
                "CapturePointId",
                "meanHFI",
            ]
        ].copy()

        predictions[
            "prediction"
        ] = 999.0

        return {
            "predictions":
                predictions,

            # Deliberately poor test performance.
            # It must have no influence on trial selection.
            "metrics":
                RegressionMetrics(
                    mae=999.0,
                    rmse=999.0,
                    r2=-999.0,
                ),

            "fit_time":
                0.0,

            "prediction_time":
                0.0,

            "n_model_features":
                1,
        }

    monkeypatch.setattr(
        evaluator,
        "_fit_final_pipeline",
        fake_fit_final_pipeline,
    )

    monkeypatch.setattr(
        evaluator,
        "_save_results",
        lambda **kwargs: None,
    )

    result = evaluator.evaluate(
        signatures
    )

    assert result[
        "final_selection"
    ][
        "selected_trial"
    ] == 1

    assert result[
        "study"
    ].best_trial.number == 1

    assert result[
        "study"
    ].best_value == pytest.approx(
        1.0
    )

    assert events == [
        "trial_0",
        "trial_1",
        "trial_2",
        "final_test",
    ]

    assert len(
        final_configurations
    ) == 1


# =============================================================================
# POINT-BALANCED WEIGHTS
# =============================================================================


def test_point_weights_give_each_point_equal_total_weight(
    evaluator: ModelEvaluator,
) -> None:
    """
    Multiple CapturePointIds from one physical Point must collectively receive
    the same total weight as a Point represented by one CapturePointId.
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

            "meanHFI": [
                0.0,
                10.0,
                50.0,
                100.0,
            ],
        }
    )

    weights = evaluator._point_weights(
        df
    )

    weighted = df[
        [
            "Point",
        ]
    ].copy()

    weighted[
        "weight"
    ] = weights

    totals = (
        weighted.groupby(
            "Point"
        )[
            "weight"
        ]
        .sum()
    )

    assert totals[
        "P1"
    ] == pytest.approx(
        totals[
            "P2"
        ]
    )

    assert totals[
        "P2"
    ] == pytest.approx(
        totals[
            "P3"
        ]
    )


# =============================================================================
# POINT-BALANCED BASELINES
# =============================================================================


def test_baselines_are_point_balanced(
    evaluator: ModelEvaluator,
) -> None:
    """
    Point-balanced baselines must not give extra influence to Points that
    contain multiple CapturePointIds.
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

            "meanHFI": [
                0.0,
                10.0,
                50.0,
                100.0,
            ],
        }
    )

    (
        mean_value,
        median_value,
    ) = evaluator._baseline_values(
        df
    )

    # P1 contributes a Point-level average of 5.
    #
    # Equal-Point mean:
    #
    #     (5 + 50 + 100) / 3

    expected_mean = (
        5.0
        + 50.0
        + 100.0
    ) / 3.0

    assert mean_value == pytest.approx(
        expected_mean
    )

    assert median_value == pytest.approx(
        50.0
    )


# =============================================================================
# POINT-BALANCED METRICS
# =============================================================================


def test_metrics_give_each_point_equal_total_weight(
    evaluator: ModelEvaluator,
) -> None:
    """
    A Point represented by multiple CapturePointIds must not dominate MAE.
    """

    df = pd.DataFrame(
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
                0.0,
                10.0,
            ],
        }
    )

    predictions = np.array(
        [
            10.0,
            10.0,
            10.0,
        ]
    )

    metrics = (
        evaluator
        ._regression_metrics(
            df=df,
            predictions=predictions,
        )
    )

    # P1 error = 10
    # P2 error = 0
    #
    # Point-balanced MAE:
    #
    #     (10 + 0) / 2 = 5

    assert metrics.mae == pytest.approx(
        5.0
    )


# =============================================================================
# FEATURE DIMENSIONALITY
# =============================================================================


def test_raw_feature_count_tracks_aggregation_dimensionality(
    evaluator: ModelEvaluator,
) -> None:
    """
    Raw dimensionality must reflect the selected representation before PCA.
    """

    feature_dimensions = {
        "mean": {
            "indices":
                60,
            "embeddings":
                512,
        },

        "mean_std": {
            "indices":
                120,
            "embeddings":
                1024,
        },

        "hierarchical": {
            "indices":
                180,
            "embeddings":
                1536,
        },
    }

    configuration = {
        "model":
            RegressionModels.SVR,

        "feature_set":
            "both",

        "aggregation":
            "hierarchical",

        "reduction":
            "none",

        "pca_indices_components":
            None,

        "pca_embeddings_components":
            None,

        "model_params":
            {},
    }

    count = evaluator._raw_feature_count(
        configuration=configuration,
        feature_dimensions=(
            feature_dimensions
        ),
    )

    assert count == (
        180
        + 1536
    )

    assert count == 1716


def test_model_feature_count_equals_raw_count_without_pca(
    evaluator: ModelEvaluator,
) -> None:
    """
    Without dimensionality reduction, the regression model receives all raw
    features.
    """

    feature_dimensions = {
        "hierarchical": {
            "indices":
                180,
            "embeddings":
                1536,
        }
    }

    configuration = {
        "model":
            RegressionModels.SVR,

        "feature_set":
            "both",

        "aggregation":
            "hierarchical",

        "reduction":
            "none",

        "pca_indices_components":
            None,

        "pca_embeddings_components":
            None,

        "model_params":
            {},
    }

    count = evaluator._model_feature_count(
        configuration=configuration,
        feature_dimensions=(
            feature_dimensions
        ),
    )

    assert count == 1716


def test_model_feature_count_uses_pca_components(
    evaluator: ModelEvaluator,
) -> None:
    """
    With PCA, model dimensionality must equal the sum of the retained
    components from the active feature blocks.
    """

    feature_dimensions = {
        "hierarchical": {
            "indices":
                180,
            "embeddings":
                1536,
        }
    }

    configuration = {
        "model":
            RegressionModels.SVR,

        "feature_set":
            "both",

        "aggregation":
            "hierarchical",

        "reduction":
            "pca",

        "pca_indices_components":
            12,

        "pca_embeddings_components":
            24,

        "model_params":
            {},
    }

    raw_count = evaluator._raw_feature_count(
        configuration=configuration,
        feature_dimensions=(
            feature_dimensions
        ),
    )

    model_count = evaluator._model_feature_count(
        configuration=configuration,
        feature_dimensions=(
            feature_dimensions
        ),
    )

    assert raw_count == 1716

    assert model_count == 36


# =============================================================================
# CONFIGURATION SERIALIZATION
# =============================================================================


def test_pipeline_configuration_serialization_roundtrip(
    evaluator: ModelEvaluator,
) -> None:
    """
    A configuration stored in Optuna user attributes must be recoverable
    without changing its meaning.
    """

    configuration = {
        "model":
            RegressionModels.GRADIENT_BOOSTING,

        "feature_set":
            "embeddings",

        "aggregation":
            "hierarchical",

        "reduction":
            "pca",

        "pca_indices_components":
            None,

        "pca_embeddings_components":
            16,

        "model_params": {
            "n_estimators":
                300,

            "learning_rate":
                0.05,

            "max_depth":
                2,
        },
    }

    serialized = (
        evaluator
        ._serializable_config(
            configuration
        )
    )

    assert serialized[
        "model"
    ] == "GRADIENT_BOOSTING"

    restored = (
        evaluator
        ._configuration_from_serialized(
            serialized
        )
    )

    assert restored[
        "model"
    ] == (
        RegressionModels
        .GRADIENT_BOOSTING
    )

    assert restored[
        "feature_set"
    ] == "embeddings"

    assert restored[
        "aggregation"
    ] == "hierarchical"

    assert restored[
        "pca_embeddings_components"
    ] == 16

    assert restored[
        "model_params"
    ] == configuration[
        "model_params"
    ]
