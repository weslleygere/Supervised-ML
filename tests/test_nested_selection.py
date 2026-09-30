import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.evaluation.evaluator import ModelEvaluator
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
    One CapturePointId per Point.

    Eight Points are enough for grouped outer and inner splitting.
    """

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P2",
                "P3",
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
            ],
            "meanHFI": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ],
            "ACI_mean": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            "Embedding": [
                np.array([1.0, 2.0]),
                np.array([2.0, 3.0]),
                np.array([3.0, 4.0]),
                np.array([4.0, 5.0]),
                np.array([5.0, 6.0]),
                np.array([6.0, 7.0]),
                np.array([7.0, 8.0]),
                np.array([8.0, 9.0]),
            ],
        }
    )


@pytest.fixture
def signatures(
    reference_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Minimal aligned representations for evaluator tests.

    These tests target nested-selection logic rather than presplit
    aggregation, so identical copies are sufficient.
    """

    return {
        "mean":
            reference_data.copy(),
        "mean_std":
            reference_data.copy(),
        "hierarchical":
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
            "mean_std",
            "hierarchical",
        ),
        reductions=(
            "none",
        ),
        pca_indices_candidates=(
            2,
        ),
        pca_embeddings_candidates=(
            2,
        ),
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
        optuna_trials=2,
    )


# =============================================================================
# INNER GROUP SEPARATION
# =============================================================================


def test_inner_cv_has_no_point_overlap(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    No physical Point may appear simultaneously in inner training and
    validation data.
    """

    splits = evaluator._make_inner_splits(
        reference_data,
        outer_repeat=1,
        outer_fold=1,
    )

    bag_to_point = (
        reference_data
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


# =============================================================================
# FIXED INNER SPLITS
# =============================================================================


def test_inner_splits_are_reproducible(
    evaluator: ModelEvaluator,
    reference_data: pd.DataFrame,
) -> None:
    """
    The same outer fold must generate the same inner partition.

    This allows all model families to be compared using identical folds.
    """

    first = evaluator._make_inner_splits(
        reference_data,
        outer_repeat=1,
        outer_fold=1,
    )

    second = evaluator._make_inner_splits(
        reference_data,
        outer_repeat=1,
        outer_fold=1,
    )

    assert first == second


# =============================================================================
# OUTER TEST EXCLUSION
# =============================================================================


def test_optuna_receives_only_outer_training_bags(
    evaluator: ModelEvaluator,
    signatures: dict[str, pd.DataFrame],
    monkeypatch,
) -> None:
    """
    Inner optimization must never receive a CapturePointId belonging to the
    current outer-test partition.
    """

    reference = signatures[
        "mean"
    ]

    checked_calls = []

    def fake_optimize_family(
        model_enum,
        signatures,
        train_bags,
        inner_splits,
        feature_dimensions,
        min_inner_train_size,
        preprocessing_cache,
        outer_repeat,
        outer_fold,
    ):

        train_bag_set = set(
            train_bags
        )

        all_bags = set(
            reference[
                "CapturePointId"
            ]
        )

        outer_test_bags = (
            all_bags
            - train_bag_set
        )

        for split in inner_splits:

            inner_bags = (
                set(
                    split.train_bags
                )
                | set(
                    split.validation_bags
                )
            )

            assert inner_bags <= train_bag_set

            assert inner_bags.isdisjoint(
                outer_test_bags
            )

        checked_calls.append(
            (
                outer_repeat,
                outer_fold,
                model_enum,
            )
        )

        return {
            "inner_mae":
                1.0,
            "configuration": {
                "model":
                    model_enum,
                "feature_set":
                    "indices",
                "aggregation":
                    "mean",
                "reduction":
                    "none",
                "pca_indices_components":
                    None,
                "pca_embeddings_components":
                    None,
                "model_params":
                    {},
            },
        }

    monkeypatch.setattr(
        evaluator,
        "_optimize_family",
        fake_optimize_family,
    )

    def fake_fit_selected_pipeline(
        signatures,
        train_bags,
        test_bags,
        configuration,
    ):

        return (
            np.zeros(
                len(
                    test_bags
                )
            ),
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        evaluator,
        "_fit_selected_pipeline",
        fake_fit_selected_pipeline,
    )

    monkeypatch.setattr(
        evaluator,
        "_select_final_configuration",
        lambda **kwargs: (
            {},
            pd.DataFrame(),
        ),
    )

    monkeypatch.setattr(
        evaluator,
        "_save_results",
        lambda **kwargs: None,
    )

    evaluator.evaluate(
        signatures
    )

    assert checked_calls


# =============================================================================
# MODEL-FAMILY SELECTION
# =============================================================================


def test_model_family_is_selected_by_inner_mae_only(
    evaluator: ModelEvaluator,
    signatures: dict[str, pd.DataFrame],
    monkeypatch,
) -> None:
    """
    The model fitted on the outer fold must be the family with the lowest
    inner-CV MAE.

    Outer-test performance must not participate in this choice.
    """

    selected_models = []

    def fake_optimize_family(
        model_enum,
        signatures,
        train_bags,
        inner_splits,
        feature_dimensions,
        min_inner_train_size,
        preprocessing_cache,
        outer_repeat,
        outer_fold,
    ):

        if (
            model_enum
            == RegressionModels.RIDGE_REGRESSION
        ):
            inner_mae = 5.0

        else:
            inner_mae = 2.0

        return {
            "inner_mae":
                inner_mae,
            "configuration": {
                "model":
                    model_enum,
                "feature_set":
                    "indices",
                "aggregation":
                    "mean",
                "reduction":
                    "none",
                "pca_indices_components":
                    None,
                "pca_embeddings_components":
                    None,
                "model_params":
                    {},
            },
        }

    monkeypatch.setattr(
        evaluator,
        "_optimize_family",
        fake_optimize_family,
    )

    def fake_fit_selected_pipeline(
        signatures,
        train_bags,
        test_bags,
        configuration,
    ):

        selected_models.append(
            configuration[
                "model"
            ]
        )

        return (
            np.zeros(
                len(
                    test_bags
                )
            ),
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        evaluator,
        "_fit_selected_pipeline",
        fake_fit_selected_pipeline,
    )

    monkeypatch.setattr(
        evaluator,
        "_select_final_configuration",
        lambda **kwargs: (
            {},
            pd.DataFrame(),
        ),
    )

    monkeypatch.setattr(
        evaluator,
        "_save_results",
        lambda **kwargs: None,
    )

    evaluator.evaluate(
        signatures
    )

    assert selected_models

    assert all(
        model
        == RegressionModels.SVR
        for model in selected_models
    )


# =============================================================================
# POINT-BALANCED BASELINES
# =============================================================================


def test_baselines_are_point_balanced(
    evaluator: ModelEvaluator,
) -> None:
    """
    Points, rather than CapturePointIds, receive equal total weight when the
    training-only mean and median baselines are calculated.
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

    mean_value, median_value = (
        evaluator._baseline_values(
            df
        )
    )

    # P1 contributes a Point-level mean of 5.
    #
    # Therefore the equal-Point mean is:
    #
    #     (5 + 50 + 100) / 3
    #
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
    A Point with multiple CapturePointIds must not dominate the reported MAE.
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
        evaluator._regression_metrics(
            df=df,
            predictions=predictions,
        )
    )

    # P1 error = 10
    # P2 error = 0
    #
    # Equal Point weighting:
    #
    #     MAE = (10 + 0) / 2 = 5
    #
    assert metrics.mae == pytest.approx(
        5.0
    )
