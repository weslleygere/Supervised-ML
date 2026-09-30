import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.processors.postsplit import (
    PostSplitProcessor,
    WeightedPCA,
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
        index_prefixes=(
            "ACI",
            "NDSI",
        ),
        embedding="Embedding",
    )


@pytest.fixture
def train_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Simple training dataset with one CapturePointId per Point.

    Equal Point weights therefore reduce to equal row weights.
    """

    X = pd.DataFrame(
        {
            "Point": [
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
            ],

            "CapturePointId": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
            ],

            "ACI_mean": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],

            "NDSI_mean": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
            ],

            "Embedding": [
                np.array(
                    [1.0, 2.0, 3.0, 4.0]
                ),
                np.array(
                    [2.0, 3.0, 4.0, 5.0]
                ),
                np.array(
                    [3.0, 4.0, 5.0, 6.0]
                ),
                np.array(
                    [4.0, 5.0, 6.0, 7.0]
                ),
                np.array(
                    [5.0, 6.0, 7.0, 8.0]
                ),
                np.array(
                    [6.0, 7.0, 8.0, 9.0]
                ),
            ],
        }
    )

    y = pd.DataFrame(
        {
            "meanHFI": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ]
        }
    )

    return X, y


@pytest.fixture
def unequal_point_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Training dataset where P1 contains two CapturePointIds.

    Point-balanced preprocessing must ensure that P1 collectively has the
    same total weight as P2 and P3.
    """

    X = pd.DataFrame(
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

            "ACI_mean": [
                0.0,
                10.0,
                20.0,
                40.0,
            ],

            "NDSI_mean": [
                0.0,
                20.0,
                40.0,
                80.0,
            ],

            "Embedding": [
                np.array(
                    [0.0, 0.0]
                ),
                np.array(
                    [10.0, 10.0]
                ),
                np.array(
                    [20.0, 20.0]
                ),
                np.array(
                    [40.0, 40.0]
                ),
            ],
        }
    )

    y = pd.DataFrame(
        {
            "meanHFI": [
                0.0,
                10.0,
                20.0,
                40.0,
            ]
        }
    )

    return X, y


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Point": [
                "P7",
                "P8",
            ],

            "CapturePointId": [
                "B7",
                "B8",
            ],

            "ACI_mean": [
                100.0,
                200.0,
            ],

            "NDSI_mean": [
                200.0,
                400.0,
            ],

            "Embedding": [
                np.array(
                    [
                        100.0,
                        200.0,
                        300.0,
                        400.0,
                    ]
                ),
                np.array(
                    [
                        200.0,
                        400.0,
                        600.0,
                        800.0,
                    ]
                ),
            ],
        }
    )


# =============================================================================
# POINT-BALANCED WEIGHTS
# =============================================================================


def test_point_weights_give_each_point_equal_total_weight(
    schema: Schema,
    unequal_point_data,
) -> None:
    """
    Multiple CapturePointIds belonging to the same Point must divide that
    Point's total preprocessing weight.
    """

    X_train, _ = (
        unequal_point_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    weights = (
        processor._point_weights(
            X_train
        )
    )

    weighted = X_train[
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


def test_point_weights_have_mean_one(
    schema: Schema,
    unequal_point_data,
) -> None:
    """
    Weight normalization must preserve relative Point weighting while keeping
    the average sample weight equal to one.
    """

    X_train, _ = (
        unequal_point_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    weights = (
        processor._point_weights(
            X_train
        )
    )

    assert weights.mean() == pytest.approx(
        1.0
    )


# =============================================================================
# TRAINING-ONLY FEATURE SCALING
# =============================================================================


def test_index_scaler_is_fitted_only_on_training_data(
    schema: Schema,
    train_data,
    test_data: pd.DataFrame,
) -> None:
    """
    Validation/test observations must never influence feature scaling.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    expected_training_mean = (
        X_train[
            [
                "ACI_mean",
                "NDSI_mean",
            ]
        ]
        .to_numpy(
            dtype=float
        )
        .mean(
            axis=0
        )
    )

    np.testing.assert_allclose(
        processor.index_scaler.mean_,
        expected_training_mean,
    )

    mean_before_test = (
        processor
        .index_scaler
        .mean_
        .copy()
    )

    processor.transform(
        test_data
    )

    np.testing.assert_allclose(
        processor.index_scaler.mean_,
        mean_before_test,
    )


# =============================================================================
# POINT-BALANCED FEATURE SCALING
# =============================================================================


def test_index_scaler_is_point_balanced(
    schema: Schema,
    unequal_point_data,
) -> None:
    """
    A Point with multiple CapturePointIds must not receive extra influence
    when estimating feature means.
    """

    X_train, y_train = (
        unequal_point_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    # Raw Point weights:
    #
    # P1:
    #     B1 = 1/2
    #     B2 = 1/2
    #
    # P2:
    #     B3 = 1
    #
    # P3:
    #     B4 = 1
    #
    # Therefore:
    #
    # ACI weighted mean =
    #
    # (0*0.5 + 10*0.5 + 20*1 + 40*1) / 3

    expected_aci_mean = (
        (
            0.0 * 0.5
            + 10.0 * 0.5
            + 20.0
            + 40.0
        )
        / 3.0
    )

    expected_ndsi_mean = (
        (
            0.0 * 0.5
            + 20.0 * 0.5
            + 40.0
            + 80.0
        )
        / 3.0
    )

    np.testing.assert_allclose(
        processor.index_scaler.mean_,
        np.array(
            [
                expected_aci_mean,
                expected_ndsi_mean,
            ]
        ),
    )

    # Confirm that this is not simply the unweighted row mean.

    unweighted_mean = (
        X_train[
            [
                "ACI_mean",
                "NDSI_mean",
            ]
        ]
        .to_numpy(
            dtype=float
        )
        .mean(
            axis=0
        )
    )

    assert not np.allclose(
        processor.index_scaler.mean_,
        unweighted_mean,
    )


def test_embedding_scaler_is_point_balanced(
    schema: Schema,
    unequal_point_data,
) -> None:
    """
    Embedding dimensions must use the same Point-balanced weighting.
    """

    X_train, y_train = (
        unequal_point_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="embeddings",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    expected_mean = (
        (
            np.array(
                [0.0, 0.0]
            )
            * 0.5
        )
        + (
            np.array(
                [10.0, 10.0]
            )
            * 0.5
        )
        + np.array(
            [20.0, 20.0]
        )
        + np.array(
            [40.0, 40.0]
        )
    ) / 3.0

    np.testing.assert_allclose(
        processor.embedding_scaler.mean_,
        expected_mean,
    )


# =============================================================================
# TARGET SCALING
# =============================================================================


def test_target_scaler_is_fitted_only_on_training_target(
    schema: Schema,
    train_data,
) -> None:
    """
    Target scaling must be fitted exclusively using training targets.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    expected_mean = (
        y_train[
            "meanHFI"
        ]
        .mean()
    )

    assert (
        processor
        .target_scaler
        .mean_[
            0
        ]
        == pytest.approx(
            expected_mean
        )
    )


def test_target_scaler_is_point_balanced(
    schema: Schema,
    unequal_point_data,
) -> None:
    """
    The target transformation must give every Point the same total influence.
    """

    X_train, y_train = (
        unequal_point_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    expected_mean = (
        (
            0.0 * 0.5
            + 10.0 * 0.5
            + 20.0
            + 40.0
        )
        / 3.0
    )

    assert (
        processor
        .target_scaler
        .mean_[
            0
        ]
        == pytest.approx(
            expected_mean
        )
    )

    unweighted_mean = (
        y_train[
            "meanHFI"
        ]
        .mean()
    )

    assert (
        processor
        .target_scaler
        .mean_[
            0
        ]
        != pytest.approx(
            unweighted_mean
        )
    )


# =============================================================================
# TARGET INVERSE TRANSFORMATION
# =============================================================================


def test_target_inverse_transform_restores_original_scale(
    schema: Schema,
    train_data,
) -> None:
    """
    Predictions transformed back from model space must exactly return to the
    original HFI scale.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    _, y_scaled = (
        processor.fit_transform(
            X_train,
            y_train,
        )
    )

    restored = (
        processor
        .inverse_transform_target(
            y_scaled
        )
    )

    np.testing.assert_allclose(
        restored[
            "meanHFI"
        ].to_numpy(),
        y_train[
            "meanHFI"
        ].to_numpy(),
    )


# =============================================================================
# NO PCA
# =============================================================================


def test_no_reduction_preserves_index_dimension(
    schema: Schema,
    train_data,
) -> None:
    """
    Without PCA, standardized acoustic indices retain their original
    dimensionality and names.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    X_processed, _ = (
        processor.fit_transform(
            X_train,
            y_train,
        )
    )

    assert list(
        X_processed.columns
    ) == [
        "ACI_mean",
        "NDSI_mean",
    ]

    assert X_processed.shape == (
        len(
            X_train
        ),
        2,
    )


def test_no_reduction_preserves_embedding_dimension(
    schema: Schema,
    train_data,
) -> None:
    """
    Without PCA, all standardized embedding dimensions are preserved.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="embeddings",
        reduction="none",
    )

    X_processed, _ = (
        processor.fit_transform(
            X_train,
            y_train,
        )
    )

    assert X_processed.shape == (
        len(
            X_train
        ),
        4,
    )

    assert list(
        X_processed.columns
    ) == [
        "embedding_1",
        "embedding_2",
        "embedding_3",
        "embedding_4",
    ]


# =============================================================================
# WEIGHTED PCA
# =============================================================================


def test_weighted_pca_uses_weighted_center() -> None:
    """
    WeightedPCA must center the matrix according to sample weights rather than
    using the ordinary row mean.
    """

    matrix = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [40.0, 40.0],
        ]
    )

    weights = np.array(
        [
            0.5,
            0.5,
            1.0,
            1.0,
        ]
    )

    pca = WeightedPCA(
        n_components=1
    )

    pca.fit(
        matrix,
        sample_weight=weights,
    )

    expected_mean = np.average(
        matrix,
        axis=0,
        weights=weights,
    )

    np.testing.assert_allclose(
        pca.mean_,
        expected_mean,
    )

    assert not np.allclose(
        pca.mean_,
        matrix.mean(
            axis=0
        ),
    )


def test_weighted_pca_transformed_training_data_has_zero_weighted_mean() -> None:
    """
    PCA scores from the training matrix must be centered under the same
    weights used to fit PCA.
    """

    matrix = np.array(
        [
            [0.0, 1.0],
            [10.0, 5.0],
            [20.0, 15.0],
            [40.0, 30.0],
        ]
    )

    weights = np.array(
        [
            0.5,
            0.5,
            1.0,
            1.0,
        ]
    )

    pca = WeightedPCA(
        n_components=2
    )

    transformed = (
        pca.fit_transform(
            matrix,
            sample_weight=weights,
        )
    )

    weighted_score_mean = np.average(
        transformed,
        axis=0,
        weights=weights,
    )

    np.testing.assert_allclose(
        weighted_score_mean,
        np.zeros(
            2
        ),
        atol=1e-12,
    )


# =============================================================================
# PCA DIMENSION
# =============================================================================


def test_pca_uses_exact_requested_dimension(
    schema: Schema,
    train_data,
) -> None:
    """
    A feasible PCA dimension must be fitted exactly as requested.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="embeddings",
        reduction="pca",
        pca_embeddings_components=3,
    )

    X_processed, _ = (
        processor.fit_transform(
            X_train,
            y_train,
        )
    )

    assert X_processed.shape == (
        len(
            X_train
        ),
        3,
    )

    assert (
        processor
        .embedding_pca
        .n_components_
        == 3
    )

    assert list(
        X_processed.columns
    ) == [
        "embedding_pc_1",
        "embedding_pc_2",
        "embedding_pc_3",
    ]


def test_invalid_pca_dimension_is_not_silently_reduced(
    schema: Schema,
    train_data,
) -> None:
    """
    PostSplitProcessor must reject an impossible selected PCA dimension.

    Feasibility must be determined by the search space rather than silently
    changing the selected pipeline.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="pca",
        pca_indices_components=3,
    )

    with pytest.raises(
        ValueError
    ):

        processor.fit_transform(
            X_train,
            y_train,
        )


# =============================================================================
# BOTH FEATURE BLOCKS
# =============================================================================


def test_both_feature_blocks_are_processed_independently(
    schema: Schema,
    train_data,
) -> None:
    """
    Indices and embeddings must receive separate scaling and PCA before being
    concatenated.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="both",
        reduction="pca",
        pca_indices_components=2,
        pca_embeddings_components=3,
    )

    X_processed, _ = (
        processor.fit_transform(
            X_train,
            y_train,
        )
    )

    assert X_processed.shape == (
        len(
            X_train
        ),
        5,
    )

    assert list(
        X_processed.columns
    ) == [
        "indices_pc_1",
        "indices_pc_2",
        "embedding_pc_1",
        "embedding_pc_2",
        "embedding_pc_3",
    ]

    assert (
        processor
        .index_pca
        .n_components_
        == 2
    )

    assert (
        processor
        .embedding_pca
        .n_components_
        == 3
    )


# =============================================================================
# VALIDATION / TEST TRANSFORMATION
# =============================================================================


def test_transform_does_not_refit_scaler(
    schema: Schema,
    train_data,
    test_data: pd.DataFrame,
) -> None:
    """
    transform() must reuse feature scaling learned from training data.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    scaler_mean_before = (
        processor
        .index_scaler
        .mean_
        .copy()
    )

    processor.transform(
        test_data
    )

    np.testing.assert_allclose(
        processor.index_scaler.mean_,
        scaler_mean_before,
    )


def test_transform_does_not_refit_pca(
    schema: Schema,
    train_data,
    test_data: pd.DataFrame,
) -> None:
    """
    transform() must reuse PCA fitted on training observations.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="embeddings",
        reduction="pca",
        pca_embeddings_components=2,
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    scaler_mean_before = (
        processor
        .embedding_scaler
        .mean_
        .copy()
    )

    pca_mean_before = (
        processor
        .embedding_pca
        .mean_
        .copy()
    )

    pca_components_before = (
        processor
        .embedding_pca
        .components_
        .copy()
    )

    X_test_processed = (
        processor.transform(
            test_data
        )
    )

    np.testing.assert_allclose(
        processor.embedding_scaler.mean_,
        scaler_mean_before,
    )

    np.testing.assert_allclose(
        processor.embedding_pca.mean_,
        pca_mean_before,
    )

    np.testing.assert_allclose(
        processor.embedding_pca.components_,
        pca_components_before,
    )

    assert X_test_processed.shape == (
        len(
            test_data
        ),
        2,
    )


# =============================================================================
# PCA CONFIGURATION
# =============================================================================


def test_pca_requires_components_for_active_index_block(
    schema: Schema,
    train_data,
) -> None:
    """
    PCA cannot be applied to indices without an explicit selected dimension.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="pca",
        pca_indices_components=None,
    )

    with pytest.raises(
        ValueError
    ):

        processor.fit_transform(
            X_train,
            y_train,
        )


def test_pca_requires_components_for_active_embedding_block(
    schema: Schema,
    train_data,
) -> None:
    """
    PCA cannot be applied to embeddings without an explicit selected
    dimension.
    """

    X_train, y_train = (
        train_data
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="embeddings",
        reduction="pca",
        pca_embeddings_components=None,
    )

    with pytest.raises(
        ValueError
    ):

        processor.fit_transform(
            X_train,
            y_train,
        )


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def test_misaligned_training_indices_are_rejected(
    schema: Schema,
    train_data,
) -> None:
    """
    Feature and target rows must remain explicitly aligned.
    """

    X_train, y_train = (
        train_data
    )

    y_train = (
        y_train.copy()
    )

    y_train.index = (
        y_train.index
        + 10
    )

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    with pytest.raises(
        ValueError,
        match="aligned indices",
    ):

        processor.fit_transform(
            X_train,
            y_train,
        )


def test_non_finite_features_are_rejected(
    schema: Schema,
    train_data,
) -> None:
    """
    NaN or infinite acoustic features must not silently enter preprocessing.
    """

    X_train, y_train = (
        train_data
    )

    X_train = (
        X_train.copy()
    )

    X_train.loc[
        0,
        "ACI_mean",
    ] = np.nan

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):

        processor.fit_transform(
            X_train,
            y_train,
        )


def test_non_finite_target_is_rejected(
    schema: Schema,
    train_data,
) -> None:
    """
    NaN or infinite HFI values must be rejected before model fitting.
    """

    X_train, y_train = (
        train_data
    )

    y_train = (
        y_train.copy()
    )

    y_train.loc[
        0,
        "meanHFI",
    ] = np.nan

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):

        processor.fit_transform(
            X_train,
            y_train,
        )
