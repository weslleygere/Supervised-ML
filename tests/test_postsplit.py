import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.processors.postsplit import PostSplitProcessor


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
        index_prefixes=("ACI", "NDSI"),
        embedding="Embedding",
    )


@pytest.fixture
def train_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
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
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([2.0, 3.0, 4.0, 5.0]),
                np.array([3.0, 4.0, 5.0, 6.0]),
                np.array([4.0, 5.0, 6.0, 7.0]),
                np.array([5.0, 6.0, 7.0, 8.0]),
                np.array([6.0, 7.0, 8.0, 9.0]),
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
                    [100.0, 200.0, 300.0, 400.0]
                ),
                np.array(
                    [200.0, 400.0, 600.0, 800.0]
                ),
            ],
        }
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
    Validation/test observations must not influence StandardScaler.
    """

    X_train, y_train = train_data

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
        .to_numpy()
        .mean(
            axis=0
        )
    )

    np.testing.assert_allclose(
        processor.index_scaler.mean_,
        expected_training_mean,
    )

    mean_before_test = (
        processor.index_scaler.mean_.copy()
    )

    processor.transform(
        test_data
    )

    np.testing.assert_allclose(
        processor.index_scaler.mean_,
        mean_before_test,
    )


# =============================================================================
# TRAINING-ONLY TARGET SCALING
# =============================================================================


def test_target_scaler_is_fitted_only_on_training_target(
    schema: Schema,
    train_data,
) -> None:
    """
    RobustScaler for HFI must be fitted exclusively from training targets.
    """

    X_train, y_train = train_data

    processor = PostSplitProcessor(
        schema=schema,
        feature_set="indices",
        reduction="none",
    )

    processor.fit_transform(
        X_train,
        y_train,
    )

    expected_median = np.median(
        y_train[
            "meanHFI"
        ]
    )

    assert processor.target_scaler.center_[0] == pytest.approx(
        expected_median
    )


# =============================================================================
# TARGET INVERSE TRANSFORMATION
# =============================================================================


def test_target_inverse_transform_restores_original_scale(
    schema: Schema,
    train_data,
) -> None:
    """
    Predictions transformed back from the model scale must return to the
    original HFI scale.
    """

    X_train, y_train = train_data

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
        processor.inverse_transform_target(
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

    X_train, y_train = train_data

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
        len(X_train),
        2,
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

    X_train, y_train = train_data

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
        len(X_train),
        3,
    )

    assert processor.embedding_pca.n_components_ == 3

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
    PostSplitProcessor must not replace an infeasible selected PCA dimension
    with a smaller value.

    Feasibility is the responsibility of the search space.
    """

    X_train, y_train = train_data

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
    Indices and embeddings must receive independent scaling and PCA before
    concatenation.
    """

    X_train, y_train = train_data

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
        len(X_train),
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

    assert processor.index_pca.n_components_ == 2
    assert processor.embedding_pca.n_components_ == 3


# =============================================================================
# TEST TRANSFORMATION
# =============================================================================


def test_transform_does_not_refit_preprocessing(
    schema: Schema,
    train_data,
    test_data: pd.DataFrame,
) -> None:
    """
    Calling transform() must reuse the scaler and PCA fitted on training data.
    """

    X_train, y_train = train_data

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
        processor.embedding_scaler.mean_.copy()
    )

    pca_components_before = (
        processor.embedding_pca
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
        processor.embedding_pca.components_,
        pca_components_before,
    )

    assert X_test_processed.shape == (
        len(test_data),
        2,
    )


# =============================================================================
# PCA CONFIGURATION
# =============================================================================


def test_pca_requires_components_for_active_index_block(
    schema: Schema,
    train_data,
) -> None:

    X_train, y_train = train_data

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

    X_train, y_train = train_data

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
