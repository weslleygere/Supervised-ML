import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.core.data.schema import Schema
from src.core.processors.feature_selection import (
    WeightedCorrelationSelector,
)


# =============================================================================
# WEIGHTED PCA
# =============================================================================


class WeightedPCA:
    """
    Principal Component Analysis with sample weights.

    The weighted covariance structure is represented through a weighted
    singular-value decomposition.

    This allows physical Points to contribute equally to PCA even when they
    contain different numbers of CapturePointIds.
    """

    def __init__(
        self,
        n_components: int,
    ) -> None:

        self.n_components = n_components

        self.n_components_: int | None = None

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        x: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "WeightedPCA":
        """
        Fit weighted PCA.
        """

        matrix = np.asarray(
            x,
            dtype=float,
        )

        weights = np.asarray(
            sample_weight,
            dtype=float,
        )

        self._validate_inputs(
            matrix,
            weights,
        )

        n_samples, n_features = (
            matrix.shape
        )

        max_components = min(
            n_features,
            n_samples - 1,
        )

        if (
            self.n_components
            > max_components
        ):
            raise ValueError(
                "Requested PCA dimensionality is not feasible. "
                f"Requested: {self.n_components}; "
                f"maximum: {max_components}."
            )

        # =====================================================================
        # WEIGHTED CENTER
        # =====================================================================

        self.mean_ = np.average(
            matrix,
            axis=0,
            weights=weights,
        )

        centered = (
            matrix
            - self.mean_
        )

        # =====================================================================
        # WEIGHTED SVD
        # =====================================================================

        normalized_weights = (
            weights
            / weights.sum()
        )

        weighted_matrix = (
            centered
            * np.sqrt(
                normalized_weights
            )[:, np.newaxis]
        )

        _, singular_values, components = (
            np.linalg.svd(
                weighted_matrix,
                full_matrices=False,
            )
        )

        self.n_components_ = (
            self.n_components
        )

        self.components_ = (
            components[
                :self.n_components
            ]
        )

        self.singular_values_ = (
            singular_values[
                :self.n_components
            ]
        )

        all_variances = (
            singular_values
            ** 2
        )

        self.explained_variance_ = (
            all_variances[
                :self.n_components
            ]
        )

        total_variance = (
            all_variances.sum()
        )

        if total_variance > 0:

            self.explained_variance_ratio_ = (
                self.explained_variance_
                / total_variance
            )

        else:

            self.explained_variance_ratio_ = (
                np.zeros(
                    self.n_components,
                    dtype=float,
                )
            )

        return self

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Project observations onto the fitted principal components.
        """

        if (
            self.mean_ is None
            or self.components_ is None
        ):

            raise RuntimeError(
                "WeightedPCA must be fitted before transform()."
            )

        matrix = np.asarray(
            x,
            dtype=float,
        )

        if matrix.ndim != 2:

            raise ValueError(
                "PCA input must be a two-dimensional matrix."
            )

        if (
            matrix.shape[
                1
            ]
            != self.mean_.shape[
                0
            ]
        ):

            raise ValueError(
                "PCA input has a different number of features "
                "from the fitted data."
            )

        return (
            matrix
            - self.mean_
        ) @ self.components_.T

    # =========================================================================
    # FIT + TRANSFORM
    # =========================================================================

    def fit_transform(
        self,
        x: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        """
        Fit PCA and transform the training matrix.
        """

        self.fit(
            x,
            sample_weight,
        )

        return self.transform(
            x
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_inputs(
        matrix: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """
        Validate PCA input and sample weights.
        """

        if matrix.ndim != 2:

            raise ValueError(
                "PCA input must be a two-dimensional matrix."
            )

        if matrix.shape[
            0
        ] < 2:

            raise ValueError(
                "PCA requires at least two observations."
            )

        if weights.ndim != 1:

            raise ValueError(
                "sample_weight must be one-dimensional."
            )

        if (
            len(
                weights
            )
            != matrix.shape[
                0
            ]
        ):

            raise ValueError(
                "sample_weight length must match "
                "the number of observations."
            )

        if not np.all(
            np.isfinite(
                weights
            )
        ):

            raise ValueError(
                "sample_weight contains non-finite values."
            )

        if np.any(
            weights <= 0
        ):

            raise ValueError(
                "sample_weight values must be positive."
            )

        if not np.all(
            np.isfinite(
                matrix
            )
        ):

            raise ValueError(
                "PCA input contains non-finite values."
            )


# =============================================================================
# POST-SPLIT PROCESSOR
# =============================================================================


class PostSplitProcessor:
    """
    Apply training-specific preprocessing.

    Feature blocks
    --------------
    indices:
        Point-balanced StandardScaler
        -> optional Point-balanced PCA or supervised feature selection

    embeddings:
        Point-balanced StandardScaler
        -> optional Point-balanced PCA or supervised feature selection

    both:
        process indices and embeddings independently
        -> normalize each processed block to equal weighted energy
        -> concatenate the resulting blocks

    Target
    ------
    HFI is standardized using a Point-balanced StandardScaler.

    Weighting
    ---------
    Every physical Point receives the same total preprocessing weight.

    If one Point contains multiple CapturePointIds, its weight is divided
    among those CapturePointIds.

    Leakage prevention
    ------------------
    All learned preprocessing parameters are fitted exclusively on the
    corresponding training fold or, for the final model, on the complete
    development set.
    """

    def __init__(
        self,
        schema: Schema,
        feature_set: str,
        reduction: str,
        pca_indices_components: int | None = None,
        pca_embeddings_components: int | None = None,
        selection_indices_features: int | None = None,
        selection_embeddings_features: int | None = None,
    ) -> None:

        self.schema = schema

        self.feature_set = (
            feature_set
        )

        self.reduction = (
            reduction
        )

        self.pca_indices_components = (
            pca_indices_components
        )

        self.pca_embeddings_components = (
            pca_embeddings_components
        )

        self.selection_indices_features = (
            selection_indices_features
        )

        self.selection_embeddings_features = (
            selection_embeddings_features
        )

        self._validate_configuration()

        # =====================================================================
        # SCALERS
        # =====================================================================

        self.index_scaler = (
            StandardScaler()
        )

        self.embedding_scaler = (
            StandardScaler()
        )

        self.target_scaler = (
            StandardScaler()
        )

        # =====================================================================
        # PCA
        # =====================================================================

        self.index_pca: (
            WeightedPCA
            | None
        ) = None

        self.embedding_pca: (
            WeightedPCA
            | None
        ) = None

        # =====================================================================
        # SUPERVISED FEATURE SELECTION
        # =====================================================================

        self.index_selector: (
            WeightedCorrelationSelector
            | None
        ) = None

        self.embedding_selector: (
            WeightedCorrelationSelector
            | None
        ) = None

        # =====================================================================
        # BLOCK NORMALIZATION
        # =====================================================================

        self.index_block_scale_: float = (
            1.0
        )

        self.embedding_block_scale_: float = (
            1.0
        )

        # =====================================================================
        # FEATURE METADATA
        # =====================================================================

        self.index_cols: list[str] = []

        self.processed_index_cols: list[str] = []

        self.processed_embedding_cols: list[str] = []

    # =========================================================================
    # FIT + TRANSFORM
    # =========================================================================

    def fit_transform(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Fit preprocessing using training data only and transform it.
        """

        self._validate_training_data(
            x_train,
            y_train,
        )

        sample_weight = (
            self._point_weights(
                x_train
            )
        )

        target = (
            y_train[
                self.schema.target
            ]
            .to_numpy(
                dtype=float
            )
        )

        # =====================================================================
        # FEATURES
        # =====================================================================

        x_transformed = (
            self._fit_transform_features(
                x_train=x_train,
                target=target,
                sample_weight=sample_weight,
            )
        )

        # =====================================================================
        # TARGET
        # =====================================================================

        target_matrix = (
            target.reshape(
                -1,
                1,
            )
        )

        self.target_scaler.fit(
            target_matrix,
            sample_weight=sample_weight,
        )

        transformed_target = (
            self.target_scaler.transform(
                target_matrix
            )
        )

        y_transformed = pd.DataFrame(
            transformed_target,
            columns=[
                self.schema.target
            ],
            index=y_train.index,
        )

        return (
            x_transformed,
            y_transformed,
        )

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform validation or test data using preprocessing fitted on
        training data.
        """

        parts: list[
            pd.DataFrame
        ] = []

        if self.feature_set in {
            "indices",
            "both",
        }:

            parts.append(
                self._transform_indices(
                    x
                )
            )

        if self.feature_set in {
            "embeddings",
            "both",
        }:

            parts.append(
                self._transform_embeddings(
                    x
                )
            )

        return pd.concat(
            parts,
            axis=1,
        )

    # =========================================================================
    # TARGET
    # =========================================================================

    def inverse_transform_target(
        self,
        y_pred: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return predictions to the original HFI scale.
        """

        target_matrix = (
            y_pred[
                [
                    self.schema.target
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        restored = (
            self.target_scaler
            .inverse_transform(
                target_matrix
            )
        )

        return pd.DataFrame(
            restored,
            columns=[
                self.schema.target
            ],
            index=y_pred.index,
        )

    # =========================================================================
    # FEATURES
    # =========================================================================

    def _fit_transform_features(
        self,
        x_train: pd.DataFrame,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        """
        Fit and transform the selected feature blocks.
        """

        parts: list[
            pd.DataFrame
        ] = []

        if self.feature_set in {
            "indices",
            "both",
        }:

            parts.append(
                self._fit_transform_indices(
                    x_train=x_train,
                    target=target,
                    sample_weight=sample_weight,
                )
            )

        if self.feature_set in {
            "embeddings",
            "both",
        }:

            parts.append(
                self._fit_transform_embeddings(
                    x_train=x_train,
                    target=target,
                    sample_weight=sample_weight,
                )
            )

        return pd.concat(
            parts,
            axis=1,
        )

    # =========================================================================
    # ACOUSTIC INDICES
    # =========================================================================

    def _fit_transform_indices(
        self,
        x_train: pd.DataFrame,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        """
        Fit the acoustic-index preprocessing block.
        """

        self.index_cols = (
            self.schema.index_columns(
                x_train.columns
            )
        )

        matrix = (
            x_train[
                self.index_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        self._validate_feature_matrix(
            matrix,
            block_name="indices",
        )

        # =====================================================================
        # WEIGHTED STANDARDIZATION
        # =====================================================================

        self.index_scaler.fit(
            matrix,
            sample_weight=sample_weight,
        )

        matrix = (
            self.index_scaler
            .transform(
                matrix
            )
        )

        # =====================================================================
        # REDUCTION
        # =====================================================================

        if self.reduction == "pca":

            self.index_pca = (
                WeightedPCA(
                    n_components=(
                        self.pca_indices_components
                    )
                )
            )

            matrix = (
                self.index_pca
                .fit_transform(
                    matrix,
                    sample_weight=(
                        sample_weight
                    ),
                )
            )

            self.processed_index_cols = [
                f"indices_pc_{i + 1}"
                for i in range(
                    self.pca_indices_components
                )
            ]

        elif (
            self.reduction
            == "supervised_selection"
        ):

            self.index_selector = (
                WeightedCorrelationSelector(
                    n_features=(
                        self.selection_indices_features
                    )
                )
            )

            matrix = (
                self.index_selector
                .fit_transform(
                    matrix,
                    target,
                    sample_weight,
                )
            )

            self.processed_index_cols = [
                self.index_cols[
                    index
                ]
                for index
                in (
                    self.index_selector
                    .selected_indices_
                )
            ]

        else:

            self.processed_index_cols = (
                self.index_cols.copy()
            )

        # =====================================================================
        # BLOCK BALANCING
        # =====================================================================

        if self.feature_set == "both":

            self.index_block_scale_ = (
                self._block_scale(
                    matrix,
                    sample_weight,
                )
            )

            matrix = (
                matrix
                / self.index_block_scale_
            )

        return pd.DataFrame(
            matrix,
            columns=(
                self.processed_index_cols
            ),
            index=x_train.index,
        )

    def _transform_indices(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform the acoustic-index block without refitting.
        """

        matrix = (
            x[
                self.index_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        matrix = (
            self.index_scaler
            .transform(
                matrix
            )
        )

        if (
            self.index_pca
            is not None
        ):

            matrix = (
                self.index_pca
                .transform(
                    matrix
                )
            )

        elif (
            self.index_selector
            is not None
        ):

            matrix = (
                self.index_selector
                .transform(
                    matrix
                )
            )

        if self.feature_set == "both":

            matrix = (
                matrix
                / self.index_block_scale_
            )

        return pd.DataFrame(
            matrix,
            columns=(
                self.processed_index_cols
            ),
            index=x.index,
        )

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    def _fit_transform_embeddings(
        self,
        x_train: pd.DataFrame,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        """
        Fit the embedding preprocessing block.
        """

        matrix = np.stack(
            x_train[
                self.schema.embedding
            ]
            .to_numpy()
        ).astype(
            float
        )

        self._validate_feature_matrix(
            matrix,
            block_name="embeddings",
        )

        # =====================================================================
        # WEIGHTED STANDARDIZATION
        # =====================================================================

        self.embedding_scaler.fit(
            matrix,
            sample_weight=sample_weight,
        )

        matrix = (
            self.embedding_scaler
            .transform(
                matrix
            )
        )

        original_columns = [
            f"embedding_{i + 1}"
            for i in range(
                matrix.shape[
                    1
                ]
            )
        ]

        # =====================================================================
        # REDUCTION
        # =====================================================================

        if self.reduction == "pca":

            self.embedding_pca = (
                WeightedPCA(
                    n_components=(
                        self.pca_embeddings_components
                    )
                )
            )

            matrix = (
                self.embedding_pca
                .fit_transform(
                    matrix,
                    sample_weight=(
                        sample_weight
                    ),
                )
            )

            self.processed_embedding_cols = [
                f"embedding_pc_{i + 1}"
                for i in range(
                    self.pca_embeddings_components
                )
            ]

        elif (
            self.reduction
            == "supervised_selection"
        ):

            self.embedding_selector = (
                WeightedCorrelationSelector(
                    n_features=(
                        self.selection_embeddings_features
                    )
                )
            )

            matrix = (
                self.embedding_selector
                .fit_transform(
                    matrix,
                    target,
                    sample_weight,
                )
            )

            self.processed_embedding_cols = [
                original_columns[
                    index
                ]
                for index
                in (
                    self.embedding_selector
                    .selected_indices_
                )
            ]

        else:

            self.processed_embedding_cols = (
                original_columns
            )

        # =====================================================================
        # BLOCK BALANCING
        # =====================================================================

        if self.feature_set == "both":

            self.embedding_block_scale_ = (
                self._block_scale(
                    matrix,
                    sample_weight,
                )
            )

            matrix = (
                matrix
                / self.embedding_block_scale_
            )

        return pd.DataFrame(
            matrix,
            columns=(
                self.processed_embedding_cols
            ),
            index=x_train.index,
        )

    def _transform_embeddings(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform the embedding block without refitting.
        """

        matrix = np.stack(
            x[
                self.schema.embedding
            ]
            .to_numpy()
        ).astype(
            float
        )

        matrix = (
            self.embedding_scaler
            .transform(
                matrix
            )
        )

        if (
            self.embedding_pca
            is not None
        ):

            matrix = (
                self.embedding_pca
                .transform(
                    matrix
                )
            )

        elif (
            self.embedding_selector
            is not None
        ):

            matrix = (
                self.embedding_selector
                .transform(
                    matrix
                )
            )

        if self.feature_set == "both":

            matrix = (
                matrix
                / self.embedding_block_scale_
            )

        return pd.DataFrame(
            matrix,
            columns=(
                self.processed_embedding_cols
            ),
            index=x.index,
        )

    # =========================================================================
    # BLOCK BALANCING
    # =========================================================================

    @staticmethod
    def _block_scale(
        matrix: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        """
        Calculate the weighted RMS Euclidean norm of one feature block.

        Dividing by this value gives the block unit weighted energy:

            weighted mean(||x_i||²) = 1

        This is applied only when indices and embeddings are used together.
        """

        scale = np.sqrt(
            np.average(
                np.sum(
                    matrix
                    ** 2,
                    axis=1,
                ),
                weights=sample_weight,
            )
        )

        if scale == 0:

            return 1.0

        return float(
            scale
        )

    # =========================================================================
    # POINT-BALANCED WEIGHTS
    # =========================================================================

    def _point_weights(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:
        """
        Give every physical Point equal total preprocessing weight.
        """

        n_captures = (
            x.groupby(
                self.schema.group
            )[
                self.schema.bag
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

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate preprocessing configuration.
        """

        if self.feature_set not in {
            "indices",
            "embeddings",
            "both",
        }:

            raise ValueError(
                "Invalid feature_set: "
                f"{self.feature_set}"
            )

        if self.reduction not in {
            "none",
            "pca",
            "supervised_selection",
        }:

            raise ValueError(
                "Invalid reduction method: "
                f"{self.reduction}"
            )

        if self.reduction == "pca":

            if (
                self.feature_set
                in {
                    "indices",
                    "both",
                }
                and self.pca_indices_components
                is None
            ):

                raise ValueError(
                    "PCA components were not defined "
                    "for the indices block."
                )

            if (
                self.feature_set
                in {
                    "embeddings",
                    "both",
                }
                and self.pca_embeddings_components
                is None
            ):

                raise ValueError(
                    "PCA components were not defined "
                    "for the embedding block."
                )

        if (
            self.reduction
            == "supervised_selection"
        ):

            if (
                self.feature_set
                in {
                    "indices",
                    "both",
                }
                and self.selection_indices_features
                is None
            ):

                raise ValueError(
                    "Selection dimension was not defined "
                    "for the indices block."
                )

            if (
                self.feature_set
                in {
                    "embeddings",
                    "both",
                }
                and self.selection_embeddings_features
                is None
            ):

                raise ValueError(
                    "Selection dimension was not defined "
                    "for the embedding block."
                )

    def _validate_training_data(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> None:
        """
        Validate feature/target alignment before fitting preprocessing.
        """

        if len(
            x_train
        ) != len(
            y_train
        ):

            raise ValueError(
                "Training features and targets have different "
                "numbers of rows."
            )

        if not x_train.index.equals(
            y_train.index
        ):

            raise ValueError(
                "Training features and targets must have aligned indices."
            )

        target = (
            y_train[
                self.schema.target
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.all(
            np.isfinite(
                target
            )
        ):

            raise ValueError(
                "Training target contains non-finite values."
            )

    @staticmethod
    def _validate_feature_matrix(
        matrix: np.ndarray,
        block_name: str,
    ) -> None:
        """
        Reject malformed or non-finite feature matrices.
        """

        if matrix.ndim != 2:

            raise ValueError(
                f"{block_name} feature block must be two-dimensional."
            )

        if matrix.shape[
            1
        ] < 1:

            raise ValueError(
                f"{block_name} feature block contains no features."
            )

        if not np.all(
            np.isfinite(
                matrix
            )
        ):

            raise ValueError(
                f"{block_name} feature block contains "
                "non-finite values."
            )