import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.core.data.schema import Schema


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

        self.components_ = components[
            :self.n_components
        ]

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
            matrix.shape[1]
            != self.mean_.shape[0]
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

        if matrix.shape[0] < 2:
            raise ValueError(
                "PCA requires at least two observations."
            )

        if weights.ndim != 1:
            raise ValueError(
                "sample_weight must be one-dimensional."
            )

        if (
            len(weights)
            != matrix.shape[0]
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
        -> optional Point-balanced PCA

    embeddings:
        Point-balanced StandardScaler
        -> optional Point-balanced PCA

    both:
        process indices and embeddings independently,
        then concatenate the resulting blocks.

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
    All preprocessing parameters are fitted exclusively on the corresponding
    training fold or, for the final model, on the complete development set.
    """

    def __init__(
        self,
        schema: Schema,
        feature_set: str,
        reduction: str,
        pca_indices_components: int | None = None,
        pca_embeddings_components: int | None = None,
    ) -> None:

        self.schema = schema

        self.feature_set = feature_set
        self.reduction = reduction

        self.pca_indices_components = (
            pca_indices_components
        )

        self.pca_embeddings_components = (
            pca_embeddings_components
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

        # =====================================================================
        # FEATURES
        # =====================================================================

        x_transformed = (
            self._fit_transform_features(
                x_train=x_train,
                sample_weight=sample_weight,
            )
        )

        # =====================================================================
        # TARGET
        # =====================================================================

        target_matrix = (
            y_train[
                [
                    self.schema.target
                ]
            ]
            .to_numpy(
                dtype=float
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

        if not parts:
            raise RuntimeError(
                "No active feature blocks were available."
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
                    sample_weight=(
                        sample_weight
                    ),
                )
            )

        if self.feature_set in {
            "embeddings",
            "both",
        }:

            parts.append(
                self._fit_transform_embeddings(
                    x_train=x_train,
                    sample_weight=(
                        sample_weight
                    ),
                )
            )

        if not parts:
            raise RuntimeError(
                "No active feature blocks were available."
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
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        """
        Point-balance the acoustic-index scaling and optionally apply
        Point-balanced PCA.
        """

        self.index_cols = (
            self.schema.index_columns(
                x_train.columns
            )
        )

        if not self.index_cols:
            raise ValueError(
                "No acoustic-index columns were found."
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
        # OPTIONAL WEIGHTED PCA
        # =====================================================================

        if self.reduction == "pca":

            if (
                self.pca_indices_components
                is None
            ):
                raise ValueError(
                    "PCA components were not defined "
                    "for the indices block."
                )

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

        else:

            self.processed_index_cols = (
                self.index_cols.copy()
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

        self._validate_feature_matrix(
            matrix,
            block_name="indices",
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
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        """
        Point-balance embedding scaling and optionally apply Point-balanced
        PCA.
        """

        if (
            self.schema.embedding
            not in x_train.columns
        ):
            raise ValueError(
                "Embedding column not found: "
                f"{self.schema.embedding}"
            )

        matrix = np.stack(
            x_train[
                self.schema.embedding
            ].to_numpy()
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

        # =====================================================================
        # OPTIONAL WEIGHTED PCA
        # =====================================================================

        if self.reduction == "pca":

            if (
                self.pca_embeddings_components
                is None
            ):
                raise ValueError(
                    "PCA components were not defined "
                    "for the embedding block."
                )

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

        else:

            self.processed_embedding_cols = [
                f"embedding_{i + 1}"
                for i in range(
                    matrix.shape[
                        1
                    ]
                )
            ]

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

        if (
            self.schema.embedding
            not in x.columns
        ):
            raise ValueError(
                "Embedding column not found: "
                f"{self.schema.embedding}"
            )

        matrix = np.stack(
            x[
                self.schema.embedding
            ].to_numpy()
        ).astype(
            float
        )

        self._validate_feature_matrix(
            matrix,
            block_name="embeddings",
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

        return pd.DataFrame(
            matrix,
            columns=(
                self.processed_embedding_cols
            ),
            index=x.index,
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

        If a Point contains N CapturePointIds, each CapturePointId receives
        weight 1/N.

        Weights are normalized to mean one. The normalization does not change
        relative weighting but keeps their numerical scale convenient.
        """

        if (
            self.schema.group
            not in x.columns
        ):
            raise ValueError(
                "Grouping column not found in preprocessing data: "
                f"{self.schema.group}"
            )

        if (
            self.schema.bag
            not in x.columns
        ):
            raise ValueError(
                "CapturePointId column not found in preprocessing data: "
                f"{self.schema.bag}"
            )

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

        if np.any(
            n_captures <= 0
        ):
            raise ValueError(
                "Invalid number of CapturePointIds per Point."
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

        valid_feature_sets = {
            "indices",
            "embeddings",
            "both",
        }

        if (
            self.feature_set
            not in valid_feature_sets
        ):
            raise ValueError(
                "Invalid feature_set: "
                f"{self.feature_set}"
            )

        valid_reductions = {
            "none",
            "pca",
        }

        if (
            self.reduction
            not in valid_reductions
        ):
            raise ValueError(
                "Invalid reduction method: "
                f"{self.reduction}"
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

        if (
            self.schema.target
            not in y_train.columns
        ):
            raise ValueError(
                "Target column not found: "
                f"{self.schema.target}"
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

        if matrix.shape[1] < 1:
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
