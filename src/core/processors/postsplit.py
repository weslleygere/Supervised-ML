import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.core.data.schema import Schema


# =============================================================================
# POST-SPLIT PROCESSOR
# =============================================================================


class PostSplitProcessor:
    """
    Apply fold-specific preprocessing.

    Feature blocks
    --------------
    indices:
        StandardScaler
        -> optional PCA

    embeddings:
        StandardScaler
        -> optional PCA

    both:
        process indices and embeddings independently,
        then concatenate the resulting blocks.

    Target
    ------
    HFI is transformed with RobustScaler.

    All scalers and PCA transformations are fitted exclusively on the
    corresponding training fold.
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

        self.index_scaler = StandardScaler()
        self.embedding_scaler = StandardScaler()
        self.target_scaler = RobustScaler()

        self.index_pca: PCA | None = None
        self.embedding_pca: PCA | None = None

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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit preprocessing on the training fold and transform it.
        """

        x_transformed = (
            self._fit_transform_features(
                x_train
            )
        )

        y_transformed = y_train.copy()

        y_transformed.loc[
            :, [self.schema.target]
        ] = self.target_scaler.fit_transform(
            y_train[
                [self.schema.target]
            ]
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
        Transform validation or test data using transformations fitted on
        the corresponding training fold.
        """

        parts: list[pd.DataFrame] = []

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

        result = y_pred.copy()

        result.loc[
            :, [self.schema.target]
        ] = self.target_scaler.inverse_transform(
            result[
                [self.schema.target]
            ]
        )

        return result

    # =========================================================================
    # FEATURES
    # =========================================================================

    def _fit_transform_features(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit and transform the selected feature blocks.
        """

        parts: list[pd.DataFrame] = []

        if self.feature_set in {
            "indices",
            "both",
        }:
            parts.append(
                self._fit_transform_indices(
                    x_train
                )
            )

        if self.feature_set in {
            "embeddings",
            "both",
        }:
            parts.append(
                self._fit_transform_embeddings(
                    x_train
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
    ) -> pd.DataFrame:
        """
        Standardize the acoustic-index block and optionally apply PCA.
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

        matrix = self.index_scaler.fit_transform(
            matrix
        )

        if self.reduction == "pca":

            if self.pca_indices_components is None:
                raise ValueError(
                    "PCA components were not defined "
                    "for the indices block."
                )

            self.index_pca = PCA(
                n_components=(
                    self.pca_indices_components
                ),
                svd_solver="full",
            )

            matrix = self.index_pca.fit_transform(
                matrix
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
            columns=self.processed_index_cols,
            index=x_train.index,
        )

    def _transform_indices(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform the acoustic-index block.
        """

        matrix = (
            x[
                self.index_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        matrix = self.index_scaler.transform(
            matrix
        )

        if self.index_pca is not None:
            matrix = self.index_pca.transform(
                matrix
            )

        return pd.DataFrame(
            matrix,
            columns=self.processed_index_cols,
            index=x.index,
        )

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    def _fit_transform_embeddings(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Standardize the embedding block and optionally apply PCA.
        """

        matrix = np.stack(
            x_train[
                self.schema.embedding
            ].to_numpy()
        )

        matrix = (
            self.embedding_scaler.fit_transform(
                matrix
            )
        )

        if self.reduction == "pca":

            if self.pca_embeddings_components is None:
                raise ValueError(
                    "PCA components were not defined "
                    "for the embedding block."
                )

            self.embedding_pca = PCA(
                n_components=(
                    self.pca_embeddings_components
                ),
                svd_solver="full",
            )

            matrix = (
                self.embedding_pca.fit_transform(
                    matrix
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
                    matrix.shape[1]
                )
            ]

        return pd.DataFrame(
            matrix,
            columns=self.processed_embedding_cols,
            index=x_train.index,
        )

    def _transform_embeddings(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform the embedding block.
        """

        matrix = np.stack(
            x[
                self.schema.embedding
            ].to_numpy()
        )

        matrix = (
            self.embedding_scaler.transform(
                matrix
            )
        )

        if self.embedding_pca is not None:
            matrix = self.embedding_pca.transform(
                matrix
            )

        return pd.DataFrame(
            matrix,
            columns=self.processed_embedding_cols,
            index=x.index,
        )
