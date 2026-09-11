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

    Feature representations
    -----------------------
    indices:
        StandardScaler -> PCA

    embeddings:
        StandardScaler -> PCA

    both:
        StandardScaler(indices) -> PCA_indices
        +
        StandardScaler(embeddings) -> PCA_embeddings

    All preprocessing steps are fitted only on the training fold.
    """

    def __init__(
        self,
        schema: Schema,
        feature_set: str,
        pca_indices_components: int | None,
        pca_embeddings_components: int | None,
    ) -> None:

        self.schema = schema
        self.feature_set = feature_set

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

        x_train_transformed = (
            self._fit_transform_features(
                x_train
            )
        )

        y_train_transformed = (
            y_train.copy()
        )

        y_train_transformed.loc[
            :, [self.schema.target]
        ] = self.target_scaler.fit_transform(
            y_train_transformed[
                [self.schema.target]
            ]
        )

        return (
            x_train_transformed,
            y_train_transformed,
        )

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(
        self,
        x_test: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform validation or test data using preprocessing
        fitted on the corresponding training fold.
        """

        parts: list[pd.DataFrame] = []

        if self.feature_set in {
            "indices",
            "both",
        }:
            parts.append(
                self._transform_indices(
                    x_test
                )
            )

        if self.feature_set in {
            "embeddings",
            "both",
        }:
            parts.append(
                self._transform_embeddings(
                    x_test
                )
            )

        if not parts:
            raise ValueError(
                f"Invalid feature set: "
                f"{self.feature_set}"
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

        y_pred = y_pred.copy()

        y_pred.loc[
            :, [self.schema.target]
        ] = self.target_scaler.inverse_transform(
            y_pred[
                [self.schema.target]
            ]
        )

        return y_pred

    # =========================================================================
    # FEATURE BLOCKS
    # =========================================================================

    def _fit_transform_features(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:

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

        if not parts:
            raise ValueError(
                f"Invalid feature set: "
                f"{self.feature_set}"
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
        Standardize aggregated acoustic-index features and apply PCA.
        """

        self.index_cols = (
            self.schema.index_columns(
                x_train.columns
            )
        )

        if not self.index_cols:
            raise ValueError(
                "No acoustic index columns were found."
            )

        matrix = (
            x_train[
                self.index_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        matrix = (
            self.index_scaler.fit_transform(
                matrix
            )
        )

        if self.pca_indices_components is not None:

            n_components = (
                self._effective_pca_components(
                    requested=(
                        self.pca_indices_components
                    ),
                    matrix=matrix,
                )
            )

            self.index_pca = PCA(
                n_components=n_components,
                svd_solver="full",
            )

            matrix = (
                self.index_pca.fit_transform(
                    matrix
                )
            )

            self.processed_index_cols = [
                f"indices_pc_{i + 1}"
                for i in range(
                    n_components
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
        x_test: pd.DataFrame,
    ) -> pd.DataFrame:

        matrix = (
            x_test[
                self.index_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        matrix = (
            self.index_scaler.transform(
                matrix
            )
        )

        if self.index_pca is not None:
            matrix = (
                self.index_pca.transform(
                    matrix
                )
            )

        return pd.DataFrame(
            matrix,
            columns=self.processed_index_cols,
            index=x_test.index,
        )

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    def _fit_transform_embeddings(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Standardize the aggregated embedding representation and apply PCA.
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

        if self.pca_embeddings_components is not None:

            n_components = (
                self._effective_pca_components(
                    requested=(
                        self.pca_embeddings_components
                    ),
                    matrix=matrix,
                )
            )

            self.embedding_pca = PCA(
                n_components=n_components,
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
                    n_components
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
        x_test: pd.DataFrame,
    ) -> pd.DataFrame:

        matrix = np.stack(
            x_test[
                self.schema.embedding
            ].to_numpy()
        )

        matrix = (
            self.embedding_scaler.transform(
                matrix
            )
        )

        if self.embedding_pca is not None:
            matrix = (
                self.embedding_pca.transform(
                    matrix
                )
            )

        return pd.DataFrame(
            matrix,
            columns=self.processed_embedding_cols,
            index=x_test.index,
        )

    # =========================================================================
    # PCA
    # =========================================================================

    @staticmethod
    def _effective_pca_components(
        requested: int,
        matrix: np.ndarray,
    ) -> int:
        """
        Determine a valid PCA dimensionality for the current training fold.

        PCA dimensionality is limited by:
        - the requested number of components;
        - the number of original features;
        - the effective rank supported by the number of training samples.
        """

        max_components = min(
            matrix.shape[0] - 1,
            matrix.shape[1],
        )

        if max_components < 1:
            raise ValueError(
                "PCA requires at least two training observations."
            )

        return min(
            requested,
            max_components,
        )
