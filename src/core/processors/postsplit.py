import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.core.data.schema import Schema


class PostSplitProcessor:
    """
    Apply fold-specific preprocessing for the selected feature representation.

    Acoustic indices and embeddings are processed separately:
    - indices: RobustScaler
    - embeddings: StandardScaler + optional PCA
    - both: concatenate the two processed blocks

    Target scaling follows the original pipeline behavior.
    """

    def __init__(
        self,
        schema: Schema,
        feature_set: str,
        pca_components: int | None,
    ) -> None:
        self.schema = schema
        self.feature_set = feature_set
        self.pca_components = pca_components

        self.index_scaler = RobustScaler()
        self.embedding_scaler = StandardScaler()
        self.target_scaler = RobustScaler()

        self.pca = (
            PCA(n_components=pca_components, svd_solver="full")
            if pca_components is not None
            else None
        )

        self.index_cols: list[str] = []

    def fit_transform(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit preprocessing using training data and transform train features/target.
        """
        x_train_transformed = self._fit_transform_features(x_train)

        y_train_transformed = y_train.copy()
        y_train_transformed.loc[:, [self.schema.target]] = (
            self.target_scaler.fit_transform(
                y_train_transformed[[self.schema.target]]
            )
        )

        return x_train_transformed, y_train_transformed

    def transform(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test features using preprocessing fitted on training data.
        """
        parts: list[pd.DataFrame] = []

        if self.feature_set in {"indices", "both"}:
            parts.append(self._transform_indices(x_test))

        if self.feature_set in {"embeddings", "both"}:
            parts.append(self._transform_embeddings(x_test))

        return pd.concat(parts, axis=1)

    def inverse_transform_target(
        self,
        y_pred: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform predicted HFI values back to the original scale.
        """
        y_pred = y_pred.copy()

        y_pred.loc[:, [self.schema.target]] = (
            self.target_scaler.inverse_transform(
                y_pred[[self.schema.target]]
            )
        )

        return y_pred

    def _fit_transform_features(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []

        if self.feature_set in {"indices", "both"}:
            parts.append(self._fit_transform_indices(x_train))

        if self.feature_set in {"embeddings", "both"}:
            parts.append(self._fit_transform_embeddings(x_train))

        return pd.concat(parts, axis=1)

    def _fit_transform_indices(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:
        self.index_cols = self.schema.index_columns(x_train.columns)

        values = self.index_scaler.fit_transform(
            x_train[self.index_cols]
        )

        return pd.DataFrame(
            values,
            columns=self.index_cols,
            index=x_train.index,
        )

    def _transform_indices(
        self,
        x_test: pd.DataFrame,
    ) -> pd.DataFrame:
        values = self.index_scaler.transform(
            x_test[self.index_cols]
        )

        return pd.DataFrame(
            values,
            columns=self.index_cols,
            index=x_test.index,
        )

    def _fit_transform_embeddings(
        self,
        x_train: pd.DataFrame,
    ) -> pd.DataFrame:
        matrix = np.stack(
            x_train[self.schema.embedding].to_numpy()
        )

        matrix = self.embedding_scaler.fit_transform(matrix)

        if self.pca is not None:
            matrix = self.pca.fit_transform(matrix)
            columns = [
                f"embedding_pc_{i + 1}"
                for i in range(matrix.shape[1])
            ]
        else:
            columns = [
                f"embedding_{i + 1}"
                for i in range(matrix.shape[1])
            ]

        return pd.DataFrame(
            matrix,
            columns=columns,
            index=x_train.index,
        )

    def _transform_embeddings(
        self,
        x_test: pd.DataFrame,
    ) -> pd.DataFrame:
        matrix = np.stack(
            x_test[self.schema.embedding].to_numpy()
        )

        matrix = self.embedding_scaler.transform(matrix)

        if self.pca is not None:
            matrix = self.pca.transform(matrix)
            columns = [
                f"embedding_pc_{i + 1}"
                for i in range(matrix.shape[1])
            ]
        else:
            columns = [
                f"embedding_{i + 1}"
                for i in range(matrix.shape[1])
            ]

        return pd.DataFrame(
            matrix,
            columns=columns,
            index=x_test.index,
        )
