import numpy as np
import pandas as pd

from src.core.data.schema import Schema


class PreSplitProcessor:
    """
    Build one acoustic signature per CapturePointId.

    Hierarchy:
        segment -> Audio_Name -> Date -> CapturePointId

    Acoustic indices and embeddings are aggregated using the mean
    at every level.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def process(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df_raw.copy()

        index_cols = self.schema.index_columns(
            df.columns
        )

        if not index_cols:
            raise ValueError(
                "No acoustic index columns were found."
            )

        df[self.schema.datetime] = pd.to_datetime(
            df[self.schema.datetime]
        )

        df["Date"] = (
            df[self.schema.datetime]
            .dt.date
        )

        targets = (
            df.groupby(
                [
                    self.schema.group,
                    self.schema.bag,
                ],
                as_index=False,
            )[self.schema.target]
            .first()
        )

        audio = self._aggregate_audio(
            df,
            index_cols,
        )

        daily = self._aggregate_daily(
            audio,
            index_cols,
        )

        capture = self._aggregate_capture(
            daily,
            index_cols,
        )

        return capture.merge(
            targets,
            on=[
                self.schema.group,
                self.schema.bag,
            ],
            how="left",
        )

    def _aggregate_audio(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
    ) -> pd.DataFrame:

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
            self.schema.audio,
        ]

        indices = (
            df.groupby(
                keys,
                as_index=False,
            )[index_cols]
            .mean()
        )

        embeddings = (
            df.groupby(
                keys,
                as_index=False,
            )[self.schema.embedding]
            .agg(self._mean_embedding)
        )

        return indices.merge(
            embeddings,
            on=keys,
            how="inner",
        )

    def _aggregate_daily(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
    ) -> pd.DataFrame:

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
        ]

        indices = (
            df.groupby(
                keys,
                as_index=False,
            )[index_cols]
            .mean()
        )

        embeddings = (
            df.groupby(
                keys,
                as_index=False,
            )[self.schema.embedding]
            .agg(self._mean_embedding)
        )

        return indices.merge(
            embeddings,
            on=keys,
            how="inner",
        )

    def _aggregate_capture(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
    ) -> pd.DataFrame:

        keys = [
            self.schema.group,
            self.schema.bag,
        ]

        indices = (
            df.groupby(
                keys,
                as_index=False,
            )[index_cols]
            .mean()
        )

        embeddings = (
            df.groupby(
                keys,
                as_index=False,
            )[self.schema.embedding]
            .agg(self._mean_embedding)
        )

        return indices.merge(
            embeddings,
            on=keys,
            how="inner",
        )

    @staticmethod
    def _mean_embedding(
        values: pd.Series,
    ) -> np.ndarray:

        return np.mean(
            np.stack(values.to_numpy()),
            axis=0,
        )
