import numpy as np
import pandas as pd

from src.core.data.schema import Schema
from src.core.processors.aggregations import AcousticAggregator


# =============================================================================
# PRE-SPLIT PROCESSOR
# =============================================================================


class PreSplitProcessor:
    """
    Prepare acoustic observations before train/test splitting.

    Processing hierarchy
    --------------------
    1. Segment -> Audio_Name
       Average valid segments from the same recording.

    2. Audio_Name -> day
       Calculate daily means and population standard deviations.

    3. CapturePointId signature
       Delegate acoustic representation construction to AcousticAggregator.
    """

    def __init__(
        self,
        schema: Schema,
    ) -> None:

        self.schema = schema

        self.aggregator = AcousticAggregator(
            schema=schema
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def process(
        self,
        df_raw: pd.DataFrame,
        aggregation: str = "hierarchical",
    ) -> pd.DataFrame:
        """
        Convert segment-level data into one acoustic signature per
        CapturePointId.
        """

        audio = self.prepare_audio(
            df_raw
        )

        daily = self.prepare_daily(
            audio
        )

        return self.aggregator.build(
            audio=audio,
            daily=daily,
            aggregation=aggregation,
        )

    def process_all(
        self,
        df_raw: pd.DataFrame,
        aggregations: tuple[str, ...],
    ) -> dict[str, pd.DataFrame]:
        """
        Build all requested acoustic representations.

        Segment and daily processing are performed only once.
        """

        audio = self.prepare_audio(
            df_raw
        )

        daily = self.prepare_daily(
            audio
        )

        return {
            aggregation:
                self.aggregator.build(
                    audio=audio,
                    daily=daily,
                    aggregation=aggregation,
                )
            for aggregation
            in aggregations
        }

    # =========================================================================
    # SEGMENT -> AUDIO_NAME
    # =========================================================================

    def prepare_audio(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Average valid segments belonging to the same Audio_Name.

        One resulting row corresponds to one atomic acoustic observation.
        The original recording timestamp is preserved.
        """

        df = df_raw.copy()

        df[
            self.schema.datetime
        ] = pd.to_datetime(
            df[
                self.schema.datetime
            ]
        )

        df["Date"] = (
            df[
                self.schema.datetime
            ]
            .dt.date
        )

        index_cols = (
            self.schema.index_columns(
                df.columns
            )
        )

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
            self.schema.audio,
            self.schema.datetime,
        ]

        grouped = df.groupby(
            keys,
            as_index=False,
            sort=False,
        )

        blocks = [
            (
                grouped[
                    self.schema.target
                ]
                .first()
            )
        ]

        if index_cols:

            blocks.append(
                grouped[
                    index_cols
                ]
                .mean()
            )

        if (
            self.schema.embedding
            in df.columns
        ):

            blocks.append(
                df.groupby(
                    keys,
                    sort=False,
                )[
                    self.schema.embedding
                ]
                .agg(
                    self._mean_embedding
                )
                .reset_index(
                    name=(
                        self.schema.embedding
                    )
                )
            )

        return self._merge_blocks(
            blocks=blocks,
            keys=keys,
        )

    # =========================================================================
    # AUDIO_NAME -> DAY
    # =========================================================================

    def prepare_daily(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate daily mean and population standard deviation across
        atomic Audio_Name observations.
        """

        index_cols = (
            self.schema.index_columns(
                audio.columns
            )
        )

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
        ]

        grouped = audio.groupby(
            keys,
            sort=False,
        )

        blocks = [
            (
                grouped[
                    self.schema.target
                ]
                .first()
                .reset_index()
            )
        ]

        if index_cols:

            blocks.append(
                grouped[
                    index_cols
                ]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        col:
                            f"{col}__daily_mean"
                        for col
                        in index_cols
                    }
                )
            )

            blocks.append(
                grouped[
                    index_cols
                ]
                .std(
                    ddof=0
                )
                .reset_index()
                .rename(
                    columns={
                        col:
                            f"{col}__daily_std"
                        for col
                        in index_cols
                    }
                )
            )

        if (
            self.schema.embedding
            in audio.columns
        ):

            blocks.append(
                grouped[
                    self.schema.embedding
                ]
                .agg(
                    self._mean_embedding
                )
                .reset_index(
                    name=(
                        "embedding__daily_mean"
                    )
                )
            )

            blocks.append(
                grouped[
                    self.schema.embedding
                ]
                .agg(
                    self._std_embedding
                )
                .reset_index(
                    name=(
                        "embedding__daily_std"
                    )
                )
            )

        return self._merge_blocks(
            blocks=blocks,
            keys=keys,
        )

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    @staticmethod
    def _mean_embedding(
        values: pd.Series,
    ) -> np.ndarray:
        """
        Element-wise mean of embedding vectors.
        """

        return np.mean(
            np.stack(
                values.to_numpy()
            ),
            axis=0,
        )

    @staticmethod
    def _std_embedding(
        values: pd.Series,
    ) -> np.ndarray:
        """
        Element-wise population standard deviation of embedding vectors.
        """

        return np.std(
            np.stack(
                values.to_numpy()
            ),
            axis=0,
            ddof=0,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _merge_blocks(
        blocks: list[pd.DataFrame],
        keys: list[str],
    ) -> pd.DataFrame:
        """
        Merge feature blocks using their common keys.
        """

        result = blocks[0]

        for block in blocks[1:]:

            result = result.merge(
                block,
                on=keys,
                how="inner",
            )

        return result
