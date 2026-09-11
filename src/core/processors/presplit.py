import numpy as np
import pandas as pd

from src.core.data.schema import Schema


# =============================================================================
# PRE-SPLIT PROCESSOR
# =============================================================================


class PreSplitProcessor:
    """
    Build one hierarchical acoustic signature per CapturePointId.

    Hierarchy
    ---------
    1-min segments
        -> mean within Audio_Name

    Audio_Name representations
        -> daily mean
        -> daily standard deviation

    Daily representations
        -> overall mean
        -> within-day variability
        -> between-day variability

    Acoustic indices
    ----------------
    Each original index produces three final features:

        index_mean
        index_within_day_std
        index_between_day_std

    Embeddings
    ----------
    The same three summaries are calculated element-wise and concatenated:

        [mean, within-day std, between-day std]

    The resulting dataframe contains one row per CapturePointId.
    """

    def __init__(
        self,
        schema: Schema,
    ) -> None:

        self.schema = schema

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def process(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert segment-level data into CapturePointId acoustic signatures.
        """

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

        # =====================================================================
        # TARGET
        # =====================================================================

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

        # =====================================================================
        # SEGMENTS -> AUDIO_NAME
        # =====================================================================

        audio = self._aggregate_audio(
            df=df,
            index_cols=index_cols,
        )

        # =====================================================================
        # AUDIO_NAME -> DAY
        # =====================================================================

        daily = self._aggregate_daily(
            df=audio,
            index_cols=index_cols,
        )

        # =====================================================================
        # DAY -> CAPTUREPOINTID
        # =====================================================================

        capture = self._aggregate_capture(
            df=daily,
            index_cols=index_cols,
        )

        # =====================================================================
        # TARGET MERGE
        # =====================================================================

        return capture.merge(
            targets,
            on=[
                self.schema.group,
                self.schema.bag,
            ],
            how="left",
        )

    # =========================================================================
    # SEGMENTS -> AUDIO_NAME
    # =========================================================================

    def _aggregate_audio(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
    ) -> pd.DataFrame:
        """
        Average valid 1-min segments belonging to the same original
        Audio_Name.

        No standard deviation is calculated at this level because an
        Audio_Name may contain only 1, 2 or 3 surviving segments.
        """

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
            self.schema.audio,
        ]

        # ---------------------------------------------------------------------
        # Acoustic indices
        # ---------------------------------------------------------------------

        indices = (
            df.groupby(
                keys,
                as_index=False,
            )[index_cols]
            .mean()
        )

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------

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

    # =========================================================================
    # AUDIO_NAME -> DAY
    # =========================================================================

    def _aggregate_daily(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
    ) -> pd.DataFrame:
        """
        Calculate daily mean and within-day standard deviation across
        original Audio_Name representations.
        """

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
        ]

        grouped = df.groupby(
            keys,
            sort=False,
        )

        # ---------------------------------------------------------------------
        # Acoustic indices - daily mean
        # ---------------------------------------------------------------------

        index_mean = (
            grouped[index_cols]
            .mean()
            .reset_index()
            .rename(
                columns={
                    col: f"{col}__daily_mean"
                    for col in index_cols
                }
            )
        )

        # ---------------------------------------------------------------------
        # Acoustic indices - within-day standard deviation
        # ---------------------------------------------------------------------

        index_std = (
            grouped[index_cols]
            .std(ddof=0)
            .reset_index()
            .rename(
                columns={
                    col: f"{col}__daily_std"
                    for col in index_cols
                }
            )
        )

        # ---------------------------------------------------------------------
        # Embeddings - daily mean
        # ---------------------------------------------------------------------

        embedding_mean = (
            grouped[
                self.schema.embedding
            ]
            .agg(self._mean_embedding)
            .reset_index(
                name="embedding__daily_mean"
            )
        )

        # ---------------------------------------------------------------------
        # Embeddings - within-day standard deviation
        # ---------------------------------------------------------------------

        embedding_std = (
            grouped[
                self.schema.embedding
            ]
            .agg(self._std_embedding)
            .reset_index(
                name="embedding__daily_std"
            )
        )

        daily = (
            index_mean
            .merge(
                index_std,
                on=keys,
                how="inner",
            )
            .merge(
                embedding_mean,
                on=keys,
                how="inner",
            )
            .merge(
                embedding_std,
                on=keys,
                how="inner",
            )
        )

        return daily

    # =========================================================================
    # DAY -> CAPTUREPOINTID
    # =========================================================================

    def _aggregate_capture(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
    ) -> pd.DataFrame:
        """
        Build the final CapturePointId acoustic signature.

        For each original feature:

        mean
            Mean of daily means.

        within_day_std
            RMS of daily standard deviations.

        between_day_std
            Standard deviation of daily means.
        """

        keys = [
            self.schema.group,
            self.schema.bag,
        ]

        daily_mean_cols = [
            f"{col}__daily_mean"
            for col in index_cols
        ]

        daily_std_cols = [
            f"{col}__daily_std"
            for col in index_cols
        ]

        grouped = df.groupby(
            keys,
            sort=False,
        )

        # ---------------------------------------------------------------------
        # Acoustic indices - overall mean
        # ---------------------------------------------------------------------

        index_mean = (
            grouped[
                daily_mean_cols
            ]
            .mean()
            .reset_index()
            .rename(
                columns={
                    f"{col}__daily_mean":
                    f"{col}_mean"
                    for col in index_cols
                }
            )
        )

        # ---------------------------------------------------------------------
        # Acoustic indices - within-day variability
        # ---------------------------------------------------------------------

        index_within = (
            grouped[
                daily_std_cols
            ]
            .agg(self._rms)
            .reset_index()
            .rename(
                columns={
                    f"{col}__daily_std":
                    f"{col}_within_day_std"
                    for col in index_cols
                }
            )
        )

        # ---------------------------------------------------------------------
        # Acoustic indices - between-day variability
        # ---------------------------------------------------------------------

        index_between = (
            grouped[
                daily_mean_cols
            ]
            .std(ddof=0)
            .reset_index()
            .rename(
                columns={
                    f"{col}__daily_mean":
                    f"{col}_between_day_std"
                    for col in index_cols
                }
            )
        )

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------

        embedding = (
            grouped[[
                "embedding__daily_mean",
                "embedding__daily_std",
            ]]
            .apply(
                lambda group: pd.Series({
                    self.schema.embedding: self._capture_embedding(group)
                }),
            )
            .reset_index()
        )

        # ---------------------------------------------------------------------
        # Final CapturePointId representation
        # ---------------------------------------------------------------------

        capture = (
            index_mean
            .merge(
                index_within,
                on=keys,
                how="inner",
            )
            .merge(
                index_between,
                on=keys,
                how="inner",
            )
            .merge(
                embedding,
                on=keys,
                how="inner",
            )
        )

        return capture

    # =========================================================================
    # EMBEDDING HELPERS
    # =========================================================================

    @staticmethod
    def _mean_embedding(
        values: pd.Series,
    ) -> np.ndarray:
        """
        Element-wise mean of embedding vectors.
        """

        matrix = np.stack(
            values.to_numpy()
        )

        return np.mean(
            matrix,
            axis=0,
        )

    @staticmethod
    def _std_embedding(
        values: pd.Series,
    ) -> np.ndarray:
        """
        Element-wise population standard deviation of embedding vectors.
        """

        matrix = np.stack(
            values.to_numpy()
        )

        return np.std(
            matrix,
            axis=0,
            ddof=0,
        )

    def _capture_embedding(
        self,
        group: pd.DataFrame,
    ) -> np.ndarray:
        """
        Construct the final embedding representation for one CapturePointId.

        The output concatenates:

            mean embedding
            within-day embedding variability
            between-day embedding variability
        """

        daily_mean = np.stack(
            group[
                "embedding__daily_mean"
            ].to_numpy()
        )

        daily_std = np.stack(
            group[
                "embedding__daily_std"
            ].to_numpy()
        )

        mean_embedding = np.mean(
            daily_mean,
            axis=0,
        )

        within_day_std = np.sqrt(
            np.mean(
                np.square(
                    daily_std
                ),
                axis=0,
            )
        )

        between_day_std = np.std(
            daily_mean,
            axis=0,
            ddof=0,
        )

        return np.concatenate(
            [
                mean_embedding,
                within_day_std,
                between_day_std,
            ]
        )

    # =========================================================================
    # STATISTICAL HELPERS
    # =========================================================================

    @staticmethod
    def _rms(
        values: pd.Series,
    ) -> float:
        """
        Root mean square.

        Used to combine daily standard deviations into one estimate of
        typical within-day variability.
        """

        values = values.to_numpy(
            dtype=float
        )

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        values
                    )
                )
            )
        )
