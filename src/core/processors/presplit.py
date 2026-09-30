import numpy as np
import pandas as pd

from src.core.data.schema import Schema


# =============================================================================
# PRE-SPLIT PROCESSOR
# =============================================================================


class PreSplitProcessor:
    """
    Build one acoustic signature per CapturePointId.

    Processing hierarchy
    --------------------
    1. Segment -> Audio_Name
       Average the 1, 2 or 3 valid segments belonging to each Audio_Name.
       The result is treated as one atomic acoustic observation.

    2. Audio_Name -> day
       Calculate the daily mean and daily population standard deviation.

    3. Day -> CapturePointId
       Build one of three representations:

       mean
           [mean]

       mean_std
           [mean, total_std]

       hierarchical
           [mean, within_day_std, between_day_std]

    Days receive equal weight.

    The variance components satisfy:

        total_std² = within_day_std² + between_day_std²

    All aggregation is performed independently within CapturePointId.
    Scaling, PCA and model fitting are performed later inside
    cross-validation.
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
        aggregation: str = "hierarchical",
    ) -> pd.DataFrame:
        """
        Convert segment-level data into one signature per CapturePointId.
        """

        audio = self.prepare_audio(
            df_raw
        )

        daily = self.prepare_daily(
            audio
        )

        return self.build_signature(
            daily=daily,
            aggregation=aggregation,
        )

    def process_all(
        self,
        df_raw: pd.DataFrame,
        aggregations: tuple[str, ...],
    ) -> dict[str, pd.DataFrame]:
        """
        Build all requested aggregation representations.

        The common segment -> Audio_Name -> day processing is performed
        only once.
        """

        audio = self.prepare_audio(
            df_raw
        )

        daily = self.prepare_daily(
            audio
        )

        return {
            aggregation: self.build_signature(
                daily=daily,
                aggregation=aggregation,
            )
            for aggregation in aggregations
        }

    def prepare_audio(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Average the valid segments belonging to each Audio_Name.

        Each Audio_Name contains 1, 2 or 3 valid segment rows and becomes
        one atomic observation after this step.
        """

        df = df_raw.copy()

        df[self.schema.datetime] = pd.to_datetime(
            df[self.schema.datetime]
        )

        df["Date"] = (
            df[self.schema.datetime]
            .dt.date
        )

        self._validate_structure(
            df
        )

        index_cols = self.schema.index_columns(
            df.columns
        )

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
            self.schema.audio,
        ]

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        blocks = [
            (
                df.groupby(
                    keys,
                    as_index=False,
                    sort=False,
                )[self.schema.target]
                .first()
            )
        ]

        # ---------------------------------------------------------------------
        # Acoustic indices
        # ---------------------------------------------------------------------

        if index_cols:
            blocks.append(
                df.groupby(
                    keys,
                    as_index=False,
                    sort=False,
                )[index_cols]
                .mean()
            )

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------

        if self.schema.embedding in df.columns:
            blocks.append(
                df.groupby(
                    keys,
                    sort=False,
                )[self.schema.embedding]
                .agg(
                    self._mean_embedding
                )
                .reset_index(
                    name=self.schema.embedding
                )
            )

        return self._merge_blocks(
            blocks=blocks,
            keys=keys,
        )

    def prepare_daily(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate daily mean and daily population standard deviation across
        Audio_Name observations.
        """

        index_cols = self.schema.index_columns(
            audio.columns
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

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        blocks = [
            (
                grouped[
                    self.schema.target
                ]
                .first()
                .reset_index()
            )
        ]

        # ---------------------------------------------------------------------
        # Acoustic indices
        # ---------------------------------------------------------------------

        if index_cols:

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

            index_std = (
                grouped[index_cols]
                .std(
                    ddof=0
                )
                .reset_index()
                .rename(
                    columns={
                        col: f"{col}__daily_std"
                        for col in index_cols
                    }
                )
            )

            blocks.extend(
                [
                    index_mean,
                    index_std,
                ]
            )

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------

        if self.schema.embedding in audio.columns:

            embedding_mean = (
                grouped[
                    self.schema.embedding
                ]
                .agg(
                    self._mean_embedding
                )
                .reset_index(
                    name="embedding__daily_mean"
                )
            )

            embedding_std = (
                grouped[
                    self.schema.embedding
                ]
                .agg(
                    self._std_embedding
                )
                .reset_index(
                    name="embedding__daily_std"
                )
            )

            blocks.extend(
                [
                    embedding_mean,
                    embedding_std,
                ]
            )

        return self._merge_blocks(
            blocks=blocks,
            keys=keys,
        )

    def build_signature(
        self,
        daily: pd.DataFrame,
        aggregation: str,
    ) -> pd.DataFrame:
        """
        Build the final CapturePointId representation.

        All strategies use the same mean. They differ only in how much
        temporal variability information is retained.
        """

        index_cols = self._index_columns_from_daily(
            daily
        )

        has_embedding = (
            "embedding__daily_mean"
            in daily.columns
        )

        keys = [
            self.schema.group,
            self.schema.bag,
        ]

        rows = []

        for key_values, group in daily.groupby(
            keys,
            sort=False,
        ):

            row = {
                keys[0]: key_values[0],
                keys[1]: key_values[1],
                self.schema.target:
                    group[self.schema.target].iloc[0],
            }

            # =================================================================
            # ACOUSTIC INDICES
            # =================================================================

            for col in index_cols:

                daily_means = group[
                    f"{col}__daily_mean"
                ].to_numpy(
                    dtype=float
                )

                daily_stds = group[
                    f"{col}__daily_std"
                ].to_numpy(
                    dtype=float
                )

                (
                    mean_value,
                    total_std,
                    within_day_std,
                    between_day_std,
                ) = self._variance_components(
                    daily_means,
                    daily_stds,
                )

                row[
                    f"{col}_mean"
                ] = float(
                    mean_value
                )

                if aggregation == "mean_std":

                    row[
                        f"{col}_total_std"
                    ] = float(
                        total_std
                    )

                elif aggregation == "hierarchical":

                    row[
                        f"{col}_within_day_std"
                    ] = float(
                        within_day_std
                    )

                    row[
                        f"{col}_between_day_std"
                    ] = float(
                        between_day_std
                    )

                elif aggregation != "mean":

                    raise ValueError(
                        f"Unknown aggregation: {aggregation}"
                    )

            # =================================================================
            # EMBEDDINGS
            # =================================================================

            if has_embedding:

                daily_means = np.stack(
                    group[
                        "embedding__daily_mean"
                    ].to_numpy()
                )

                daily_stds = np.stack(
                    group[
                        "embedding__daily_std"
                    ].to_numpy()
                )

                (
                    mean_embedding,
                    total_std_embedding,
                    within_day_std_embedding,
                    between_day_std_embedding,
                ) = self._variance_components(
                    daily_means,
                    daily_stds,
                )

                if aggregation == "mean":

                    embedding = (
                        mean_embedding
                    )

                elif aggregation == "mean_std":

                    embedding = np.concatenate(
                        [
                            mean_embedding,
                            total_std_embedding,
                        ]
                    )

                elif aggregation == "hierarchical":

                    embedding = np.concatenate(
                        [
                            mean_embedding,
                            within_day_std_embedding,
                            between_day_std_embedding,
                        ]
                    )

                else:

                    raise ValueError(
                        f"Unknown aggregation: {aggregation}"
                    )

                row[
                    self.schema.embedding
                ] = embedding

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================================
    # VARIANCE DECOMPOSITION
    # =========================================================================

    @staticmethod
    def _variance_components(
        daily_means: np.ndarray,
        daily_stds: np.ndarray,
    ):
        """
        Calculate equal-day temporal variance components.

        total_variance =
            within_day_variance + between_day_variance
        """

        mean_value = np.mean(
            daily_means,
            axis=0,
        )

        within_variance = np.mean(
            np.square(
                daily_stds
            ),
            axis=0,
        )

        between_variance = np.mean(
            np.square(
                daily_means
                - mean_value
            ),
            axis=0,
        )

        total_variance = (
            within_variance
            + between_variance
        )

        return (
            mean_value,
            np.sqrt(
                total_variance
            ),
            np.sqrt(
                within_variance
            ),
            np.sqrt(
                between_variance
            ),
        )

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
    # VALIDATION
    # =========================================================================

    def _validate_structure(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate structural assumptions required by the aggregation.

        Each CapturePointId must belong to exactly one Point and have
        exactly one target value.

        Each Audio_Name must contain between 1 and 3 valid segment rows.
        """

        capture_structure = (
            df.groupby(
                self.schema.bag,
                dropna=False,
            )
            .agg(
                n_points=(
                    self.schema.group,
                    lambda values:
                    values.nunique(
                        dropna=False
                    ),
                ),
                n_targets=(
                    self.schema.target,
                    lambda values:
                    values.nunique(
                        dropna=False
                    ),
                ),
            )
        )

        invalid_captures = (
            capture_structure[
                (
                    capture_structure[
                        "n_points"
                    ] != 1
                )
                | (
                    capture_structure[
                        "n_targets"
                    ] != 1
                )
            ]
        )

        if not invalid_captures.empty:
            raise ValueError(
                "Each CapturePointId must belong to exactly one "
                "Point and have exactly one target value."
            )

        segment_counts = (
            df.groupby(
                [
                    self.schema.group,
                    self.schema.bag,
                    "Date",
                    self.schema.audio,
                ],
                dropna=False,
            )
            .size()
        )

        if not segment_counts.between(
            1,
            3,
        ).all():
            raise ValueError(
                "Each Audio_Name must contain between "
                "1 and 3 valid segment rows."
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
        Merge feature blocks using their common grouping keys.
        """

        result = blocks[0]

        for block in blocks[1:]:

            result = result.merge(
                block,
                on=keys,
                how="inner",
                validate="one_to_one",
            )

        return result

    @staticmethod
    def _index_columns_from_daily(
        daily: pd.DataFrame,
    ) -> list[str]:
        """
        Recover original acoustic-index names from daily feature names.
        """

        suffix = "__daily_mean"

        return [
            col[
                :-len(
                    suffix
                )
            ]
            for col in daily.columns
            if (
                col.endswith(
                    suffix
                )
                and col
                != "embedding__daily_mean"
            )
        ]
