import numpy as np
import pandas as pd

from src.core.data.schema import Schema


# =============================================================================
# ACOUSTIC AGGREGATIONS
# =============================================================================


class AcousticAggregator:
    """
    Build one acoustic signature per CapturePointId.

    Representations
    ---------------
    mean
        Mean acoustic state across equally weighted days.

    mean_std
        Mean + total temporal variability.

    hierarchical
        Mean + within-day variability + between-day variability.

    robust_daily
        Median daily state + within-day IQR + between-day IQR.

    dawn_profile
        Equal-day means in four 30-minute bins between 04:00 and 06:00.

    dawn_trend
        Mean level at 05:00 + mean dawn slope + between-day slope SD.
    """

    DAWN_LABELS = (
        "0400_0430",
        "0430_0500",
        "0500_0530",
        "0530_0600",
    )

    def __init__(
        self,
        schema: Schema,
    ) -> None:

        self.schema = schema

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def build(
        self,
        audio: pd.DataFrame,
        daily: pd.DataFrame,
        aggregation: str,
    ) -> pd.DataFrame:

        if aggregation in {
            "mean",
            "mean_std",
            "hierarchical",
        }:
            return self._variance_representation(
                daily,
                aggregation,
            )

        if aggregation == "robust_daily":
            return self._robust_daily(
                audio
            )

        if aggregation == "dawn_profile":
            return self._dawn_profile(
                audio
            )

        if aggregation == "dawn_trend":
            return self._dawn_trend(
                audio
            )

        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    # =========================================================================
    # MEAN / MEAN_STD / HIERARCHICAL
    # =========================================================================

    def _variance_representation(
        self,
        daily: pd.DataFrame,
        aggregation: str,
    ) -> pd.DataFrame:

        index_cols = self._daily_index_columns(
            daily
        )

        has_embedding = (
            "embedding__daily_mean"
            in daily.columns
        )

        rows = []

        for key, group in self._capture_groups(
            daily
        ):

            row = self._metadata(
                key,
                group,
            )

            # -----------------------------------------------------------------
            # Acoustic indices
            # -----------------------------------------------------------------

            if index_cols:

                means = group[
                    [
                        f"{col}__daily_mean"
                        for col in index_cols
                    ]
                ].to_numpy(
                    dtype=float
                )

                stds = group[
                    [
                        f"{col}__daily_std"
                        for col in index_cols
                    ]
                ].to_numpy(
                    dtype=float
                )

                (
                    mean,
                    total_std,
                    within_std,
                    between_std,
                ) = self._variance_components(
                    means,
                    stds,
                )

                for i, col in enumerate(
                    index_cols
                ):

                    row[
                        f"{col}_mean"
                    ] = mean[i]

                    if aggregation == "mean_std":

                        row[
                            f"{col}_total_std"
                        ] = total_std[i]

                    elif aggregation == "hierarchical":

                        row[
                            f"{col}_within_day_std"
                        ] = within_std[i]

                        row[
                            f"{col}_between_day_std"
                        ] = between_std[i]

            # -----------------------------------------------------------------
            # Embeddings
            # -----------------------------------------------------------------

            if has_embedding:

                (
                    mean,
                    total_std,
                    within_std,
                    between_std,
                ) = self._variance_components(
                    np.stack(
                        group[
                            "embedding__daily_mean"
                        ]
                    ),
                    np.stack(
                        group[
                            "embedding__daily_std"
                        ]
                    ),
                )

                if aggregation == "mean":

                    embedding = mean

                elif aggregation == "mean_std":

                    embedding = np.concatenate(
                        [
                            mean,
                            total_std,
                        ]
                    )

                else:

                    embedding = np.concatenate(
                        [
                            mean,
                            within_std,
                            between_std,
                        ]
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
    # ROBUST DAILY
    # =========================================================================

    def _robust_daily(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:

        index_cols = self.schema.index_columns(
            audio.columns
        )

        has_embedding = (
            self.schema.embedding
            in audio.columns
        )

        rows = []

        for key, group in self._capture_groups(
            audio
        ):

            row = self._metadata(
                key,
                group,
            )

            days = list(
                group.groupby(
                    "Date",
                    sort=False,
                )
            )

            # -----------------------------------------------------------------
            # Acoustic indices
            # -----------------------------------------------------------------

            if index_cols:

                daily_medians = np.stack(
                    [
                        day[
                            index_cols
                        ]
                        .median()
                        .to_numpy(
                            dtype=float
                        )
                        for _, day
                        in days
                    ]
                )

                daily_iqrs = np.stack(
                    [
                        self._iqr(
                            day[
                                index_cols
                            ].to_numpy(
                                dtype=float
                            ),
                            axis=0,
                        )
                        for _, day
                        in days
                    ]
                )

                median = np.median(
                    daily_medians,
                    axis=0,
                )

                within_iqr = np.median(
                    daily_iqrs,
                    axis=0,
                )

                between_iqr = self._iqr(
                    daily_medians,
                    axis=0,
                )

                for i, col in enumerate(
                    index_cols
                ):

                    row[
                        f"{col}_median"
                    ] = median[i]

                    row[
                        f"{col}_within_day_iqr"
                    ] = within_iqr[i]

                    row[
                        f"{col}_between_day_iqr"
                    ] = between_iqr[i]

            # -----------------------------------------------------------------
            # Embeddings
            # -----------------------------------------------------------------

            if has_embedding:

                daily_medians = np.stack(
                    [
                        np.median(
                            np.stack(
                                day[
                                    self.schema.embedding
                                ]
                            ),
                            axis=0,
                        )
                        for _, day
                        in days
                    ]
                )

                daily_iqrs = np.stack(
                    [
                        self._iqr(
                            np.stack(
                                day[
                                    self.schema.embedding
                                ]
                            ),
                            axis=0,
                        )
                        for _, day
                        in days
                    ]
                )

                row[
                    self.schema.embedding
                ] = np.concatenate(
                    [
                        np.median(
                            daily_medians,
                            axis=0,
                        ),
                        np.median(
                            daily_iqrs,
                            axis=0,
                        ),
                        self._iqr(
                            daily_medians,
                            axis=0,
                        ),
                    ]
                )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================================
    # DAWN PROFILE
    # =========================================================================

    def _dawn_profile(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:

        df = audio.copy()

        df["_dawn_bin"] = self._dawn_bins(
            df[
                self.schema.datetime
            ]
        )

        index_cols = self.schema.index_columns(
            df.columns
        )

        has_embedding = (
            self.schema.embedding
            in df.columns
        )

        rows = []

        for key, group in self._capture_groups(
            df
        ):

            row = self._metadata(
                key,
                group,
            )

            # -----------------------------------------------------------------
            # Acoustic indices
            # -----------------------------------------------------------------

            if index_cols:

                daily_profile = (
                    group.groupby(
                        [
                            "Date",
                            "_dawn_bin",
                        ],
                        observed=True,
                    )[
                        index_cols
                    ]
                    .mean()
                )

                profile = (
                    daily_profile.groupby(
                        "_dawn_bin",
                        observed=True,
                    )
                    .mean()
                    .reindex(
                        self.DAWN_LABELS
                    )
                )

                for col in index_cols:

                    for label in self.DAWN_LABELS:

                        row[
                            f"{col}_dawn_{label}"
                        ] = profile.loc[
                            label,
                            col,
                        ]

            # -----------------------------------------------------------------
            # Embeddings
            # -----------------------------------------------------------------

            if has_embedding:

                daily_profile = (
                    group.groupby(
                        [
                            "Date",
                            "_dawn_bin",
                        ],
                        observed=True,
                    )[
                        self.schema.embedding
                    ]
                    .agg(
                        self._mean_embedding
                    )
                )

                profile = (
                    daily_profile.groupby(
                        "_dawn_bin",
                        observed=True,
                    )
                    .agg(
                        self._mean_embedding
                    )
                    .reindex(
                        self.DAWN_LABELS
                    )
                )

                row[
                    self.schema.embedding
                ] = np.concatenate(
                    profile.to_list()
                )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================================
    # DAWN TREND
    # =========================================================================

    def _dawn_trend(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:

        index_cols = self.schema.index_columns(
            audio.columns
        )

        has_embedding = (
            self.schema.embedding
            in audio.columns
        )

        rows = []

        for key, group in self._capture_groups(
            audio
        ):

            row = self._metadata(
                key,
                group,
            )

            days = list(
                group.groupby(
                    "Date",
                    sort=False,
                )
            )

            # -----------------------------------------------------------------
            # Acoustic indices
            # -----------------------------------------------------------------

            if index_cols:

                levels = []
                slopes = []

                for _, day in days:

                    time = self._hours_from_five(
                        day[
                            self.schema.datetime
                        ]
                    )

                    if np.unique(
                        time
                    ).size < 2:
                        continue

                    level, slope = (
                        self._linear_trend(
                            time,
                            day[
                                index_cols
                            ].to_numpy(
                                dtype=float
                            ),
                        )
                    )

                    levels.append(
                        level
                    )

                    slopes.append(
                        slope
                    )

                levels = np.stack(
                    levels
                )

                slopes = np.stack(
                    slopes
                )

                mean_level = np.mean(
                    levels,
                    axis=0,
                )

                mean_slope = np.mean(
                    slopes,
                    axis=0,
                )

                slope_std = np.std(
                    slopes,
                    axis=0,
                    ddof=0,
                )

                for i, col in enumerate(
                    index_cols
                ):

                    row[
                        f"{col}_dawn_level"
                    ] = mean_level[i]

                    row[
                        f"{col}_dawn_slope"
                    ] = mean_slope[i]

                    row[
                        f"{col}_dawn_slope_std"
                    ] = slope_std[i]

            # -----------------------------------------------------------------
            # Embeddings
            # -----------------------------------------------------------------

            if has_embedding:

                levels = []
                slopes = []

                for _, day in days:

                    time = self._hours_from_five(
                        day[
                            self.schema.datetime
                        ]
                    )

                    if np.unique(
                        time
                    ).size < 2:
                        continue

                    level, slope = (
                        self._linear_trend(
                            time,
                            np.stack(
                                day[
                                    self.schema.embedding
                                ]
                            ),
                        )
                    )

                    levels.append(
                        level
                    )

                    slopes.append(
                        slope
                    )

                levels = np.stack(
                    levels
                )

                slopes = np.stack(
                    slopes
                )

                row[
                    self.schema.embedding
                ] = np.concatenate(
                    [
                        np.mean(
                            levels,
                            axis=0,
                        ),
                        np.mean(
                            slopes,
                            axis=0,
                        ),
                        np.std(
                            slopes,
                            axis=0,
                            ddof=0,
                        ),
                    ]
                )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _variance_components(
        daily_means: np.ndarray,
        daily_stds: np.ndarray,
    ):

        mean = np.mean(
            daily_means,
            axis=0,
        )

        within_variance = np.mean(
            daily_stds**2,
            axis=0,
        )

        between_variance = np.mean(
            (
                daily_means
                - mean
            )
            ** 2,
            axis=0,
        )

        return (
            mean,
            np.sqrt(
                within_variance
                + between_variance
            ),
            np.sqrt(
                within_variance
            ),
            np.sqrt(
                between_variance
            ),
        )

    @staticmethod
    def _linear_trend(
        time: np.ndarray,
        values: np.ndarray,
    ):

        time = np.asarray(
            time,
            dtype=float,
        )

        values = np.asarray(
            values,
            dtype=float,
        )

        centered_time = (
            time
            - time.mean()
        )

        slope = (
            centered_time
            @ values
        ) / (
            centered_time
            @ centered_time
        )

        level = (
            values.mean(
                axis=0
            )
            - slope
            * time.mean()
        )

        return (
            level,
            slope,
        )

    @staticmethod
    def _hours_from_five(
        datetime: pd.Series,
    ) -> np.ndarray:

        return (
            datetime.dt.hour
            + datetime.dt.minute
            / 60.0
            - 5.0
        ).to_numpy(
            dtype=float
        )

    @staticmethod
    def _dawn_bins(
        datetime: pd.Series,
    ) -> pd.Series:

        minutes = (
            datetime.dt.hour
            * 60
            + datetime.dt.minute
        )

        return pd.cut(
            minutes,
            bins=(
                240,
                270,
                300,
                330,
                360,
            ),
            labels=(
                "0400_0430",
                "0430_0500",
                "0500_0530",
                "0530_0600",
            ),
            right=False,
        )

    @staticmethod
    def _iqr(
        values: np.ndarray,
        axis=None,
    ):

        return (
            np.percentile(
                values,
                75,
                axis=axis,
            )
            - np.percentile(
                values,
                25,
                axis=axis,
            )
        )

    @staticmethod
    def _mean_embedding(
        values: pd.Series,
    ) -> np.ndarray:

        return np.mean(
            np.stack(
                values.to_numpy()
            ),
            axis=0,
        )

    def _capture_groups(
        self,
        df: pd.DataFrame,
    ):

        return df.groupby(
            [
                self.schema.group,
                self.schema.bag,
            ],
            sort=False,
        )

    def _metadata(
        self,
        key,
        group: pd.DataFrame,
    ) -> dict:

        return {
            self.schema.group:
                key[0],
            self.schema.bag:
                key[1],
            self.schema.target:
                group[
                    self.schema.target
                ].iloc[0],
        }

    @staticmethod
    def _daily_index_columns(
        daily: pd.DataFrame,
    ) -> list[str]:

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
