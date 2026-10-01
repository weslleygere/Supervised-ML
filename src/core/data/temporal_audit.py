import pandas as pd

from src.core.data.schema import Schema


# =============================================================================
# TEMPORAL AUDIT
# =============================================================================


class TemporalAudit:
    """
    Describe temporal sampling coverage of atomic Audio_Name observations.

    The audit operates on the output of PreSplitProcessor.prepare_audio(),
    where each row represents one atomic acoustic observation.

    The recording window is divided into fixed temporal bins so that the
    availability of repeated observations across days can be inspected
    before introducing time-dependent acoustic representations.
    """

    def __init__(
        self,
        schema: Schema,
        start_hour: int = 4,
        end_hour: int = 6,
        bin_minutes: int = 30,
    ) -> None:

        self.schema = schema

        self.start_minute = (
            start_hour * 60
        )

        self.end_minute = (
            end_hour * 60
        )

        self.bin_minutes = (
            bin_minutes
        )

        self.bin_labels = (
            self._make_bin_labels()
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def build(
        self,
        audio: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Build daily and CapturePointId-level temporal summaries.
        """

        daily = self.daily_coverage(
            audio
        )

        capture = self.capture_summary(
            daily
        )

        return {
            "daily":
                daily,

            "capture":
                capture,
        }

    # =========================================================================
    # DAILY COVERAGE
    # =========================================================================

    def daily_coverage(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize temporal coverage for each sampled day.
        """

        df = self._add_time_bins(
            audio
        )

        keys = [
            self.schema.group,
            self.schema.bag,
            "Date",
        ]

        grouped = df.groupby(
            keys,
            sort=False,
        )

        daily = (
            grouped
            .agg(
                **{
                    self.schema.target: (
                        self.schema.target,
                        "first",
                    ),

                    "n_audio": (
                        self.schema.audio,
                        "size",
                    ),

                    "first_datetime": (
                        self.schema.datetime,
                        "min",
                    ),

                    "last_datetime": (
                        self.schema.datetime,
                        "max",
                    ),
                }
            )
            .reset_index()
        )

        # ---------------------------------------------------------------------
        # Number of observations in each temporal bin
        # ---------------------------------------------------------------------

        in_window = (
            df[
                df["_time_bin"]
                .notna()
            ]
        )

        bin_counts = (
            in_window
            .groupby(
                keys
                + [
                    "_time_bin"
                ],
                sort=False,
            )
            .size()
            .unstack(
                fill_value=0
            )
            .reindex(
                columns=(
                    self.bin_labels
                ),
                fill_value=0,
            )
            .reset_index()
        )

        bin_counts = (
            bin_counts.rename(
                columns={
                    label:
                        f"audio_{label}"
                    for label
                    in self.bin_labels
                }
            )
        )

        daily = daily.merge(
            bin_counts,
            on=keys,
            how="left",
        )

        bin_columns = [
            f"audio_{label}"
            for label
            in self.bin_labels
        ]

        daily[
            bin_columns
        ] = (
            daily[
                bin_columns
            ]
            .fillna(
                0
            )
            .astype(
                int
            )
        )

        # ---------------------------------------------------------------------
        # Coverage summaries
        # ---------------------------------------------------------------------

        daily[
            "n_audio_in_window"
        ] = (
            daily[
                bin_columns
            ]
            .sum(
                axis=1
            )
        )

        daily[
            "n_audio_outside_window"
        ] = (
            daily[
                "n_audio"
            ]
            - daily[
                "n_audio_in_window"
            ]
        )

        daily[
            "n_bins_covered"
        ] = (
            daily[
                bin_columns
            ]
            .gt(
                0
            )
            .sum(
                axis=1
            )
        )

        daily[
            "complete_bin_coverage"
        ] = (
            daily[
                "n_bins_covered"
            ]
            == len(
                self.bin_labels
            )
        )

        return (
            daily.sort_values(
                [
                    self.schema.group,
                    self.schema.bag,
                    "Date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # CAPTUREPOINTID SUMMARY
    # =========================================================================

    def capture_summary(
        self,
        daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize temporal sampling across days for each CapturePointId.
        """

        keys = [
            self.schema.group,
            self.schema.bag,
        ]

        grouped = daily.groupby(
            keys,
            sort=False,
        )

        summary = (
            grouped
            .agg(
                **{
                    self.schema.target: (
                        self.schema.target,
                        "first",
                    ),

                    "n_days": (
                        "Date",
                        "nunique",
                    ),

                    "n_audio": (
                        "n_audio",
                        "sum",
                    ),

                    "min_audio_per_day": (
                        "n_audio",
                        "min",
                    ),

                    "median_audio_per_day": (
                        "n_audio",
                        "median",
                    ),

                    "mean_audio_per_day": (
                        "n_audio",
                        "mean",
                    ),

                    "max_audio_per_day": (
                        "n_audio",
                        "max",
                    ),

                    "n_complete_days": (
                        "complete_bin_coverage",
                        "sum",
                    ),

                    "first_datetime": (
                        "first_datetime",
                        "min",
                    ),

                    "last_datetime": (
                        "last_datetime",
                        "max",
                    ),
                }
            )
            .reset_index()
        )

        summary[
            "complete_day_fraction"
        ] = (
            summary[
                "n_complete_days"
            ]
            / summary[
                "n_days"
            ]
        )

        # ---------------------------------------------------------------------
        # Coverage of each temporal bin across days
        # ---------------------------------------------------------------------

        for label in self.bin_labels:

            column = (
                f"audio_{label}"
            )

            total = (
                grouped[
                    column
                ]
                .sum()
                .reset_index(
                    name=column
                )
            )

            days = (
                grouped[
                    column
                ]
                .apply(
                    lambda values:
                        int(
                            (
                                values
                                > 0
                            )
                            .sum()
                        )
                )
                .reset_index(
                    name=(
                        f"days_{label}"
                    )
                )
            )

            summary = summary.merge(
                total,
                on=keys,
                how="left",
            )

            summary = summary.merge(
                days,
                on=keys,
                how="left",
            )

            summary[
                f"coverage_{label}"
            ] = (
                summary[
                    f"days_{label}"
                ]
                / summary[
                    "n_days"
                ]
            )

        return (
            summary.sort_values(
                keys
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # TEMPORAL BINS
    # =========================================================================

    def _add_time_bins(
        self,
        audio: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign each Audio_Name to one temporal bin.

        The configured end of the recording window is exclusive.
        For the default configuration:

            04:00 <= time < 06:00
        """

        df = audio.copy()

        datetime = pd.to_datetime(
            df[
                self.schema.datetime
            ]
        )

        minute_of_day = (
            datetime.dt.hour
            * 60
            + datetime.dt.minute
        )

        bin_index = (
            (
                minute_of_day
                - self.start_minute
            )
            // self.bin_minutes
        )

        label_map = {
            index:
                label
            for index, label
            in enumerate(
                self.bin_labels
            )
        }

        df[
            "_time_bin"
        ] = (
            bin_index.map(
                label_map
            )
        )

        inside_window = (
            (
                minute_of_day
                >= self.start_minute
            )
            &
            (
                minute_of_day
                < self.end_minute
            )
        )

        df.loc[
            ~inside_window,
            "_time_bin",
        ] = pd.NA

        return df

    def _make_bin_labels(
        self,
    ) -> tuple[str, ...]:
        """
        Create compact labels for the temporal bins.
        """

        labels = []

        for start in range(
            self.start_minute,
            self.end_minute,
            self.bin_minutes,
        ):

            end = min(
                start
                + self.bin_minutes,
                self.end_minute,
            )

            labels.append(
                (
                    f"{start // 60:02d}"
                    f"{start % 60:02d}"
                    "_"
                    f"{end // 60:02d}"
                    f"{end % 60:02d}"
                )
            )

        return tuple(
            labels
        )