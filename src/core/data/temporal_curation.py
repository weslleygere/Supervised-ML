import pandas as pd

from src.core.data.schema import Schema


class TemporalCurator:
    """
    Apply pre-specified temporal eligibility rules per CapturePointId.

    A CapturePointId is retained only when at least ``min_observed_days``
    distinct recording dates are available. A date counts as observed when
    at least one atomic Audio_Name observation is present.

    When the temporal span exceeds ``max_window_days`` calendar days, the
    first chronological window of that length containing at least the minimum
    number of observed days is retained. All observed days inside the selected
    window are preserved.
    """

    def __init__(
        self,
        schema: Schema,
        min_observed_days: int = 7,
        max_window_days: int = 28,
    ) -> None:

        if min_observed_days < 1:
            raise ValueError(
                "min_observed_days must be >= 1."
            )

        if max_window_days < min_observed_days:
            raise ValueError(
                "max_window_days must be >= min_observed_days."
            )

        self.schema = schema
        self.min_observed_days = min_observed_days
        self.max_window_days = max_window_days

    def curate(
        self,
        audio: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if audio.empty:
            raise ValueError(
                "Cannot curate an empty acoustic dataset."
            )

        df = audio.copy()

        df["_curation_date"] = (
            pd.to_datetime(
                df["Date"]
            )
            .dt.normalize()
        )

        retained_frames = []
        manifest_rows = []

        keys = [
            self.schema.group,
            self.schema.bag,
        ]

        for (
            point,
            bag,
        ), capture in df.groupby(
            keys,
            sort=False,
        ):

            dates = pd.DatetimeIndex(
                capture[
                    "_curation_date"
                ]
                .drop_duplicates()
                .sort_values()
            )

            n_original_days = len(
                dates
            )

            first_date = dates.min()
            last_date = dates.max()

            original_span_days = int(
                (
                    last_date
                    - first_date
                ).days
                + 1
            )

            selected_start = pd.NaT
            selected_end = pd.NaT
            status = "excluded_min_days"

            retained = capture.iloc[
                0:0
            ]

            if (
                n_original_days
                >= self.min_observed_days
            ):

                if (
                    original_span_days
                    <= self.max_window_days
                ):

                    retained = capture

                    selected_start = (
                        first_date
                    )

                    selected_end = (
                        last_date
                    )

                    status = "retained_full"

                else:

                    window = (
                        self._first_eligible_window(
                            dates
                        )
                    )

                    if window is not None:

                        (
                            selected_start,
                            selected_end,
                        ) = window

                        retained = capture[
                            capture[
                                "_curation_date"
                            ]
                            .between(
                                selected_start,
                                selected_end,
                                inclusive="both",
                            )
                        ]

                        status = "windowed"

                    else:

                        status = (
                            "excluded_no_eligible_window"
                        )

            retained_days = (
                retained[
                    "_curation_date"
                ]
                .nunique()
            )

            if not retained.empty:
                retained_frames.append(
                    retained
                )

            manifest_rows.append(
                {
                    self.schema.group:
                        point,

                    self.schema.bag:
                        bag,

                    self.schema.target:
                        capture[
                            self.schema.target
                        ].iloc[0],

                    "status":
                        status,

                    "original_n_days":
                        int(
                            n_original_days
                        ),

                    "original_first_date":
                        first_date,

                    "original_last_date":
                        last_date,

                    "original_span_days":
                        original_span_days,

                    "selected_window_start":
                        selected_start,

                    "selected_window_end":
                        selected_end,

                    "retained_n_days":
                        int(
                            retained_days
                        ),

                    "original_n_audio":
                        int(
                            len(
                                capture
                            )
                        ),

                    "retained_n_audio":
                        int(
                            len(
                                retained
                            )
                        ),
                }
            )

        curated = pd.concat(
            retained_frames,
            ignore_index=True,
        )

        curated = (
            curated.sort_values(
                [
                    self.schema.group,
                    self.schema.bag,
                    self.schema.datetime,
                ]
            )
            .drop(
                columns=[
                    "_curation_date"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        manifest = (
            pd.DataFrame(
                manifest_rows
            )
            .sort_values(
                [
                    self.schema.group,
                    self.schema.bag,
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return (
            curated,
            manifest,
        )

    def _first_eligible_window(
        self,
        dates: pd.DatetimeIndex,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:

        for start in dates:

            end = (
                pd.Timestamp(
                    start
                )
                + pd.Timedelta(
                    days=(
                        self.max_window_days
                        - 1
                    )
                )
            )

            n_days = int(
                (
                    (
                        dates
                        >= start
                    )
                    & (
                        dates
                        <= end
                    )
                )
                .sum()
            )

            if (
                n_days
                >= self.min_observed_days
            ):

                return (
                    pd.Timestamp(
                        start
                    ),
                    end,
                )

        return None
