import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.data.temporal_audit import TemporalAudit


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def schema() -> Schema:
    return Schema(
        target="meanHFI",
        group="Point",
        bag="CapturePointId",
        audio="Audio_Name",
        datetime="AudioDate",
        index_prefixes=("s_", "t_"),
        embedding="embedding",
    )


@pytest.fixture
def audio_data() -> pd.DataFrame:
    """
    Atomic Audio_Name observations for two recording days.

    Day 1
    -----
    04:05
    04:35
    05:05
    05:35

    All four 30-minute bins are represented.

    Day 2
    -----
    04:15
    05:45

    Only the first and fourth bins are represented.
    """

    return pd.DataFrame(
        {
            "Point": [
                "P1",
                "P1",
                "P1",
                "P1",
                "P1",
                "P1",
            ],

            "CapturePointId": [
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
            ],

            "Date": [
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-02").date(),
                pd.Timestamp("2026-01-02").date(),
            ],

            "Audio_Name": [
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
                "A6",
            ],

            "AudioDate": pd.to_datetime(
                [
                    "2026-01-01 04:05",
                    "2026-01-01 04:35",
                    "2026-01-01 05:05",
                    "2026-01-01 05:35",
                    "2026-01-02 04:15",
                    "2026-01-02 05:45",
                ]
            ),

            "meanHFI": [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ],
        }
    )


# =============================================================================
# BIN DEFINITION
# =============================================================================


def test_default_bins(
    schema: Schema,
) -> None:

    audit = TemporalAudit(
        schema=schema
    )

    assert audit.bin_labels == (
        "0400_0430",
        "0430_0500",
        "0500_0530",
        "0530_0600",
    )


# =============================================================================
# DAILY COVERAGE
# =============================================================================


def test_daily_coverage_counts_observations_by_bin(
    schema: Schema,
    audio_data: pd.DataFrame,
) -> None:

    audit = TemporalAudit(
        schema=schema
    )

    daily = (
        audit.daily_coverage(
            audio_data
        )
        .set_index(
            "Date"
        )
    )

    day_1 = pd.Timestamp(
        "2026-01-01"
    ).date()

    day_2 = pd.Timestamp(
        "2026-01-02"
    ).date()

    # Day 1: one observation in every bin.

    assert daily.loc[
        day_1,
        "audio_0400_0430",
    ] == 1

    assert daily.loc[
        day_1,
        "audio_0430_0500",
    ] == 1

    assert daily.loc[
        day_1,
        "audio_0500_0530",
    ] == 1

    assert daily.loc[
        day_1,
        "audio_0530_0600",
    ] == 1

    assert daily.loc[
        day_1,
        "n_audio_in_window",
    ] == 4

    assert daily.loc[
        day_1,
        "n_bins_covered",
    ] == 4

    assert bool(
        daily.loc[
            day_1,
            "complete_bin_coverage",
        ]
    )

    # Day 2: only first and fourth bins.

    assert daily.loc[
        day_2,
        "audio_0400_0430",
    ] == 1

    assert daily.loc[
        day_2,
        "audio_0430_0500",
    ] == 0

    assert daily.loc[
        day_2,
        "audio_0500_0530",
    ] == 0

    assert daily.loc[
        day_2,
        "audio_0530_0600",
    ] == 1

    assert daily.loc[
        day_2,
        "n_audio_in_window",
    ] == 2

    assert daily.loc[
        day_2,
        "n_bins_covered",
    ] == 2

    assert not bool(
        daily.loc[
            day_2,
            "complete_bin_coverage",
        ]
    )


# =============================================================================
# CAPTUREPOINTID SUMMARY
# =============================================================================


def test_capture_summary_describes_coverage_across_days(
    schema: Schema,
    audio_data: pd.DataFrame,
) -> None:

    audit = TemporalAudit(
        schema=schema
    )

    result = audit.build(
        audio_data
    )

    summary = (
        result[
            "capture"
        ]
        .iloc[0]
    )

    assert summary[
        "n_days"
    ] == 2

    assert summary[
        "n_audio"
    ] == 6

    assert summary[
        "min_audio_per_day"
    ] == 2

    assert summary[
        "median_audio_per_day"
    ] == pytest.approx(
        3.0
    )

    assert summary[
        "mean_audio_per_day"
    ] == pytest.approx(
        3.0
    )

    assert summary[
        "max_audio_per_day"
    ] == 4

    assert summary[
        "n_complete_days"
    ] == 1

    assert summary[
        "complete_day_fraction"
    ] == pytest.approx(
        0.5
    )

    # First bin occurs on both days.

    assert summary[
        "audio_0400_0430"
    ] == 2

    assert summary[
        "days_0400_0430"
    ] == 2

    assert summary[
        "coverage_0400_0430"
    ] == pytest.approx(
        1.0
    )

    # Second bin occurs on only one of the two days.

    assert summary[
        "audio_0430_0500"
    ] == 1

    assert summary[
        "days_0430_0500"
    ] == 1

    assert summary[
        "coverage_0430_0500"
    ] == pytest.approx(
        0.5
    )

    # Fourth bin occurs on both days.

    assert summary[
        "audio_0530_0600"
    ] == 2

    assert summary[
        "days_0530_0600"
    ] == 2

    assert summary[
        "coverage_0530_0600"
    ] == pytest.approx(
        1.0
    )


# =============================================================================
# WINDOW BOUNDARIES
# =============================================================================


def test_recording_window_is_left_closed_right_open(
    schema: Schema,
) -> None:

    audio = pd.DataFrame(
        {
            "Point": [
                "P1",
                "P1",
                "P1",
            ],

            "CapturePointId": [
                "C1",
                "C1",
                "C1",
            ],

            "Date": [
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-01").date(),
            ],

            "Audio_Name": [
                "A1",
                "A2",
                "A3",
            ],

            "AudioDate": pd.to_datetime(
                [
                    "2026-01-01 04:00",
                    "2026-01-01 05:59",
                    "2026-01-01 06:00",
                ]
            ),

            "meanHFI": [
                10.0,
                10.0,
                10.0,
            ],
        }
    )

    audit = TemporalAudit(
        schema=schema
    )

    daily = (
        audit.daily_coverage(
            audio
        )
        .iloc[0]
    )

    assert daily[
        "n_audio"
    ] == 3

    assert daily[
        "n_audio_in_window"
    ] == 2

    assert daily[
        "n_audio_outside_window"
    ] == 1

    assert daily[
        "audio_0400_0430"
    ] == 1

    assert daily[
        "audio_0530_0600"
    ] == 1