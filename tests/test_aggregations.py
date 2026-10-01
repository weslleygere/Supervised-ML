import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.processors.aggregations import AcousticAggregator
from src.core.processors.presplit import PreSplitProcessor


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
        index_prefixes=("ACI",),
        embedding="Embedding",
    )


def _make_audio(
    times,
    values,
    embeddings=None,
) -> pd.DataFrame:

    times = pd.to_datetime(
        times
    )

    data = {
        "Point": ["P1"] * len(times),
        "CapturePointId": ["C1"] * len(times),
        "Date": times.date,
        "Audio_Name": [
            f"A{i}"
            for i in range(
                len(times)
            )
        ],
        "AudioDate": times,
        "meanHFI": [10.0] * len(times),
        "ACI": values,
    }

    if embeddings is not None:
        data[
            "Embedding"
        ] = embeddings

    return pd.DataFrame(
        data
    )


# =============================================================================
# EXISTING VARIANCE REPRESENTATIONS
# =============================================================================


def test_variance_representations_preserve_expected_decomposition(
    schema: Schema,
) -> None:

    audio = _make_audio(
        times=[
            "2026-01-01 04:00",
            "2026-01-01 04:10",
            "2026-01-02 04:00",
        ],
        values=[
            2.0,
            4.0,
            7.0,
        ],
    )

    processor = PreSplitProcessor(
        schema=schema
    )

    daily = processor.prepare_daily(
        audio
    )

    aggregator = AcousticAggregator(
        schema=schema
    )

    mean_std = aggregator.build(
        audio=audio,
        daily=daily,
        aggregation="mean_std",
    ).iloc[0]

    hierarchical = aggregator.build(
        audio=audio,
        daily=daily,
        aggregation="hierarchical",
    ).iloc[0]

    # Day means:
    #
    # Day 1 = 3
    # Day 2 = 7
    #
    # Final equal-day mean = 5
    #
    # Within-day variance:
    # (1² + 0²) / 2 = 0.5
    #
    # Between-day variance:
    # ((3 - 5)² + (7 - 5)²) / 2 = 4

    assert mean_std[
        "ACI_mean"
    ] == pytest.approx(
        5.0
    )

    assert hierarchical[
        "ACI_mean"
    ] == pytest.approx(
        5.0
    )

    assert hierarchical[
        "ACI_within_day_std"
    ] == pytest.approx(
        np.sqrt(0.5)
    )

    assert hierarchical[
        "ACI_between_day_std"
    ] == pytest.approx(
        2.0
    )

    assert mean_std[
        "ACI_total_std"
    ] ** 2 == pytest.approx(
        hierarchical[
            "ACI_within_day_std"
        ] ** 2
        + hierarchical[
            "ACI_between_day_std"
        ] ** 2
    )


# =============================================================================
# ROBUST DAILY
# =============================================================================


def test_robust_daily_representation(
    schema: Schema,
) -> None:

    audio = _make_audio(
        times=[
            "2026-01-01 04:00",
            "2026-01-01 04:10",
            "2026-01-01 04:20",
            "2026-01-02 04:00",
            "2026-01-02 04:10",
            "2026-01-02 04:20",
        ],
        values=[
            0.0,
            2.0,
            4.0,
            10.0,
            12.0,
            14.0,
        ],
    )

    aggregator = AcousticAggregator(
        schema=schema
    )

    result = aggregator.build(
        audio=audio,
        daily=pd.DataFrame(),
        aggregation="robust_daily",
    ).iloc[0]

    # Daily medians:
    # [2, 12]
    #
    # Median of daily medians:
    # 7
    #
    # Daily IQRs:
    # [2, 2]
    #
    # Median within-day IQR:
    # 2
    #
    # IQR of daily medians [2, 12]:
    # 9.5 - 4.5 = 5

    assert result[
        "ACI_median"
    ] == pytest.approx(
        7.0
    )

    assert result[
        "ACI_within_day_iqr"
    ] == pytest.approx(
        2.0
    )

    assert result[
        "ACI_between_day_iqr"
    ] == pytest.approx(
        5.0
    )


# =============================================================================
# DAWN PROFILE
# =============================================================================


def test_dawn_profile_uses_equal_day_bin_means(
    schema: Schema,
) -> None:

    audio = _make_audio(
        times=[
            # Day 1
            "2026-01-01 04:00",
            "2026-01-01 04:10",
            "2026-01-01 04:30",
            "2026-01-01 05:00",
            "2026-01-01 05:30",

            # Day 2
            "2026-01-02 04:00",
            "2026-01-02 04:30",
            "2026-01-02 05:00",
            "2026-01-02 05:30",
        ],
        values=[
            # Day 1
            2.0,
            4.0,
            10.0,
            20.0,
            30.0,

            # Day 2
            9.0,
            14.0,
            24.0,
            34.0,
        ],
    )

    aggregator = AcousticAggregator(
        schema=schema
    )

    result = aggregator.build(
        audio=audio,
        daily=pd.DataFrame(),
        aggregation="dawn_profile",
    ).iloc[0]

    # First bin:
    #
    # Day 1 mean = (2 + 4) / 2 = 3
    # Day 2 mean = 9
    #
    # Equal-day profile value:
    # (3 + 9) / 2 = 6
    #
    # Importantly, this is not the raw observation mean:
    # (2 + 4 + 9) / 3 = 5

    assert result[
        "ACI_dawn_0400_0430"
    ] == pytest.approx(
        6.0
    )

    assert result[
        "ACI_dawn_0430_0500"
    ] == pytest.approx(
        12.0
    )

    assert result[
        "ACI_dawn_0500_0530"
    ] == pytest.approx(
        22.0
    )

    assert result[
        "ACI_dawn_0530_0600"
    ] == pytest.approx(
        32.0
    )


# =============================================================================
# DAWN TREND
# =============================================================================


def test_dawn_trend_recovers_known_linear_trajectory(
    schema: Schema,
) -> None:

    times = pd.to_datetime(
        [
            # Day 1
            "2026-01-01 04:00",
            "2026-01-01 04:30",
            "2026-01-01 05:00",
            "2026-01-01 05:30",

            # Day 2
            "2026-01-02 04:00",
            "2026-01-02 04:30",
            "2026-01-02 05:00",
            "2026-01-02 05:30",
        ]
    )

    time_from_five = (
        times.hour
        + times.minute / 60.0
        - 5.0
    )

    # Day 1:
    #
    # ACI = 10 + 2 * time
    #
    # level at 05:00 = 10
    # slope = 2 / hour

    day_1 = (
        10.0
        + 2.0
        * time_from_five[
            :4
        ]
    )

    # Day 2:
    #
    # ACI = 20 + 4 * time
    #
    # level at 05:00 = 20
    # slope = 4 / hour

    day_2 = (
        20.0
        + 4.0
        * time_from_five[
            4:
        ]
    )

    values = np.concatenate(
        [
            day_1,
            day_2,
        ]
    )

    embeddings = [
        np.array(
            [
                value,
                2.0 * value,
            ]
        )
        for value in values
    ]

    audio = _make_audio(
        times=times,
        values=values,
        embeddings=embeddings,
    )

    aggregator = AcousticAggregator(
        schema=schema
    )

    result = aggregator.build(
        audio=audio,
        daily=pd.DataFrame(),
        aggregation="dawn_trend",
    ).iloc[0]

    # Final index representation:
    #
    # mean level = (10 + 20) / 2 = 15
    # mean slope = (2 + 4) / 2 = 3
    # population SD of slopes [2, 4] = 1

    assert result[
        "ACI_dawn_level"
    ] == pytest.approx(
        15.0
    )

    assert result[
        "ACI_dawn_slope"
    ] == pytest.approx(
        3.0
    )

    assert result[
        "ACI_dawn_slope_std"
    ] == pytest.approx(
        1.0
    )

    # Embedding:
    #
    # [mean_level,
    #  mean_slope,
    #  slope_std]
    #
    # for each original embedding dimension.

    expected_embedding = np.array(
        [
            15.0,
            30.0,
            3.0,
            6.0,
            1.0,
            2.0,
        ]
    )

    np.testing.assert_allclose(
        result[
            "Embedding"
        ],
        expected_embedding,
    )