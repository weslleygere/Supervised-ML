import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
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
        datetime="Datetime",
        index_prefixes=("ACI",),
        embedding="Embedding",
    )


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """
    Synthetic segment-level data with known aggregation results.

    Day 1
    -----
    A1:
        ACI segments = [1, 3]
        atomic mean = 2

    A2:
        ACI segment = [4]
        atomic mean = 4

    Daily mean = 3
    Daily population SD = 1

    Day 2
    -----
    A3:
        ACI segments = [5, 7, 9]
        atomic mean = 7

    Daily mean = 7
    Daily population SD = 0

    Equal-day final representation
    ------------------------------
    mean = 5

    within-day variance =
        (1² + 0²) / 2
        = 0.5

    between-day variance =
        ((3 - 5)² + (7 - 5)²) / 2
        = 4

    total variance =
        0.5 + 4
        = 4.5
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
                "B1",
                "B1",
                "B1",
                "B1",
                "B1",
                "B1",
            ],

            "Audio_Name": [
                "A1",
                "A1",
                "A2",
                "A3",
                "A3",
                "A3",
            ],

            "Datetime": pd.to_datetime(
                [
                    "2026-01-01 04:10",
                    "2026-01-01 04:10",
                    "2026-01-01 04:40",
                    "2026-01-02 05:20",
                    "2026-01-02 05:20",
                    "2026-01-02 05:20",
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

            "ACI": [
                1.0,
                3.0,
                4.0,
                5.0,
                7.0,
                9.0,
            ],

            "Embedding": [
                np.array([1.0, 3.0]),
                np.array([3.0, 5.0]),
                np.array([4.0, 6.0]),
                np.array([6.0, 8.0]),
                np.array([8.0, 10.0]),
                np.array([10.0, 12.0]),
            ],
        }
    )


# =============================================================================
# SEGMENT -> AUDIO_NAME
# =============================================================================


def test_prepare_audio_creates_atomic_observations(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    assert len(audio) == 3

    assert set(
        audio["Audio_Name"]
    ) == {
        "A1",
        "A2",
        "A3",
    }


def test_prepare_audio_averages_segments(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = (
        processor.prepare_audio(
            raw_data
        )
        .set_index(
            "Audio_Name"
        )
    )

    assert audio.loc[
        "A1",
        "ACI",
    ] == pytest.approx(
        2.0
    )

    assert audio.loc[
        "A2",
        "ACI",
    ] == pytest.approx(
        4.0
    )

    assert audio.loc[
        "A3",
        "ACI",
    ] == pytest.approx(
        7.0
    )

    np.testing.assert_allclose(
        audio.loc[
            "A1",
            "Embedding",
        ],
        [2.0, 4.0],
    )

    np.testing.assert_allclose(
        audio.loc[
            "A3",
            "Embedding",
        ],
        [8.0, 10.0],
    )


def test_prepare_audio_preserves_timestamp(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = (
        processor.prepare_audio(
            raw_data
        )
        .set_index(
            "Audio_Name"
        )
    )

    assert (
        "Datetime"
        in audio.columns
    )

    assert audio.loc[
        "A1",
        "Datetime",
    ] == pd.Timestamp(
        "2026-01-01 04:10"
    )

    assert audio.loc[
        "A2",
        "Datetime",
    ] == pd.Timestamp(
        "2026-01-01 04:40"
    )

    assert audio.loc[
        "A3",
        "Datetime",
    ] == pd.Timestamp(
        "2026-01-02 05:20"
    )


def test_prepare_audio_preserves_date(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    assert audio[
        "Date"
    ].nunique() == 2

    assert set(
        audio["Date"]
    ) == {
        pd.Timestamp(
            "2026-01-01"
        ).date(),
        pd.Timestamp(
            "2026-01-02"
        ).date(),
    }


# =============================================================================
# AUDIO_NAME -> DAY
# =============================================================================


def test_prepare_daily_uses_atomic_observations(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    daily = (
        processor.prepare_daily(
            audio
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

    assert daily.loc[
        day_1,
        "ACI__daily_mean",
    ] == pytest.approx(
        3.0
    )

    assert daily.loc[
        day_1,
        "ACI__daily_std",
    ] == pytest.approx(
        1.0
    )

    assert daily.loc[
        day_2,
        "ACI__daily_mean",
    ] == pytest.approx(
        7.0
    )

    assert daily.loc[
        day_2,
        "ACI__daily_std",
    ] == pytest.approx(
        0.0
    )


# =============================================================================
# DAY -> CAPTUREPOINTID
# =============================================================================


def test_mean_uses_equal_day_weighting(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    result = processor.process(
        raw_data,
        aggregation="mean",
    ).iloc[0]

    # Day means are 3 and 7.
    #
    # Equal-day mean:
    #
    # (3 + 7) / 2 = 5
    #
    # This differs from averaging the three Audio_Name observations
    # directly:
    #
    # (2 + 4 + 7) / 3 = 4.333...

    assert result[
        "ACI_mean"
    ] == pytest.approx(
        5.0
    )


def test_variance_decomposition(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = processor.process_all(
        df_raw=raw_data,
        aggregations=(
            "mean_std",
            "hierarchical",
        ),
    )

    mean_std = (
        signatures[
            "mean_std"
        ]
        .iloc[0]
    )

    hierarchical = (
        signatures[
            "hierarchical"
        ]
        .iloc[0]
    )

    total_std = (
        mean_std[
            "ACI_total_std"
        ]
    )

    within_std = (
        hierarchical[
            "ACI_within_day_std"
        ]
    )

    between_std = (
        hierarchical[
            "ACI_between_day_std"
        ]
    )

    assert total_std == pytest.approx(
        np.sqrt(4.5)
    )

    assert within_std == pytest.approx(
        np.sqrt(0.5)
    )

    assert between_std == pytest.approx(
        2.0
    )

    assert total_std**2 == pytest.approx(
        within_std**2
        + between_std**2
    )


def test_aggregation_strategies_share_same_mean(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = processor.process_all(
        df_raw=raw_data,
        aggregations=(
            "mean",
            "mean_std",
            "hierarchical",
        ),
    )

    values = [
        signatures[
            aggregation
        ].iloc[0][
            "ACI_mean"
        ]
        for aggregation in signatures
    ]

    np.testing.assert_allclose(
        values,
        [
            5.0,
            5.0,
            5.0,
        ],
    )


# =============================================================================
# EMBEDDINGS
# =============================================================================


def test_embedding_aggregation(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = processor.process_all(
        df_raw=raw_data,
        aggregations=(
            "mean",
            "mean_std",
            "hierarchical",
        ),
    )

    mean_embedding = (
        signatures[
            "mean"
        ]
        .iloc[0][
            "Embedding"
        ]
    )

    mean_std_embedding = (
        signatures[
            "mean_std"
        ]
        .iloc[0][
            "Embedding"
        ]
    )

    hierarchical_embedding = (
        signatures[
            "hierarchical"
        ]
        .iloc[0][
            "Embedding"
        ]
    )

    np.testing.assert_allclose(
        mean_embedding,
        [
            5.5,
            7.5,
        ],
    )

    assert len(
        mean_embedding
    ) == 2

    assert len(
        mean_std_embedding
    ) == 4

    assert len(
        hierarchical_embedding
    ) == 6

    np.testing.assert_allclose(
        mean_std_embedding[
            2:
        ]
        ** 2,
        (
            hierarchical_embedding[
                2:4
            ]
            ** 2
        )
        + (
            hierarchical_embedding[
                4:6
            ]
            ** 2
        ),
    )


# =============================================================================
# REPRESENTATION ALIGNMENT
# =============================================================================


def test_aggregation_strategies_preserve_same_units(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = processor.process_all(
        df_raw=raw_data,
        aggregations=(
            "mean",
            "mean_std",
            "hierarchical",
        ),
    )

    metadata = [
        "Point",
        "CapturePointId",
        "meanHFI",
    ]

    reference = (
        signatures[
            "mean"
        ][
            metadata
        ]
        .reset_index(
            drop=True
        )
    )

    for aggregation in (
        "mean_std",
        "hierarchical",
    ):

        pd.testing.assert_frame_equal(
            reference,
            signatures[
                aggregation
            ][
                metadata
            ]
            .reset_index(
                drop=True
            ),
        )
