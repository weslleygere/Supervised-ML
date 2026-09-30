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
    Synthetic dataset with known aggregation results.

    CapturePointId B1 contains:

    Day 1
        A1 -> two valid segments:
              ACI = 1, 3
              aggregated Audio_Name = 2

        A2 -> one valid segment:
              ACI = 4
              aggregated Audio_Name = 4

        daily mean = 3
        daily population SD = 1

    Day 2
        A3 -> three valid segments:
              ACI = 5, 7, 9
              aggregated Audio_Name = 7

        daily mean = 7
        daily population SD = 0

    Equal-day representation:

        mean = (3 + 7) / 2 = 5

        within variance
            = (1² + 0²) / 2
            = 0.5

        between variance
            = ((3 - 5)² + (7 - 5)²) / 2
            = 4

        total variance
            = 0.5 + 4
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
                    "2026-01-01 08:00",
                    "2026-01-01 08:00",
                    "2026-01-01 09:00",
                    "2026-01-02 08:00",
                    "2026-01-02 08:00",
                    "2026-01-02 08:00",
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
# AUDIO-NAME AGGREGATION
# =============================================================================


def test_audio_name_rows_are_averaged_before_temporal_aggregation(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    One Audio_Name must become exactly one atomic observation,
    independently of whether it contains 1, 2 or 3 valid segment rows.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    a1 = audio[
        audio["Audio_Name"] == "A1"
    ].iloc[0]

    a2 = audio[
        audio["Audio_Name"] == "A2"
    ].iloc[0]

    a3 = audio[
        audio["Audio_Name"] == "A3"
    ].iloc[0]

    assert a1["ACI"] == pytest.approx(
        2.0
    )

    assert a2["ACI"] == pytest.approx(
        4.0
    )

    assert a3["ACI"] == pytest.approx(
        7.0
    )

    np.testing.assert_allclose(
        a1["Embedding"],
        np.array(
            [2.0, 4.0]
        ),
    )

    np.testing.assert_allclose(
        a2["Embedding"],
        np.array(
            [4.0, 6.0]
        ),
    )

    np.testing.assert_allclose(
        a3["Embedding"],
        np.array(
            [8.0, 10.0]
        ),
    )


# =============================================================================
# EQUAL-DAY MEAN
# =============================================================================


def test_mean_uses_equal_day_weighting(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Days receive equal weight regardless of how many Audio_Name observations
    are available within each day.
    """

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

    expected_mean = 5.0

    for aggregation in signatures:

        result = signatures[
            aggregation
        ].iloc[0]

        assert result[
            "ACI_mean"
        ] == pytest.approx(
            expected_mean
        )


# =============================================================================
# NESTED REPRESENTATIONS
# =============================================================================


def test_three_aggregation_strategies_share_the_same_mean(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Aggregation strategies may differ only in their representation of
    temporal variability, not in the definition of the mean.
    """

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

    mean = signatures[
        "mean"
    ].iloc[0]

    mean_std = signatures[
        "mean_std"
    ].iloc[0]

    hierarchical = signatures[
        "hierarchical"
    ].iloc[0]

    assert mean[
        "ACI_mean"
    ] == pytest.approx(
        mean_std[
            "ACI_mean"
        ]
    )

    assert mean[
        "ACI_mean"
    ] == pytest.approx(
        hierarchical[
            "ACI_mean"
        ]
    )


# =============================================================================
# VARIANCE DECOMPOSITION
# =============================================================================


def test_hierarchical_variance_decomposition(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Total temporal variance must equal within-day variance plus
    between-day variance.
    """

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

    mean_std = signatures[
        "mean_std"
    ].iloc[0]

    hierarchical = signatures[
        "hierarchical"
    ].iloc[0]

    total_std = mean_std[
        "ACI_total_std"
    ]

    within_std = hierarchical[
        "ACI_within_day_std"
    ]

    between_std = hierarchical[
        "ACI_between_day_std"
    ]

    assert within_std == pytest.approx(
        np.sqrt(
            0.5
        )
    )

    assert between_std == pytest.approx(
        2.0
    )

    assert total_std == pytest.approx(
        np.sqrt(
            4.5
        )
    )

    assert total_std**2 == pytest.approx(
        within_std**2
        + between_std**2
    )


# =============================================================================
# EMBEDDINGS
# =============================================================================


def test_embedding_aggregation_dimensions_and_means(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Embeddings use the same mean/std/hierarchical definitions element-wise.
    """

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

    mean_embedding = signatures[
        "mean"
    ].iloc[0][
        "Embedding"
    ]

    mean_std_embedding = signatures[
        "mean_std"
    ].iloc[0][
        "Embedding"
    ]

    hierarchical_embedding = signatures[
        "hierarchical"
    ].iloc[0][
        "Embedding"
    ]

    assert len(
        mean_embedding
    ) == 2

    assert len(
        mean_std_embedding
    ) == 4

    assert len(
        hierarchical_embedding
    ) == 6

    expected_mean = np.array(
        [
            5.5,
            7.5,
        ]
    )

    np.testing.assert_allclose(
        mean_embedding,
        expected_mean,
    )

    np.testing.assert_allclose(
        mean_std_embedding[:2],
        expected_mean,
    )

    np.testing.assert_allclose(
        hierarchical_embedding[:2],
        expected_mean,
    )


# =============================================================================
# SOURCE-SEGMENT ASSUMPTION
# =============================================================================


def test_more_than_three_segments_per_audio_name_is_rejected(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    The input dataset is expected to contain 1-3 valid segment rows per
    Audio_Name after upstream QC.
    """

    extra = raw_data.iloc[
        [0]
    ].copy()

    invalid = pd.concat(
        [
            raw_data,
            extra,
            extra,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError
    ):
        PreSplitProcessor(
            schema=schema
        ).process(
            invalid,
            aggregation="mean",
        )


# =============================================================================
# TARGET / GROUP CONSISTENCY
# =============================================================================


def test_capture_point_cannot_have_multiple_targets(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    invalid = raw_data.copy()

    invalid.loc[
        invalid.index[-1],
        "meanHFI",
    ] = 20.0

    with pytest.raises(
        ValueError
    ):
        PreSplitProcessor(
            schema=schema
        ).process(
            invalid,
            aggregation="mean",
        )


def test_capture_point_cannot_belong_to_multiple_points(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:

    invalid = raw_data.copy()

    invalid.loc[
        invalid.index[-1],
        "Point",
    ] = "P2"

    with pytest.raises(
        ValueError
    ):
        PreSplitProcessor(
            schema=schema
        ).process(
            invalid,
            aggregation="mean",
        )
