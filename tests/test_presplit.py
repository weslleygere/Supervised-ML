import numpy as np
import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.processors.presplit import (
    PreSplitProcessor,
)


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
        index_prefixes=(
            "ACI",
        ),
        embedding="Embedding",
    )


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """
    Synthetic dataset with known aggregation results.

    CapturePointId B1 contains:

    Day 1
    -----
    A1 -> two valid segments

        ACI = 1, 3
        Audio_Name mean = 2

        embedding:
            [1, 3]
            [3, 5]

        Audio_Name mean:
            [2, 4]

    A2 -> one valid segment

        ACI = 4

        embedding:
            [4, 6]

    Therefore, Day 1:

        ACI daily mean = (2 + 4) / 2 = 3

        ACI daily population SD = 1

        embedding daily mean:
            ([2, 4] + [4, 6]) / 2
            = [3, 5]

        embedding daily population SD:
            [1, 1]


    Day 2
    -----
    A3 -> three valid segments

        ACI = 5, 7, 9
        Audio_Name mean = 7

        embedding:
            [6, 8]
            [8, 10]
            [10, 12]

        Audio_Name mean:
            [8, 10]

    Since Day 2 contains one Audio_Name:

        ACI daily mean = 7
        ACI daily population SD = 0

        embedding daily mean = [8, 10]
        embedding daily population SD = [0, 0]


    Equal-day CapturePointId representation
    ----------------------------------------

    ACI mean:

        (3 + 7) / 2 = 5

    ACI within-day variance:

        (1² + 0²) / 2
        = 0.5

    ACI between-day variance:

        ((3 - 5)² + (7 - 5)²) / 2
        = 4

    ACI total variance:

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

            "Datetime":
                pd.to_datetime(
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
                np.array(
                    [1.0, 3.0]
                ),
                np.array(
                    [3.0, 5.0]
                ),
                np.array(
                    [4.0, 6.0]
                ),
                np.array(
                    [6.0, 8.0]
                ),
                np.array(
                    [8.0, 10.0]
                ),
                np.array(
                    [10.0, 12.0]
                ),
            ],
        }
    )


# =============================================================================
# AUDIO_NAME = ATOMIC OBSERVATION
# =============================================================================


def test_each_audio_name_becomes_one_atomic_observation(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    After segment aggregation, every Audio_Name must correspond to exactly
    one row.

    This is the atomic acoustic observation used by all subsequent analyses,
    including the future recording-effort analysis.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    assert len(
        audio
    ) == 3

    assert (
        audio[
            "Audio_Name"
        ]
        .nunique()
        == 3
    )

    counts = (
        audio.groupby(
            [
                "Point",
                "CapturePointId",
                "Date",
                "Audio_Name",
            ]
        )
        .size()
    )

    assert (
        counts
        == 1
    ).all()


def test_audio_name_rows_are_averaged_before_temporal_aggregation(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    One Audio_Name must receive equal weight as one atomic observation,
    independently of whether it originally contained 1, 2 or 3 valid
    segment rows.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    a1 = (
        audio[
            audio[
                "Audio_Name"
            ]
            == "A1"
        ]
        .iloc[
            0
        ]
    )

    a2 = (
        audio[
            audio[
                "Audio_Name"
            ]
            == "A2"
        ]
        .iloc[
            0
        ]
    )

    a3 = (
        audio[
            audio[
                "Audio_Name"
            ]
            == "A3"
        ]
        .iloc[
            0
        ]
    )

    # =========================================================================
    # ACOUSTIC INDEX
    # =========================================================================

    assert a1[
        "ACI"
    ] == pytest.approx(
        2.0
    )

    assert a2[
        "ACI"
    ] == pytest.approx(
        4.0
    )

    assert a3[
        "ACI"
    ] == pytest.approx(
        7.0
    )

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    np.testing.assert_allclose(
        a1[
            "Embedding"
        ],
        np.array(
            [
                2.0,
                4.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        a2[
            "Embedding"
        ],
        np.array(
            [
                4.0,
                6.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        a3[
            "Embedding"
        ],
        np.array(
            [
                8.0,
                10.0,
            ]
        ),
    )


def test_audio_table_preserves_date_information(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    The atomic Audio_Name table must retain its recording day so the same
    observations can later be reaggregated after effort subsampling.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    assert (
        "Date"
        in audio.columns
    )

    assert (
        audio[
            "Date"
        ]
        .nunique()
        == 2
    )


# =============================================================================
# DAILY AGGREGATION
# =============================================================================


def test_daily_statistics_use_audio_names_not_raw_segments(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Daily statistics must be calculated after segment rows have been collapsed
    to atomic Audio_Name observations.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    daily = processor.prepare_daily(
        audio
    )

    day_1 = (
        daily[
            daily[
                "Date"
            ]
            == pd.Timestamp(
                "2026-01-01"
            ).date()
        ]
        .iloc[
            0
        ]
    )

    assert day_1[
        "ACI__daily_mean"
    ] == pytest.approx(
        3.0
    )

    assert day_1[
        "ACI__daily_std"
    ] == pytest.approx(
        1.0
    )


def test_daily_standard_deviation_is_population_sd(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Daily variability must use population SD (ddof=0).

    Day 1 contains atomic ACI observations [2, 4], whose population SD is 1.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    daily = processor.prepare_daily(
        audio
    )

    day_1 = (
        daily.iloc[
            0
        ]
    )

    assert day_1[
        "ACI__daily_std"
    ] == pytest.approx(
        1.0
    )


def test_single_audio_day_has_zero_daily_variability(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    A day represented by a single Audio_Name must have population SD = 0,
    rather than NaN.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    audio = processor.prepare_audio(
        raw_data
    )

    daily = processor.prepare_daily(
        audio
    )

    day_2 = (
        daily[
            daily[
                "Date"
            ]
            == pd.Timestamp(
                "2026-01-02"
            ).date()
        ]
        .iloc[
            0
        ]
    )

    assert day_2[
        "ACI__daily_std"
    ] == pytest.approx(
        0.0
    )

    np.testing.assert_allclose(
        day_2[
            "embedding__daily_std"
        ],
        np.zeros(
            2
        ),
    )


# =============================================================================
# EQUAL-DAY AGGREGATION
# =============================================================================


def test_mean_uses_equal_day_weighting(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Days receive equal weight regardless of how many Audio_Name observations
    occur within each day.

    Day 1 has two Audio_Name observations and mean 3.
    Day 2 has one Audio_Name observation and mean 7.

    Final mean must therefore be:

        (3 + 7) / 2 = 5

    rather than the observation-weighted value:

        (2 + 4 + 7) / 3
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean",
                "mean_std",
                "hierarchical",
            ),
        )
    )

    expected_mean = 5.0

    observation_weighted_mean = (
        2.0
        + 4.0
        + 7.0
    ) / 3.0

    for aggregation in signatures:

        result = (
            signatures[
                aggregation
            ]
            .iloc[
                0
            ]
        )

        assert result[
            "ACI_mean"
        ] == pytest.approx(
            expected_mean
        )

        assert result[
            "ACI_mean"
        ] != pytest.approx(
            observation_weighted_mean
        )


# =============================================================================
# AGGREGATION REPRESENTATIONS
# =============================================================================


def test_three_aggregation_strategies_share_same_mean(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Aggregation strategies differ only in the temporal-variability information
    retained.

    The underlying acoustic mean must be identical.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean",
                "mean_std",
                "hierarchical",
            ),
        )
    )

    mean = (
        signatures[
            "mean"
        ]
        .iloc[
            0
        ]
    )

    mean_std = (
        signatures[
            "mean_std"
        ]
        .iloc[
            0
        ]
    )

    hierarchical = (
        signatures[
            "hierarchical"
        ]
        .iloc[
            0
        ]
    )

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


def test_all_aggregation_strategies_have_identical_units_and_targets(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Every aggregation representation must describe exactly the same
    CapturePointIds, Points and target values.

    This is required so candidate pipelines are evaluated on the same
    statistical units.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean",
                "mean_std",
                "hierarchical",
            ),
        )
    )

    metadata_columns = [
        "Point",
        "CapturePointId",
        "meanHFI",
    ]

    reference = (
        signatures[
            "mean"
        ][
            metadata_columns
        ]
        .sort_values(
            "CapturePointId"
        )
        .reset_index(
            drop=True
        )
    )

    for aggregation in (
        "mean_std",
        "hierarchical",
    ):

        current = (
            signatures[
                aggregation
            ][
                metadata_columns
            ]
            .sort_values(
                "CapturePointId"
            )
            .reset_index(
                drop=True
            )
        )

        pd.testing.assert_frame_equal(
            reference,
            current,
        )


# =============================================================================
# VARIANCE DECOMPOSITION
# =============================================================================


def test_hierarchical_variance_decomposition(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Equal-day temporal variance must satisfy:

        total variance
            =
        within-day variance
            +
        between-day variance
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean_std",
                "hierarchical",
            ),
        )
    )

    mean_std = (
        signatures[
            "mean_std"
        ]
        .iloc[
            0
        ]
    )

    hierarchical = (
        signatures[
            "hierarchical"
        ]
        .iloc[
            0
        ]
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

    assert (
        total_std
        ** 2
    ) == pytest.approx(
        (
            within_std
            ** 2
        )
        + (
            between_std
            ** 2
        )
    )


def test_variance_component_helper_matches_expected_values() -> None:
    """
    Test the equal-day variance decomposition independently from dataframe
    processing.
    """

    daily_means = np.array(
        [
            3.0,
            7.0,
        ]
    )

    daily_stds = np.array(
        [
            1.0,
            0.0,
        ]
    )

    (
        mean_value,
        total_std,
        within_std,
        between_std,
    ) = (
        PreSplitProcessor
        ._variance_components(
            daily_means,
            daily_stds,
        )
    )

    assert mean_value == pytest.approx(
        5.0
    )

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


# =============================================================================
# EMBEDDINGS
# =============================================================================


def test_embedding_aggregation_dimensions_and_means(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Embeddings use exactly the same aggregation definitions element-wise.

    For an original two-dimensional embedding:

        mean:
            2 dimensions

        mean_std:
            4 dimensions

        hierarchical:
            6 dimensions
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean",
                "mean_std",
                "hierarchical",
            ),
        )
    )

    mean_embedding = (
        signatures[
            "mean"
        ]
        .iloc[
            0
        ][
            "Embedding"
        ]
    )

    mean_std_embedding = (
        signatures[
            "mean_std"
        ]
        .iloc[
            0
        ][
            "Embedding"
        ]
    )

    hierarchical_embedding = (
        signatures[
            "hierarchical"
        ]
        .iloc[
            0
        ][
            "Embedding"
        ]
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
        mean_std_embedding[
            :2
        ],
        expected_mean,
    )

    np.testing.assert_allclose(
        hierarchical_embedding[
            :2
        ],
        expected_mean,
    )


def test_embedding_variance_decomposition(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Embedding variability must obey the same element-wise decomposition as
    acoustic indices.
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean_std",
                "hierarchical",
            ),
        )
    )

    mean_std_embedding = (
        signatures[
            "mean_std"
        ]
        .iloc[
            0
        ][
            "Embedding"
        ]
    )

    hierarchical_embedding = (
        signatures[
            "hierarchical"
        ]
        .iloc[
            0
        ][
            "Embedding"
        ]
    )

    # Original embedding dimension = 2.
    #
    # mean_std:
    #
    # [mean_1, mean_2, total_std_1, total_std_2]
    #
    # hierarchical:
    #
    # [mean_1, mean_2,
    #  within_1, within_2,
    #  between_1, between_2]

    total_std = (
        mean_std_embedding[
            2:
        ]
    )

    within_std = (
        hierarchical_embedding[
            2:4
        ]
    )

    between_std = (
        hierarchical_embedding[
            4:6
        ]
    )

    np.testing.assert_allclose(
        total_std
        ** 2,
        (
            within_std
            ** 2
        )
        + (
            between_std
            ** 2
        ),
    )


# =============================================================================
# REPRESENTATION DIMENSION
# =============================================================================


def test_index_representation_dimensions(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    With one original acoustic index:

        mean         -> 1 index feature
        mean_std     -> 2 index features
        hierarchical -> 3 index features
    """

    processor = PreSplitProcessor(
        schema=schema
    )

    signatures = (
        processor.process_all(
            df_raw=raw_data,
            aggregations=(
                "mean",
                "mean_std",
                "hierarchical",
            ),
        )
    )

    mean_indices = (
        schema.index_columns(
            signatures[
                "mean"
            ].columns
        )
    )

    mean_std_indices = (
        schema.index_columns(
            signatures[
                "mean_std"
            ].columns
        )
    )

    hierarchical_indices = (
        schema.index_columns(
            signatures[
                "hierarchical"
            ].columns
        )
    )

    assert len(
        mean_indices
    ) == 1

    assert len(
        mean_std_indices
    ) == 2

    assert len(
        hierarchical_indices
    ) == 3


# =============================================================================
# SOURCE-SEGMENT ASSUMPTIONS
# =============================================================================


def test_more_than_three_segments_per_audio_name_is_rejected(
    schema: Schema,
    raw_data: pd.DataFrame,
) -> None:
    """
    Input is expected to contain between one and three valid segment rows per
    Audio_Name after upstream quality control.
    """

    extra = (
        raw_data.iloc[
            [
                0
            ]
        ]
        .copy()
    )

    invalid = pd.concat(
        [
            raw_data,
            extra,
            extra,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="between 1 and 3",
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
    """
    One CapturePointId must have exactly one HFI target.
    """

    invalid = (
        raw_data.copy()
    )

    invalid.loc[
        invalid.index[
            -1
        ],
        "meanHFI",
    ] = 20.0

    with pytest.raises(
        ValueError,
        match="exactly one target",
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
    """
    One CapturePointId must belong to exactly one physical Point.
    """

    invalid = (
        raw_data.copy()
    )

    invalid.loc[
        invalid.index[
            -1
        ],
        "Point",
    ] = "P2"

    with pytest.raises(
        ValueError,
        match="exactly one Point",
    ):

        PreSplitProcessor(
            schema=schema
        ).process(
            invalid,
            aggregation="mean",
        )
