import pandas as pd
import pytest

from src.core.data.schema import Schema
from src.core.data.temporal_curation import TemporalCurator


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


def _rows(
    point: str,
    bag: str,
    offsets: list[int],
) -> list[dict]:

    rows = []

    for offset in offsets:
        dt = (
            pd.Timestamp(
                "2026-01-01 04:00"
            )
            + pd.Timedelta(
                days=offset
            )
        )

        rows.append(
            {
                "Point": point,
                "CapturePointId": bag,
                "Date": dt.date(),
                "Audio_Name": f"{bag}_{offset}",
                "AudioDate": dt,
                "meanHFI": 1.0,
            }
        )

    return rows


def test_minimum_seven_days_is_per_capturepoint(
    schema: Schema,
) -> None:

    audio = pd.DataFrame(
        _rows(
            "P1",
            "C1",
            list(
                range(
                    6
                )
            ),
        )
        + _rows(
            "P1",
            "C2",
            list(
                range(
                    7
                )
            ),
        )
    )

    curated, manifest = TemporalCurator(
        schema=schema,
        min_observed_days=7,
        max_window_days=28,
    ).curate(
        audio
    )

    assert set(
        curated[
            "CapturePointId"
        ]
    ) == {
        "C2",
    }

    status = (
        manifest.set_index(
            "CapturePointId"
        )[
            "status"
        ]
        .to_dict()
    )

    assert status[
        "C1"
    ] == "excluded_min_days"

    assert status[
        "C2"
    ] == "retained_full"


def test_long_capture_is_limited_to_first_28_day_window(
    schema: Schema,
) -> None:

    audio = pd.DataFrame(
        _rows(
            "P2",
            "C3",
            list(
                range(
                    35
                )
            ),
        )
    )

    curated, manifest = TemporalCurator(
        schema=schema,
        min_observed_days=7,
        max_window_days=28,
    ).curate(
        audio
    )

    assert curated[
        "Date"
    ].nunique() == 28

    row = manifest.iloc[
        0
    ]

    assert row[
        "status"
    ] == "windowed"

    assert row[
        "retained_n_days"
    ] == 28
