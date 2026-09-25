import numpy as np
import pandas as pd
import pytest
from src.core.data.schema import Schema


@pytest.fixture
def schema():
    return Schema(target="meanHFI", group="Point", bag="CapturePointId", audio="Audio_Name", datetime="AudioDate", index_prefixes=("s_", "t_"), embedding="embedding")


def make_raw(points=8, captures=2, days=3, recordings=4):
    rng = np.random.default_rng(17)
    rows = []
    for point in range(points):
        for capture in range(captures):
            installation = point * captures + capture
            target = 0.4 + point * 0.35 + capture * 0.03
            base = np.array([target, target ** 2 / 3, np.sin(target), np.cos(target), point % 3, capture])
            for day in range(days):
                for audio in range(recordings):
                    # Vary surviving segment count; one Audio_Name still counts once.
                    for segment in range(1 + (audio % 3)):
                        vector = base + rng.normal(0, 0.15, 6)
                        rows.append({"Point": point, "CapturePointId": installation,
                                     "Audio_Name": f"{installation}_{day}_{audio}", "Segment_Name": f"{installation}_{day}_{audio}_{segment}",
                                     "AudioDate": pd.Timestamp("2025-01-01 04:00", tz="America/Fortaleza") + pd.Timedelta(days=day, minutes=audio * 10),
                                     "meanHFI": target, "s_a": vector[0], "s_b": vector[2], "t_c": vector[1], "embedding": vector})
    return pd.DataFrame(rows)


@pytest.fixture
def raw():
    return make_raw()
