"""Reproducible sampling of aggregated Audio_Name representations."""

import hashlib
import json

import numpy as np
import pandas as pd

from .schema import Schema


def stable_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([seed, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sample_submissions(
    audio: pd.DataFrame,
    schema: Schema,
    efforts: tuple[int, ...],
    repeats: int,
    seed: int,
    namespace: str,
) -> pd.DataFrame:
    """Sample nested subsets on a common installation cohort.

    sample_id identifies an aggregated recording, not an original segment.
    Sorting and per-installation seeds make selections independent of row order
    and of the other installations present in a fold.
    """
    rows = []
    for installation, group in audio.groupby(schema.bag, sort=True):
        group = group.sort_values("sample_id")
        if len(group) < max(efforts):
            continue
        for repeat in range(repeats):
            rng = np.random.default_rng(stable_seed(seed, namespace, installation, repeat))
            order = rng.permutation(len(group))[:max(efforts)]
            for effort in efforts:
                selected = group.iloc[order[:effort]]
                rows.append({
                    schema.group: selected[schema.group].iloc[0],
                    schema.bag: installation,
                    schema.target: float(selected[schema.target].iloc[0]),
                    "submission_id": f"{namespace}:{installation}:{repeat}:{effort}",
                    "sampling_repeat": repeat,
                    "n_recordings": effort,
                    "n_days": int(selected["Date"].nunique()),
                    "collection_span_days": int(
                        (selected[schema.datetime].max() - selected[schema.datetime].min()).total_seconds() // 86400
                    ),
                    "sample_ids": selected["sample_id"].tolist(),
                })
    if not rows:
        raise ValueError(f"No installations support all requested efforts {efforts} in {namespace}.")
    return pd.DataFrame(rows)
