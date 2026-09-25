"""Deterministic recording preparation and subset-only acoustic aggregation."""
import hashlib
import json
from datetime import time

import numpy as np
import pandas as pd

from src.core.data.schema import Schema


class PreSplitProcessor:
    """One Audio_Name mean is one operational sample, regardless of segment count."""

    def __init__(self, schema: Schema):
        self.schema = schema

    def prepare_audio(self, raw: pd.DataFrame, start: str = "04:00", end: str = "06:00") -> pd.DataFrame:
        s = self.schema
        required = {s.group, s.bag, s.audio, s.datetime, s.target}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        df = raw.copy()
        if df[list(required)].isna().any().any():
            raise ValueError("Identifiers, timestamps and HFI labels must not be missing.")
        df[s.target] = pd.to_numeric(df[s.target], errors="raise")
        if not np.isfinite(df[s.target]).all():
            raise ValueError("HFI labels must be finite.")
        installation = df.groupby(s.bag).agg({s.target: "nunique", s.group: "nunique"})
        if (installation > 1).any().any():
            raise ValueError("Each CapturePointId must have one HFI label and belong to one Point.")
        df[s.datetime] = pd.to_datetime(df[s.datetime], errors="raise")
        try:
            local_time = df[s.datetime].dt.time
        except AttributeError as exc:
            raise ValueError("Recording timestamps must use a consistent local timezone.") from exc
        # Compare local wall time; do not silently convert morning recordings to UTC.
        df = df[(local_time >= time.fromisoformat(start)) & (local_time < time.fromisoformat(end))].copy()
        if df.empty:
            raise ValueError("No recordings fall within the configured local-time window.")
        keys = [s.group, s.bag, s.audio]
        if df.groupby(keys)[s.datetime].nunique().gt(1).any():
            raise ValueError("Audio_Name segments must share their parent recording timestamp.")
        index_cols = s.index_columns(df.columns)
        if not index_cols and s.embedding not in df:
            raise ValueError("No acoustic features are present.")
        if index_cols:
            df[index_cols] = df[index_cols].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(df[index_cols].to_numpy()).all():
                raise ValueError("Acoustic indices must be finite; audit invalid features before running.")
        if s.embedding in df:
            vectors = df[s.embedding].map(lambda v: np.asarray(v, dtype=float))
            lengths = {v.shape for v in vectors}
            if len(lengths) != 1 or any(v.ndim != 1 or len(v) == 0 or not np.isfinite(v).all() for v in vectors):
                raise ValueError("Embeddings must be finite, nonempty vectors of equal length.")
            df[s.embedding] = vectors
        rows = []
        for identity, group in df.groupby(keys, sort=True):
            row = dict(zip(keys, identity))
            row[s.datetime] = group[s.datetime].iloc[0]
            row[s.target] = float(group[s.target].iloc[0])
            row["Date"] = row[s.datetime].date().isoformat()
            row["n_segments"] = len(group)
            row["sample_id"] = hashlib.sha256(json.dumps(list(map(str, identity))).encode()).hexdigest()
            row.update(group[index_cols].mean().to_dict())
            if s.embedding in df:
                row[s.embedding] = np.stack(group[s.embedding]).mean(axis=0)
            rows.append(row)
        return pd.DataFrame(rows).sort_values([s.bag, "sample_id"]).reset_index(drop=True)

    def process(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Convenience entry point retaining the original full-signature behavior."""
        audio = self.prepare_audio(raw)
        return SignatureBuilder(audio, self.schema).full("hierarchical")


class SignatureBuilder:
    """Build full training signatures or signatures using exact sample manifests."""

    def __init__(self, audio: pd.DataFrame, schema: Schema):
        self.audio = audio
        self.schema = schema
        self.index_cols = schema.index_columns(audio.columns)
        self.indexed = audio.set_index("sample_id", drop=False)
        if not self.indexed.index.is_unique:
            raise ValueError("Aggregated sample IDs must be unique.")
        self._full_cache = {}

    @staticmethod
    def _summarize(matrix: np.ndarray, dates: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == "mean":
            return matrix.mean(axis=0)
        if strategy == "mean_std":
            return np.concatenate([matrix.mean(axis=0), matrix.std(axis=0, ddof=0)])
        if strategy != "hierarchical":
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
        days = [matrix[dates == day] for day in np.unique(dates)]
        daily_mean = np.stack([values.mean(axis=0) for values in days])
        daily_std = np.stack([values.std(axis=0, ddof=0) for values in days])
        return np.concatenate([
            daily_mean.mean(axis=0),
            np.sqrt(np.mean(daily_std ** 2, axis=0)),
            daily_mean.std(axis=0, ddof=0),
        ])

    def _row(self, group: pd.DataFrame, strategy: str) -> dict:
        s = self.schema
        dates = group["Date"].to_numpy()
        row = {s.group: group[s.group].iloc[0], s.bag: group[s.bag].iloc[0], s.target: float(group[s.target].iloc[0]) if s.target in group else np.nan}
        suffixes = {"mean": ["mean"], "mean_std": ["mean", "std"], "hierarchical": ["mean", "within_day_std", "between_day_std"]}[strategy]
        if self.index_cols:
            values = self._summarize(group[self.index_cols].to_numpy(float), dates, strategy)
            row.update(zip([f"{col}__{suffix}" for suffix in suffixes for col in self.index_cols], values))
        if s.embedding in group:
            row[s.embedding] = self._summarize(np.stack(group[s.embedding]), dates, strategy)
        if strategy == "mean_std":
            row["availability_std"] = float(len(group) > 1)
        elif strategy == "hierarchical":
            sizes = group.groupby("Date").size()
            row["availability_within_day_fraction"] = float((sizes > 1).mean())
            row["availability_between_day"] = float(len(sizes) > 1)
        return row

    def full(self, strategy: str) -> pd.DataFrame:
        if strategy not in self._full_cache:
            rows = [self._row(g, strategy) for _, g in self.audio.groupby(self.schema.bag, sort=True)]
            self._full_cache[strategy] = pd.DataFrame(rows)
        return self._full_cache[strategy]

    def submissions(self, manifest: pd.DataFrame, strategy: str) -> pd.DataFrame:
        rows = []
        for entry in manifest.to_dict("records"):
            ids = list(entry["sample_ids"])
            if len(ids) != entry["n_recordings"] or len(set(ids)) != len(ids):
                raise ValueError("Submission must contain exactly the requested number of distinct samples.")
            selected = self.indexed.loc[ids]
            if not selected[self.schema.bag].eq(entry[self.schema.bag]).all():
                raise ValueError("A submission cannot mix installations.")
            if not selected[self.schema.group].eq(entry[self.schema.group]).all():
                raise ValueError("Submission Point does not match its samples.")
            row = self._row(selected, strategy)
            row.update({key: value for key, value in entry.items() if key != "sample_ids"})
            rows.append(row)
        return pd.DataFrame(rows)
