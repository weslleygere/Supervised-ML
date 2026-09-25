"""Training-only block preprocessing and a serializable inference pipeline."""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.core.data.schema import Schema
from src.core.models.factory import ModelFactory
from .presplit import SignatureBuilder


class ModelConvergenceError(RuntimeError):
    pass


class PostSplitProcessor:
    def __init__(self, schema: Schema, config: dict):
        self.schema = schema
        self.config = config
        self.blocks = {}
        self.target_scaler = RobustScaler()

    def _matrix(self, df: pd.DataFrame, block: str, columns=None):
        if block == "embeddings":
            return np.stack(df[self.schema.embedding]).astype(float)
        return df[columns].to_numpy(float)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        parts = []
        self.availability_columns = sorted(c for c in df if c.startswith("availability_"))
        for block in ("indices", "embeddings"):
            if self.config["feature_set"] not in (block, "both"):
                continue
            columns = self.schema.index_columns(df.columns) if block == "indices" else None
            if block == "indices" and not columns:
                raise ValueError("The indices representation has no index features.")
            matrix = self._matrix(df, block, columns)
            scaling = self.config["scaling"]
            scaler = StandardScaler() if scaling == "standard" else RobustScaler() if scaling == "robust" else None
            if scaler is not None:
                matrix = scaler.fit_transform(matrix)
            components = self.config.get(f"pca_{block}")
            pca = None
            if components is not None:
                if isinstance(components, int) and components > min(len(df) - 1, matrix.shape[1]):
                    raise ValueError("Infeasible PCA count; candidates must be filtered before optimization.")
                pca = PCA(n_components=components, svd_solver="full")
                matrix = pca.fit_transform(matrix)
            self.blocks[block] = (columns, scaler, pca)
            parts.append(matrix)
        if self.availability_columns:
            parts.append(df[self.availability_columns].to_numpy(float))
        x = np.concatenate(parts, axis=1)
        y = self.target_scaler.fit_transform(df[[self.schema.target]].to_numpy(float)).ravel()
        self.diagnostics = {
            block: {"n_features_out": int(pca.n_components_) if pca is not None else int(scaler.n_features_in_) if scaler is not None else (len(columns) if columns is not None else len(df[self.schema.embedding].iloc[0])),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist() if pca is not None else None}
            for block, (columns, scaler, pca) in self.blocks.items()
        }
        return x, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for block, (columns, scaler, pca) in self.blocks.items():
            matrix = self._matrix(df, block, columns)
            if scaler is not None:
                matrix = scaler.transform(matrix)
            if pca is not None:
                matrix = pca.transform(matrix)
            parts.append(matrix)
        if self.availability_columns:
            parts.append(df[self.availability_columns].to_numpy(float))
        return np.concatenate(parts, axis=1)

    def inverse_target(self, values) -> np.ndarray:
        return self.target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def fit_estimator(model, x, y) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    messages = [str(w.message) for w in caught]
    # GPR bound warnings diagnose the kernel search; SVR/linear solver failure
    # means a candidate was not actually optimized to its stated solution.
    if getattr(model, "fit_status_", 0) != 0:
        raise ModelConvergenceError("SVR reached its iteration limit.")
    if model.__class__.__name__ != "GaussianProcessRegressor" and any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ModelConvergenceError("; ".join(messages))
    return messages


@dataclass
class FittedPipeline:
    schema: Schema
    config: dict
    processor: PostSplitProcessor
    estimator: object
    warnings: list[str]
    recording_start: str = "04:00"
    recording_end: str = "06:00"

    @classmethod
    def fit(cls, schema, config, train, seed, jobs=1, recording_start="04:00", recording_end="06:00"):
        processor = PostSplitProcessor(schema, config)
        x, y = processor.fit_transform(train)
        model = ModelFactory.create_model(config["model"], config["model_params"], seed, jobs)
        messages = fit_estimator(model, x, y)
        return cls(schema, config, processor, model, messages, recording_start, recording_end)

    def predict(self, signatures: pd.DataFrame) -> np.ndarray:
        values = self.processor.inverse_target(self.estimator.predict(self.processor.transform(signatures)))
        if not np.isfinite(values).all():
            raise ValueError("Model produced non-finite predictions.")
        return values

    def predict_recordings(self, audio: pd.DataFrame) -> pd.DataFrame:
        """Predict grouped, already prepared Audio_Name samples (no fitting)."""
        signatures = SignatureBuilder(audio, self.schema).full(self.config["aggregation"])
        result = signatures[[self.schema.group, self.schema.bag]].copy()
        result["prediction"] = self.predict(signatures)
        return result

    def predict_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Inference from schema-compatible segment features, without HFI labels.

        Raw audio feature extraction is external to this repository.
        """
        from .presplit import PreSplitProcessor
        raw = raw.copy()
        raw[self.schema.target] = 0.0  # placeholder never used by transform/predict
        audio = PreSplitProcessor(self.schema).prepare_audio(raw, self.recording_start, self.recording_end)
        return self.predict_recordings(audio)
