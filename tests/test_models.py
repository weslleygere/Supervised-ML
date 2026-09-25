import numpy as np
import pandas as pd
import pytest
from src.core.models.factory import RegressionModels
from src.core.processors.postsplit import FittedPipeline
from src.core.processors.presplit import PreSplitProcessor, SignatureBuilder


@pytest.mark.parametrize("model", [m.name for m in RegressionModels])
def test_every_family_fits_and_predicts_without_target_at_inference(raw, schema, model):
    params = {
        "PLS": {"n_components": 2},
        "GAUSSIAN_PROCESS": {"kernel": "matern15", "length_scale": 2.0, "amplitude": 1.0, "noise": 0.1},
        "XGBOOST": {"n_estimators": 10, "max_depth": 2},
        "CATBOOST": {"iterations": 10, "depth": 2},
        "RANDOM_FOREST": {"n_estimators": 10}, "EXTRA_TREES": {"n_estimators": 10},
        "ELASTIC_NET": {"alpha": 0.1},
    }.get(model, {})
    audio = PreSplitProcessor(schema).prepare_audio(raw)
    full = SignatureBuilder(audio, schema).full("mean")
    config = {"model": model, "model_params": params, "feature_set": "indices", "aggregation": "mean", "scaling": "standard", "pca_indices": None, "pca_embeddings": None}
    pipeline = FittedPipeline.fit(schema, config, full, 42)
    predicted = pipeline.predict_raw(raw.drop(columns="meanHFI"))
    assert len(predicted) == raw.CapturePointId.nunique()
    assert np.isfinite(predicted.prediction).all()


def test_deployment_preserves_configured_recording_window(raw, schema):
    shifted = raw.copy()
    shifted["AudioDate"] = shifted.AudioDate + pd.Timedelta(hours=3)
    audio = PreSplitProcessor(schema).prepare_audio(shifted, "07:00", "09:00")
    full = SignatureBuilder(audio, schema).full("mean")
    config = {"model": "RIDGE_REGRESSION", "model_params": {}, "feature_set": "indices", "aggregation": "mean", "scaling": "standard", "pca_indices": None, "pca_embeddings": None}
    pipeline = FittedPipeline.fit(schema, config, full, 42, recording_start="07:00", recording_end="09:00")
    assert np.allclose(pipeline.predict_raw(shifted.drop(columns="meanHFI")).prediction,
                       pipeline.predict(full))
