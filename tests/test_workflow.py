from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.core.evaluation.evaluator import ModelEvaluator
from src.core.processors.postsplit import FittedPipeline
from src.pipeline import Pipeline


def make_settings(tmp_path, raw, schema):
    data = tmp_path / "data.parquet"
    raw.to_parquet(data)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"target": schema.target, "group": schema.group, "bag": schema.bag, "audio": schema.audio, "datetime": schema.datetime, "index_prefixes": schema.index_prefixes, "embedding": schema.embedding}))
    return Settings(data_path=str(data), schema_path=str(schema_path), output_dir=str(tmp_path / "runs"),
                    models=("RIDGE_REGRESSION", "SVR"), feature_sets=("indices", "embeddings"),
                    aggregation_strategies=("mean", "mean_std"), reduction_methods=("none", "pca"),
                    pca_component_candidates=(1, 2), selection_efforts=(1, 3), selection_sampling_repeats=2,
                    outer_splits=2, outer_repeats=1, inner_splits=2, optuna_trials=3,
                    effort_analysis_counts=(1, 2, 3), effort_analysis_repeats=3, bootstrap_repeats=20)


def test_complete_workflow_and_resume_without_refitting(tmp_path, raw, schema, monkeypatch):
    settings = make_settings(tmp_path, raw, schema)
    report = Pipeline(settings).check()
    assert report["eligible_installations"] == raw.CapturePointId.nunique()
    assert not Path(settings.output_dir).exists()
    source = Pipeline(settings).run()
    assert json.loads((source / "run_manifest.json").read_text())["status"] == "complete"
    splits = pd.read_parquet(source / "outer_splits.parquet")
    for _, frame in splits.groupby(["repeat", "outer_fold"]):
        train, test = frame[frame.role == "train"], frame[frame.role == "test"]
        assert not set(train.Point) & set(test.Point)
        assert set(train.CapturePointId) | set(test.CapturePointId) == set(raw.CapturePointId)
    for path in source.glob("repeat_*/search/inner_splits.parquet"):
        inner = pd.read_parquet(path)
        bundle = joblib.load(path.parent.parent / "pipeline.joblib")
        assert set(inner.Point) <= set(bundle["train_points"])
        assert not set(inner.Point) & set(bundle["test_points"])
        for _, fold in inner.groupby("inner_fold"):
            assert not set(fold[fold.role == "train"].Point) & set(fold[fold.role == "validation"].Point)
    predictions = pd.read_parquet(source / "oof_predictions.parquet")
    assert set(predictions.n_recordings) == {1, 3}
    for path in source.glob("repeat_*/search/*_trials.json"):
        completed = [t for t in json.loads(path.read_text()) if t["state"] == "COMPLETE"]
        assert completed
        assert {"feature_set", "aggregation", "scaling"} <= completed[0]["parameters"].keys()
    for path in source.glob("repeat_*/pipeline.joblib"):
        bundle = joblib.load(path)
        expected = raw[raw.CapturePointId.isin(bundle["train_installations"])].Audio_Name.nunique()
        assert bundle["training_recordings"] == expected
        assert bundle["pipeline"].processor.target_scaler.n_features_in_ == 1
    assert Pipeline(replace(settings, resume_run=str(source))).run() == source
    # Detailed analysis must not call fit, select, or re-tune a model.
    with monkeypatch.context() as patch:
        patch.setattr(FittedPipeline, "fit", lambda *a, **kw: pytest.fail("effort analysis attempted fit"))
        patch.setattr(ModelEvaluator, "_search", lambda *a, **kw: pytest.fail("effort analysis attempted search"))
        effort = Pipeline(replace(settings, run_mode="effort_analysis", source_run=str(source))).run()
    summary = pd.read_csv(effort / "effort_summary.csv")
    assert summary.n_recordings.tolist() == [1, 2, 3]
    assert (effort / "effort_curve.pdf").is_file()
    assert summary.n_points.nunique() == 1
    final = Pipeline(replace(settings, run_mode="final_fit", source_run=str(source))).run()
    manifest = json.loads((final / "deployment_manifest.json").read_text())
    assert manifest["training_installations"] == raw.CapturePointId.nunique()
    assert manifest["training_audio_samples"] == raw.Audio_Name.nunique()
    pipeline = joblib.load(final / "deployment.joblib")
    assert np.isfinite(pipeline.predict_raw(raw.drop(columns="meanHFI")).prediction).all()
    # A changed dataset cannot be evaluated with old fitted artifacts.
    changed = raw.copy()
    changed["s_a"] += 0.1
    changed.to_parquet(settings.data_path)
    with pytest.raises(ValueError, match="changed"):
        Pipeline(replace(settings, run_mode="effort_analysis", source_run=str(source)))


def test_completed_outer_folds_resume_from_incomplete_run(tmp_path, raw, schema, monkeypatch):
    settings = replace(make_settings(tmp_path, raw, schema), models=("RIDGE_REGRESSION",), optuna_trials=1, feature_sets=("indices",), reduction_methods=("none",))
    source = Pipeline(settings).run()
    status = json.loads((source / "run_manifest.json").read_text())
    status["status"] = "failed"
    (source / "run_manifest.json").write_text(json.dumps(status))
    monkeypatch.setattr(ModelEvaluator, "_search", lambda *a, **kw: pytest.fail("completed fold searched again"))
    assert Pipeline(replace(settings, resume_run=str(source))).run() == source


def test_optuna_configuration_and_hyperparameters_for_every_family(tmp_path, raw, schema):
    from threadpoolctl import threadpool_limits
    from src.core.models.factory import RegressionModels
    from src.core.processors.presplit import PreSplitProcessor

    settings = replace(make_settings(tmp_path, raw, schema), models=("all",), optuna_trials=2,
                       feature_sets=("both",), aggregation_strategies=("hierarchical",),
                       reduction_methods=("pca_variance",))
    audio = PreSplitProcessor(schema).prepare_audio(raw)
    evaluator = ModelEvaluator(settings, schema, audio, tmp_path / "search")
    with threadpool_limits(limits=1):
        selected = evaluator._search(evaluator.metadata, tmp_path / "search", "all_families", 17)
    results = json.loads((tmp_path / "search" / "candidate_results.json").read_text())
    assert {r["model"] for r in results} == {m.name for m in RegressionModels}
    assert all(r["completed_trials"] > 0 for r in results)
    assert selected["inner_score"] == min(r["inner_score"] for r in results)
    for result in results:
        config = result["configuration"]
        assert config["feature_set"] == "both"
        assert config["aggregation"] == "hierarchical"
        assert config["model_params"]
        assert config["reduction"] == ("none" if result["model"] == "PLS" else "pca_variance")
