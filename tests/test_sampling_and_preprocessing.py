from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.core.data.sampling import sample_submissions
from src.core.evaluation.metrics import regression_metrics
from src.core.processors.presplit import PreSplitProcessor, SignatureBuilder
from src.core.processors.postsplit import FittedPipeline, PostSplitProcessor


def test_audio_means_preserve_operational_unit_and_labels(raw, schema):
    audio = PreSplitProcessor(schema).prepare_audio(raw)
    assert len(audio) == raw.Audio_Name.nunique()
    assert set(audio.n_segments) == {1, 2, 3}
    original = raw[raw.Audio_Name == audio.Audio_Name.iloc[0]]
    assert audio.s_a.iloc[0] == pytest.approx(original.s_a.mean())
    assert np.allclose(audio.embedding.iloc[0], np.stack(original.embedding).mean(axis=0))
    # Different installation labels within one Point are allowed.
    assert audio.groupby("Point").meanHFI.nunique().max() == 2
    bad = raw.copy()
    bad.loc[0, "meanHFI"] += 1
    with pytest.raises(ValueError, match="one HFI"):
        PreSplitProcessor(schema).prepare_audio(bad)


def test_nested_sampling_is_order_invariant_and_without_replacement(raw, schema):
    audio = PreSplitProcessor(schema).prepare_audio(raw)
    first = sample_submissions(audio, schema, (1, 3, 7), 3, 99, "test")
    second = sample_submissions(audio.sample(frac=1, random_state=4), schema, (1, 3, 7), 3, 99, "test")
    pd.testing.assert_frame_equal(first, second)
    for _, group in first.groupby([schema.bag, "sampling_repeat"]):
        sets = [set(ids) for ids in group.sample_ids]
        assert sets[0] < sets[1] < sets[2]
    assert all(len(ids) == len(set(ids)) == n for ids, n in zip(first.sample_ids, first.n_recordings))
    one = audio[audio.CapturePointId == 0].iloc[:2]
    with pytest.raises(ValueError, match="No installations"):
        sample_submissions(one, schema, (1, 3), 2, 99, "short")


def test_subset_aggregation_never_uses_unselected_recordings(raw, schema):
    audio = PreSplitProcessor(schema).prepare_audio(raw)
    manifest = sample_submissions(audio, schema, (1, 3), 1, 1, "subset")
    first = manifest.iloc[[0]]
    selected_id = first.sample_ids.iloc[0][0]
    changed = audio.copy()
    changed.loc[changed.sample_id != selected_id, "s_a"] = 1e8
    original = SignatureBuilder(audio, schema).submissions(first, "mean_std")
    altered = SignatureBuilder(changed, schema).submissions(first, "mean_std")
    assert original.s_a__mean.iloc[0] == altered.s_a__mean.iloc[0]
    assert original.s_a__std.iloc[0] == 0
    assert original.availability_std.iloc[0] == 0
    assert SignatureBuilder(audio, schema).full("mean").shape[0] == raw.CapturePointId.nunique()


def test_preprocessor_fits_only_training_statistics(raw, schema):
    full = SignatureBuilder(PreSplitProcessor(schema).prepare_audio(raw), schema).full("mean")
    train = full.iloc[:8]
    config = {"feature_set": "indices", "scaling": "standard", "pca_indices": None}
    processor = PostSplitProcessor(schema, config)
    x, y = processor.fit_transform(train)
    expected = train[schema.index_columns(train.columns)].mean().to_numpy()
    assert np.allclose(processor.blocks["indices"][1].mean_, expected)
    changed = full.iloc[8:].copy()
    changed["s_a__mean"] = 1e9
    processor.transform(changed)
    assert np.allclose(processor.blocks["indices"][1].mean_, expected)
    assert np.allclose(processor.inverse_target(y), train.meanHFI)
    with pytest.raises(ValueError, match="Infeasible PCA"):
        PostSplitProcessor(schema, {**config, "pca_indices": 100}).fit_transform(train)


def test_metrics_balance_submissions_installations_and_points(schema):
    df = pd.DataFrame({"Point": [1, 1, 1] + [2]*10, "CapturePointId": [10, 10, 11] + [20]*10,
                       "meanHFI": [0]*13, "prediction": [0, 0, 2] + [4]*10})
    assert regression_metrics(df, schema)["MAE"] == pytest.approx(2.5)
    assert regression_metrics(pd.concat([df, df]), schema)["MAE"] == pytest.approx(2.5)


def test_settings_validate_without_creating_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Settings()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        Settings(selection_efforts=(5, 1))
    with pytest.raises(ValueError, match="SOURCE_RUN"):
        Settings(run_mode="effort_analysis")
    env = tmp_path / "bad.env"
    env.write_text("FEATURE_SET=indices\n")
    with pytest.raises(ValueError, match="legacy"):
        Settings.from_env(str(env))
