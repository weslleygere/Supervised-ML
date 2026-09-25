from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reporting.analysis import performance_tables
from reporting.comparison import compare_families
from reporting.report import build_report
from src.config.settings import Settings
from src.core.evaluation.evaluator import ModelEvaluator
from src.core.evaluation.persistence import file_hash
from src.core.processors.postsplit import FittedPipeline
from src.pipeline import Pipeline


@pytest.fixture
def completed_run(tmp_path, raw, schema):
    data = tmp_path / 'data.parquet'
    raw.to_parquet(data)
    schema_path = tmp_path / 'schema.json'
    schema_path.write_text(json.dumps(asdict(schema)))
    settings = Settings(data_path=str(data), schema_path=str(schema_path), output_dir=str(tmp_path / 'runs'),
                        models=('RIDGE_REGRESSION', 'SVR'), feature_sets=('indices',), aggregation_strategies=('mean',),
                        reduction_methods=('none',), selection_efforts=(1, 3), selection_sampling_repeats=2,
                        outer_splits=2, outer_repeats=1, inner_splits=2, optuna_trials=1, bootstrap_repeats=20)
    return settings, Pipeline(settings).run()


def test_report_is_artifact_only_and_preserves_original_files(completed_run, monkeypatch):
    settings, source = completed_run
    original = {p: file_hash(p) for p in source.rglob('*') if p.is_file()}
    Path(settings.data_path).rename(Path(settings.data_path).with_suffix('.unavailable'))
    monkeypatch.setattr(FittedPipeline, 'fit', lambda *a, **kw: pytest.fail('Report fitted a model'))
    monkeypatch.setattr(ModelEvaluator, '_search', lambda *a, **kw: pytest.fail('Report searched'))
    report = build_report(source)
    assert all(file_hash(p) == digest for p, digest in original.items())
    assert report.exists()
    assert 'No final deployment model is selected in this run' in report.read_text()
    assert 'not an outer-test' in (report.parent / 'summary.md').read_text() or 'not be read as held-out' in report.read_text()
    summary = json.loads((report.parent / 'experiment_summary.json').read_text())
    scores = pd.read_csv(source / 'selection_procedure_scores.csv')
    assert summary['macro_MAE'] == pytest.approx(scores[scores.evaluation == 'selected'].macro_point_balanced_mae.mean())
    assert not summary['outer_family_comparison_available']
    assert not (report.parent / 'outer_model_comparison.csv').exists()
    assert len(list((report.parent / 'figures').glob('*.pdf'))) >= 5
    assert 'inner_search' in set(pd.read_csv(report.parent / 'model_search_summary.csv').evaluation_level)


def test_optional_comparison_reuses_search_and_exact_test_submissions(completed_run, monkeypatch):
    settings, source = completed_run
    fits = []
    original_fit = FittedPipeline.fit

    def track_fit(cls, schema, config, train, *args, **kwargs):
        fits.append(set(train[schema.bag]))
        return original_fit(schema, config, train, *args, **kwargs)

    monkeypatch.setattr(FittedPipeline, 'fit', classmethod(track_fit))
    monkeypatch.setattr(ModelEvaluator, '_search', lambda *a, **kw: pytest.fail('Comparison repeated Optuna'))
    directory = compare_families(source)
    assert len(fits) == 2  # one additional family per fold
    splits = pd.read_parquet(source / 'outer_splits.parquet')
    for train_ids in fits:
        matching = [fold for _, fold in splits.groupby('outer_fold') if set(fold[fold.role == 'train'].CapturePointId) == train_ids]
        assert len(matching) == 1
        assert not train_ids & set(matching[0][matching[0].role == 'test'].CapturePointId)
    predictions = pd.read_parquet(directory / 'predictions.parquet')
    assert set(predictions.evaluation) == {'selected', 'DUMMY_MEAN', 'DUMMY_MEDIAN', *settings.models}
    keys = ['repeat', 'submission_id', 'CapturePointId', 'n_recordings']
    selected = predictions[predictions.evaluation == 'selected'][keys].sort_values(keys).reset_index(drop=True)
    for model in settings.models:
        pd.testing.assert_frame_equal(selected, predictions[predictions.evaluation == model][keys].sort_values(keys).reset_index(drop=True))
    assert compare_families(source) == directory
    assert len(fits) == 2
    report = build_report(source)
    assert json.loads((report.parent / 'experiment_summary.json').read_text())['outer_family_comparison_available']
    assert (report.parent / 'outer_model_comparison.csv').exists()
    assert (report.parent / 'figures/outer_model_comparison.pdf').exists()
    assert 'Model families on the same unseen Points' in report.read_text()


def test_comparison_rejects_changed_source_data(completed_run, raw):
    settings, source = completed_run
    changed = raw.copy()
    changed['s_a'] += 1
    changed.to_parquet(settings.data_path)
    with pytest.raises(ValueError, match='changed'):
        compare_families(source)


def test_final_fit_report_identifies_actual_deployment_choice(completed_run):
    settings, source = completed_run
    final = Pipeline(replace(settings, run_mode='final_fit', source_run=str(source))).run()
    report = build_report(final)
    summary = json.loads((report.parent / 'experiment_summary.json').read_text())
    actual = json.loads((final / 'deployment_manifest.json').read_text())
    assert summary['deployment_model'] == actual['configuration']['model']
    assert 'No final deployment model' not in report.read_text()
    assert 'Held-out MAE' not in report.read_text()
    assert not (report.parent / 'performance_summary.csv').exists()


def test_point_bootstrap_does_not_treat_repetitions_as_new_points(schema):
    rows = []
    for repeat in range(2):
        for point, error in enumerate([1., 2., 4.]):
            for count in (1, 3):
                for evaluation, prediction in [('selected', error/count), ('DUMMY_MEAN', error/count+1)]:
                    rows.append({'Point': point, 'CapturePointId': point, 'meanHFI': 0., 'prediction': prediction,
                                 'repeat': repeat, 'n_recordings': count, 'submission_id': f'{point}-{count}', 'evaluation': evaluation})
    predictions = pd.DataFrame(rows)
    first = performance_tables(predictions[predictions['repeat'] == 0], schema, 100, .95, 1)
    repeated = performance_tables(predictions, schema, 100, .95, 1)
    for column in ['macro_MAE', 'MAE_ci_low', 'MAE_ci_high']:
        assert np.allclose(first[1][column], repeated[1][column])
    gains = repeated[-1].iloc[0]
    assert gains.MAE_reduction == pytest.approx(1.)
    assert gains.ci_low == pytest.approx(1.)
    assert gains.ci_high == pytest.approx(1.)
    broken = predictions.iloc[1:]
    with pytest.raises(ValueError, match='same held-out'):
        performance_tables(broken, schema, 10)


def test_effort_report_uses_available_efforts_without_search_tables(completed_run):
    settings, source = completed_run
    effort = Pipeline(replace(settings, run_mode='effort_analysis', source_run=str(source),
                              effort_analysis_counts=(1, 2, 3), effort_analysis_repeats=2)).run()
    report = build_report(effort)
    table = pd.read_csv(report.parent / 'performance_by_effort.csv')
    assert table.n_recordings.tolist() == [1, 2, 3]
    assert set(table.evaluation) == {'selected'}
    assert not (report.parent / 'model_search_summary.csv').exists()
    assert not (report.parent / 'paired_comparisons_to_selected.csv').exists()


def test_run_experiment_cli_connects_original_pipeline_and_report(completed_run, monkeypatch):
    import results
    settings, source = completed_run
    calls = []
    monkeypatch.setattr(Settings, 'from_env', classmethod(lambda cls, path: settings))
    monkeypatch.setattr(Pipeline, 'run', lambda self: calls.append(self.settings) or source)
    monkeypatch.setattr('sys.argv', ['results.py', '--run-experiment'])
    results.main()
    assert calls == [settings]
    assert (source / 'reports/report.html').exists()
