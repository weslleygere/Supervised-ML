"""Optional matched outer-test refits; no Optuna search or deployment selection."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from src.config.settings import Settings
from src.core.data.data_loader import DataLoader
from src.core.data.sampling import stable_seed
from src.core.evaluation.persistence import canonical_json, environment, file_hash, identity, write_frame, write_json
from src.core.models.factory import RegressionModels
from src.core.processors.presplit import PreSplitProcessor, SignatureBuilder
from src.core.processors.postsplit import FittedPipeline
from .analysis import Evidence, performance_tables

COMPARISON_VERSION = '1'


def compare_families(run_dir, output_dir=None):
    source = Evidence(run_dir)
    run = source.read('run_manifest.json')
    if run.get('status') != 'complete' or run.get('run_mode') != 'nested_selection':
        raise ValueError('Family comparison requires a completed nested_selection run.')
    settings = Settings.from_dict(source.read('resolved_config.json'))
    if identity(settings)['fingerprint'] != run['identity']['fingerprint']:
        raise ValueError('Source data, schema, training configuration or implementation changed. Refusing incompatible refits.')
    if environment()['packages'] != run['environment']['packages']:
        raise ValueError('Package versions differ from the source run.')
    predictions = source.read('oof_predictions.parquet')
    splits = source.read('outer_splits.parquet')
    selections = source.read('outer_selections.json')
    schema_dict = source.read('schema.json')
    families = [m.name for m in RegressionModels] if settings.models == ('all',) else list(settings.models)
    assignments = []
    for selection in selections:
        repeat, fold = selection['repeat'], selection['outer_fold']
        namespace = f'repeat_{repeat:02d}_fold_{fold:02d}'
        candidates = source.read(f'{namespace}/search/candidate_results.json')
        manifest = source.read(f'{namespace}/submissions.parquet')
        if len(candidates) != len(families) or {c['model'] for c in candidates} != set(families):
            raise ValueError(f'{namespace} does not contain one selected configuration for every family.')
        winning = next(c for c in candidates if c['model'] == selection['model'])
        if canonical_json(winning['configuration']) != canonical_json(selection['configuration']):
            raise ValueError(f'{namespace}: selected pipeline differs from saved candidate configuration.')
        assignments.append((repeat, fold, namespace, candidates, manifest, selection))
    expected_folds = settings.outer_repeats * settings.outer_splits
    if len(assignments) != expected_folds or len({(a[0], a[1]) for a in assignments}) != expected_folds:
        raise ValueError('Incomplete or duplicate outer-fold assignments.')
    output = Path(output_dir).resolve() if output_dir else source.root / 'reports/family_comparison'
    output.mkdir(parents=True, exist_ok=True)
    signature = {'source_fingerprint': run['identity']['fingerprint'], 'source_artifacts': source.hashes,
                 'comparison_version': COMPARISON_VERSION, 'comparison_code_hash': file_hash(__file__),
                 'environment': environment()['packages']}
    status_path = output / 'comparison_manifest.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if any(previous.get(k) != v for k, v in signature.items()):
            raise ValueError('Existing comparison has different provenance; choose a new report directory.')
        if previous['status'] == 'complete':
            return output
    status = {**signature, 'status': 'running', 'source_run': str(source.root),
              'additional_fits_expected': (len(families)-1) * expected_folds,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    try:
        loader = DataLoader(settings.data_path, settings.schema_path)
        schema = loader.load_schema()
        if canonical_json(schema_dict) != canonical_json(json.loads(Path(settings.schema_path).read_text())):
            raise ValueError('Stored and current schemas differ.')
        audio = PreSplitProcessor(schema).prepare_audio(loader.load_data(), settings.recording_start, settings.recording_end)
        builder = SignatureBuilder(audio, schema)
        metadata = audio.groupby(schema.bag)[schema.group].first()
        fold_frames, fit_rows = [], []
        with threadpool_limits(limits=settings.model_jobs):
            for repeat, fold, namespace, candidates, manifest, selection in assignments:
                directory = output / namespace
                directory.mkdir(exist_ok=True)
                if (directory / 'complete.json').exists():
                    fold_frames.append(pd.read_parquet(directory / 'predictions.parquet'))
                    fit_rows.extend(json.loads((directory / 'fits.json').read_text()))
                    continue
                assignment = splits[(splits['repeat'] == repeat) & (splits.outer_fold == fold)]
                train, test = assignment[assignment.role == 'train'], assignment[assignment.role == 'test']
                if set(train[schema.group]) & set(test[schema.group]):
                    raise ValueError('Point leakage in saved outer split.')
                if set(train[schema.bag]) & set(test[schema.bag]) or set(train[schema.bag]) | set(test[schema.bag]) != set(metadata.index):
                    raise ValueError('Saved installation partitions do not match the source data.')
                for frame in (train, test):
                    if not frame[schema.bag].map(metadata).eq(frame[schema.group]).all():
                        raise ValueError('Installation/Point assignments changed.')
                if not set(manifest[schema.bag]) <= set(test[schema.bag]):
                    raise ValueError('Submissions contain a training installation.')
                saved = predictions[(predictions['repeat'] == repeat) & (predictions.outer_fold == fold) & (predictions.evaluation == 'selected')].copy()
                common = [schema.group, schema.bag, schema.target, 'submission_id', 'n_recordings', 'sampling_repeat']
                if not saved[common].sort_values('submission_id').reset_index(drop=True).equals(manifest[common].sort_values('submission_id').reset_index(drop=True)):
                    raise ValueError('Saved predictions and submission manifests do not match.')
                frames, records, validation_cache = [], [], {}
                for candidate in candidates:
                    model, config = candidate['model'], candidate['configuration']
                    started = time.perf_counter()
                    reused = model == selection['model']
                    if reused:
                        frame = saved.copy()
                    else:
                        full = builder.full(config['aggregation'])
                        full = full[full[schema.bag].isin(train[schema.bag])].reset_index(drop=True)
                        pipeline = FittedPipeline.fit(schema, config, full, stable_seed(settings.model_seed, namespace, 'refit'),
                                                      settings.model_jobs, settings.recording_start, settings.recording_end)
                        if config['aggregation'] not in validation_cache:
                            validation_cache[config['aggregation']] = builder.submissions(manifest, config['aggregation'])
                        frame = manifest.drop(columns='sample_ids').copy()
                        frame['prediction'] = pipeline.predict(validation_cache[config['aggregation']])
                        frame['repeat'], frame['outer_fold'] = repeat, fold
                    frame['evaluation'], frame['selected_model'] = model, model
                    frames.append(frame)
                    records.append({'repeat': repeat, 'outer_fold': fold, 'model': model,
                                    'reused_selected_predictions': reused, 'seconds': time.perf_counter()-started,
                                    'training_installations': len(train), 'training_recordings': int(train.available_recordings.sum()),
                                    'configuration': config})
                    print(f'{namespace}: {model} — {"reused" if reused else "refitted"}', flush=True)
                combined = pd.concat(frames, ignore_index=True)
                write_frame(directory / 'predictions.parquet', combined)
                write_json(directory / 'fits.json', records)
                write_json(directory / 'complete.json', {'complete': True})
                fold_frames.append(combined)
                fit_rows.extend(records)
        combined = pd.concat([predictions, *fold_frames], ignore_index=True)
        if set(combined.evaluation) != set(families) | set(predictions.evaluation):
            raise ValueError('Incomplete family predictions; refusing a partial comparison.')
        # Validate matching cohorts and sample identities before labeling this complete.
        performance_tables(combined, schema, bootstrap_repeats=2)
        write_frame(output / 'predictions.parquet', combined)
        write_json(output / 'fits.json', fit_rows)
        status['status'] = 'complete'
        status['additional_fits_completed'] = sum(not row['reused_selected_predictions'] for row in fit_rows)
    except BaseException as exc:
        status['status'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(status_path, status)
    return output
