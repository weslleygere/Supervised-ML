"""Summaries of saved evidence; this module never fits a model."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.core.evaluation.metrics import metric_table, point_loss_matrix, regression_metrics
from src.core.evaluation.persistence import file_hash

NAMES = {
    'selected': 'Selected procedure', 'DUMMY_MEAN': 'Mean baseline', 'DUMMY_MEDIAN': 'Median baseline',
    'RIDGE_REGRESSION': 'Ridge', 'ELASTIC_NET': 'Elastic Net', 'HUBER': 'Huber', 'PLS': 'PLS',
    'SVR': 'SVR', 'KERNEL_RIDGE': 'Kernel Ridge', 'GAUSSIAN_PROCESS': 'Gaussian Process',
    'RANDOM_FOREST': 'Random Forest', 'EXTRA_TREES': 'Extra Trees', 'XGBOOST': 'XGBoost', 'CATBOOST': 'CatBoost',
}


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def read(self, relative):
        path = self.root / relative
        self.hashes[str(relative)] = file_hash(path)
        if path.suffix == '.json':
            return json.loads(path.read_text())
        return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def configuration_row(config):
    return {**{k: config.get(k) for k in ('model', 'feature_set', 'aggregation', 'scaling', 'reduction', 'pca_indices', 'pca_embeddings')},
            'hyperparameters': json.dumps(config.get('model_params', {}), sort_keys=True)}


def search_tables(evidence, selections):
    rows, trials = [], []
    paths = sorted(evidence.root.glob('repeat_*/search/candidate_results.json'))
    if not paths and (evidence.root / 'search/candidate_results.json').exists():
        paths = [evidence.root / 'search/candidate_results.json']
    for path in paths:
        fold = path.parent.parent.name if path.parent.parent != evidence.root else 'all_data'
        candidates = evidence.read(path.relative_to(evidence.root))
        ordered = sorted(candidates, key=lambda c: (c['inner_score'], c['model']))
        for rank, candidate in enumerate(ordered, 1):
            rows.append({'partition': fold, **configuration_row(candidate['configuration']),
                         'inner_MAE': candidate['inner_score'], 'inner_rank': rank,
                         'gap_to_inner_winner': candidate['inner_score'] - ordered[0]['inner_score'],
                         'completed_trials': candidate['completed_trials'], 'total_trials': candidate['total_trials'],
                         'best_trial': candidate['trial_number'], 'evaluation_level': 'inner_search'})
        for trial_path in sorted(path.parent.glob('*_trials.json')):
            model = trial_path.name.removesuffix('_trials.json')
            for trial in evidence.read(trial_path.relative_to(evidence.root)):
                trials.append({'partition': fold, 'model': model, 'trial': trial['number'], 'state': trial['state'],
                               'inner_MAE': trial['value'], 'seconds': trial.get('duration_seconds') or 0,
                               'failure': trial.get('attributes', {}).get('failure', '')})
    candidates, trials = pd.DataFrame(rows), pd.DataFrame(trials)
    if candidates.empty:
        return candidates, pd.DataFrame(), trials, pd.DataFrame()
    summary = candidates.groupby('model').agg(
        mean_best_inner_MAE=('inner_MAE', 'mean'), sd_best_inner_MAE=('inner_MAE', 'std'),
        mean_inner_rank=('inner_rank', 'mean'), mean_gap_to_winner=('gap_to_inner_winner', 'mean'),
        completed_trials=('completed_trials', 'sum'), total_trials=('total_trials', 'sum'),
        partitions=('partition', 'nunique')).reset_index()
    wins = pd.Series([s['model'] for s in selections], dtype='object').value_counts()
    summary['selected_partitions'] = summary.model.map(wins).fillna(0).astype(int)
    summary['evaluation_level'] = 'inner_search'
    health = trials.groupby('model').agg(trial_seconds=('seconds', 'sum')).join(pd.crosstab(trials.model, trials.state)).reset_index()
    for state in ('COMPLETE', 'FAIL', 'PRUNED'):
        if state not in health:
            health[state] = 0
    return candidates, summary.sort_values(['mean_best_inner_MAE', 'model']), trials, health


def performance_tables(predictions, schema, bootstrap_repeats=1000, confidence=.95, seed=1042):
    """Pair entire Points across methods and efforts, averaging CV repetitions first."""
    if bootstrap_repeats < 1 or not 0 < confidence < 1:
        raise ValueError('Bootstrap repetitions must be positive and confidence must be between 0 and 1.')
    keys = ['repeat', 'submission_id']
    if predictions.duplicated(['evaluation', *keys]).any():
        raise ValueError('Duplicate predictions for the same evaluation/submission.')
    reference = predictions[predictions.evaluation == 'selected']
    if reference.empty:
        raise ValueError('Predictions must contain the selected procedure.')
    identity_columns = [*keys, 'n_recordings', schema.group, schema.bag, schema.target]
    expected = reference[identity_columns].sort_values(keys).reset_index(drop=True)
    matrices = {}
    for method, group in predictions.groupby('evaluation'):
        actual = group[identity_columns].sort_values(keys).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f'{method} does not have the same held-out submissions, labels and cohort as selected.')
        matrices[method] = point_loss_matrix(group, schema)
    base = matrices['selected']
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(base), size=(bootstrap_repeats, len(base)))
    tail = (1 - confidence) / 2
    metrics = metric_table(predictions, schema)
    efforts, overall, boots, point_losses = [], [], {}, []
    for method, matrix in matrices.items():
        matrix = matrix.reindex(index=base.index, columns=base.columns)
        if matrix.isna().any().any():
            raise ValueError('Methods must share the same Point/effort cohort.')
        values = matrix.to_numpy()
        boot = values[draws].mean(axis=1)
        boots[method] = boot
        mean = values.mean(axis=0)
        low, high = np.quantile(boot, [tail, 1-tail], axis=0)
        for j, count in enumerate(matrix.columns):
            repeated = metrics[(metrics.evaluation == method) & (metrics.n_recordings == count)]
            row = {'evaluation': method, 'evaluation_level': 'outer_test', 'n_recordings': int(count),
                   'MAE': mean[j], 'MAE_ci_low': low[j], 'MAE_ci_high': high[j],
                   'n_points': len(matrix), 'n_installations': int(repeated.n_installations.iloc[0])}
            row.update({col: repeated[col].mean() for col in ('RMSE', 'R2', 'bias', 'median_absolute_error', 'p90_absolute_error')})
            efforts.append(row)
        ci = np.quantile(boot.mean(axis=1), [tail, 1-tail])
        overall.append({'evaluation': method, 'evaluation_level': 'outer_test', 'macro_MAE': mean.mean(),
                        'MAE_ci_low': ci[0], 'MAE_ci_high': ci[1], 'n_points': len(matrix),
                        'n_efforts': len(matrix.columns), 'n_cv_repeats': int(predictions['repeat'].nunique())})
        point_losses.append(matrix.rename_axis(schema.group).reset_index().melt(id_vars=schema.group, var_name='n_recordings', value_name='MAE').assign(evaluation=method))
    selected_mean = base.to_numpy().mean(axis=0)
    counts = base.columns.to_numpy(int)
    delta = boots['selected'][:, :-1] - boots['selected'][:, 1:]
    ci = np.quantile(delta, [tail, 1-tail], axis=0) if len(counts) > 1 else np.empty((2, 0))
    improvements = pd.DataFrame({'from_recordings': counts[:-1], 'to_recordings': counts[1:],
                                 'MAE_improvement': selected_mean[:-1] - selected_mean[1:],
                                 'ci_low': ci[0], 'ci_high': ci[1]})
    overall = pd.DataFrame(overall).sort_values('macro_MAE')
    differences = []
    for method in sorted(boots):
        if method == 'selected':
            continue
        gain = boots[method].mean(axis=1) - boots['selected'].mean(axis=1)
        ci = np.quantile(gain, [tail, 1-tail])
        row = overall[overall.evaluation == method].iloc[0]
        selected = overall[overall.evaluation == 'selected'].iloc[0]
        differences.append({'reference': method, 'comparison': 'selected',
                            'MAE_reduction': row.macro_MAE - selected.macro_MAE,
                            'ci_low': ci[0], 'ci_high': ci[1]})
    return pd.DataFrame(efforts), overall, improvements, pd.concat(point_losses), pd.DataFrame(differences)


def diagnostic_table(predictions, schema):
    selected = predictions[predictions.evaluation == 'selected'].copy()
    selected['absolute_error'] = abs(selected.prediction - selected[schema.target])
    selected['residual'] = selected.prediction - selected[schema.target]
    return selected.groupby(['n_recordings', schema.group, schema.bag], as_index=False).agg(
        observed_hfi=(schema.target, 'first'), mean_prediction=('prediction', 'mean'),
        prediction_p10=('prediction', lambda x: x.quantile(.1)), prediction_p90=('prediction', lambda x: x.quantile(.9)),
        MAE=('absolute_error', 'mean'), bias=('residual', 'mean'), n_predictions=('prediction', 'size'))
