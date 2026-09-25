"""Exportable scientific figures for the experiment report."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analysis import NAMES

TEAL = '#087f8c'
ORANGE = '#d67828'


def save(fig, directory, name):
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{extension}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    return name


def model_search(candidates, summary, directory):
    names = summary.model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(names) * .42)), gridspec_kw={'width_ratios': [3, 1]})
    for i, model in enumerate(names):
        values = candidates[candidates.model == model].inner_MAE.to_numpy()
        axes[0].scatter(values, i + np.linspace(-.14, .14, len(values)), s=22, color='#a5cbd0', alpha=.8)
        axes[0].scatter(values.mean(), i, marker='D', s=45, color=TEAL, zorder=3)
    axes[0].set(yticks=range(len(names)), yticklabels=[NAMES.get(m, m) for m in names], xlabel='Best inner-search MAE (HFI); lower is better', title='Tuning performance — not outer-test performance')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=.2)
    axes[1].barh(range(len(names)), summary.selected_partitions, color=TEAL)
    for i, count in enumerate(summary.selected_partitions):
        axes[1].text(count + .06, i, str(count), va='center')
    axes[1].set(yticks=range(len(names)), yticklabels=[], xlabel='Partitions selected', title='Selection frequency', xlim=(0, max(1, summary.selected_partitions.max()) + 1))
    axes[1].set_ylim(axes[0].get_ylim())
    return save(fig, directory, 'model_search')


def search_heatmap(candidates, summary, directory):
    matrix = candidates.pivot(index='model', columns='partition', values='gap_to_inner_winner').reindex(summary.model)
    fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * .55), max(4, len(matrix) * .42)))
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0)
    ax.set(yticks=range(len(matrix)), yticklabels=[NAMES.get(m, m) for m in matrix.index],
           xticks=range(len(matrix.columns)), xticklabels=[str(c).replace('repeat_', 'R').replace('_fold_', '/F') for c in matrix.columns],
           title='Inner MAE above the best candidate in each partition (0 = winner)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.colorbar(im, ax=ax, label='MAE gap (HFI)')
    return save(fig, directory, 'model_search_by_fold')


def configuration_choices(selections, settings, directory):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = {'indices': 'Indices', 'embeddings': 'Embeddings', 'both': 'Both', 'mean': 'Mean', 'mean_std': 'Mean + SD', 'hierarchical': 'Hierarchical', 'none': 'No PCA', 'pca': 'PCA count', 'pca_variance': 'PCA variance'}
    for ax, key, options, title in zip(axes, ['feature_set', 'aggregation', 'reduction'], ['feature_sets', 'aggregation_strategies', 'reduction_methods'], ['Representation', 'Aggregation', 'PCA preprocessing']):
        choices = list(dict.fromkeys([*settings.get(options, []), *selections[key].tolist()]))
        counts = selections[key].value_counts().reindex(choices, fill_value=0)
        ax.bar(range(len(choices)), counts, color=TEAL)
        for i, count in enumerate(counts):
            ax.text(i, count + .12, str(count), ha='center')
        ax.set(xticks=range(len(choices)), xticklabels=[labels.get(c, c) for c in choices], ylabel='Selected partitions', title=title, ylim=(0, len(selections) + 1))
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    return save(fig, directory, 'configuration_choices')


def effort_curve(efforts, gains, directory, confidence):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    selected = efforts[efforts.evaluation == 'selected'].sort_values('n_recordings')
    axes[0].plot(selected.n_recordings, selected.MAE, marker='o', color=TEAL, label='Selected procedure')
    axes[0].fill_between(selected.n_recordings, selected.MAE_ci_low, selected.MAE_ci_high, color=TEAL, alpha=.15, label=f'{confidence:.0%} conditional Point-bootstrap interval')
    for name, group in efforts[efforts.evaluation.isin(['DUMMY_MEAN', 'DUMMY_MEDIAN'])].groupby('evaluation'):
        axes[0].plot(group.n_recordings, group.MAE, linestyle='--', label=NAMES[name], alpha=.7)
    axes[0].set(xlabel='Aggregated recordings', ylabel='Point-balanced MAE (HFI)', title='Prediction error on unseen Points')
    axes[0].legend(fontsize=9)
    if len(gains):
        x = np.arange(len(gains))
        axes[1].bar(x, gains.MAE_improvement, color=TEAL, alpha=.75)
        axes[1].vlines(x, gains.ci_low, gains.ci_high, color='#263238')
        axes[1].scatter(x, gains.MAE_improvement, color='#263238', s=14)
        axes[1].set(xticks=x, xticklabels=[f'{int(a)} → {int(b)}' for a, b in zip(gains.from_recordings, gains.to_recordings)], xlabel='Increase in recording count')
    axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].set(ylabel='MAE reduction (HFI)', title='Benefit of moving to the next evaluated effort')
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
    return save(fig, directory, 'recording_effort')


def diagnostics(table, directory):
    efforts = sorted(table.n_recordings.unique())
    chosen = list(dict.fromkeys([efforts[0], efforts[len(efforts)//2], efforts[-1]]))
    fig, axes = plt.subplots(2, len(chosen), squeeze=False, figsize=(4.4*len(chosen), 7))
    for col, effort in enumerate(chosen):
        d = table[table.n_recordings == effort]
        lo = min(d.observed_hfi.min(), d.prediction_p10.min())
        hi = max(d.observed_hfi.max(), d.prediction_p90.max())
        axes[0, col].vlines(d.observed_hfi, d.prediction_p10, d.prediction_p90, alpha=.22, color=TEAL)
        axes[0, col].scatter(d.observed_hfi, d.mean_prediction, s=19, alpha=.75, color=TEAL)
        axes[0, col].plot([lo, hi], [lo, hi], linestyle='--', color='#555555', linewidth=1)
        axes[0, col].set(title=f'{effort} recording(s)', xlabel='Observed HFI', ylabel='Mean held-out prediction', xlim=(lo-.1, hi+.1), ylim=(lo-.1, hi+.1))
        axes[1, col].vlines(d.observed_hfi, d.prediction_p10-d.observed_hfi, d.prediction_p90-d.observed_hfi, alpha=.22, color=TEAL)
        axes[1, col].scatter(d.observed_hfi, d.bias, s=19, alpha=.75, color=TEAL)
        axes[1, col].axhline(0, color='#555555', linestyle='--')
        axes[1, col].set(xlabel='Observed HFI', ylabel='Mean prediction − observed HFI')
    return save(fig, directory, 'prediction_diagnostics')


def search_progress(trials, directory):
    models = sorted(trials.model.unique())
    columns = min(3, len(models))
    fig, axes = plt.subplots(math.ceil(len(models)/columns), columns, squeeze=False, figsize=(4.2*columns, 2.7*math.ceil(len(models)/columns)))
    for ax, model in zip(axes.flat, models):
        data = trials[trials.model == model].copy()
        data.loc[data.state != 'COMPLETE', 'inner_MAE'] = np.nan
        data = data.sort_values('trial')
        data['best'] = data.groupby('partition').inner_MAE.transform(lambda s: s.cummin().ffill())
        by_trial = data.groupby('trial').best
        med = by_trial.median()
        ax.plot(med.index + 1, med, color=TEAL)
        ax.fill_between(med.index + 1, by_trial.quantile(.25), by_trial.quantile(.75), color=TEAL, alpha=.18)
        ax.set(title=NAMES.get(model, model), xlabel='Trial attempt', ylabel='Best inner MAE')
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    return save(fig, directory, 'search_progress')


def outer_comparison(summary, directory):
    table = summary.sort_values('macro_MAE')
    fig, ax = plt.subplots(figsize=(10, max(4, .43*len(table))))
    for i, row in enumerate(table.itertuples()):
        color = ORANGE if row.evaluation == 'selected' else TEAL
        ax.hlines(i, row.MAE_ci_low, row.MAE_ci_high, color=color, linewidth=2)
        ax.scatter(row.macro_MAE, i, color=color, zorder=3)
    ax.set(yticks=range(len(table)), yticklabels=[NAMES.get(m, m) for m in table.evaluation], xlabel='Equal-effort Point-balanced MAE (HFI)', title='Matched outer-test submissions — conditional Point-bootstrap intervals')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    return save(fig, directory, 'outer_model_comparison')
