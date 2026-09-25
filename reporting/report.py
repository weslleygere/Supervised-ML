"""Create a compact human-facing report from existing experiment artifacts."""
import base64
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.data.schema import Schema
from src.core.evaluation.persistence import file_hash, write_frame, write_json
from .analysis import Evidence, NAMES, configuration_row, diagnostic_table, performance_tables, search_tables
from . import charts

REPORT_VERSION = '1'
UNCERTAINTY = ('Intervals resample whole Points, preserving paired methods and efforts. They are conditional on '
               'the fitted models and sampled submissions; they exclude retraining uncertainty, are not individual '
               'prediction intervals, and have no simultaneous-coverage guarantee.')


def _table(frame, names=None):
    display = frame.copy()
    for column in ('model', 'evaluation'):
        if column in display:
            display[column] = display[column].map(lambda value: NAMES.get(value, value))
    return display.rename(columns=names or {}).to_html(index=False, border=0, float_format=lambda x: f'{x:.3f}', na_rep='—', escape=True)


def _markdown(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return '—' if not np.isfinite(value) else f'{value:.3f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]])


def build_report(run_dir, output_dir=None, bootstrap_repeats=None, confidence=None, seed=None):
    evidence = Evidence(run_dir)
    manifest = evidence.read('run_manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Report requires a completed experiment; partial results must not look like a final comparison.')
    mode = manifest['run_mode']
    if mode not in {'nested_selection', 'effort_analysis', 'final_fit'}:
        raise ValueError(f'Unsupported run mode: {mode}')
    settings = evidence.read('resolved_config.json')
    schema = Schema.from_dict(evidence.read('schema.json'))
    bootstrap_repeats = settings.get('bootstrap_repeats', 1000) if bootstrap_repeats is None else bootstrap_repeats
    confidence = settings.get('confidence_level', .95) if confidence is None else confidence
    seed = settings.get('sampling_seed', 1042) if seed is None else seed
    if bootstrap_repeats < 1 or not 0 < confidence < 1 or seed < 0:
        raise ValueError('Use a positive bootstrap count, confidence in (0,1), and a nonnegative seed.')
    output = Path(output_dir).resolve() if output_dir else evidence.root / 'reports'
    if output == evidence.root:
        raise ValueError('Use a separate reports directory to preserve original experiment artifacts.')
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    sections, markdown, conclusions, exports, figure_names = [], [], [], [], []
    comparison_provenance = None
    summary = {'run': evidence.root.name, 'run_mode': mode, 'evaluation_level': 'outer_test' if mode != 'final_fit' else 'inner_search',
               'confidence_level': confidence, 'bootstrap_repeats': bootstrap_repeats}

    def csv(name, frame):
        write_frame(output / name, frame)
        exports.append(name)

    def section(title, description, frame=None, image=None, names=None, advanced=False):
        content = f'<section><h2>{escape(title)}</h2><p>{escape(description)}</p>'
        markdown.extend([f'## {title}', '', description, ''])
        if image:
            figure_names.append(image)
            encoded = base64.b64encode((figures / f'{image}.png').read_bytes()).decode()
            content += f'<img alt="{escape(title)}" src="data:image/png;base64,{encoded}">'
            markdown.extend([f'![{title}](figures/{image}.png)', ''])
        if frame is not None and not frame.empty:
            content += '<div class="table-wrap">' + _table(frame, names) + '</div>'
            markdown.extend([_markdown(frame.rename(columns=names or {})), ''])
        body = content + '</section>'
        sections.append(f'<details><summary>{escape(title)}</summary>{body}</details>' if advanced else body)

    selections = evidence.read('outer_selections.json') if (evidence.root / 'outer_selections.json').exists() else []
    deployment = None
    if mode == 'final_fit':
        deployment = evidence.read('deployment_manifest.json')
        chosen = evidence.read('search/selected.json')
        selections = [chosen]
        config = deployment['configuration']
        summary.update(deployment_model=config['model'], training_installations=deployment['training_installations'], training_points=deployment['training_points'])
        conclusions.append(f"The all-data inner search selected {NAMES.get(config['model'], config['model'])} for deployment, using {config['feature_set']}, {config['aggregation']} aggregation and {config['reduction']} preprocessing.")
        conclusions.append(f"It was fitted on {deployment['training_installations']} installations and {deployment['training_audio_samples']} aggregated recordings. Its inner selection MAE is {deployment['inner_selection_score']:.3f}; this is a tuning score, not independent validation of this artifact.")
        section('Deployment decision', 'This is the pipeline selected by the final all-data search. Generalization evidence belongs to the source nested experiment.', pd.DataFrame([configuration_row(config)]).drop(columns='hyperparameters'))
    else:
        conclusions.append('No final deployment model is selected in this run. The selected procedure uses the pipeline chosen within each outer training partition. The final_fit stage selects and fits the deployment model on all development data.')

    candidates, search_summary, trials, health = search_tables(evidence, selections)
    if not search_summary.empty:
        csv('model_search_summary.csv', search_summary)
        csv('configurations_by_family_and_fold.csv', candidates)
        csv('search_health.csv', health)
        leading = search_summary.iloc[0]
        conclusions.append(f"{NAMES.get(leading.model, leading.model)} has the lowest mean best inner-search MAE ({leading.mean_best_inner_MAE:.3f}). This descriptive tuning ranking does not establish a unique outer-test winner.")
        failures = int(health.FAIL.sum())
        summary['failed_trials'] = failures
        summary['total_trial_attempts'] = int(search_summary.total_trials.sum())
        summary['inner_search_leader'] = leading.model
        runner_up = candidates[candidates.inner_rank == 2].gap_to_inner_winner
        if len(runner_up):
            summary['median_runner_up_gap'] = float(runner_up.median())
            conclusions.append(f'The median inner-score gap between each winner and runner-up is {runner_up.median():.4f} HFI units. This describes selection margins, not statistical evidence of superiority.')
        if failures:
            conclusions.append(f"{failures} of {summary['total_trial_attempts']} trial attempts failed. Search health lists the affected families; failed attempts reduce successful search coverage.")
            csv('failed_trial_reasons.csv', trials[trials.state == 'FAIL'].groupby(['model', 'failure']).size().rename('count').reset_index())
        section('Which model families were strongest during tuning?',
                'Each dot is one partition’s best inner-search score; diamonds show means. These scores selected configurations and must not be read as held-out performance. Selection frequency alone does not determine the deployment model.',
                search_summary[['model', 'mean_best_inner_MAE', 'mean_inner_rank', 'selected_partitions', 'completed_trials', 'total_trials']],
                charts.model_search(candidates, search_summary, figures),
                {'mean_best_inner_MAE': 'Mean best inner MAE', 'mean_inner_rank': 'Mean inner rank', 'selected_partitions': 'Times selected'})
        section('How stable was the tuning result across partitions?',
                'Every column compares candidates using the same inner folds and submissions. Low values mean close to that partition’s winner. R/F identifiers use the original zero-based repeat/fold numbering.',
                image=charts.search_heatmap(candidates, search_summary, figures), advanced=True)

    if selections:
        selected_rows = []
        for item in selections:
            partition = f"repeat_{item['repeat']:02d}_fold_{item['outer_fold']:02d}" if 'repeat' in item else 'all_data'
            selected_rows.append({'partition': partition, **configuration_row(item['configuration']), 'inner_selection_MAE': item['inner_score']})
        selected_table = pd.DataFrame(selected_rows)
        csv('selected_pipelines.csv', selected_table)
        if mode != 'final_fit':
            counts = selected_table.feature_set.value_counts()
            means = int((selected_table.aggregation == 'mean').sum())
            conclusions.append(f"Among {len(selected_table)} selected pipelines: embeddings-only was selected {int(counts.get('embeddings', 0))} times, both representations {int(counts.get('both', 0))} times, and indices-only {int(counts.get('indices', 0))} times. Mean aggregation was selected {means} times. These are selection patterns, not controlled ablation results.")
            section('What configurations were selected?', 'All choices below were made using inner validation only. The detailed CSV also contains PCA settings and hyperparameters. No-PCA PLS still performs its own supervised reduction.',
                    selected_table.drop(columns=['hyperparameters', 'scaling']), charts.configuration_choices(selected_table, settings, figures))

    if mode != 'final_fit':
        predictions = evidence.read('oof_predictions.parquet')
        comparison_manifest = output / 'family_comparison/comparison_manifest.json'
        comparison_evidence = None
        if comparison_manifest.exists():
            comparison_evidence = Evidence(comparison_manifest.parent)
            cm = comparison_evidence.read('comparison_manifest.json')
            if cm.get('status') == 'complete':
                if cm['source_fingerprint'] != manifest['identity']['fingerprint']:
                    raise ValueError('Family comparison belongs to a different source experiment.')
                # The fitted comparison must still refer to exactly these source artifacts.
                for name, digest in cm['source_artifacts'].items():
                    if file_hash(evidence.root / name) != digest:
                        raise ValueError(f'Source artifact changed after comparison: {name}')
                predictions = comparison_evidence.read('predictions.parquet')
        efforts, overall, gains, point_losses, differences = performance_tables(predictions, schema, bootstrap_repeats, confidence, seed)
        csv('performance_by_effort.csv', efforts)
        csv('performance_summary.csv', overall)
        csv('recording_effort_gains.csv', gains)
        if not differences.empty:
            csv('paired_comparisons_to_selected.csv', differences)
        csv('point_errors.csv', point_losses)
        chosen = overall[overall.evaluation == 'selected'].iloc[0]
        selected_efforts = efforts[efforts.evaluation == 'selected'].sort_values('n_recordings')
        summary.update(macro_MAE=float(chosen.macro_MAE), MAE_ci_low=float(chosen.MAE_ci_low), MAE_ci_high=float(chosen.MAE_ci_high),
                       evaluated_points=int(chosen.n_points), evaluated_installations=int(selected_efforts.n_installations.iloc[0]))
        score_sentence = f"Held-out MAE, averaged equally across the evaluated efforts, is {chosen.macro_MAE:.3f} HFI units ({confidence:.0%} conditional interval {chosen.MAE_ci_low:.3f}–{chosen.MAE_ci_high:.3f}). Evaluation covers {summary['evaluated_installations']} installations across {summary['evaluated_points']} Points."
        conclusions.insert(0, score_sentence)
        baselines = overall[overall.evaluation.isin(['DUMMY_MEAN', 'DUMMY_MEDIAN'])]
        if len(baselines):
            baseline = baselines.sort_values('macro_MAE').iloc[0]
            improvement = (1 - chosen.macro_MAE / baseline.macro_MAE) * 100 if baseline.macro_MAE else float('nan')
            summary['baseline_improvement_percent'] = float(improvement)
            conclusions.insert(1, f"Average MAE is {improvement:.1f}% lower than the stronger constant baseline ({NAMES[baseline.evaluation]}, MAE {baseline.macro_MAE:.3f}).")
        if len(selected_efforts) > 1:
            first, last = selected_efforts.iloc[0], selected_efforts.iloc[-1]
            last_gain = gains.iloc[-1]
            conclusions.append(f"MAE changes from {first.MAE:.3f} at {int(first.n_recordings)} recording(s) to {last.MAE:.3f} at {int(last.n_recordings)}. The last evaluated increase, {int(last_gain.from_recordings)} to {int(last_gain.to_recordings)}, reduces MAE by {last_gain.MAE_improvement:.3f} (conditional interval {last_gain.ci_low:.3f}–{last_gain.ci_high:.3f}). This alone does not establish a plateau.")
        section('How accurate are the held-out predictions?', UNCERTAINTY + ' MAE averages submissions within installations, installations within Points, then Points and efforts. Other reported effort metrics are averages of the per-CV-repetition metrics.', overall)
        section('How much do additional recordings help?',
                'This uses only the effort counts already evaluated in this run; it does not launch the denser effort experiment. Negative gains mean error increased. Average error is not a guarantee for an individual recording or location.',
                selected_efforts[['n_recordings', 'MAE', 'MAE_ci_low', 'MAE_ci_high', 'RMSE', 'R2', 'bias']], charts.effort_curve(efforts, gains, figures, confidence))
        diagnostics = diagnostic_table(predictions, schema)
        csv('installation_diagnostics.csv', diagnostics)
        section('Where do predictions fail?',
                'Each dot averages an installation’s held-out predictions. Whiskers span the 10th–90th percentiles across its submissions and CV repetitions; these are not confidence intervals. Negative residuals indicate underestimation. Reported errors are computed before averaging predictions, so they include sampling variability.',
                image=charts.diagnostics(diagnostics, figures))
        highest = int(diagnostics.n_recordings.max())
        hard = diagnostics[diagnostics.n_recordings == highest].groupby(schema.group).agg(MAE=('MAE', 'mean'), bias=('bias', 'mean'), mean_HFI=('observed_hfi', 'mean'), installations=(schema.bag, 'nunique')).reset_index().sort_values('MAE', ascending=False)
        section(f'Points with the largest errors at {highest} recordings',
                'Descriptive diagnostics on the evaluated cohort. These observations should not be removed or used to retune the model on the same outer test results.', hard.head(10))
        family_rows = overall[~overall.evaluation.isin(['selected', 'DUMMY_MEAN', 'DUMMY_MEDIAN'])]
        if len(family_rows):
            csv('outer_model_comparison.csv', overall)
            section('Model families on the same unseen Points',
                    'Every family uses its stored inner-selected configuration and exactly the same outer-test submissions. This is a descriptive comparison of tuned families. Picking the best row retrospectively would introduce another selection step; retain the prescribed all-data inner search for deployment.',
                    image=charts.outer_comparison(overall, figures,))
            summary['outer_family_comparison_available'] = True
        else:
            summary['outer_family_comparison_available'] = False
            section('Why is there no outer-test score for every family?',
                    'The original run evaluated only each partition’s selected pipeline and the constant baselines. Grouping selected predictions by family would compare different locations. Run results.py with --compare-models on the source nested_selection directory to add the missing matched evaluations from stored configurations, without repeating Optuna.')
        if comparison_evidence:
            comparison_provenance = {'directory': str(comparison_evidence.root), 'artifacts': comparison_evidence.hashes}

    if not health.empty:
        section('Did the searches finish successfully?',
                'Failed or pruned attempts consume search budget. The progress curves show the median best-so-far score and the interquartile range across partitions; shading is descriptive, not a confidence interval. Trial seconds include preprocessing on cache misses, so family totals are not pure estimator-speed benchmarks.',
                health, charts.search_progress(trials, figures), advanced=True)
    summary['interpretations'] = ' '.join(conclusions)
    csv('experiment_summary.csv', pd.DataFrame([summary]))
    write_json(output / 'experiment_summary.json', {**summary, 'interpretations': conclusions})
    exports.append('experiment_summary.json')
    cards = []
    if 'macro_MAE' in summary:
        cards = [('Held-out MAE', f"{summary['macro_MAE']:.3f}"), ('Evaluated Points', str(summary['evaluated_points'])),
                 ('Installations', str(summary['evaluated_installations'])), ('Deployment decision', 'Pending final_fit')]
    elif deployment:
        cards = [('Deployment model', NAMES.get(summary['deployment_model'], summary['deployment_model'])),
                 ('Training Points', str(summary['training_points'])), ('Training installations', str(summary['training_installations']))]
    intro = '<section class="intro"><h2>What this run tells you</h2><ul>' + ''.join(f'<li>{escape(c)}</li>' for c in conclusions) + '</ul></section>'
    downloads = ''.join(f'<li><a href="{escape(name)}">{escape(name)}</a></li>' for name in exports)
    image_links = ''.join(f'<li><a href="figures/{name}.pdf">{name}.pdf</a> · <a href="figures/{name}.png">PNG</a></li>' for name in figure_names)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>HFI experiment results</title><style>
    body{font:16px/1.6 system-ui,sans-serif;color:#20323b;background:#f2f6f8;margin:0}main{max-width:1180px;margin:auto;padding:30px 22px}h1{font-size:2.2rem;line-height:1.2}h2{line-height:1.3;color:#125762}p{max-width:100ch}.meta{color:#526570;overflow-wrap:anywhere}.cards{display:flex;flex-wrap:wrap;gap:14px}.card{background:#0e626d;color:white;border-radius:10px;padding:18px;flex:1;min-width:170px}.card strong{display:block;font-size:1.6rem}.card span{font-size:.85rem}section{background:white;padding:24px;margin:22px 0;border-radius:10px;border:1px solid #dde6e9}.intro li{margin:12px 0}img{display:block;width:100%;height:auto;margin:18px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dce5e8}th{background:#edf5f6;white-space:nowrap}a{color:#087781}details{background:white;padding:18px;margin:14px 0;border-radius:8px}summary{cursor:pointer;font-weight:600}footer{font-size:.85rem;color:#526570}@media print{body{background:white}main{padding:0}section{break-inside:avoid}.cards{display:none}}
    </style></head><body><main>'''
    html += f'<h1>Acoustic HFI · Experiment results</h1><p class="meta">{escape(evidence.root.name)} · {escape(mode)}</p>'
    html += '<div class="cards">' + ''.join(f'<div class="card"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>' for k, v in cards) + '</div>'
    html += intro + ''.join(sections)
    html += f'<details><summary>Download tables and publication figures</summary><ul>{downloads}{image_links}</ul></details>'
    html += f'<footer>{escape(UNCERTAINTY)}<p>Source: {escape(str(evidence.root))}. Report version {REPORT_VERSION}. Original experiment artifacts remain unchanged.</p></footer></main></body></html>'
    (output / 'report.html').write_text(html, encoding='utf-8')
    (output / 'summary.md').write_text('# HFI experiment results\n\n' + '\n\n'.join(conclusions) + '\n\n' + '\n'.join(markdown), encoding='utf-8')
    write_json(output / 'report_manifest.json', {'report_version': REPORT_VERSION, 'source_run': str(evidence.root),
               'source_fingerprint': manifest['identity']['fingerprint'], 'generated_utc': datetime.now(timezone.utc).isoformat(),
               'source_artifacts': evidence.hashes, 'comparison_artifacts': comparison_provenance, 'bootstrap_repeats': bootstrap_repeats, 'confidence_level': confidence, 'bootstrap_seed': seed,
               'report_code': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}, 'uncertainty': UNCERTAINTY})
    return output / 'report.html'
