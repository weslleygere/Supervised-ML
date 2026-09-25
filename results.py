"""Human-facing experiment results. Original training code and artifacts stay frozen."""
import argparse
from pathlib import Path

from reporting.report import build_report


def main():
    parser = argparse.ArgumentParser(description='Create useful HFI experiment tables, interpretations and plots from a completed run.')
    parser.add_argument('run_dir', nargs='?', help='Completed nested_selection, effort_analysis or final_fit directory')
    parser.add_argument('--run-experiment', action='store_true', help='Run the experiment configured in .env and automatically generate its report')
    parser.add_argument('--env-file', default='.env', help='Experiment settings, used only with --run-experiment')
    parser.add_argument('--output-dir', help='Report directory (default: RUN_DIR/reports)')
    parser.add_argument('--compare-models', action='store_true', help='Also refit stored inner-selected configurations for every family on the original outer splits; no new Optuna search')
    parser.add_argument('--bootstrap-repeats', type=int)
    parser.add_argument('--confidence-level', type=float)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if bool(args.run_dir) == args.run_experiment:
        parser.error('Provide either a completed run directory or --run-experiment.')
    if args.bootstrap_repeats is not None and args.bootstrap_repeats < 1:
        parser.error('--bootstrap-repeats must be positive.')
    if args.confidence_level is not None and not 0 < args.confidence_level < 1:
        parser.error('--confidence-level must be between 0 and 1.')
    if args.seed is not None and args.seed < 0:
        parser.error('--seed cannot be negative.')
    if args.run_experiment:
        from src.config.settings import Settings
        from src.pipeline import Pipeline
        settings = Settings.from_env(args.env_file)
        if args.compare_models and settings.run_mode != 'nested_selection':
            parser.error('--compare-models requires a nested_selection experiment.')
        args.run_dir = str(Pipeline(settings).run())
    output = Path(args.output_dir).resolve() if args.output_dir else Path(args.run_dir).resolve() / 'reports'
    if args.compare_models:
        from reporting.comparison import compare_families
        compare_families(args.run_dir, output / 'family_comparison')
    report = build_report(args.run_dir, output, args.bootstrap_repeats, args.confidence_level, args.seed)
    print(f'Open report: {report}')
    print(f'VS Code summary: {report.parent / "summary.md"}')


if __name__ == '__main__':
    main()
