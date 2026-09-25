"""Command-line entry point; importing this module never starts an experiment."""
import argparse
import json

from src.config.settings import Settings
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Nested HFI pipeline selection and recording-effort evaluation")
    parser.add_argument("--env-file", default=".env", help="Explicit dotenv file; process environment overrides its values")
    parser.add_argument("--run-mode", choices=["nested_selection", "effort_analysis", "final_fit"])
    parser.add_argument("--source-run")
    parser.add_argument("--resume-run")
    parser.add_argument("--output-dir")
    parser.add_argument("--check", action="store_true", help="Validate data and split feasibility without fitting or creating run files")
    args = parser.parse_args()
    overrides = {key: getattr(args, key) for key in ("run_mode", "source_run", "resume_run", "output_dir")}
    settings = Settings.from_env(args.env_file, overrides)
    pipeline = Pipeline(settings)
    if args.check:
        print(json.dumps(pipeline.check(), indent=2))
    else:
        print(f"Run artifacts: {pipeline.run()}")


if __name__ == "__main__":
    main()
