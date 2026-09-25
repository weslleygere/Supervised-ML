"""Orchestrate selection, saved-fold effort analysis, and final deployment fitting."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import uuid

from threadpoolctl import threadpool_limits

from .config.logging import configure_logging
from .config.settings import Settings
from .core.data.data_loader import DataLoader
from .core.evaluation.evaluator import ModelEvaluator
from .core.evaluation.persistence import canonical_json, environment, identity, write_frame, write_json
from .core.processors.presplit import PreSplitProcessor

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, settings: Settings):
        self.requested = settings
        self.source_dir = Path(settings.source_run).resolve() if settings.source_run else None
        self.settings = settings
        if settings.run_mode != "nested_selection":
            manifest_path = self.source_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise ValueError("SOURCE_RUN must identify a completed nested-selection run.")
            source = json.loads(manifest_path.read_text())
            if source["run_mode"] != "nested_selection" or source["status"] != "complete":
                raise ValueError("SOURCE_RUN must be a completed nested_selection run.")
            self.settings = Settings.from_dict(json.loads((self.source_dir / "resolved_config.json").read_text()))
            current_identity = identity(self.settings)
            if source["identity"]["fingerprint"] != current_identity["fingerprint"]:
                raise ValueError("Source data, schema, search protocol or implementation changed. Refusing to mix incompatible artifacts.")
            if source["environment"]["packages"] != environment()["packages"]:
                raise ValueError("Source package versions differ from the current environment.")

    def _load(self):
        loader = DataLoader(self.settings.data_path, self.settings.schema_path)
        raw = loader.load_data()
        schema = loader.load_schema()
        audio = PreSplitProcessor(schema).prepare_audio(raw, self.settings.recording_start, self.settings.recording_end)
        logger.info("Prepared %d aggregated recordings across %d installations and %d Points", len(audio), audio[schema.bag].nunique(), audio[schema.group].nunique())
        return schema, audio

    def check(self):
        schema, audio = self._load()
        evaluator = ModelEvaluator(self.settings, schema, audio, Path("."))
        audit = evaluator.validate_design()
        return {
            "run_mode": self.requested.run_mode, "aggregated_recordings": len(audio),
            "installations": len(audit), "Points": int(audit[schema.group].nunique()),
            "eligible_installations": int(audit.eligible_for_all_efforts.sum()),
            "selection_efforts": self.settings.selection_efforts,
            "models": evaluator.models, "outer_fits": self.settings.outer_splits * self.settings.outer_repeats,
            "maximum_inner_estimator_fits": self.settings.outer_splits * self.settings.outer_repeats * len(evaluator.models) * self.settings.optuna_trials * self.settings.inner_splits,
            "training_policy": "all recordings, one row per installation, equal-installation fitting",
            "selection_metric": "equal-effort average of point-balanced MAE",
        }

    def run(self):
        settings = self.settings
        experiment_identity = identity(settings)
        run_environment = environment()
        if self.requested.resume_run:
            output = Path(self.requested.resume_run).resolve()
            manifest = json.loads((output / "run_manifest.json").read_text())
            if manifest["identity"]["fingerprint"] != experiment_identity["fingerprint"] or manifest["run_mode"] != self.requested.run_mode:
                raise ValueError("RESUME_RUN has a different data/configuration/code identity or execution mode.")
            previous_request = json.loads((output / "request_config.json").read_text())
            ignored = {"resume_run", "output_dir", "log_level"}
            if canonical_json({k: v for k, v in previous_request.items() if k not in ignored}) != canonical_json({k: v for k, v in self.requested.to_dict().items() if k not in ignored}):
                raise ValueError("Resume request changed. Resume must preserve the original protocol.")
            if manifest["environment"]["packages"] != run_environment["packages"]:
                raise ValueError("Package versions changed since this run.")
            if manifest["status"] == "complete":
                return output
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = Path(self.requested.output_dir).resolve() / f"{stamp}_{self.requested.run_mode}_{uuid.uuid4().hex[:8]}"
            output.mkdir(parents=True, exist_ok=False)
            manifest = {"run_mode": self.requested.run_mode, "identity": experiment_identity,
                        "environment": run_environment, "source_run": str(self.source_dir) if self.source_dir else None}
            write_json(output / "resolved_config.json", settings.to_dict())
            write_json(output / "request_config.json", self.requested.to_dict())
        configure_logging(output, self.requested.log_level)
        manifest.pop("error", None)
        manifest["status"] = "running"
        write_json(output / "run_manifest.json", manifest)
        try:
            schema, audio = self._load()
            write_json(output / "schema.json", {"target": schema.target, "group": schema.group, "bag": schema.bag, "audio": schema.audio, "datetime": schema.datetime, "index_prefixes": schema.index_prefixes, "embedding": schema.embedding})
            evaluator = ModelEvaluator(settings, schema, audio, output)
            write_frame(output / "data_audit.csv", evaluator.audit())
            with threadpool_limits(limits=settings.model_jobs):
                if self.requested.run_mode == "nested_selection":
                    evaluator.evaluate()
                elif self.requested.run_mode == "effort_analysis":
                    evaluator.effort_analysis(self.source_dir, self.requested)
                else:
                    evaluator.final_fit()
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Run interrupted or failed; artifacts retained in %s", output)
            raise
        finally:
            write_json(output / "run_manifest.json", manifest)
        logger.info("Completed %s: %s", self.requested.run_mode, output)
        return output
