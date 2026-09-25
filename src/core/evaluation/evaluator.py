"""Nested selection of complete pipelines, with full-recording training only."""
from collections import OrderedDict
import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.core.data.sampling import sample_submissions, stable_seed
from src.core.models.factory import ModelFactory, RegressionModels
from src.core.models.search_space import suggest_configuration
from src.core.processors.postsplit import FittedPipeline, ModelConvergenceError, PostSplitProcessor, fit_estimator
from src.core.processors.presplit import SignatureBuilder
from .metrics import macro_mae, metric_table
from .persistence import canonical_json, write_bundle, write_frame, write_json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, settings, schema, audio, output_dir):
        self.settings = settings
        self.schema = schema
        self.audio = audio
        self.output_dir = Path(output_dir)
        self.builder = SignatureBuilder(audio, schema)
        self.metadata = audio.groupby(schema.bag, as_index=False).agg({schema.group: "first", schema.target: "first", "sample_id": "size"}).rename(columns={"sample_id": "available_recordings"})
        self.models = [m.name for m in RegressionModels] if settings.models == ("all",) else list(settings.models)
        self.eligible = set(self.metadata.loc[self.metadata.available_recordings >= max(settings.selection_efforts), schema.bag])
        self.dimensions = {}
        for aggregation in settings.aggregation_strategies:
            full = self.builder.full(aggregation)
            self.dimensions[aggregation] = {
                "indices": len(schema.index_columns(full.columns)),
                "embeddings": len(full[schema.embedding].iloc[0]) if schema.embedding in full else 0,
                "availability": sum(c.startswith("availability_") for c in full),
            }
        for feature in settings.feature_sets:
            for block in ("indices", "embeddings"):
                if feature in (block, "both") and not all(d[block] > 0 for d in self.dimensions.values()):
                    raise ValueError(f"Requested {feature}, but {block} features are absent.")

    def _splits(self, metadata, count, seed):
        if metadata[self.schema.group].nunique() < count:
            raise ValueError(f"Need at least {count} distinct Points for this CV split.")
        cv = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
        result = []
        for train_idx, test_idx in cv.split(metadata, groups=metadata[self.schema.group]):
            train, test = metadata.iloc[train_idx].copy(), metadata.iloc[test_idx].copy()
            if not set(test[self.schema.bag]) & self.eligible:
                raise ValueError("A validation fold has no installations eligible for every effort. Reduce maximum effort or revise the frozen split design.")
            if set(train[self.schema.group]) & set(test[self.schema.group]):
                raise AssertionError("Point leakage detected.")
            result.append((train, test))
        return result

    def validate_design(self):
        if not self.eligible:
            raise ValueError("No installations support the largest selection effort.")
        for repeat in range(self.settings.outer_repeats):
            outer = self._splits(self.metadata, self.settings.outer_splits, stable_seed(self.settings.split_seed, "outer", repeat))
            for fold, (train, _) in enumerate(outer):
                self._splits(train, self.settings.inner_splits, stable_seed(self.settings.split_seed, "inner", repeat, fold))
        # The final all-data selection uses the same inner search procedure.
        self._splits(self.metadata, self.settings.inner_splits, stable_seed(self.settings.split_seed, "final_inner"))
        return self.audit()

    def audit(self):
        table = self.metadata.copy()
        table["eligible_for_all_efforts"] = table[self.schema.bag].isin(self.eligible)
        return table

    def _audio_for(self, metadata):
        return self.audio[self.audio[self.schema.bag].isin(metadata[self.schema.bag])]

    def _manifest(self, metadata, namespace, efforts=None, repeats=None):
        # Preserve the original common cohort even when an analysis requests only
        # a subset of the selection efforts.
        metadata = metadata[metadata[self.schema.bag].isin(self.eligible)]
        return sample_submissions(self._audio_for(metadata), self.schema,
                                  efforts or self.settings.selection_efforts,
                                  repeats or self.settings.selection_sampling_repeats,
                                  self.settings.sampling_seed, namespace)

    def _full(self, metadata, aggregation):
        full = self.builder.full(aggregation)
        return full[full[self.schema.bag].isin(metadata[self.schema.bag])].reset_index(drop=True)

    def _search(self, metadata, directory, namespace, split_seed):
        directory.mkdir(parents=True, exist_ok=True)
        folds = self._splits(metadata, self.settings.inner_splits, split_seed)
        manifests = []
        split_rows = []
        for fold, (train, validation) in enumerate(folds):
            manifest = self._manifest(validation, f"{namespace}:inner:{fold}")
            manifests.append(manifest)
            write_frame(directory / f"inner_{fold}_submissions.parquet", manifest)
            for role, frame in (("train", train), ("validation", validation)):
                split_rows.append(frame.assign(inner_fold=fold, role=role))
        write_frame(directory / "inner_splits.parquet", pd.concat(split_rows, ignore_index=True))
        min_train = min(len(train) for train, _ in folds)
        aggregated = {}
        prepared_cache = OrderedDict()

        def prepared(config):
            key = canonical_json({k: v for k, v in config.items() if k not in {"model", "model_params"}})
            if key in prepared_cache:
                prepared_cache.move_to_end(key)
                return prepared_cache[key]
            aggregation = config["aggregation"]
            if aggregation not in aggregated:
                aggregated[aggregation] = [
                    (self._full(train, aggregation), self.builder.submissions(manifest, aggregation))
                    for (train, _), manifest in zip(folds, manifests)
                ]
            data = []
            for train, validation in aggregated[aggregation]:
                processor = PostSplitProcessor(self.schema, config)
                x, y = processor.fit_transform(train)
                data.append((x, y, processor.transform(validation), processor))
            prepared_cache[key] = data
            while len(prepared_cache) > self.settings.cache_entries:
                prepared_cache.popitem(last=False)
            return data

        results = []
        for model_name in self.models:
            seed = stable_seed(self.settings.search_seed, namespace, model_name)
            sampler_path = directory / f"{model_name}_sampler.joblib"
            sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, max(1, self.settings.optuna_trials // 4)))
            storage = optuna.storages.RDBStorage(url=f"sqlite:///{(directory / 'studies.sqlite3').resolve()}", engine_kwargs={"connect_args": {"timeout": 60}})
            study = optuna.create_study(study_name=model_name, direction="minimize", sampler=sampler, storage=storage, load_if_exists=True)
            for old in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
                study.tell(old.number, state=optuna.trial.TrialState.FAIL)
                logger.warning("Marked interrupted %s trial %d as failed", model_name, old.number)

            def objective(trial):
                started = time.perf_counter()
                config = suggest_configuration(trial, model_name, self.settings, self.dimensions, min_train)
                trial.set_user_attr("configuration", config)
                predictions, warning_messages = [], []
                try:
                    for fold, ((x, y, xv, processor), manifest) in enumerate(zip(prepared(config), manifests)):
                        estimator = ModelFactory.create_model(model_name, config["model_params"], stable_seed(self.settings.model_seed, namespace, model_name, fold), self.settings.model_jobs)
                        warning_messages.extend(fit_estimator(estimator, x, y))
                        predicted = processor.inverse_target(estimator.predict(xv))
                        if not np.isfinite(predicted).all():
                            raise FloatingPointError("Candidate produced non-finite validation predictions.")
                        frame = manifest.drop(columns="sample_ids").copy()
                        frame["prediction"] = predicted
                        predictions.append(frame)
                    combined = pd.concat(predictions, ignore_index=True)
                    score = macro_mae(combined, self.schema, self.settings.selection_efforts)
                    trial.set_user_attr("warnings", sorted(set(warning_messages)))
                    trial.set_user_attr("seconds", time.perf_counter() - started)
                    return score
                except (ModelConvergenceError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    trial.set_user_attr("failure", str(exc))
                    raise

            def save_sampler(study, trial):
                write_bundle(sampler_path, study.sampler)

            remaining = max(0, self.settings.optuna_trials - len(study.trials))
            logger.info("%s: %s — %d remaining trials (configuration + hyperparameters)", namespace, model_name, remaining)
            try:
                study.optimize(objective, n_trials=remaining, n_jobs=1,
                               callbacks=[save_sampler], catch=(ModelConvergenceError, np.linalg.LinAlgError, FloatingPointError))
            finally:
                trials = study.trials
                write_json(directory / f"{model_name}_trials.json", [
                    {"number": t.number, "state": t.state.name, "value": t.value, "parameters": t.params,
                     "attributes": t.user_attrs, "duration_seconds": t.duration.total_seconds() if t.duration else None}
                    for t in trials
                ])
                storage.remove_session()
                storage.engine.dispose()
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and np.isfinite(t.value)]
            if not completed:
                raise RuntimeError(f"No completed trials for {model_name} in {namespace}; inspect saved trial diagnostics. Do not silently omit this family.")
            best = min(completed, key=lambda t: (t.value, t.number))
            results.append({"model": model_name, "inner_score": best.value, "trial_number": best.number,
                            "configuration": best.user_attrs["configuration"], "completed_trials": len(completed), "total_trials": len(trials)})
        results.sort(key=lambda result: (result["inner_score"], result["model"]))
        write_json(directory / "candidate_results.json", results)
        write_json(directory / "selected.json", results[0])
        return results[0]

    def _predict(self, pipeline, manifest, repeat, fold, evaluation="selected"):
        signatures = self.builder.submissions(manifest, pipeline.config["aggregation"])
        result = manifest.drop(columns="sample_ids").copy()
        result["prediction"] = pipeline.predict(signatures)
        result["repeat"] = repeat
        result["outer_fold"] = fold
        result["evaluation"] = evaluation
        result["selected_model"] = pipeline.config["model"]
        return result

    def evaluate(self):
        self.validate_design()
        write_frame(self.output_dir / "data_audit.csv", self.audit())
        all_predictions, outer_splits, selections = [], [], []
        for repeat in range(self.settings.outer_repeats):
            splits = self._splits(self.metadata, self.settings.outer_splits, stable_seed(self.settings.split_seed, "outer", repeat))
            for fold, (train, test) in enumerate(splits):
                namespace = f"repeat_{repeat:02d}_fold_{fold:02d}"
                directory = self.output_dir / namespace
                directory.mkdir(parents=True, exist_ok=True)
                for role, data in (("train", train), ("test", test)):
                    outer_splits.append(data.assign(repeat=repeat, outer_fold=fold, role=role))
                # Persist all assignments incrementally for inspection after a crash.
                write_frame(self.output_dir / "outer_splits.parquet", pd.concat(outer_splits, ignore_index=True))
                if (directory / "complete.json").exists():
                    all_predictions.append(pd.read_parquet(directory / "predictions.parquet"))
                    selections.append(json.loads((directory / "selection.json").read_text()))
                    logger.info("Reusing completed %s", namespace)
                    continue
                selected = self._search(train, directory / "search", namespace, stable_seed(self.settings.split_seed, "inner", repeat, fold))
                started = time.perf_counter()
                full = self._full(train, selected["configuration"]["aggregation"])
                pipeline = FittedPipeline.fit(self.schema, selected["configuration"], full, stable_seed(self.settings.model_seed, namespace, "refit"), self.settings.model_jobs, self.settings.recording_start, self.settings.recording_end)
                bundle = {"pipeline": pipeline, "repeat": repeat, "outer_fold": fold,
                          "train_installations": train[self.schema.bag].tolist(), "test_installations": test[self.schema.bag].tolist(),
                          "train_points": train[self.schema.group].unique().tolist(), "test_points": test[self.schema.group].unique().tolist(),
                          "training_recordings": int(train.available_recordings.sum())}
                write_bundle(directory / "pipeline.joblib", bundle)
                manifest = self._manifest(test, f"{namespace}:outer")
                write_frame(directory / "submissions.parquet", manifest)
                predictions = [self._predict(pipeline, manifest, repeat, fold)]
                for baseline, value in (("DUMMY_MEAN", train[self.schema.target].mean()), ("DUMMY_MEDIAN", train[self.schema.target].median())):
                    frame = manifest.drop(columns="sample_ids").copy()
                    frame["prediction"] = float(value)
                    frame["repeat"], frame["outer_fold"] = repeat, fold
                    frame["evaluation"], frame["selected_model"] = baseline, baseline
                    predictions.append(frame)
                combined = pd.concat(predictions, ignore_index=True)
                selection = {**selected, "repeat": repeat, "outer_fold": fold,
                             "refit_and_prediction_seconds": time.perf_counter() - started,
                             "preprocessing": pipeline.processor.diagnostics, "refit_warnings": pipeline.warnings}
                write_json(directory / "selection.json", selection)
                write_frame(directory / "predictions.parquet", combined)
                write_json(directory / "complete.json", {"complete": True})
                all_predictions.append(combined)
                selections.append(selection)
        combined = pd.concat(all_predictions, ignore_index=True)
        write_frame(self.output_dir / "oof_predictions.parquet", combined)
        metrics = metric_table(combined, self.schema)
        write_frame(self.output_dir / "metrics_by_effort.csv", metrics)
        scores = metrics.groupby(["evaluation", "repeat"], as_index=False).MAE.mean().rename(columns={"MAE": "macro_point_balanced_mae"})
        write_frame(self.output_dir / "selection_procedure_scores.csv", scores)
        write_json(self.output_dir / "outer_selections.json", selections)
        frequencies = pd.DataFrame({"model": [s["model"] for s in selections]}).value_counts().rename("selected_folds").reset_index()
        write_frame(self.output_dir / "selection_frequency.csv", frequencies)
        return combined

    def effort_analysis(self, source_dir, analysis_settings):
        from .plots import save_effort_analysis
        if max(analysis_settings.effort_analysis_counts) > max(self.settings.selection_efforts):
            raise ValueError("Detailed efforts exceed the source experiment's common cohort range. Run a new prespecified selection design for an extended range.")
        selections = json.loads((source_dir / "outer_selections.json").read_text())
        predictions = []
        for selection in selections:
            repeat, fold = selection["repeat"], selection["outer_fold"]
            namespace = f"repeat_{repeat:02d}_fold_{fold:02d}"
            bundle = joblib.load(source_dir / namespace / "pipeline.joblib")
            if set(bundle["train_points"]) & set(bundle["test_points"]):
                raise ValueError("Source pipeline contains overlapping Point partitions.")
            test = self.metadata[self.metadata[self.schema.bag].isin(bundle["test_installations"])]
            manifest = self._manifest(test, f"{namespace}:effort", analysis_settings.effort_analysis_counts, analysis_settings.effort_analysis_repeats)
            write_frame(self.output_dir / f"{namespace}_submissions.parquet", manifest)
            predictions.append(self._predict(bundle["pipeline"], manifest, repeat, fold))
        result = pd.concat(predictions, ignore_index=True)
        write_frame(self.output_dir / "oof_predictions.parquet", result)
        save_effort_analysis(result, self.schema, analysis_settings, self.output_dir)
        return result

    def final_fit(self):
        selected = self._search(self.metadata, self.output_dir / "search", "final", stable_seed(self.settings.split_seed, "final_inner"))
        full = self._full(self.metadata, selected["configuration"]["aggregation"])
        pipeline = FittedPipeline.fit(self.schema, selected["configuration"], full, stable_seed(self.settings.model_seed, "final_refit"), self.settings.model_jobs, self.settings.recording_start, self.settings.recording_end)
        write_bundle(self.output_dir / "deployment.joblib", pipeline)
        write_json(self.output_dir / "deployment_manifest.json", {
            "configuration": selected["configuration"], "inner_selection_score": selected["inner_score"],
            "training_installations": len(full), "training_points": int(full[self.schema.group].nunique()),
            "training_audio_samples": len(self.audio), "recording_start": self.settings.recording_start,
            "recording_end": self.settings.recording_end, "sample_unit": "mean within Audio_Name",
            "preprocessing": pipeline.processor.diagnostics, "refit_warnings": pipeline.warnings,
            "performance_claim": "The source nested run estimates the selection procedure. This final artifact requires independent test data for direct validation.",
        })
        return pipeline
