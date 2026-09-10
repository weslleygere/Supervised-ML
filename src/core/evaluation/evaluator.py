import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold

from src.core.data.schema import Schema
from src.core.models.factory import (
    ModelFactory,
    RegressionModels,
)
from src.core.evaluation.plots import save_evaluation_plots
from src.core.models.search_space import suggest_parameters
from src.core.models.definitions import ModelConvergenceError
from src.core.processors.postsplit import PostSplitProcessor


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MIRMetrics:
    """
    Metrics obtained after instance-level predictions are aggregated
    within CapturePointId.

    Each physical Point receives the same total evaluation weight.
    """

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class PreparedInnerFold:
    """Training-only preprocessing reused across trials in one outer fold."""

    validation_indices: np.ndarray
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_validation: pd.DataFrame
    weights: np.ndarray
    processor: PostSplitProcessor


class ModelEvaluator:
    """
    Nested grouped cross-validation for the instance-MIR experiment.

    Experimental structure
    ----------------------
    - one dataframe row = one 1-minute recording;
    - outer CV grouped by Point;
    - inner CV grouped by Point;
    - Optuna selects hyperparameters using inner OOF predictions;
    - models are fitted with hierarchical Point/CapturePointId weights;
    - predictions are produced independently for every 1-minute recording;
    - k recordings are randomly sampled within each CapturePointId;
    - sampled predictions are averaged to obtain a bag-level prediction;
    - evaluation is balanced so every physical Point has equal total weight.
    """

    def __init__(
        self,
        schema: Schema,
        output_dir: str,
        models: list[RegressionModels],
        random_state: int,
        feature_set: str,
        pca_components: int | None,
        outer_splits: int,
        inner_splits: int,
        optuna_trials: int,
        inference_k: int,
        inference_repeats: int,
    ) -> None:
        self.schema = schema
        self.output_dir = output_dir
        self.models = models
        self.random_state = random_state

        self.feature_set = feature_set
        self.pca_components = pca_components

        self.outer_splits = outer_splits
        self.inner_splits = inner_splits

        self.optuna_trials = optuna_trials

        self.inference_k = inference_k
        self.inference_repeats = inference_repeats

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def evaluate(
        self,
        df_processed: pd.DataFrame,
    ) -> None:
        """
        Evaluate every selected model using nested grouped cross-validation.
        """

        X = df_processed.drop(
            columns=[self.schema.target]
        )

        y = df_processed[
            [self.schema.target]
        ]

        groups = df_processed[
            self.schema.group
        ].to_numpy()

        logger.info(
            "Starting instance-MIR evaluation: "
            "%d recordings, %d Points, feature_set=%s",
            len(df_processed),
            df_processed[self.schema.group].nunique(),
            self.feature_set,
        )

        outer_metrics_rows: list[dict] = []
        best_params_rows: list[dict] = []
        oof_rows: list[pd.DataFrame] = []

        for model_idx, model_enum in enumerate(
            self.models,
            start=1,
        ):
            logger.info(
                "Evaluating model %d/%d: %s",
                model_idx,
                len(self.models),
                model_enum.name,
            )

            (
                model_metrics,
                model_params,
                model_oof,
            ) = self._evaluate_model(
                model_enum=model_enum,
                X=X,
                y=y,
                df=df_processed,
                groups=groups,
            )

            outer_metrics_rows.extend(
                model_metrics
            )

            best_params_rows.extend(
                model_params
            )

            oof_rows.append(
                model_oof
            )

        outer_metrics = pd.DataFrame(
            outer_metrics_rows
        )

        best_params = pd.DataFrame(
            best_params_rows
        )

        oof_predictions = pd.concat(
            oof_rows,
            ignore_index=True,
        )

        summary = self._summarize_results(
            outer_metrics=outer_metrics,
            oof_predictions=oof_predictions,
        )

        self._save_results(
            outer_metrics=outer_metrics,
            best_params=best_params,
            oof_predictions=oof_predictions,
            summary=summary,
        )

        save_evaluation_plots(
            outer_metrics=outer_metrics,
            summary=summary,
            oof_predictions=oof_predictions,
            output_dir=self.output_dir,
            target_column=self.schema.target,
            group_column=self.schema.group,
            bag_column=self.schema.bag,
            inference_k=self.inference_k,
            inference_repeats=self.inference_repeats,
            seed=self._pooled_inference_seed(),
        )

    # =========================================================================
    # OUTER CROSS-VALIDATION
    # =========================================================================

    def _evaluate_model(
        self,
        model_enum: RegressionModels,
        X: pd.DataFrame,
        y: pd.DataFrame,
        df: pd.DataFrame,
        groups: np.ndarray,
    ) -> tuple[
        list[dict],
        list[dict],
        pd.DataFrame,
    ]:
        """
        Evaluate one model across all outer folds.
        """

        outer_cv = GroupKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        metrics_rows: list[dict] = []
        params_rows: list[dict] = []
        prediction_rows: list[pd.DataFrame] = []

        for outer_fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            outer_cv.split(
                X,
                y,
                groups=groups,
            ),
            start=1,
        ):
            logger.info(
                "%s — outer fold %d/%d",
                model_enum.name,
                outer_fold,
                self.outer_splits,
            )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            best_params = self._optimize_hyperparameters(
                model_enum=model_enum,
                X=X_train,
                y=y_train,
                df=df_train,
                outer_fold=outer_fold,
            )

            try:
                (
                    predictions,
                    fit_time,
                    prediction_time,
                ) = self._fit_outer_model(
                    model_enum=model_enum,
                    params=best_params,
                    X_train=X_train,
                    y_train=y_train,
                    df_train=df_train,
                    X_test=X_test,
                )
            except ModelConvergenceError as exc:
                message = (
                    f"{model_enum.name} — outer fold {outer_fold}: "
                    "the selected configuration did not converge on the "
                    "outer training data. Evaluation stopped; this fold "
                    "was not scored."
                )
                logger.error("%s %s", message, exc)
                raise ModelConvergenceError(message) from exc

            metrics = self._instance_mir_metrics(
                df=df_test,
                predictions=predictions,
                seed=self._outer_inference_seed(
                    outer_fold
                ),
            )

            metrics_rows.append(
                {
                    "model": model_enum.name,
                    "feature_set": self.feature_set,
                    "outer_fold": outer_fold,
                    "n_train_points": (
                        df_train[
                            self.schema.group
                        ].nunique()
                    ),
                    "n_test_points": (
                        df_test[
                            self.schema.group
                        ].nunique()
                    ),
                    "MAE": metrics.mae,
                    "RMSE": metrics.rmse,
                    "R2": metrics.r2,
                    "fit_time": fit_time,
                    "prediction_time": prediction_time,
                }
            )

            params_rows.append(
                {
                    "model": model_enum.name,
                    "feature_set": self.feature_set,
                    "outer_fold": outer_fold,
                    "best_params": json.dumps(
                        best_params,
                        sort_keys=True,
                    ),
                }
            )

            prediction_rows.append(
                self._oof_dataframe(
                    df_test=df_test,
                    predictions=predictions,
                    model_enum=model_enum,
                    outer_fold=outer_fold,
                )
            )

        return (
            metrics_rows,
            params_rows,
            pd.concat(
                prediction_rows,
                ignore_index=True,
            ),
        )

    # =========================================================================
    # OPTUNA / INNER CROSS-VALIDATION
    # =========================================================================

    def _optimize_hyperparameters(
        self,
        model_enum: RegressionModels,
        X: pd.DataFrame,
        y: pd.DataFrame,
        df: pd.DataFrame,
        outer_fold: int,
    ) -> dict:
        """Select a converged configuration using inner OOF predictions.

        Prepared folds are local to this search and released when it ends.
        No preprocessing is fitted on inner validation or outer test data.
        """
        prepared_folds = self._prepare_inner_folds(
            X=X,
            y=y,
            df=df,
            outer_fold=outer_fold,
        )

        sampler = optuna.samplers.TPESampler(
            seed=self._optuna_seed(model_enum, outer_fold)
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            params = suggest_parameters(trial, model_enum)
            trial.set_user_attr("model_params", params)

            try:
                return self._inner_cv_score(
                    model_enum=model_enum,
                    params=params,
                    prepared_folds=prepared_folds,
                    df=df,
                    outer_fold=outer_fold,
                )
            except ModelConvergenceError as exc:
                trial.set_user_attr("convergence_failure", str(exc))
                raise

        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            n_jobs=1,
            show_progress_bar=False,
            catch=(ModelConvergenceError,),
        )

        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )
        convergence_failures = sum(
            "convergence_failure" in trial.user_attrs
            for trial in study.trials
        )
        logger.info(
            "%s — outer fold %d — trials: %d completed, "
            "%d failed to converge, %d total",
            model_enum.name,
            outer_fold,
            completed,
            convergence_failures,
            len(study.trials),
        )

        if completed == 0:
            message = (
                f"{model_enum.name} — outer fold {outer_fold}: "
                f"no completed trials out of {len(study.trials)} "
                f"({convergence_failures} convergence failures). "
                "No configuration can be selected; evaluation stopped."
            )
            logger.error(message)
            raise RuntimeError(message)

        best_params = study.best_trial.user_attrs["model_params"]
        logger.info(
            "%s — outer fold %d — "
            "best inner MIR-MAE: %.4f — params: %s",
            model_enum.name,
            outer_fold,
            study.best_value,
            best_params,
        )
        return best_params

    def _prepare_inner_folds(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        df: pd.DataFrame,
        outer_fold: int,
    ) -> list[PreparedInnerFold]:
        """Fit each inner fold's preprocessing and weights once per search."""
        inner_cv = GroupKFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=self._inner_cv_seed(outer_fold),
        )
        prepared_folds = []
        for train_idx, validation_idx in inner_cv.split(
            X, y, groups=df[self.schema.group].to_numpy()
        ):
            processor = PostSplitProcessor(
                schema=self.schema,
                feature_set=self.feature_set,
                pca_components=self.pca_components,
            )
            X_train, y_train = processor.fit_transform(
                X.iloc[train_idx], y.iloc[train_idx]
            )
            prepared_folds.append(
                PreparedInnerFold(
                    validation_indices=validation_idx,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=processor.transform(X.iloc[validation_idx]),
                    weights=self._hierarchical_weights(df.iloc[train_idx]),
                    processor=processor,
                )
            )
        return prepared_folds

    def _inner_cv_score(
        self,
        model_enum: RegressionModels,
        params: dict,
        prepared_folds: list[PreparedInnerFold],
        df: pd.DataFrame,
        outer_fold: int,
    ) -> float:
        """Score one configuration using the cached, fold-specific inputs."""
        inner_predictions = np.full(len(df), np.nan, dtype=float)

        for inner_fold, fold in enumerate(prepared_folds, start=1):
            model = ModelFactory.create_model(model_enum, params=params)
            try:
                model.fit(
                    fold.X_train,
                    fold.y_train,
                    sample_weight=fold.weights,
                )
            except ModelConvergenceError as exc:
                raise ModelConvergenceError(
                    f"{model_enum.name} — outer fold {outer_fold}, "
                    f"inner fold {inner_fold}: {exc}"
                ) from exc

            y_pred, _ = model.predict(fold.X_validation)
            y_pred = fold.processor.inverse_transform_target(y_pred)
            inner_predictions[fold.validation_indices] = (
                y_pred[self.schema.target].to_numpy()
            )

        metrics = self._instance_mir_metrics(
            df=df,
            predictions=inner_predictions,
            seed=self._inner_inference_seed(outer_fold),
        )
        return metrics.mae

    # =========================================================================
    # FINAL FIT WITHIN EACH OUTER FOLD
    # =========================================================================

    def _fit_outer_model(
        self,
        model_enum: RegressionModels,
        params: dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        df_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:
        """
        Fit the selected inner-CV configuration on the complete outer
        training set and predict every individual outer-test recording.
        """

        processor = PostSplitProcessor(
            schema=self.schema,
            feature_set=self.feature_set,
            pca_components=self.pca_components,
        )

        (
            X_train_processed,
            y_train_processed,
        ) = processor.fit_transform(
            X_train,
            y_train,
        )

        X_test_processed = processor.transform(
            X_test
        )

        weights = self._hierarchical_weights(
            df_train
        )

        model = ModelFactory.create_model(
            model_enum,
            params=params,
        )

        fit_time = model.fit(
            X_train_processed,
            y_train_processed,
            sample_weight=weights,
        )

        y_pred, prediction_time = model.predict(
            X_test_processed
        )

        y_pred = processor.inverse_transform_target(
            y_pred
        )

        predictions = y_pred[
            self.schema.target
        ].to_numpy()

        return (
            predictions,
            fit_time,
            prediction_time,
        )

    # =========================================================================
    # HIERARCHICAL TRAINING WEIGHTS
    # =========================================================================

    def _hierarchical_weights(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Give every Point the same total training weight.

        Within each Point:
        - weight is divided equally among CapturePointIds;
        - each CapturePointId weight is divided equally among its recordings.
        """

        n_captures = (
            df.groupby(
                self.schema.group
            )[self.schema.bag]
            .transform("nunique")
            .to_numpy()
        )

        n_instances = (
            df.groupby(
                [
                    self.schema.group,
                    self.schema.bag,
                ]
            )[self.schema.bag]
            .transform("size")
            .to_numpy()
        )

        weights = (
            1.0
            / (
                n_captures
                * n_instances
            )
        )

        return weights / weights.mean()

    # =========================================================================
    # INSTANCE-MIR INFERENCE
    # =========================================================================

    def _instance_mir_metrics(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        seed: int,
    ) -> MIRMetrics:
        """
        Evaluate predictions using the fixed-k instance-MIR rule.

        For each repetition:
        1. randomly select k recordings within every CapturePointId;
        2. average their individual HFI predictions;
        3. evaluate CapturePointId predictions with Point-balanced weights.

        The returned metrics are averages across the repeated random samples.
        """

        inference_df = df[
            [
                self.schema.group,
                self.schema.bag,
                self.schema.target,
            ]
        ].copy()

        inference_df["prediction"] = (
            predictions
        )

        repeated_metrics: list[
            MIRMetrics
        ] = []

        for repeat in range(
            self.inference_repeats
        ):
            rng = np.random.default_rng(
                seed + repeat
            )

            capture_rows: list[dict] = []

            for (
                point,
                capture,
            ), capture_df in inference_df.groupby(
                [
                    self.schema.group,
                    self.schema.bag,
                ],
                sort=False,
            ):
                selected_positions = rng.choice(
                    len(capture_df),
                    size=self.inference_k,
                    replace=False,
                )

                selected = capture_df.iloc[
                    selected_positions
                ]

                capture_rows.append(
                    {
                        self.schema.group: point,
                        self.schema.bag: capture,
                        "observed": (
                            selected[
                                self.schema.target
                            ].mean()
                        ),
                        "prediction": (
                            selected[
                                "prediction"
                            ].mean()
                        ),
                    }
                )

            capture_predictions = (
                pd.DataFrame(
                    capture_rows
                )
            )

            point_weights = (
                1.0
                / capture_predictions.groupby(
                    self.schema.group
                )[self.schema.bag]
                .transform("nunique")
                .to_numpy()
            )

            observed = capture_predictions[
                "observed"
            ].to_numpy()

            predicted = capture_predictions[
                "prediction"
            ].to_numpy()

            repeated_metrics.append(
                MIRMetrics(
                    mae=mean_absolute_error(
                        observed,
                        predicted,
                        sample_weight=point_weights,
                    ),
                    rmse=root_mean_squared_error(
                        observed,
                        predicted,
                        sample_weight=point_weights,
                    ),
                    r2=r2_score(
                        observed,
                        predicted,
                        sample_weight=point_weights,
                    ),
                )
            )

        return MIRMetrics(
            mae=float(
                np.mean(
                    [
                        metric.mae
                        for metric
                        in repeated_metrics
                    ]
                )
            ),
            rmse=float(
                np.mean(
                    [
                        metric.rmse
                        for metric
                        in repeated_metrics
                    ]
                )
            ),
            r2=float(
                np.mean(
                    [
                        metric.r2
                        for metric
                        in repeated_metrics
                    ]
                )
            ),
        )

    # =========================================================================
    # OOF PREDICTIONS
    # =========================================================================

    def _oof_dataframe(
        self,
        df_test: pd.DataFrame,
        predictions: np.ndarray,
        model_enum: RegressionModels,
        outer_fold: int,
    ) -> pd.DataFrame:
        """
        Store individual recording-level outer-fold predictions.

        No prediction aggregation is performed here. These predictions are
        retained for the later k=1,...,K accumulation experiment.
        """

        columns = [
            self.schema.group,
            self.schema.bag,
            *self.schema.instance_columns,
            self.schema.target,
        ]

        result = df_test[
            columns
        ].copy()

        result["prediction"] = predictions
        result["outer_fold"] = outer_fold
        result["model"] = model_enum.name
        result["feature_set"] = (
            self.feature_set
        )

        return result

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def _summarize_results(
        self,
        outer_metrics: pd.DataFrame,
        oof_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize outer-fold performance and calculate pooled OOF
        instance-MIR performance for each model.
        """

        rows: list[dict] = []

        for model_enum in self.models:
            model_name = model_enum.name

            fold_data = outer_metrics[
                outer_metrics["model"]
                == model_name
            ]

            model_oof = oof_predictions[
                oof_predictions["model"]
                == model_name
            ]

            pooled_metrics = (
                self._instance_mir_metrics(
                    df=model_oof,
                    predictions=model_oof[
                        "prediction"
                    ].to_numpy(),
                    seed=self._pooled_inference_seed(),
                )
            )

            rows.append(
                {
                    "model": model_name,
                    "feature_set": (
                        self.feature_set
                    ),
                    "MAE_mean": (
                        fold_data["MAE"].mean()
                    ),
                    "MAE_std": (
                        fold_data["MAE"].std()
                    ),
                    "RMSE_mean": (
                        fold_data["RMSE"].mean()
                    ),
                    "RMSE_std": (
                        fold_data["RMSE"].std()
                    ),
                    "R2_mean": (
                        fold_data["R2"].mean()
                    ),
                    "R2_std": (
                        fold_data["R2"].std()
                    ),
                    "OOF_MAE": (
                        pooled_metrics.mae
                    ),
                    "OOF_RMSE": (
                        pooled_metrics.rmse
                    ),
                    "OOF_R2": (
                        pooled_metrics.r2
                    ),
                    "fit_time_mean": (
                        fold_data[
                            "fit_time"
                        ].mean()
                    ),
                    "prediction_time_mean": (
                        fold_data[
                            "prediction_time"
                        ].mean()
                    ),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                "OOF_MAE",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def _save_results(
        self,
        outer_metrics: pd.DataFrame,
        best_params: pd.DataFrame,
        oof_predictions: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> None:
        """
        Save the results required for model comparison and later
        accumulation analysis.
        """

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        summary.to_csv(
            os.path.join(
                self.output_dir,
                "eval_summary.csv",
            ),
            index=False,
        )

        outer_metrics.to_csv(
            os.path.join(
                self.output_dir,
                "outer_fold_metrics.csv",
            ),
            index=False,
        )

        best_params.to_csv(
            os.path.join(
                self.output_dir,
                "best_params.csv",
            ),
            index=False,
        )

        oof_predictions.to_csv(
            os.path.join(
                self.output_dir,
                "oof_predictions.csv",
            ),
            index=False,
        )

        logger.info(
            "Evaluation results saved to %s",
            self.output_dir,
        )

    # =========================================================================
    # REPRODUCIBLE RANDOM SEEDS
    # =========================================================================

    def _inner_cv_seed(
        self,
        outer_fold: int,
    ) -> int:
        return (
            self.random_state
            + 1_000
            + outer_fold
        )

    def _inner_inference_seed(
        self,
        outer_fold: int,
    ) -> int:
        """
        Same sampling seed for every Optuna trial within an outer fold.

        This ensures hyperparameter configurations are compared using
        exactly the same sampled recordings.
        """
        return (
            self.random_state
            + 10_000
            + outer_fold
        )

    def _outer_inference_seed(
        self,
        outer_fold: int,
    ) -> int:
        """
        Same outer-test sampling for every model.
        """
        return (
            self.random_state
            + 20_000
            + outer_fold
        )

    def _pooled_inference_seed(
        self,
    ) -> int:
        """
        Same pooled-OOF sampling for every model.
        """
        return (
            self.random_state
            + 30_000
        )

    def _optuna_seed(
        self,
        model_enum: RegressionModels,
        outer_fold: int,
    ) -> int:
        return (
            self.random_state
            + 40_000
            + 100 * model_enum.value
            + outer_fold
        )
