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
from src.core.evaluation.plots import save_evaluation_plots
from src.core.models.definitions import ModelConvergenceError
from src.core.models.factory import (
    ModelFactory,
    RegressionModels,
)
from src.core.models.search_space import suggest_parameters
from src.core.processors.postsplit import PostSplitProcessor


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class RegressionMetrics:
    """
    Regression metrics calculated at CapturePointId level.
    """

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class PreparedInnerFold:
    """
    Preprocessed inner fold reused across Optuna trials.
    """

    validation_indices: np.ndarray
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_validation: pd.DataFrame
    weights: np.ndarray
    processor: PostSplitProcessor


# =============================================================================
# MODEL EVALUATOR
# =============================================================================


class ModelEvaluator:
    """
    Nested grouped cross-validation for aggregated acoustic signatures.

    Experimental structure
    ----------------------
    - one dataframe row = one CapturePointId;
    - target = meanHFI;
    - outer CV grouped by Point;
    - inner CV grouped by Point;
    - Optuna selects hyperparameters using inner OOF predictions;
    - every Point receives the same total weight.
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

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def evaluate(
        self,
        df_processed: pd.DataFrame,
    ) -> None:
        """
        Evaluate all selected models using nested GroupKFold.
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
            "Starting evaluation: "
            "%d CapturePointIds, %d Points, feature_set=%s",
            len(df_processed),
            df_processed[
                self.schema.group
            ].nunique(),
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

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            df_train = df.iloc[
                train_idx
            ]

            df_test = df.iloc[
                test_idx
            ]

            best_params = (
                self._optimize_hyperparameters(
                    model_enum=model_enum,
                    X=X_train,
                    y=y_train,
                    df=df_train,
                    outer_fold=outer_fold,
                )
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
                    f"{model_enum.name} — outer fold "
                    f"{outer_fold}: selected configuration "
                    "did not converge on the outer "
                    "training data."
                )

                logger.error(
                    "%s %s",
                    message,
                    exc,
                )

                raise ModelConvergenceError(
                    message
                ) from exc

            metrics = self._regression_metrics(
                df=df_test,
                predictions=predictions,
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
                    "n_train_captures": (
                        len(df_train)
                    ),
                    "n_test_captures": (
                        len(df_test)
                    ),
                    "MAE": metrics.mae,
                    "RMSE": metrics.rmse,
                    "R2": metrics.r2,
                    "fit_time": fit_time,
                    "prediction_time": (
                        prediction_time
                    ),
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
    # OPTUNA
    # =========================================================================

    def _optimize_hyperparameters(
        self,
        model_enum: RegressionModels,
        X: pd.DataFrame,
        y: pd.DataFrame,
        df: pd.DataFrame,
        outer_fold: int,
    ) -> dict:
        """
        Select hyperparameters using inner grouped cross-validation.

        Preprocessing is fitted once for each inner fold and reused
        across Optuna trials.
        """

        prepared_folds = (
            self._prepare_inner_folds(
                X=X,
                y=y,
                df=df,
                outer_fold=outer_fold,
            )
        )

        sampler = optuna.samplers.TPESampler(
            seed=self._optuna_seed(
                model_enum,
                outer_fold,
            )
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        def objective(
            trial: optuna.Trial,
        ) -> float:

            params = suggest_parameters(
                trial,
                model_enum,
            )

            trial.set_user_attr(
                "model_params",
                params,
            )

            try:
                return self._inner_cv_score(
                    model_enum=model_enum,
                    params=params,
                    prepared_folds=prepared_folds,
                    df=df,
                    outer_fold=outer_fold,
                )

            except ModelConvergenceError as exc:
                trial.set_user_attr(
                    "convergence_failure",
                    str(exc),
                )

                raise

        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            n_jobs=1,
            show_progress_bar=False,
            catch=(ModelConvergenceError,),
        )

        completed = sum(
            trial.state
            == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )

        convergence_failures = sum(
            "convergence_failure"
            in trial.user_attrs
            for trial in study.trials
        )

        logger.info(
            "%s — outer fold %d — trials: "
            "%d completed, %d convergence failures, "
            "%d total",
            model_enum.name,
            outer_fold,
            completed,
            convergence_failures,
            len(study.trials),
        )

        if completed == 0:
            message = (
                f"{model_enum.name} — outer fold "
                f"{outer_fold}: no completed Optuna "
                f"trials out of {len(study.trials)}."
            )

            logger.error(
                message
            )

            raise RuntimeError(
                message
            )

        best_params = (
            study.best_trial.user_attrs[
                "model_params"
            ]
        )

        logger.info(
            "%s — outer fold %d — "
            "best inner MAE: %.4f — params: %s",
            model_enum.name,
            outer_fold,
            study.best_value,
            best_params,
        )

        return best_params

    # =========================================================================
    # INNER CROSS-VALIDATION
    # =========================================================================

    def _prepare_inner_folds(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        df: pd.DataFrame,
        outer_fold: int,
    ) -> list[PreparedInnerFold]:
        """
        Fit fold-specific preprocessing once before Optuna trials.
        """

        inner_cv = GroupKFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=self._inner_cv_seed(
                outer_fold
            ),
        )

        prepared_folds: list[
            PreparedInnerFold
        ] = []

        groups = df[
            self.schema.group
        ].to_numpy()

        for (
            train_idx,
            validation_idx,
        ) in inner_cv.split(
            X,
            y,
            groups=groups,
        ):

            processor = PostSplitProcessor(
                schema=self.schema,
                feature_set=self.feature_set,
                pca_components=self.pca_components,
            )

            (
                X_train_processed,
                y_train_processed,
            ) = processor.fit_transform(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            X_validation_processed = (
                processor.transform(
                    X.iloc[
                        validation_idx
                    ]
                )
            )

            prepared_folds.append(
                PreparedInnerFold(
                    validation_indices=(
                        validation_idx
                    ),
                    X_train=(
                        X_train_processed
                    ),
                    y_train=(
                        y_train_processed
                    ),
                    X_validation=(
                        X_validation_processed
                    ),
                    weights=self._point_weights(
                        df.iloc[train_idx]
                    ),
                    processor=processor,
                )
            )

        return prepared_folds

    def _inner_cv_score(
        self,
        model_enum: RegressionModels,
        params: dict,
        prepared_folds: list[
            PreparedInnerFold
        ],
        df: pd.DataFrame,
        outer_fold: int,
    ) -> float:
        """
        Score one Optuna configuration using complete inner OOF predictions.
        """

        inner_predictions = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        for inner_fold, fold in enumerate(
            prepared_folds,
            start=1,
        ):

            model = ModelFactory.create_model(
                model_enum,
                params=params,
            )

            try:
                model.fit(
                    fold.X_train,
                    fold.y_train,
                    sample_weight=fold.weights,
                )

            except ModelConvergenceError as exc:
                raise ModelConvergenceError(
                    f"{model_enum.name} — outer fold "
                    f"{outer_fold}, inner fold "
                    f"{inner_fold}: {exc}"
                ) from exc

            y_pred, _ = model.predict(
                fold.X_validation
            )

            y_pred = (
                fold.processor
                .inverse_transform_target(
                    y_pred
                )
            )

            inner_predictions[
                fold.validation_indices
            ] = y_pred[
                self.schema.target
            ].to_numpy()

        metrics = self._regression_metrics(
            df=df,
            predictions=inner_predictions,
        )

        return metrics.mae

    # =========================================================================
    # OUTER-FOLD FINAL FIT
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
        Fit the selected configuration on all outer-training
        CapturePointIds and predict the outer-test CapturePointIds.
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

        X_test_processed = (
            processor.transform(
                X_test
            )
        )

        weights = self._point_weights(
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

        (
            y_pred,
            prediction_time,
        ) = model.predict(
            X_test_processed
        )

        y_pred = (
            processor.inverse_transform_target(
                y_pred
            )
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
    # POINT-BALANCED WEIGHTS
    # =========================================================================

    def _point_weights(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Give every physical Point the same total weight.

        Since each row represents one CapturePointId, the weight of each
        row is divided by the number of CapturePointIds in that Point.
        """

        n_captures = (
            df.groupby(
                self.schema.group
            )[self.schema.bag]
            .transform("nunique")
            .to_numpy()
        )

        weights = (
            1.0 / n_captures
        )

        return (
            weights / weights.mean()
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    def _regression_metrics(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> RegressionMetrics:
        """
        Calculate Point-balanced metrics at CapturePointId level.
        """

        observed = df[
            self.schema.target
        ].to_numpy()

        weights = self._point_weights(
            df
        )

        return RegressionMetrics(
            mae=mean_absolute_error(
                observed,
                predictions,
                sample_weight=weights,
            ),
            rmse=root_mean_squared_error(
                observed,
                predictions,
                sample_weight=weights,
            ),
            r2=r2_score(
                observed,
                predictions,
                sample_weight=weights,
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
        Store one outer OOF prediction per CapturePointId.
        """

        result = df_test[
            [
                self.schema.group,
                self.schema.bag,
                self.schema.target,
            ]
        ].copy()

        result["prediction"] = (
            predictions
        )

        result["outer_fold"] = (
            outer_fold
        )

        result["model"] = (
            model_enum.name
        )

        result["feature_set"] = (
            self.feature_set
        )

        return result

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================

    def _summarize_results(
        self,
        outer_metrics: pd.DataFrame,
        oof_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize outer-fold metrics and pooled OOF performance.
        """

        rows: list[dict] = []

        for model_enum in self.models:

            model_name = (
                model_enum.name
            )

            fold_data = outer_metrics[
                outer_metrics["model"]
                == model_name
            ]

            model_oof = oof_predictions[
                oof_predictions["model"]
                == model_name
            ]

            pooled_metrics = (
                self._regression_metrics(
                    df=model_oof,
                    predictions=model_oof[
                        "prediction"
                    ].to_numpy(),
                )
            )

            rows.append(
                {
                    "model": model_name,
                    "feature_set": (
                        self.feature_set
                    ),

                    "MAE_mean": (
                        fold_data[
                            "MAE"
                        ].mean()
                    ),
                    "MAE_std": (
                        fold_data[
                            "MAE"
                        ].std()
                    ),

                    "RMSE_mean": (
                        fold_data[
                            "RMSE"
                        ].mean()
                    ),
                    "RMSE_std": (
                        fold_data[
                            "RMSE"
                        ].std()
                    ),

                    "R2_mean": (
                        fold_data[
                            "R2"
                        ].mean()
                    ),
                    "R2_std": (
                        fold_data[
                            "R2"
                        ].std()
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
            pd.DataFrame(
                rows
            )
            .sort_values(
                "OOF_MAE",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    def _save_results(
        self,
        outer_metrics: pd.DataFrame,
        best_params: pd.DataFrame,
        oof_predictions: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> None:
        """
        Save all model-selection outputs.
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
    # RANDOM SEEDS
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
