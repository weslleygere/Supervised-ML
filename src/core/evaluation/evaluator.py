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
from src.core.models.definitions import (
    ModelConvergenceError,
)
from src.core.models.factory import (
    ModelFactory,
    RegressionModels,
)
from src.core.models.search_space import (
    suggest_pipeline_configuration,
)
from src.core.processors.postsplit import (
    PostSplitProcessor,
)


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class CVSplit:
    """
    One grouped train/validation partition.

    CapturePointIds are stored explicitly, while grouping independence is
    enforced at the Point level.
    """

    repeat: int
    fold: int

    train_bags: tuple
    validation_bags: tuple

    n_train_points: int
    n_validation_points: int


# =============================================================================
# MODEL EVALUATOR
# =============================================================================


class ModelEvaluator:
    """
    Select and independently evaluate one complete ML pipeline.

    Experimental design
    -------------------
    1. Split independent Points once into development and final test sets.

    2. Keep the final test set completely isolated during pipeline selection.

    3. Precompute repeated grouped cross-validation partitions using only
       development Points.

    4. Run one conditional Optuna CASH study.

       Every trial jointly defines:

           feature representation
           acoustic aggregation
           dimensionality reduction
           reduction dimensionality, when applicable
           model family
           model hyperparameters

    5. Evaluate every candidate using exactly the same repeated CV
       partitions.

    6. For each CV repetition, pool all out-of-fold predictions and calculate
       Point-balanced regression metrics.

    7. Select the pipeline with the lowest mean repeated OOF MAE.

    8. Reconstruct OOF predictions only for the selected pipeline.

    9. Fit the selected pipeline on the complete development set.

    10. Evaluate it once on the isolated final test set.

    11. Save scientific outputs and reproducibility manifests.
    """

    def __init__(
        self,
        schema: Schema,
        output_dir: str,
        models: list[RegressionModels],
        random_state: int,
        feature_sets: tuple[str, ...],
        aggregations: tuple[str, ...],
        reductions: tuple[str, ...],
        pca_indices_candidates: tuple[int, ...],
        pca_embeddings_candidates: tuple[int, ...],
        test_size: float,
        cv_splits: int,
        cv_repeats: int,
        optuna_trials: int,
        selection_indices_candidates: tuple[int, ...] | None = None,
        selection_embeddings_candidates: tuple[int, ...] | None = None,
    ) -> None:

        self.schema = schema
        self.output_dir = output_dir

        self.models = models
        self.random_state = random_state

        self.feature_sets = feature_sets
        self.aggregations = aggregations
        self.reductions = reductions

        self.pca_indices_candidates = (
            pca_indices_candidates
        )

        self.pca_embeddings_candidates = (
            pca_embeddings_candidates
        )

        self.selection_indices_candidates = (
            selection_indices_candidates
            if selection_indices_candidates is not None
            else pca_indices_candidates
        )

        self.selection_embeddings_candidates = (
            selection_embeddings_candidates
            if selection_embeddings_candidates is not None
            else pca_embeddings_candidates
        )

        self.test_size = test_size

        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats

        self.optuna_trials = optuna_trials

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def evaluate(
        self,
        signatures: dict[str, pd.DataFrame],
    ) -> dict:
        """
        Select a complete pipeline on development data and evaluate it once
        on the independent final test set.
        """

        # =====================================================================
        # INPUT VALIDATION
        # =====================================================================

        self._check_signature_alignment(
            signatures
        )

        reference = signatures[
            self.aggregations[
                0
            ]
        ]

        feature_dimensions = (
            self._feature_dimensions(
                signatures
            )
        )

        logger.info(
            "Dataset contains %d CapturePointIds across %d Points",
            len(
                reference
            ),
            reference[
                self.schema.group
            ].nunique(),
        )

        # =====================================================================
        # DEVELOPMENT / FINAL TEST
        # =====================================================================

        (
            development_bags,
            test_bags,
            split_manifest,
        ) = self._make_development_test_split(
            reference
        )

        development_reference = (
            self._select_bags(
                reference,
                development_bags,
            )
        )

        test_reference = (
            self._select_bags(
                reference,
                test_bags,
            )
        )

        n_development_points = (
            development_reference[
                self.schema.group
            ]
            .nunique()
        )

        n_test_points = (
            test_reference[
                self.schema.group
            ]
            .nunique()
        )

        if (
            n_development_points
            < self.cv_splits
        ):
            raise ValueError(
                "Development set contains fewer Points "
                "than CV_SPLITS. "
                f"Development Points: {n_development_points}; "
                f"CV_SPLITS: {self.cv_splits}."
            )

        logger.info(
            "Development/test split: "
            "%d development Points, %d final-test Points",
            n_development_points,
            n_test_points,
        )

        # =====================================================================
        # FIXED REPEATED GROUPED CV
        # =====================================================================

        cv_splits = (
            self._make_repeated_cv_splits(
                development_reference
            )
        )

        cv_split_manifest = (
            self._cv_split_manifest(
                development_reference,
                cv_splits,
            )
        )

        min_cv_train_size = min(
            len(
                split.train_bags
            )
            for split in cv_splits
        )

        min_cv_train_points = min(
            split.n_train_points
            for split in cv_splits
        )

        logger.info(
            "Model-selection CV: %d repeats x %d folds "
            "(minimum training set: %d CapturePointIds / %d Points)",
            self.cv_repeats,
            self.cv_splits,
            min_cv_train_size,
            min_cv_train_points,
        )

        # =====================================================================
        # DEVELOPMENT-CV BASELINES
        # =====================================================================

        baseline_cv = (
            self._evaluate_cv_baselines(
                reference=(
                    development_reference
                ),
                cv_splits=(
                    cv_splits
                ),
            )
        )

        baseline_repeat_metrics = (
            baseline_cv[
                "repeat_metrics"
            ]
        )

        baseline_oof_predictions = (
            baseline_cv[
                "oof_predictions"
            ]
        )

        # =====================================================================
        # OPTUNA STORAGE
        # =====================================================================

        trial_repeat_rows: list[
            dict
        ] = []

        # =====================================================================
        # SINGLE CASH STUDY
        # =====================================================================

        sampler = (
            optuna.samplers.TPESampler(
                seed=(
                    self.random_state
                ),
            )
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        def objective(
            trial: optuna.Trial,
        ) -> float:
            """
            Evaluate one complete fixed pipeline candidate.
            """

            configuration = (
                suggest_pipeline_configuration(
                    trial=trial,
                    models=(
                        self.models
                    ),
                    feature_sets=(
                        self.feature_sets
                    ),
                    aggregations=(
                        self.aggregations
                    ),
                    reductions=(
                        self.reductions
                    ),
                    pca_indices_candidates=(
                        self.pca_indices_candidates
                    ),
                    pca_embeddings_candidates=(
                        self.pca_embeddings_candidates
                    ),
                    selection_indices_candidates=(
                        self.selection_indices_candidates
                    ),
                    selection_embeddings_candidates=(
                        self.selection_embeddings_candidates
                    ),
                    feature_dimensions=(
                        feature_dimensions
                    ),
                    min_cv_train_size=(
                        min_cv_train_size
                    ),
                )
            )

            serializable_configuration = (
                self._serializable_config(
                    configuration
                )
            )

            trial.set_user_attr(
                "configuration",
                serializable_configuration,
            )

            raw_feature_count = (
                self._raw_feature_count(
                    configuration=(
                        configuration
                    ),
                    feature_dimensions=(
                        feature_dimensions
                    ),
                )
            )

            model_feature_count = (
                self._model_feature_count(
                    configuration=(
                        configuration
                    ),
                    feature_dimensions=(
                        feature_dimensions
                    ),
                )
            )

            trial.set_user_attr(
                "raw_feature_count",
                int(
                    raw_feature_count
                ),
            )

            trial.set_user_attr(
                "model_feature_count",
                int(
                    model_feature_count
                ),
            )

            trial.set_user_attr(
                "min_cv_train_points",
                int(
                    min_cv_train_points
                ),
            )

            try:

                result = (
                    self._evaluate_configuration(
                        signatures=(
                            signatures
                        ),
                        configuration=(
                            configuration
                        ),
                        cv_splits=(
                            cv_splits
                        ),
                        trial_number=(
                            trial.number
                        ),
                    )
                )

            except ModelConvergenceError as exc:

                trial.set_user_attr(
                    "failure_reason",
                    str(
                        exc
                    ),
                )

                raise

            summary = result[
                "summary"
            ]

            for key, attr in (
                (
                    "CV_MAE",
                    "cv_mae",
                ),
                (
                    "CV_MAE_std",
                    "cv_mae_std",
                ),
                (
                    "CV_RMSE",
                    "cv_rmse",
                ),
                (
                    "CV_RMSE_std",
                    "cv_rmse_std",
                ),
                (
                    "CV_R2",
                    "cv_r2",
                ),
                (
                    "CV_R2_std",
                    "cv_r2_std",
                ),
                (
                    "mean_fit_time",
                    "mean_fit_time",
                ),
                (
                    "mean_prediction_time",
                    "mean_prediction_time",
                ),
            ):

                trial.set_user_attr(
                    attr,
                    float(
                        summary[
                            key
                        ]
                    ),
                )

            trial_repeat_rows.extend(
                result[
                    "repeat_metrics"
                ]
            )

            logger.info(
                "Trial %d complete — %s — "
                "features=%s, aggregation=%s, reduction=%s — "
                "CV MAE %.4f",
                trial.number,
                configuration[
                    "model"
                ].name,
                configuration[
                    "feature_set"
                ],
                configuration[
                    "aggregation"
                ],
                configuration[
                    "reduction"
                ],
                summary[
                    "CV_MAE"
                ],
            )

            return float(
                summary[
                    "CV_MAE"
                ]
            )

        logger.info(
            "Starting Optuna complete-pipeline search: %d trials",
            self.optuna_trials,
        )

        study.optimize(
            objective,
            n_trials=(
                self.optuna_trials
            ),
            n_jobs=1,
            show_progress_bar=False,
            catch=(
                ModelConvergenceError,
            ),
        )

        # =====================================================================
        # BEST TRIAL
        # =====================================================================

        completed_trials = [
            trial
            for trial in study.trials
            if (
                trial.state
                == optuna.trial.TrialState.COMPLETE
            )
        ]

        if not completed_trials:
            raise RuntimeError(
                "No Optuna trial completed successfully."
            )

        best_trial = (
            study.best_trial
        )

        best_configuration = (
            self._configuration_from_serialized(
                best_trial.user_attrs[
                    "configuration"
                ]
            )
        )

        logger.info(
            "Selected trial %d — %s — CV MAE %.4f",
            best_trial.number,
            best_configuration[
                "model"
            ].name,
            best_trial.value,
        )

        # =====================================================================
        # COMPLETE TRIAL TABLE
        # =====================================================================

        trial_repeat_metrics = (
            pd.DataFrame(
                trial_repeat_rows
            )
        )

        model_selection_trials = (
            self._build_model_selection_trials(
                study=(
                    study
                ),
                repeat_metrics=(
                    trial_repeat_metrics
                ),
            )
        )

        # =====================================================================
        # SELECTED PIPELINE CV METRICS
        # =====================================================================

        selected_repeat_metrics = (
            trial_repeat_metrics[
                trial_repeat_metrics[
                    "trial"
                ]
                == best_trial.number
            ]
            .copy()
        )

        # =====================================================================
        # RECONSTRUCT SELECTED PIPELINE OOF PREDICTIONS
        # =====================================================================

        logger.info(
            "Reconstructing OOF predictions for selected trial %d "
            "using the frozen CV partitions.",
            best_trial.number,
        )

        selected_cv_result = (
            self._evaluate_configuration(
                signatures=(
                    signatures
                ),
                configuration=(
                    best_configuration
                ),
                cv_splits=(
                    cv_splits
                ),
                trial_number=(
                    best_trial.number
                ),
            )
        )

        reconstructed_cv_mae = (
            selected_cv_result[
                "summary"
            ][
                "CV_MAE"
            ]
        )

        logger.info(
            "Selected trial OOF reconstruction — "
            "original CV MAE %.6f, reconstructed CV MAE %.6f",
            best_trial.value,
            reconstructed_cv_mae,
        )

        selected_model_oof = (
            selected_cv_result[
                "oof_predictions"
            ]
            .copy()
        )

        selected_cv_predictions = (
            self._build_selected_cv_predictions(
                selected_model_oof=(
                    selected_model_oof
                ),
                baseline_oof=(
                    baseline_oof_predictions
                ),
            )
        )

        # =====================================================================
        # FINAL FIT
        # =====================================================================

        final_result = (
            self._fit_final_pipeline(
                signatures=(
                    signatures
                ),
                development_bags=(
                    development_bags
                ),
                test_bags=(
                    test_bags
                ),
                configuration=(
                    best_configuration
                ),
            )
        )

        selected_test_metrics = (
            final_result[
                "metrics"
            ]
        )

        # =====================================================================
        # FINAL-TEST BASELINES
        # =====================================================================

        (
            mean_baseline,
            median_baseline,
        ) = (
            self._baseline_values(
                development_reference
            )
        )

        mean_predictions = np.full(
            len(
                test_reference
            ),
            mean_baseline,
            dtype=float,
        )

        median_predictions = np.full(
            len(
                test_reference
            ),
            median_baseline,
            dtype=float,
        )

        mean_test_metrics = (
            self._regression_metrics(
                df=(
                    test_reference
                ),
                predictions=(
                    mean_predictions
                ),
            )
        )

        median_test_metrics = (
            self._regression_metrics(
                df=(
                    test_reference
                ),
                predictions=(
                    median_predictions
                ),
            )
        )

        final_test_predictions = (
            self._build_final_test_predictions(
                model_predictions=(
                    final_result[
                        "predictions"
                    ]
                ),
                mean_baseline=(
                    mean_baseline
                ),
                median_baseline=(
                    median_baseline
                ),
            )
        )

        # =====================================================================
        # PERFORMANCE SUMMARY
        # =====================================================================

        performance_summary = (
            self._build_performance_summary(
                selected_repeat_metrics=(
                    selected_repeat_metrics
                ),
                baseline_repeat_metrics=(
                    baseline_repeat_metrics
                ),
                selected_test_metrics=(
                    selected_test_metrics
                ),
                mean_test_metrics=(
                    mean_test_metrics
                ),
                median_test_metrics=(
                    median_test_metrics
                ),
                development_reference=(
                    development_reference
                ),
                test_reference=(
                    test_reference
                ),
            )
        )

        # =====================================================================
        # SELECTED PIPELINE
        # =====================================================================

        selected_pipeline = {
            "selected_trial":
                int(
                    best_trial.number
                ),

            "selection_objective":
                {
                    "metric":
                        "point_balanced_MAE",

                    "CV_MAE":
                        float(
                            best_trial.value
                        ),

                    "CV_MAE_std":
                        float(
                            best_trial.user_attrs.get(
                                "cv_mae_std",
                                np.nan,
                            )
                        ),
                },

            "pipeline":
                self._serializable_config(
                    best_configuration
                ),

            "raw_feature_count":
                int(
                    best_trial.user_attrs[
                        "raw_feature_count"
                    ]
                ),

            "model_feature_count":
                int(
                    best_trial.user_attrs[
                        "model_feature_count"
                    ]
                ),

            "final_fit_time":
                float(
                    final_result[
                        "fit_time"
                    ]
                ),

            "final_prediction_time":
                float(
                    final_result[
                        "prediction_time"
                    ]
                ),
        }

        # =====================================================================
        # SAVE SCIENTIFIC OUTPUTS
        # =====================================================================

        self._save_results(
            performance_summary=(
                performance_summary
            ),
            model_selection_trials=(
                model_selection_trials
            ),
            selected_cv_predictions=(
                selected_cv_predictions
            ),
            final_test_predictions=(
                final_test_predictions
            ),
            selected_pipeline=(
                selected_pipeline
            ),
            split_manifest=(
                split_manifest
            ),
            cv_split_manifest=(
                cv_split_manifest
            ),
        )

        logger.info(
            "Final independent test performance — "
            "MAE %.4f, RMSE %.4f, R2 %.4f",
            selected_test_metrics.mae,
            selected_test_metrics.rmse,
            selected_test_metrics.r2,
        )

        return {
            "study":
                study,

            "best_configuration":
                best_configuration,

            "selected_pipeline":
                selected_pipeline,

            "final_selection":
                selected_pipeline,

            "performance_summary":
                performance_summary,
        }

    # =========================================================================
    # DEVELOPMENT / TEST SPLIT
    # =========================================================================

    def _make_development_test_split(
        self,
        reference: pd.DataFrame,
    ) -> tuple[
        tuple,
        tuple,
        pd.DataFrame,
    ]:

        points = np.array(
            sorted(
                reference[
                    self.schema.group
                ]
                .drop_duplicates()
                .tolist(),
                key=str,
            ),
            dtype=object,
        )

        n_points = len(
            points
        )

        if n_points < 2:

            raise ValueError(
                "At least two independent Points are required."
            )

        n_test_points = int(
            round(
                n_points
                * self.test_size
            )
        )

        n_test_points = max(
            1,
            n_test_points,
        )

        if (
            n_test_points
            >= n_points
        ):

            raise ValueError(
                "TEST_SIZE leaves no Points for development."
            )

        rng = np.random.default_rng(
            self.random_state
        )

        shuffled_points = (
            rng.permutation(
                points
            )
        )

        test_points = set(
            shuffled_points[
                :n_test_points
            ].tolist()
        )

        development_points = set(
            shuffled_points[
                n_test_points:
            ].tolist()
        )

        development = reference[
            reference[
                self.schema.group
            ].isin(
                development_points
            )
        ]

        test = reference[
            reference[
                self.schema.group
            ].isin(
                test_points
            )
        ]

        development_bags = tuple(
            development[
                self.schema.bag
            ].tolist()
        )

        test_bags = tuple(
            test[
                self.schema.bag
            ].tolist()
        )

        manifest = reference[
            [
                self.schema.group,
                self.schema.bag,
                self.schema.target,
            ]
        ].copy()

        manifest[
            "split"
        ] = np.where(
            manifest[
                self.schema.group
            ].isin(
                test_points
            ),
            "test",
            "development",
        )

        manifest = (
            manifest.sort_values(
                [
                    "split",
                    self.schema.group,
                    self.schema.bag,
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return (
            development_bags,
            test_bags,
            manifest,
        )

    # =========================================================================
    # REPEATED GROUPED CV
    # =========================================================================

    def _make_repeated_cv_splits(
        self,
        development_reference: pd.DataFrame,
    ) -> list[CVSplit]:

        groups = (
            development_reference[
                self.schema.group
            ]
            .to_numpy()
        )

        splits: list[
            CVSplit
        ] = []

        for repeat in range(
            1,
            self.cv_repeats + 1,
        ):

            cv = GroupKFold(
                n_splits=(
                    self.cv_splits
                ),
                shuffle=True,
                random_state=(
                    self._cv_seed(
                        repeat
                    )
                ),
            )

            for fold, (
                train_idx,
                validation_idx,
            ) in enumerate(
                cv.split(
                    development_reference,
                    groups=groups,
                ),
                start=1,
            ):

                train = (
                    development_reference.iloc[
                        train_idx
                    ]
                )

                validation = (
                    development_reference.iloc[
                        validation_idx
                    ]
                )

                train_points = set(
                    train[
                        self.schema.group
                    ]
                )

                validation_points = set(
                    validation[
                        self.schema.group
                    ]
                )

                if not (
                    train_points.isdisjoint(
                        validation_points
                    )
                ):

                    raise RuntimeError(
                        "Point leakage detected between "
                        "CV training and validation sets."
                    )

                splits.append(
                    CVSplit(
                        repeat=repeat,
                        fold=fold,
                        train_bags=tuple(
                            train[
                                self.schema.bag
                            ].tolist()
                        ),
                        validation_bags=tuple(
                            validation[
                                self.schema.bag
                            ].tolist()
                        ),
                        n_train_points=len(
                            train_points
                        ),
                        n_validation_points=len(
                            validation_points
                        ),
                    )
                )

        return splits

    # =========================================================================
    # CANDIDATE EVALUATION
    # =========================================================================

    def _evaluate_configuration(
        self,
        signatures: dict[
            str,
            pd.DataFrame,
        ],
        configuration: dict,
        cv_splits: list[CVSplit],
        trial_number: int,
    ) -> dict:

        df = signatures[
            configuration[
                "aggregation"
            ]
        ]

        repeat_rows = []
        all_oof_frames = []

        fit_times = []
        prediction_times = []

        for repeat in range(
            1,
            self.cv_repeats + 1,
        ):

            repeat_splits = [
                split
                for split in cv_splits
                if (
                    split.repeat
                    == repeat
                )
            ]

            prediction_frames = []

            for split in repeat_splits:

                df_train = (
                    self._select_bags(
                        df,
                        split.train_bags,
                    )
                )

                df_validation = (
                    self._select_bags(
                        df,
                        split.validation_bags,
                    )
                )

                X_train = (
                    df_train.drop(
                        columns=[
                            self.schema.target
                        ]
                    )
                )

                y_train = (
                    df_train[
                        [
                            self.schema.target
                        ]
                    ]
                )

                X_validation = (
                    df_validation.drop(
                        columns=[
                            self.schema.target
                        ]
                    )
                )

                processor = (
                    PostSplitProcessor(
                        schema=(
                            self.schema
                        ),
                        feature_set=(
                            configuration[
                                "feature_set"
                            ]
                        ),
                        reduction=(
                            configuration[
                                "reduction"
                            ]
                        ),
                        pca_indices_components=(
                            configuration[
                                "pca_indices_components"
                            ]
                        ),
                        pca_embeddings_components=(
                            configuration[
                                "pca_embeddings_components"
                            ]
                        ),
                        selection_indices_features=(
                            configuration.get(
                                "selection_indices_features"
                            )
                        ),
                        selection_embeddings_features=(
                            configuration.get(
                                "selection_embeddings_features"
                            )
                        ),
                    )
                )

                (
                    X_train_processed,
                    y_train_processed,
                ) = (
                    processor.fit_transform(
                        X_train,
                        y_train,
                    )
                )

                X_validation_processed = (
                    processor.transform(
                        X_validation
                    )
                )

                model = (
                    ModelFactory.create_model(
                        configuration[
                            "model"
                        ],
                        params=(
                            configuration[
                                "model_params"
                            ]
                        ),
                    )
                )

                fit_time = model.fit(
                    X_train_processed,
                    y_train_processed,
                    sample_weight=(
                        self._point_weights(
                            df_train
                        )
                    ),
                )

                (
                    y_pred,
                    prediction_time,
                ) = model.predict(
                    X_validation_processed
                )

                fit_times.append(
                    fit_time
                )

                prediction_times.append(
                    prediction_time
                )

                y_pred = (
                    processor.inverse_transform_target(
                        y_pred
                    )
                )

                result = (
                    df_validation[
                        [
                            self.schema.group,
                            self.schema.bag,
                            self.schema.target,
                        ]
                    ]
                    .copy()
                )

                result[
                    "prediction"
                ] = (
                    y_pred[
                        self.schema.target
                    ]
                    .to_numpy()
                )

                result[
                    "trial"
                ] = trial_number

                result[
                    "repeat"
                ] = repeat

                result[
                    "fold"
                ] = split.fold

                prediction_frames.append(
                    result
                )

            repeat_oof = pd.concat(
                prediction_frames,
                ignore_index=True,
            )

            repeat_metrics = (
                self._regression_metrics(
                    df=(
                        repeat_oof
                    ),
                    predictions=(
                        repeat_oof[
                            "prediction"
                        ]
                        .to_numpy()
                    ),
                )
            )

            repeat_rows.append(
                {
                    "trial":
                        trial_number,

                    "repeat":
                        repeat,

                    "MAE":
                        repeat_metrics.mae,

                    "RMSE":
                        repeat_metrics.rmse,

                    "R2":
                        repeat_metrics.r2,
                }
            )

            all_oof_frames.append(
                repeat_oof
            )

        repeat_metrics_df = (
            pd.DataFrame(
                repeat_rows
            )
        )

        summary = {
            "CV_MAE":
                float(
                    repeat_metrics_df[
                        "MAE"
                    ].mean()
                ),

            "CV_MAE_std":
                self._sample_sd(
                    repeat_metrics_df[
                        "MAE"
                    ]
                ),

            "CV_RMSE":
                float(
                    repeat_metrics_df[
                        "RMSE"
                    ].mean()
                ),

            "CV_RMSE_std":
                self._sample_sd(
                    repeat_metrics_df[
                        "RMSE"
                    ]
                ),

            "CV_R2":
                float(
                    repeat_metrics_df[
                        "R2"
                    ].mean()
                ),

            "CV_R2_std":
                self._sample_sd(
                    repeat_metrics_df[
                        "R2"
                    ]
                ),

            "mean_fit_time":
                float(
                    np.mean(
                        fit_times
                    )
                ),

            "mean_prediction_time":
                float(
                    np.mean(
                        prediction_times
                    )
                ),
        }

        return {
            "summary":
                summary,

            "repeat_metrics":
                repeat_rows,

            "oof_predictions":
                pd.concat(
                    all_oof_frames,
                    ignore_index=True,
                ),
        }

    # =========================================================================
    # FINAL PIPELINE FIT
    # =========================================================================

    def _fit_final_pipeline(
        self,
        signatures: dict[
            str,
            pd.DataFrame,
        ],
        development_bags: tuple,
        test_bags: tuple,
        configuration: dict,
    ) -> dict:

        df = signatures[
            configuration[
                "aggregation"
            ]
        ]

        df_development = (
            self._select_bags(
                df,
                development_bags,
            )
        )

        df_test = (
            self._select_bags(
                df,
                test_bags,
            )
        )

        X_development = (
            df_development.drop(
                columns=[
                    self.schema.target
                ]
            )
        )

        y_development = (
            df_development[
                [
                    self.schema.target
                ]
            ]
        )

        X_test = (
            df_test.drop(
                columns=[
                    self.schema.target
                ]
            )
        )

        processor = (
            PostSplitProcessor(
                schema=(
                    self.schema
                ),
                feature_set=(
                    configuration[
                        "feature_set"
                    ]
                ),
                reduction=(
                    configuration[
                        "reduction"
                    ]
                ),
                pca_indices_components=(
                    configuration[
                        "pca_indices_components"
                    ]
                ),
                pca_embeddings_components=(
                    configuration[
                        "pca_embeddings_components"
                    ]
                ),
                selection_indices_features=(
                    configuration.get(
                        "selection_indices_features"
                    )
                ),
                selection_embeddings_features=(
                    configuration.get(
                        "selection_embeddings_features"
                    )
                ),
            )
        )

        (
            X_development_processed,
            y_development_processed,
        ) = (
            processor.fit_transform(
                X_development,
                y_development,
            )
        )

        X_test_processed = (
            processor.transform(
                X_test
            )
        )

        model = (
            ModelFactory.create_model(
                configuration[
                    "model"
                ],
                params=(
                    configuration[
                        "model_params"
                    ]
                ),
            )
        )

        fit_time = model.fit(
            X_development_processed,
            y_development_processed,
            sample_weight=(
                self._point_weights(
                    df_development
                )
            ),
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

        predictions = (
            df_test[
                [
                    self.schema.group,
                    self.schema.bag,
                    self.schema.target,
                ]
            ]
            .copy()
        )

        predictions[
            "prediction"
        ] = (
            y_pred[
                self.schema.target
            ]
            .to_numpy()
        )

        metrics = (
            self._regression_metrics(
                df=(
                    df_test
                ),
                predictions=(
                    predictions[
                        "prediction"
                    ]
                    .to_numpy()
                ),
            )
        )

        return {
            "predictions":
                predictions,

            "metrics":
                metrics,

            "fit_time":
                float(
                    fit_time
                ),

            "prediction_time":
                float(
                    prediction_time
                ),
        }

    # =========================================================================
    # CV BASELINES
    # =========================================================================

    def _evaluate_cv_baselines(
        self,
        reference: pd.DataFrame,
        cv_splits: list[CVSplit],
    ) -> dict:

        repeat_rows = []
        all_oof_frames = []

        for repeat in range(
            1,
            self.cv_repeats + 1,
        ):

            prediction_frames = []

            repeat_splits = [
                split
                for split in cv_splits
                if (
                    split.repeat
                    == repeat
                )
            ]

            for split in repeat_splits:

                df_train = (
                    self._select_bags(
                        reference,
                        split.train_bags,
                    )
                )

                df_validation = (
                    self._select_bags(
                        reference,
                        split.validation_bags,
                    )
                )

                (
                    mean_value,
                    median_value,
                ) = (
                    self._baseline_values(
                        df_train
                    )
                )

                result = (
                    df_validation[
                        [
                            self.schema.group,
                            self.schema.bag,
                            self.schema.target,
                        ]
                    ]
                    .copy()
                )

                result[
                    "mean_baseline_prediction"
                ] = (
                    mean_value
                )

                result[
                    "median_baseline_prediction"
                ] = (
                    median_value
                )

                result[
                    "repeat"
                ] = repeat

                result[
                    "fold"
                ] = split.fold

                prediction_frames.append(
                    result
                )

            repeat_oof = pd.concat(
                prediction_frames,
                ignore_index=True,
            )

            mean_metrics = (
                self._regression_metrics(
                    df=(
                        repeat_oof
                    ),
                    predictions=(
                        repeat_oof[
                            "mean_baseline_prediction"
                        ]
                        .to_numpy()
                    ),
                )
            )

            median_metrics = (
                self._regression_metrics(
                    df=(
                        repeat_oof
                    ),
                    predictions=(
                        repeat_oof[
                            "median_baseline_prediction"
                        ]
                        .to_numpy()
                    ),
                )
            )

            repeat_rows.append(
                {
                    "repeat":
                        repeat,

                    "mean_MAE":
                        mean_metrics.mae,

                    "mean_RMSE":
                        mean_metrics.rmse,

                    "mean_R2":
                        mean_metrics.r2,

                    "median_MAE":
                        median_metrics.mae,

                    "median_RMSE":
                        median_metrics.rmse,

                    "median_R2":
                        median_metrics.r2,
                }
            )

            all_oof_frames.append(
                repeat_oof
            )

        return {
            "repeat_metrics":
                pd.DataFrame(
                    repeat_rows
                ),

            "oof_predictions":
                pd.concat(
                    all_oof_frames,
                    ignore_index=True,
                ),
        }

    # =========================================================================
    # ARTICLE-ORIENTED CV PREDICTIONS
    # =========================================================================

    def _build_selected_cv_predictions(
        self,
        selected_model_oof: pd.DataFrame,
        baseline_oof: pd.DataFrame,
    ) -> pd.DataFrame:

        keys = [
            "repeat",
            "fold",
            self.schema.group,
            self.schema.bag,
            self.schema.target,
        ]

        model = (
            selected_model_oof[
                keys
                + [
                    "prediction",
                ]
            ]
            .copy()
        )

        baseline = (
            baseline_oof[
                keys
                + [
                    "mean_baseline_prediction",
                    "median_baseline_prediction",
                ]
            ]
            .copy()
        )

        result = model.merge(
            baseline,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        result = result.rename(
            columns={
                self.schema.target:
                    "observed_HFI",

                "prediction":
                    "predicted_HFI",
            }
        )

        result = (
            self._add_prediction_errors(
                result
            )
        )

        return (
            result.sort_values(
                [
                    "repeat",
                    "fold",
                    self.schema.group,
                    self.schema.bag,
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # ARTICLE-ORIENTED FINAL TEST PREDICTIONS
    # =========================================================================

    def _build_final_test_predictions(
        self,
        model_predictions: pd.DataFrame,
        mean_baseline: float,
        median_baseline: float,
    ) -> pd.DataFrame:

        result = (
            model_predictions.copy()
        )

        result = result.rename(
            columns={
                self.schema.target:
                    "observed_HFI",

                "prediction":
                    "predicted_HFI",
            }
        )

        result[
            "mean_baseline_prediction"
        ] = float(
            mean_baseline
        )

        result[
            "median_baseline_prediction"
        ] = float(
            median_baseline
        )

        result = (
            self._add_prediction_errors(
                result
            )
        )

        return (
            result.sort_values(
                [
                    self.schema.group,
                    self.schema.bag,
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # PREDICTION ERRORS
    # =========================================================================

    @staticmethod
    def _add_prediction_errors(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = (
            df.copy()
        )

        result[
            "residual"
        ] = (
            result[
                "observed_HFI"
            ]
            - result[
                "predicted_HFI"
            ]
        )

        result[
            "absolute_error"
        ] = (
            result[
                "residual"
            ].abs()
        )

        result[
            "squared_error"
        ] = (
            result[
                "residual"
            ]
            ** 2
        )

        for baseline in (
            "mean",
            "median",
        ):

            prediction_col = (
                f"{baseline}_baseline_prediction"
            )

            residual_col = (
                f"{baseline}_baseline_residual"
            )

            absolute_col = (
                f"{baseline}_baseline_absolute_error"
            )

            squared_col = (
                f"{baseline}_baseline_squared_error"
            )

            result[
                residual_col
            ] = (
                result[
                    "observed_HFI"
                ]
                - result[
                    prediction_col
                ]
            )

            result[
                absolute_col
            ] = (
                result[
                    residual_col
                ]
                .abs()
            )

            result[
                squared_col
            ] = (
                result[
                    residual_col
                ]
                ** 2
            )

        return result

    # =========================================================================
    # PERFORMANCE SUMMARY
    # =========================================================================

    def _build_performance_summary(
        self,
        selected_repeat_metrics: pd.DataFrame,
        baseline_repeat_metrics: pd.DataFrame,
        selected_test_metrics: RegressionMetrics,
        mean_test_metrics: RegressionMetrics,
        median_test_metrics: RegressionMetrics,
        development_reference: pd.DataFrame,
        test_reference: pd.DataFrame,
    ) -> pd.DataFrame:

        development_points = (
            development_reference[
                self.schema.group
            ]
            .nunique()
        )

        development_captures = (
            development_reference[
                self.schema.bag
            ]
            .nunique()
        )

        test_points = (
            test_reference[
                self.schema.group
            ]
            .nunique()
        )

        test_captures = (
            test_reference[
                self.schema.bag
            ]
            .nunique()
        )

        rows = []

        rows.append(
            self._summary_row_from_repeats(
                stage="development_cv",
                method="selected_pipeline",
                mae=(
                    selected_repeat_metrics[
                        "MAE"
                    ]
                ),
                rmse=(
                    selected_repeat_metrics[
                        "RMSE"
                    ]
                ),
                r2=(
                    selected_repeat_metrics[
                        "R2"
                    ]
                ),
                n_points=(
                    development_points
                ),
                n_capture_points=(
                    development_captures
                ),
            )
        )

        rows.append(
            self._summary_row_from_repeats(
                stage="development_cv",
                method="mean_baseline",
                mae=(
                    baseline_repeat_metrics[
                        "mean_MAE"
                    ]
                ),
                rmse=(
                    baseline_repeat_metrics[
                        "mean_RMSE"
                    ]
                ),
                r2=(
                    baseline_repeat_metrics[
                        "mean_R2"
                    ]
                ),
                n_points=(
                    development_points
                ),
                n_capture_points=(
                    development_captures
                ),
            )
        )

        rows.append(
            self._summary_row_from_repeats(
                stage="development_cv",
                method="median_baseline",
                mae=(
                    baseline_repeat_metrics[
                        "median_MAE"
                    ]
                ),
                rmse=(
                    baseline_repeat_metrics[
                        "median_RMSE"
                    ]
                ),
                r2=(
                    baseline_repeat_metrics[
                        "median_R2"
                    ]
                ),
                n_points=(
                    development_points
                ),
                n_capture_points=(
                    development_captures
                ),
            )
        )

        for method, metrics in (
            (
                "selected_pipeline",
                selected_test_metrics,
            ),
            (
                "mean_baseline",
                mean_test_metrics,
            ),
            (
                "median_baseline",
                median_test_metrics,
            ),
        ):

            rows.append(
                {
                    "stage":
                        "final_test",

                    "method":
                        method,

                    "MAE":
                        metrics.mae,

                    "MAE_SD":
                        np.nan,

                    "RMSE":
                        metrics.rmse,

                    "RMSE_SD":
                        np.nan,

                    "R2":
                        metrics.r2,

                    "R2_SD":
                        np.nan,

                    "n_points":
                        test_points,

                    "n_capture_points":
                        test_captures,

                    "n_repeats":
                        1,
                }
            )

        return pd.DataFrame(
            rows
        )

    def _summary_row_from_repeats(
        self,
        stage: str,
        method: str,
        mae: pd.Series,
        rmse: pd.Series,
        r2: pd.Series,
        n_points: int,
        n_capture_points: int,
    ) -> dict:

        return {
            "stage":
                stage,

            "method":
                method,

            "MAE":
                float(
                    mae.mean()
                ),

            "MAE_SD":
                self._sample_sd(
                    mae
                ),

            "RMSE":
                float(
                    rmse.mean()
                ),

            "RMSE_SD":
                self._sample_sd(
                    rmse
                ),

            "R2":
                float(
                    r2.mean()
                ),

            "R2_SD":
                self._sample_sd(
                    r2
                ),

            "n_points":
                int(
                    n_points
                ),

            "n_capture_points":
                int(
                    n_capture_points
                ),

            "n_repeats":
                int(
                    len(
                        mae
                    )
                ),
        }

    @staticmethod
    def _sample_sd(
        values: pd.Series,
    ) -> float:

        values = (
            pd.Series(
                values
            )
            .dropna()
        )

        if len(
            values
        ) <= 1:

            return 0.0

        return float(
            values.std(
                ddof=1
            )
        )

    # =========================================================================
    # MODEL-SELECTION TRIAL TABLE
    # =========================================================================

    def _build_model_selection_trials(
        self,
        study: optuna.Study,
        repeat_metrics: pd.DataFrame,
    ) -> pd.DataFrame:

        base = (
            self._study_results_dataframe(
                study
            )
        )

        if repeat_metrics.empty:

            return base

        repeat_wide = (
            repeat_metrics.pivot(
                index="trial",
                columns="repeat",
                values=[
                    "MAE",
                    "RMSE",
                    "R2",
                ],
            )
        )

        repeat_wide.columns = [
            f"{metric}_repeat_{repeat}"
            for (
                metric,
                repeat,
            ) in repeat_wide.columns
        ]

        repeat_wide = (
            repeat_wide.reset_index()
        )

        result = base.merge(
            repeat_wide,
            on="trial",
            how="left",
            validate="one_to_one",
        )

        preferred = [
            "trial",
            "state",
            "objective",
            "model",
            "feature_set",
            "aggregation",
            "reduction",
            "pca_indices_components",
            "pca_embeddings_components",
            "selection_indices_features",
            "selection_embeddings_features",
            "model_params",
            "CV_MAE",
            "CV_MAE_std",
            "CV_RMSE",
            "CV_RMSE_std",
            "CV_R2",
            "CV_R2_std",
            "raw_feature_count",
            "model_feature_count",
            "min_cv_train_points",
            "feature_to_point_ratio",
            "mean_fit_time",
            "mean_prediction_time",
            "duration_seconds",
            "failure_reason",
        ]

        repeat_columns = [
            col
            for col in result.columns
            if (
                "_repeat_"
                in col
            )
        ]

        remaining = [
            col
            for col in result.columns
            if (
                col not in preferred
                and col not in repeat_columns
            )
        ]

        return result[
            preferred
            + sorted(
                repeat_columns
            )
            + remaining
        ]

    # =========================================================================
    # BASELINES
    # =========================================================================

    def _baseline_values(
        self,
        df_train: pd.DataFrame,
    ) -> tuple[
        float,
        float,
    ]:

        y = (
            df_train[
                self.schema.target
            ]
            .to_numpy(
                dtype=float
            )
        )

        weights = (
            self._point_weights(
                df_train
            )
        )

        mean_value = float(
            np.average(
                y,
                weights=(
                    weights
                ),
            )
        )

        median_value = (
            self._weighted_median(
                values=(
                    y
                ),
                weights=(
                    weights
                ),
            )
        )

        return (
            mean_value,
            median_value,
        )

    @staticmethod
    def _weighted_median(
        values: np.ndarray,
        weights: np.ndarray,
    ) -> float:

        order = (
            np.argsort(
                values
            )
        )

        values = (
            values[
                order
            ]
        )

        weights = (
            weights[
                order
            ]
        )

        cumulative = (
            np.cumsum(
                weights
            )
        )

        cutoff = (
            0.5
            * weights.sum()
        )

        index = (
            np.searchsorted(
                cumulative,
                cutoff,
                side="left",
            )
        )

        return float(
            values[
                index
            ]
        )

    # =========================================================================
    # POINT-BALANCED WEIGHTS
    # =========================================================================

    def _point_weights(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        n_captures = (
            df.groupby(
                self.schema.group
            )[
                self.schema.bag
            ]
            .transform(
                "nunique"
            )
            .to_numpy(
                dtype=float
            )
        )

        weights = (
            1.0
            / n_captures
        )

        return (
            weights
            / weights.mean()
        )

    # =========================================================================
    # REGRESSION METRICS
    # =========================================================================

    def _regression_metrics(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> RegressionMetrics:

        observed = (
            df[
                self.schema.target
            ]
            .to_numpy(
                dtype=float
            )
        )

        weights = (
            self._point_weights(
                df
            )
        )

        return RegressionMetrics(
            mae=float(
                mean_absolute_error(
                    observed,
                    predictions,
                    sample_weight=(
                        weights
                    ),
                )
            ),

            rmse=float(
                root_mean_squared_error(
                    observed,
                    predictions,
                    sample_weight=(
                        weights
                    ),
                )
            ),

            r2=float(
                r2_score(
                    observed,
                    predictions,
                    sample_weight=(
                        weights
                    ),
                )
            ),
        )

    # =========================================================================
    # FEATURE DIMENSIONALITY
    # =========================================================================

    def _feature_dimensions(
        self,
        signatures: dict[
            str,
            pd.DataFrame,
        ],
    ) -> dict[
        str,
        dict[
            str,
            int,
        ],
    ]:

        dimensions = {}

        for aggregation in (
            self.aggregations
        ):

            df = signatures[
                aggregation
            ]

            index_dimension = len(
                self.schema.index_columns(
                    df.columns
                )
            )

            embedding_dimension = 0

            if (
                self.schema.embedding
                in df.columns
            ):

                embedding_dimension = len(
                    np.asarray(
                        df[
                            self.schema.embedding
                        ].iloc[
                            0
                        ]
                    )
                )

            dimensions[
                aggregation
            ] = {
                "indices":
                    index_dimension,

                "embeddings":
                    embedding_dimension,
            }

        return dimensions

    def _raw_feature_count(
        self,
        configuration: dict,
        feature_dimensions: dict[
            str,
            dict[
                str,
                int,
            ],
        ],
    ) -> int:

        dimensions = (
            feature_dimensions[
                configuration[
                    "aggregation"
                ]
            ]
        )

        total = 0

        if configuration[
            "feature_set"
        ] in {
            "indices",
            "both",
        }:

            total += (
                dimensions[
                    "indices"
                ]
            )

        if configuration[
            "feature_set"
        ] in {
            "embeddings",
            "both",
        }:

            total += (
                dimensions[
                    "embeddings"
                ]
            )

        return int(
            total
        )

    def _model_feature_count(
        self,
        configuration: dict,
        feature_dimensions: dict[
            str,
            dict[
                str,
                int,
            ],
        ],
    ) -> int:

        reduction = (
            configuration[
                "reduction"
            ]
        )

        if reduction == "none":

            return (
                self._raw_feature_count(
                    configuration=(
                        configuration
                    ),
                    feature_dimensions=(
                        feature_dimensions
                    ),
                )
            )

        if reduction == "pca":

            indices_key = (
                "pca_indices_components"
            )

            embeddings_key = (
                "pca_embeddings_components"
            )

        else:

            indices_key = (
                "selection_indices_features"
            )

            embeddings_key = (
                "selection_embeddings_features"
            )

        total = 0

        if configuration[
            "feature_set"
        ] in {
            "indices",
            "both",
        }:

            total += int(
                configuration[
                    indices_key
                ]
            )

        if configuration[
            "feature_set"
        ] in {
            "embeddings",
            "both",
        }:

            total += int(
                configuration[
                    embeddings_key
                ]
            )

        return total

    # =========================================================================
    # SIGNATURE ALIGNMENT
    # =========================================================================

    def _check_signature_alignment(
        self,
        signatures: dict[
            str,
            pd.DataFrame,
        ],
    ) -> None:

        missing = (
            set(
                self.aggregations
            )
            - set(
                signatures
            )
        )

        if missing:

            raise ValueError(
                "Missing aggregation representations: "
                f"{sorted(missing)}"
            )

        reference = (
            signatures[
                self.aggregations[
                    0
                ]
            ][
                [
                    self.schema.bag,
                    self.schema.group,
                    self.schema.target,
                ]
            ]
            .sort_values(
                self.schema.bag
            )
            .reset_index(
                drop=True
            )
        )

        for aggregation in (
            self.aggregations[
                1:
            ]
        ):

            current = (
                signatures[
                    aggregation
                ][
                    [
                        self.schema.bag,
                        self.schema.group,
                        self.schema.target,
                    ]
                ]
                .sort_values(
                    self.schema.bag
                )
                .reset_index(
                    drop=True
                )
            )

            if not (
                reference.equals(
                    current
                )
            ):

                raise ValueError(
                    "Aggregation representations do not contain "
                    "identical CapturePointIds, Points and targets."
                )

    # =========================================================================
    # DATAFRAME HELPERS
    # =========================================================================

    def _select_bags(
        self,
        df: pd.DataFrame,
        bags: tuple,
    ) -> pd.DataFrame:

        bag_set = set(
            bags
        )

        return (
            df[
                df[
                    self.schema.bag
                ].isin(
                    bag_set
                )
            ]
            .copy()
        )

    # =========================================================================
    # CV MANIFEST
    # =========================================================================

    def _cv_split_manifest(
        self,
        development_reference: pd.DataFrame,
        cv_splits: list[CVSplit],
    ) -> pd.DataFrame:

        bag_to_point = (
            development_reference
            .set_index(
                self.schema.bag
            )[
                self.schema.group
            ]
            .to_dict()
        )

        rows = []

        for split in (
            cv_splits
        ):

            for bag in (
                split.train_bags
            ):

                rows.append(
                    {
                        "repeat":
                            split.repeat,

                        "fold":
                            split.fold,

                        self.schema.bag:
                            bag,

                        self.schema.group:
                            bag_to_point[
                                bag
                            ],

                        "role":
                            "train",
                    }
                )

            for bag in (
                split.validation_bags
            ):

                rows.append(
                    {
                        "repeat":
                            split.repeat,

                        "fold":
                            split.fold,

                        self.schema.bag:
                            bag,

                        self.schema.group:
                            bag_to_point[
                                bag
                            ],

                        "role":
                            "validation",
                    }
                )

        return pd.DataFrame(
            rows
        )

    # =========================================================================
    # CONFIGURATION SERIALIZATION
    # =========================================================================

    @staticmethod
    def _serializable_config(
        configuration: dict,
    ) -> dict:

        result = dict(
            configuration
        )

        result[
            "model"
        ] = (
            configuration[
                "model"
            ].name
        )

        return result

    @staticmethod
    def _configuration_from_serialized(
        configuration: dict,
    ) -> dict:

        result = dict(
            configuration
        )

        result[
            "model"
        ] = RegressionModels[
            result[
                "model"
            ]
        ]

        return result

    def _config_columns(
        self,
        configuration: dict,
    ) -> dict:

        model = (
            configuration[
                "model"
            ]
        )

        if isinstance(
            model,
            RegressionModels,
        ):

            model = (
                model.name
            )

        return {
            "model":
                model,

            "feature_set":
                configuration[
                    "feature_set"
                ],

            "aggregation":
                configuration[
                    "aggregation"
                ],

            "reduction":
                configuration[
                    "reduction"
                ],

            "pca_indices_components":
                configuration.get(
                    "pca_indices_components"
                ),

            "pca_embeddings_components":
                configuration.get(
                    "pca_embeddings_components"
                ),

            "selection_indices_features":
                configuration.get(
                    "selection_indices_features"
                ),

            "selection_embeddings_features":
                configuration.get(
                    "selection_embeddings_features"
                ),

            "model_params":
                json.dumps(
                    configuration[
                        "model_params"
                    ],
                    sort_keys=True,
                ),
        }

    # =========================================================================
    # OPTUNA HISTORY
    # =========================================================================

    def _study_results_dataframe(
        self,
        study: optuna.Study,
    ) -> pd.DataFrame:

        rows = []

        for trial in (
            study.trials
        ):

            row = {
                "trial":
                    trial.number,

                "state":
                    trial.state.name,

                "objective":
                    (
                        float(
                            trial.value
                        )
                        if (
                            trial.value
                            is not None
                        )
                        else np.nan
                    ),
            }

            configuration = (
                trial.user_attrs.get(
                    "configuration"
                )
            )

            if configuration is not None:

                row.update(
                    self._config_columns(
                        configuration
                    )
                )

            else:

                row.update(
                    {
                        "model":
                            None,

                        "feature_set":
                            None,

                        "aggregation":
                            None,

                        "reduction":
                            None,

                        "pca_indices_components":
                            None,

                        "pca_embeddings_components":
                            None,

                        "selection_indices_features":
                            None,

                        "selection_embeddings_features":
                            None,

                        "model_params":
                            None,
                    }
                )

            row[
                "CV_MAE"
            ] = (
                trial.user_attrs.get(
                    "cv_mae",
                    np.nan,
                )
            )

            row[
                "CV_MAE_std"
            ] = (
                trial.user_attrs.get(
                    "cv_mae_std",
                    np.nan,
                )
            )

            row[
                "CV_RMSE"
            ] = (
                trial.user_attrs.get(
                    "cv_rmse",
                    np.nan,
                )
            )

            row[
                "CV_RMSE_std"
            ] = (
                trial.user_attrs.get(
                    "cv_rmse_std",
                    np.nan,
                )
            )

            row[
                "CV_R2"
            ] = (
                trial.user_attrs.get(
                    "cv_r2",
                    np.nan,
                )
            )

            row[
                "CV_R2_std"
            ] = (
                trial.user_attrs.get(
                    "cv_r2_std",
                    np.nan,
                )
            )

            row[
                "raw_feature_count"
            ] = (
                trial.user_attrs.get(
                    "raw_feature_count",
                    np.nan,
                )
            )

            row[
                "model_feature_count"
            ] = (
                trial.user_attrs.get(
                    "model_feature_count",
                    np.nan,
                )
            )

            row[
                "min_cv_train_points"
            ] = (
                trial.user_attrs.get(
                    "min_cv_train_points",
                    np.nan,
                )
            )

            if (
                pd.notna(
                    row[
                        "model_feature_count"
                    ]
                )
                and pd.notna(
                    row[
                        "min_cv_train_points"
                    ]
                )
                and (
                    row[
                        "min_cv_train_points"
                    ]
                    > 0
                )
            ):

                row[
                    "feature_to_point_ratio"
                ] = (
                    float(
                        row[
                            "model_feature_count"
                        ]
                    )
                    / float(
                        row[
                            "min_cv_train_points"
                        ]
                    )
                )

            else:

                row[
                    "feature_to_point_ratio"
                ] = np.nan

            row[
                "mean_fit_time"
            ] = (
                trial.user_attrs.get(
                    "mean_fit_time",
                    np.nan,
                )
            )

            row[
                "mean_prediction_time"
            ] = (
                trial.user_attrs.get(
                    "mean_prediction_time",
                    np.nan,
                )
            )

            row[
                "failure_reason"
            ] = (
                trial.user_attrs.get(
                    "failure_reason",
                    None,
                )
            )

            if (
                trial.duration
                is not None
            ):

                row[
                    "duration_seconds"
                ] = (
                    trial.duration
                    .total_seconds()
                )

            else:

                row[
                    "duration_seconds"
                ] = np.nan

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================================
    # SAVE SCIENTIFIC OUTPUTS
    # =========================================================================

    def _save_results(
        self,
        performance_summary: pd.DataFrame,
        model_selection_trials: pd.DataFrame,
        selected_cv_predictions: pd.DataFrame,
        final_test_predictions: pd.DataFrame,
        selected_pipeline: dict,
        split_manifest: pd.DataFrame,
        cv_split_manifest: pd.DataFrame,
    ) -> None:

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        reproducibility_dir = (
            os.path.join(
                self.output_dir,
                "reproducibility",
            )
        )

        os.makedirs(
            reproducibility_dir,
            exist_ok=True,
        )

        performance_summary.to_csv(
            os.path.join(
                self.output_dir,
                "performance_summary.csv",
            ),
            index=False,
        )

        model_selection_trials.to_csv(
            os.path.join(
                self.output_dir,
                "model_selection_trials.csv",
            ),
            index=False,
        )

        selected_cv_predictions.to_csv(
            os.path.join(
                self.output_dir,
                "selected_cv_predictions.csv",
            ),
            index=False,
        )

        final_test_predictions.to_csv(
            os.path.join(
                self.output_dir,
                "final_test_predictions.csv",
            ),
            index=False,
        )

        with open(
            os.path.join(
                self.output_dir,
                "selected_pipeline.json",
            ),
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                selected_pipeline,
                file,
                indent=2,
            )

        split_manifest.to_csv(
            os.path.join(
                reproducibility_dir,
                "data_split.csv",
            ),
            index=False,
        )

        cv_split_manifest.to_csv(
            os.path.join(
                reproducibility_dir,
                "cv_splits.csv",
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

    def _cv_seed(
        self,
        repeat: int,
    ) -> int:

        return (
            self.random_state
            + 1_000
            + repeat
        )