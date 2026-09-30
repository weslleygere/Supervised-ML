import logging

from .config.settings import Settings
from .core.data.data_loader import DataLoader
from .core.evaluation.evaluator import ModelEvaluator
from .core.evaluation.plots import save_evaluation_plots
from .core.processors.presplit import PreSplitProcessor


logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE
# =============================================================================


class Pipeline:
    """
    Orchestrate the supervised machine-learning experiment.

    Steps
    -----
    1. Load data and schema.

    2. Build all candidate acoustic representations at the
       CapturePointId level.

    3. Split independent Points once into:

           development set
           final test set

    4. Select one complete machine-learning pipeline on the development set
       using repeated grouped cross-validation and a single Optuna study.

    5. Fit the selected pipeline using all development data.

    6. Evaluate the selected pipeline once on the isolated final test set.

    7. Generate manuscript-oriented figures from the saved scientific
       outputs.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:

        self.settings = settings

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run(
        self,
    ) -> dict:
        """
        Run the complete supervised machine-learning experiment.

        Returns
        -------
        dict
            Main model-selection and final-evaluation artifacts returned by
            ModelEvaluator.evaluate().
        """

        # =====================================================================
        # DATA
        # =====================================================================

        loader = DataLoader(
            data_path=(
                self.settings.data.data_path
            ),
            schema_path=(
                self.settings.data.schema_path
            ),
        )

        df_raw = loader.load_data()

        schema = loader.load_schema()

        logger.info(
            "Data loaded with shape %s",
            df_raw.shape,
        )

        # =====================================================================
        # PRE-SPLIT ACOUSTIC REPRESENTATIONS
        # =====================================================================

        processor = PreSplitProcessor(
            schema=schema
        )

        signatures = processor.process_all(
            df_raw=df_raw,
            aggregations=(
                self.settings
                .model
                .aggregation_strategies
            ),
        )

        reference_aggregation = (
            self.settings
            .model
            .aggregation_strategies[
                0
            ]
        )

        reference = signatures[
            reference_aggregation
        ]

        logger.info(
            "Acoustic signatures created for %d CapturePointIds "
            "across %d Points",
            reference[
                schema.bag
            ].nunique(),
            reference[
                schema.group
            ].nunique(),
        )

        for aggregation, dataframe in (
            signatures.items()
        ):

            logger.info(
                "Aggregation '%s': shape=%s",
                aggregation,
                dataframe.shape,
            )

        # =====================================================================
        # EXPERIMENT CONFIGURATION
        # =====================================================================

        logger.info(
            "Experiment design: "
            "test_size=%.3f, "
            "CV=%d folds x %d repeats, "
            "Optuna trials=%d",
            self.settings.validation.test_size,
            self.settings.validation.cv_splits,
            self.settings.validation.cv_repeats,
            self.settings.validation.optuna_trials,
        )

        logger.info(
            "Candidate pipeline space: "
            "%d model families, "
            "%d feature sets, "
            "%d aggregation strategies, "
            "%d reduction methods",
            len(
                self.settings.model.models
            ),
            len(
                self.settings.model.feature_sets
            ),
            len(
                self.settings
                .model
                .aggregation_strategies
            ),
            len(
                self.settings
                .model
                .reduction_methods
            ),
        )

        # =====================================================================
        # COMPLETE PIPELINE SELECTION
        # =====================================================================

        evaluator = ModelEvaluator(
            schema=schema,

            output_dir=(
                self.settings.data.output_dir
            ),

            models=(
                self.settings.model.models
            ),

            random_state=(
                self.settings.model.random_state
            ),

            feature_sets=(
                self.settings.model.feature_sets
            ),

            aggregations=(
                self.settings
                .model
                .aggregation_strategies
            ),

            reductions=(
                self.settings
                .model
                .reduction_methods
            ),

            pca_indices_candidates=(
                self.settings
                .model
                .pca_indices_candidates
            ),

            pca_embeddings_candidates=(
                self.settings
                .model
                .pca_embeddings_candidates
            ),

            test_size=(
                self.settings
                .validation
                .test_size
            ),

            cv_splits=(
                self.settings
                .validation
                .cv_splits
            ),

            cv_repeats=(
                self.settings
                .validation
                .cv_repeats
            ),

            optuna_trials=(
                self.settings
                .validation
                .optuna_trials
            ),
        )

        results = evaluator.evaluate(
            signatures
        )

        logger.info(
            "Pipeline selection and final evaluation complete."
        )

        # =====================================================================
        # MANUSCRIPT-ORIENTED FIGURES
        # =====================================================================

        logger.info(
            "Generating evaluation figures..."
        )

        save_evaluation_plots(
            output_dir=(
                self.settings.data.output_dir
            )
        )

        logger.info(
            "Evaluation figures saved to %s",
            (
                self.settings.data.output_dir
                + "/figures"
            ),
        )

        return results