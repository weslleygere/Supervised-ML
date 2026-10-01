import logging
import os

from .config.settings import Settings
from .core.data.data_loader import DataLoader
from .core.data.temporal_audit import TemporalAudit
from .core.evaluation.evaluator import ModelEvaluator
from .core.evaluation.plots import save_evaluation_plots
from .core.evaluation.result_analysis import (
    save_result_analysis,
)
from .core.processors.aggregations import AcousticAggregator
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
    2. Build atomic Audio_Name observations.
    3. Audit temporal sampling coverage.
    4. Build daily acoustic statistics.
    5. Build candidate CapturePointId representations.
    6. Select one complete pipeline using repeated grouped CV.
    7. Fit the selected pipeline on the complete development set.
    8. Evaluate once on the isolated final test set.
    9. Build report-oriented scientific analysis tables.
    10. Generate publication-oriented figures.
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
        # PRE-SPLIT PROCESSING
        # =====================================================================

        processor = PreSplitProcessor(
            schema=schema
        )

        audio = processor.prepare_audio(
            df_raw
        )

        daily = processor.prepare_daily(
            audio
        )

        # =====================================================================
        # TEMPORAL AUDIT
        # =====================================================================

        audit = TemporalAudit(
            schema=schema
        ).build(
            audio
        )

        reproducibility_dir = os.path.join(
            self.settings.data.output_dir,
            "reproducibility",
        )

        os.makedirs(
            reproducibility_dir,
            exist_ok=True,
        )

        audit[
            "daily"
        ].to_csv(
            os.path.join(
                reproducibility_dir,
                "temporal_audit_daily.csv",
            ),
            index=False,
        )

        audit[
            "capture"
        ].to_csv(
            os.path.join(
                reproducibility_dir,
                "temporal_audit_capture.csv",
            ),
            index=False,
        )

        logger.info(
            "Temporal audit saved for %d atomic recordings",
            len(
                audio
            ),
        )

        # =====================================================================
        # ACOUSTIC REPRESENTATIONS
        # =====================================================================

        aggregator = AcousticAggregator(
            schema=schema
        )

        signatures = {
            aggregation:
                aggregator.build(
                    audio=audio,
                    daily=daily,
                    aggregation=aggregation,
                )
            for aggregation
            in (
                self.settings
                .model
                .aggregation_strategies
            )
        }

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

            selection_indices_candidates=(
                self.settings
                .model
                .selection_indices_candidates
            ),

            selection_embeddings_candidates=(
                self.settings
                .model
                .selection_embeddings_candidates
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
        # SCIENTIFIC RESULT ANALYSIS
        # =====================================================================

        logger.info(
            "Building report-oriented scientific analysis tables..."
        )

        analysis = save_result_analysis(
            output_dir=(
                self.settings.data.output_dir
            ),
            schema=schema,
        )

        logger.info(
            "Scientific analysis complete: %d analysis tables created.",
            len(
                analysis
            ),
        )

        # =====================================================================
        # FIGURES
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
            os.path.join(
                self.settings.data.output_dir,
                "figures",
            ),
        )

        # =====================================================================
        # RETURN
        # =====================================================================

        results[
            "analysis"
        ] = analysis

        return results