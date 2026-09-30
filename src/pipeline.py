import logging

from .config.settings import Settings
from .core.data.data_loader import DataLoader
from .core.evaluation.evaluator import ModelEvaluator
from .core.processors.presplit import PreSplitProcessor
from src.core.evaluation.plots import save_evaluation_plots


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
    2. Build the candidate acoustic representations per CapturePointId.
    3. Run nested grouped cross-validation for complete pipeline selection.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:

        self.settings = settings

    def run(
        self,
    ) -> None:

        # =====================================================================
        # DATA
        # =====================================================================

        loader = DataLoader(
            data_path=self.settings.data.data_path,
            schema_path=self.settings.data.schema_path,
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
                self.settings.model.aggregation_strategies
            ),
        )

        reference = signatures[
            self.settings.model.aggregation_strategies[0]
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
        # NESTED PIPELINE SELECTION
        # =====================================================================

        evaluator = ModelEvaluator(
            schema=schema,
            output_dir=self.settings.data.output_dir,
            models=self.settings.model.models,
            random_state=self.settings.model.random_state,
            feature_sets=(
                self.settings.model.feature_sets
            ),
            aggregations=(
                self.settings.model.aggregation_strategies
            ),
            reductions=(
                self.settings.model.reduction_methods
            ),
            pca_indices_candidates=(
                self.settings.model.pca_indices_candidates
            ),
            pca_embeddings_candidates=(
                self.settings.model.pca_embeddings_candidates
            ),
            outer_splits=(
                self.settings.validation.outer_splits
            ),
            outer_repeats=(
                self.settings.validation.outer_repeats
            ),
            inner_splits=(
                self.settings.validation.inner_splits
            ),
            optuna_trials=(
                self.settings.validation.optuna_trials
            ),
        )

        evaluator.evaluate(
            signatures
        )

        save_evaluation_plots(
            self.settings.data.output_dir
        )
