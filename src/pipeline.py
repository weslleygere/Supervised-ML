import logging

from .config.settings import Settings
from .core.data.data_loader import DataLoader
from .core.evaluation.evaluator import ModelEvaluator
from .core.processors.presplit import PreSplitProcessor


logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE
# =============================================================================


class Pipeline:
    """
    Orchestrate the end-to-end supervised machine learning pipeline.

    Steps
    -----
    1. Load raw data and schema.
    2. Build one hierarchical acoustic signature per CapturePointId.
    3. Evaluate the selected regression models using nested GroupKFold.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

    def run(self) -> None:

        # =====================================================================
        # DATA LOADING
        # =====================================================================

        loader = DataLoader(
            data_path=self.settings.data.data_path,
            schema_path=self.settings.data.schema_path,
        )

        df_raw = loader.load_data()
        schema = loader.load_schema()

        logger.info(
            "Data loaded successfully with shape %s",
            df_raw.shape,
        )

        # =====================================================================
        # HIERARCHICAL AGGREGATION
        # =====================================================================

        pre_split_processor = PreSplitProcessor(
            schema=schema
        )

        df_processed = pre_split_processor.process(
            df_raw
        )

        logger.info(
            "Hierarchical acoustic signatures created "
            "successfully with shape %s",
            df_processed.shape,
        )

        logger.info(
            "Processed dataset contains %d CapturePointIds "
            "across %d Points",
            df_processed[
                schema.bag
            ].nunique(),
            df_processed[
                schema.group
            ].nunique(),
        )

        # =====================================================================
        # MODEL EVALUATION
        # =====================================================================

        evaluator = ModelEvaluator(
            schema=schema,
            output_dir=self.settings.data.output_dir,
            models=self.settings.model.models,
            random_state=self.settings.model.random_state,
            feature_set=self.settings.model.feature_set,
            pca_components=self.settings.model.pca_components,
            outer_splits=self.settings.validation.outer_splits,
            inner_splits=self.settings.validation.inner_splits,
            optuna_trials=self.settings.validation.optuna_trials,
        )

        evaluator.evaluate(
            df_processed
        )
