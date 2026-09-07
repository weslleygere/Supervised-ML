import logging

from .config.settings import Settings
from .core.data.data_loader import DataLoader
from .core.evaluation.evaluator import ModelEvaluator
from .core.processors.presplit import PreSplitProcessor


logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrate the instance-MIR machine learning experiment.

    Parameters
    ----------
    settings : Settings
        Experiment configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> None:
        """
        Execute the complete experiment.

        Steps
        -----
        1. Load dataset and schema.
        2. Select the columns required by the experiment.
        3. Run nested grouped instance-MIR evaluation.
        """

        # ---------------------------------------------------------------------
        # 1. Load data and schema
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # 2. Pre-split processing
        # ---------------------------------------------------------------------

        pre_split_processor = PreSplitProcessor(
            schema=schema,
        )

        df_processed = pre_split_processor.process(
            df_raw
        )

        logger.info(
            "Data prepared successfully with shape %s",
            df_processed.shape,
        )

        # ---------------------------------------------------------------------
        # 3. Nested instance-MIR evaluation
        # ---------------------------------------------------------------------

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
            inference_k=self.settings.validation.inference_k,
            inference_repeats=self.settings.validation.inference_repeats,
        )

        evaluator.evaluate(
            df_processed
        )

        logger.info(
            "Instance-MIR experiment completed successfully."
        )
