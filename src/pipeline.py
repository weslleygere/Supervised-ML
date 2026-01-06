import logging

from .core.data.data_loader import DataLoader
from .core.evaluation.evaluator import ModelEvaluator
from .core.processors.presplit import PreSplitProcessor
from .config.settings import Settings

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the end-to-end machine learning pipeline, including:
    - Data loading
    - Preprocessing
    - Model evaluation

    Parameters
    ----------
    settings : Settings
        Configuration settings object containing data paths, model parameters, and evaluation settings.
    """
    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    def run(self) -> None:
        """
        Execute the full pipeline:
        1. Load data and schema.
        2. Preprocess the dataset according to the schema.
        3. Evaluate models on the processed data.

        Logs each step of the process for traceability and debugging.
        """
        # Step 1: Load raw data and schema
        loader = DataLoader(
            data_path=self.settings.data.data_path,
            schema_path=self.settings.data.schema_path
        )
        df_raw = loader.load_data()
        schema = loader.load_schema()

        logger.info(f"Data loaded successfully with shape {df_raw.shape}")

        # Step 2: Preprocess the data
        pre_split_processor = PreSplitProcessor(schema)
        df_processed = pre_split_processor.process(df_raw)

        logger.info(f"Data preprocessed successfully with shape {df_processed.shape}")

        # Step 3: Evaluate models
        evaluator = ModelEvaluator(
            schema = schema,
            output_dir=self.settings.data.output_dir,
            models = self.settings.model.models,
            random_state= self.settings.model.random_state,
            n_splits= self.settings.validation.n_splits,
            cv_strategy= self.settings.validation.cv_strategy,
            group_column= self.settings.validation.group_column
        )

        evaluator.evaluate(df_processed)
