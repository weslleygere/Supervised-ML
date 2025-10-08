import logging
import pandas as pd
from typing import List
from .core.data.schema import Schema
from .core.data.data_loader import DataLoader
from .core.models.factory import RegressionModels
from .core.evaluation.evaluator import ModelEvaluator
from .core.processors.presplit import PreSplitProcessor

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the end-to-end machine learning pipeline, including:
    - Data loading
    - Preprocessing
    - Model evaluation

    Parameters
    -----------
    data_path : str
        Path to the input dataset.
    schema_path : str
        Path to the JSON schema defining the dataset structure.
    output_dir : str
        Directory where evaluation outputs and logs will be stored.
    models : list[RegressionModels]
        List of models to evaluate.
    random_state : int
        Random seed for reproducibility.
    n_splits : int
        Number of cross-validation folds.
    """
    def __init__(self, 
                 data_path: str, 
                 schema_path: str, 
                 output_dir: str,
                 models: List[RegressionModels],
                 random_state: int,
                 n_splits: int) -> None:
        self.data_path    = data_path
        self.schema_path  = schema_path
        self.output_dir   = output_dir
        self.models       = models
        self.random_state = random_state
        self.n_splits     = n_splits

    def run(self) -> tuple[pd.DataFrame, Schema]:
        """
        Execute the full pipeline:
        1. Load data and schema.
        2. Preprocess the dataset according to the schema.
        3. Evaluate models on the processed data.

        Logs each step of the process for traceability and debugging.
        """
        # Step 1: Load raw data and schema
        loader = DataLoader(self.data_path, self.schema_path)
        df_raw = loader.load_data()
        schema = loader.load_schema()
        logger.info(f"Data loaded successfully with shape {df_raw.shape}")

        # Step 2: Preprocess the data
        pre_split_processor = PreSplitProcessor(schema)
        df_processed = pre_split_processor.process(df_raw)
        logger.info(f"Data preprocessed successfully with shape {df_processed.shape}")

        # Step 3: Evaluate models
        evaluator = ModelEvaluator(schema, self.output_dir, self.models, self.random_state, self.n_splits)
        evaluator.evaluate(df_processed)

        return df_raw, schema
