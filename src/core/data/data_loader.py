import json
import pandas as pd
from .schema import Schema
from typing import Dict, Callable, ClassVar

from dataclasses import dataclass

@dataclass
class DataLoader:
    """
    Utility class for loading datasets and their corresponding schema definitions.

    Parameters
    ----------
    data_path : str
        Path to the dataset file. Supported formats: xlsx, csv, parquet, json.
    schema_path : str
        Path to the JSON file containing the dataset schema.
    """
    SUPPORTED_READERS: ClassVar[Dict[str, Callable[[str], pd.DataFrame]]] = {
        'xlsx'   : pd.read_excel,
        'csv'    : pd.read_csv,
        'parquet': pd.read_parquet,
        'json'   : pd.read_json
    }

    data_path   : str
    schema_path : str

    def __post_init__(self) -> None:
        """Validate file formats"""
        self._extension = self.data_path.split('.')[-1].lower()
        if self._extension not in self.SUPPORTED_READERS:
            supported = ', '.join(self.SUPPORTED_READERS.keys())
            raise ValueError(f"Unsupported file extension '{self._extension}'. Supported: {supported}")
        if not self.schema_path.endswith('.json'):
            raise ValueError(f"Schema file must be JSON: {self.schema_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from the specified path based on file extension.

        Returns
        -------
        df : pandas.DataFrame
            The loaded dataset.
        """
        reader_func = self.SUPPORTED_READERS[self._extension]

        try:
            df = reader_func(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to load file '{self.data_path}'") from e

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df).__name__}")

        if df.empty:
            raise ValueError(f"Loaded data is empty: {self.data_path}")

        return df

    def load_schema(self) -> Schema:
        """
        Load and parse the schema definition from the specified JSON file.

        Returns
        -------
        schema : Schema
            Schema object created from the loaded dictionary containing target and feature definitions.
        """
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as file:
                schema_dict = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file '{self.schema_path}'") from e
        except Exception as e:
            raise ValueError(f"Failed to load schema file '{self.schema_path}'") from e

        if not schema_dict:
            raise ValueError(f"Schema file is empty: {self.schema_path}")

        return Schema.from_dict(schema_dict)