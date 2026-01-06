import json
from dataclasses import dataclass
from typing import Dict, Callable, ClassVar

import pandas as pd

from .schema import Schema


@dataclass
class DataLoader:
    """
    Utility class for loading datasets and their corresponding schema definitions.

    Parameters
    ----------
    data_path : str
        Path to the dataset file.
    schema_path : str
        Path to the JSON schema file.
    """
    SUPPORTED_READERS: ClassVar[Dict[str, Callable[[str], pd.DataFrame]]] = {
        'xlsx'   : pd.read_excel,
        'csv'    : pd.read_csv,
        'parquet': pd.read_parquet,
        'json'   : pd.read_json
    }

    data_path  : str
    schema_path: str

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from the specified path based on file extension.

        Returns
        -------
        df : pd.DataFrame
            The loaded dataset.
        """
        extension = self.data_path.split('.')[-1].lower()
        reader_func = self.SUPPORTED_READERS[extension]

        try:
            df = reader_func(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to load file '{self.data_path}'") from e

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df).__name__}")

        if df.empty:
            raise ValueError(f"Loaded data is empty: {self.data_path}")

        return df

    def load_schema(self) -> 'Schema':
        """
        Load and parse the schema definition from the specified JSON file.

        Returns
        -------
        schema : Schema
            Schema object created from the JSON file.
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

        if not isinstance(schema_dict, dict):
            raise ValueError(
                f"Schema file must contain a JSON object at the top level. "
                f"Got {type(schema_dict).__name__} instead."
            )

        required_keys = {"targets", "features"}
        missing = required_keys - schema_dict.keys()
        if missing:
            raise ValueError(
                f"Schema file is missing required fields: {', '.join(sorted(missing))}"
            )

        unknown_keys = schema_dict.keys() - required_keys
        if unknown_keys:
            raise ValueError(
                f"Schema contains unknown top-level fields: {', '.join(sorted(unknown_keys))}"
            )

        if not isinstance(schema_dict["targets"], list):
            raise ValueError("'targets' field must be a list.")
        if not isinstance(schema_dict["features"], list):
            raise ValueError("'features' field must be a list.")

        return Schema.from_dict(schema_dict)
    