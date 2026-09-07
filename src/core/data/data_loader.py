import json
from dataclasses import dataclass
from typing import Callable, ClassVar

import pandas as pd

from .schema import Schema


@dataclass
class DataLoader:
    """
    Load the dataset and its schema definition.

    Parameters
    ----------
    data_path : str
        Path to the dataset file.

    schema_path : str
        Path to the JSON schema file.
    """

    SUPPORTED_READERS: ClassVar[
        dict[str, Callable[[str], pd.DataFrame]]
    ] = {
        "xlsx": pd.read_excel,
        "csv": pd.read_csv,
        "parquet": pd.read_parquet,
        "json": pd.read_json,
    }

    data_path: str
    schema_path: str

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset based on its file extension.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        extension = self.data_path.rsplit(".", 1)[-1].lower()
        reader = self.SUPPORTED_READERS[extension]

        try:
            df = reader(self.data_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to load file '{self.data_path}'."
            ) from exc

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Expected DataFrame, got {type(df).__name__}."
            )

        if df.empty:
            raise ValueError(
                f"Loaded data is empty: {self.data_path}"
            )

        return df

    def load_schema(self) -> Schema:
        """
        Load the instance-MIR schema from JSON.

        Returns
        -------
        Schema
            Parsed dataset schema.
        """
        try:
            with open(
                self.schema_path,
                "r",
                encoding="utf-8",
            ) as file:
                schema_dict = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in schema file "
                f"'{self.schema_path}'."
            ) from exc

        except Exception as exc:
            raise ValueError(
                f"Failed to load schema file "
                f"'{self.schema_path}'."
            ) from exc

        if not isinstance(schema_dict, dict):
            raise ValueError(
                "Schema file must contain a JSON object."
            )

        required_keys = {
            "target",
            "group",
            "bag",
            "instance_columns",
            "index_prefixes",
            "embedding",
        }

        missing = required_keys - schema_dict.keys()

        if missing:
            raise ValueError(
                "Schema file is missing required fields: "
                f"{', '.join(sorted(missing))}"
            )

        return Schema.from_dict(schema_dict)
