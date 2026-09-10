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

    data_path: str
    schema_path: str

    SUPPORTED_READERS: ClassVar[
        dict[str, Callable[[str], pd.DataFrame]]
    ] = {
        "xlsx": pd.read_excel,
        "csv": pd.read_csv,
        "parquet": pd.read_parquet,
        "json": pd.read_json,
    }

    def load_data(self) -> pd.DataFrame:
        extension = self.data_path.split(".")[-1].lower()

        if extension not in self.SUPPORTED_READERS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        try:
            df = self.SUPPORTED_READERS[extension](
                self.data_path
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load file '{self.data_path}'"
            ) from e

        if df.empty:
            raise ValueError(
                f"Loaded data is empty: {self.data_path}"
            )

        return df

    def load_schema(self) -> Schema:
        try:
            with open(
                self.schema_path,
                "r",
                encoding="utf-8",
            ) as file:
                schema_dict = json.load(file)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in schema file "
                f"'{self.schema_path}'"
            ) from e

        except Exception as e:
            raise ValueError(
                f"Failed to load schema file "
                f"'{self.schema_path}'"
            ) from e

        if not isinstance(schema_dict, dict):
            raise ValueError(
                "Schema file must contain a JSON object."
            )

        required_keys = {
            "target",
            "group",
            "bag",
            "audio",
            "datetime",
            "index_prefixes",
            "embedding",
        }

        missing = required_keys - schema_dict.keys()

        if missing:
            raise ValueError(
                "Schema is missing required fields: "
                + ", ".join(sorted(missing))
            )

        return Schema.from_dict(
            schema_dict
        )
