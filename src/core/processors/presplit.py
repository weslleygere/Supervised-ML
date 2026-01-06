import logging

import pandas as pd

from src.core.data.schema import Schema

logger = logging.getLogger(__name__)


class PreSplitProcessor:
    """
    Applies pre-split preprocessing to raw input data, including:
    - Selecting only schema-defined features and targets, dropping others.
    - Validating that all schema-defined columns exist in the dataset.
    - Setting column data types based on schema definitions.
    
    This ensures that the dataset is correctly formatted before
    splitting into train/test sets.

    Parameters
    ----------
    schema : Schema
        Schema object containing target and feature definitions.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema: Schema = schema

    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps in order:
        1. Select schema-defined features/targets and validate missing columns.
        2. Convert column data types based on schema.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input dataset.

        Returns
        -------
        df : pd.DataFrame
            Preprocessed dataset ready for splitting and modeling.
        """
        df = df_raw.copy()
        df = self._trim(df)
        df = self._set_dtypes(df)
        return df

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retain only columns defined in the schema (features + targets).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be validated and trimmed.

        Returns
        -------
        df : pd.DataFrame
            Trimmed DataFrame with schema-defined features + targets.
        """
        self._validate_schema_columns(df)

        if self.schema.ignored_feature_names:
            logger.warning(
                f"The following columns were marked as 'ignore' in the schema and "
                f"will be ignored: {self.schema.ignored_feature_names}"
            )

        cols_def = self.schema.feature_names + self.schema.target_names 
        cols_not_def = df.columns.difference(cols_def)
        
        if len(cols_not_def) > 0:
            logger.warning(
                f"The following columns were not defined in the schema and "
                f"will be ignored: {list(cols_not_def)}"
            )
        
        cols_to_keep = self.schema.valid_feature_names_id + self.schema.target_names
        
        return df[cols_to_keep]

    def _validate_schema_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all schema-defined columns exist in the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to validate.
            
        Raises
        ------
        ValueError
            If any schema-defined target or feature columns are missing.
        """
        missing_targets = [t for t in self.schema.target_names if t not in df.columns]
        if missing_targets:
            raise ValueError(
                f"The following target columns were defined in the schema but "
                f"are missing in the dataset: {missing_targets}"
            )

        missing_features = [f for f in self.schema.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"The following feature columns were defined in the schema but "
                f"are missing in the dataset: {missing_features}"
            )

    def _set_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for features and targets based on schema:
        - Numerical features → float
        - Numerical targets → float

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame whose columns will be cast.

        Returns
        -------
        pd.DataFrame
            DataFrame with updated column datatypes.
        """
        for feature in self.schema.features:
            col = feature.name

            if feature.type == "ignore":
                continue

            if feature.type == "numerical":
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except Exception as e:
                    raise ValueError(
                        f"Feature column '{col}' contains non-numeric values or "
                        f"cannot be converted to float."
                    ) from e

        for target in self.schema.target:
            col = target.name

            if target.type == "numerical":
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except Exception as e:
                    raise ValueError(
                        f"Target column '{col}' contains non-numeric values or "
                        f"cannot be converted to float."
                    ) from e

        return df
