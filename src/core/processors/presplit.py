import re
import logging
import pandas as pd
from src.core.data.schema import Schema, FeatureSchema, TargetSchema

logger = logging.getLogger(__name__)

class PreSplitProcessor:
    """
    Applies pre-split preprocessing to raw input data, including:
    - Removing rows with missing target values.
    - Dropping ignored features.
    - Setting column data types.
    - Sanitizing column names to remove special characters.

    This ensures that the dataset is cleaned and formatted correctly before splitting into train/test sets.

    Parameters
    -----------
    schema : Schema
        Schema object containing target and feature definitions.

    Attributes
    ----------
    target : TargetSchema
        Schema object for the target variable.
    features : list[FeatureSchema]
        List of feature schema objects.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema   : Schema              = schema
        self.target   : TargetSchema        = schema.target
        self.features : list[FeatureSchema] = schema.features

    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps in order:
        - Remove rows with missing targets.
        - Drop ignored columns.
        - Convert column data types.
        - Sanitize column names.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input dataset.

        Returns
        -------
        df : pd.DataFrame
            Preprocessed dataset ready for train/test split and modeling.
        """
        df = df_raw.copy()
        df = self._trim(df)
        df = self._set_dtypes(df)
        df = self._sanitize_column_names(df)
        return df

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing target values and select relevant variables.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be trimmed.

        Returns
        -------
        df : pd.DataFrame
            Trimmed DataFrame with only relevant features and complete targets.
        """
        df = df.dropna(subset=self.target.names)
        
        cols_def = [f.name for f in self.features] + self.target.names        
        cols_not_def = df.columns.difference(cols_def)
        
        if len(cols_not_def) > 0:
            logger.warning(f"The following columns were not defined in the schema and will be ignored: {list(cols_not_def)}")

        cols_to_keep = [f.name for f in self.features if f.type != 'ignore'] + self.target.names

        return df[cols_to_keep]

    def _set_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set column data types based on schema:
        - Numerical columns are cast to float.
        - Ignored columns are skipped (but already dropped in `_trim`).

        Target columns are also cast to float.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame whose columns will be cast.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with updated column data types.
        """
        for feature in self.features:
            col = feature.name

            match feature.type:
                case "numerical":
                    df[col] = pd.to_numeric(df[col])
                case "ignore":
                    pass
        for col in self.target.names:
            df[col] = pd.to_numeric(df[col])

        return df

    def _sanitize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove special characters from column names to avoid incompatibility
        with some ML frameworks (e.g., LightGBM JSON parsing errors).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with potentially unsafe column names.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with sanitized column names.
        """
        sanitized_columns = [
            re.sub(r'[\"\'{}]', '', col) for col in df.columns
        ]
        df.columns = sanitized_columns
        return df