import pandas as pd

from src.core.data.schema import Schema


class PreSplitProcessor:
    """
    Select the columns required by the instance-MIR experiments.

    The input dataframe is assumed to be already prepared and validated.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Retain target, hierarchy metadata, acoustic indices and embeddings.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the columns required for modeling.
        """
        index_columns = self.schema.index_columns(df_raw.columns)

        columns = [
            self.schema.group,
            self.schema.bag,
            *self.schema.instance_columns,
            self.schema.target,
            *index_columns,
            self.schema.embedding,
        ]

        return df_raw.loc[:, columns].copy()
