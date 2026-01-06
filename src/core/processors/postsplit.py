import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.core.data.schema import Schema


class PostSplitProcessor:
    """
    Applies post-split preprocessing transformations to numerical features and targets,
    including robust scaling.

    Parameters
    ----------
    schema : Schema
        Schema object defining features and targets.

    Attributes
    ----------
    schema : Schema
        Schema object for features and targets.
    feature_numerical_cols : list[str]
        List of numerical feature column names.
    target_numerical_cols : list[str]
        List of target column names.
    feature_scaler : RobustScaler
        Scaler for numerical features using robust scaling.
    target_scaler : RobustScaler
        Scaler for target variables.
    """

    def __init__(self, schema: Schema) -> None:

        self.feature_numerical_cols: list[str] = schema.valid_feature_names
        self.target_numerical_cols : list[str] = schema.target_names

        self.feature_scaler: RobustScaler = RobustScaler()
        self.target_scaler : RobustScaler = RobustScaler()

    def fit_transform(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the scalers on training data and apply transformations.

        Parameters
        ----------
        x_train : pd.DataFrame
            Training feature set.
        y_train : pd.DataFrame
            Training target values.

        Returns
        -------
        x_train_transformed : pd.DataFrame
            Transformed training feature set.
        y_train_transformed : pd.DataFrame
            Transformed training target values.
        """
        x_train = x_train.copy()
        y_train = y_train.copy()

        feature_cols_to_scale = [
            col for col in self.feature_numerical_cols if col in x_train.columns
        ]
        target_cols_to_scale = [
            col for col in self.target_numerical_cols if col in y_train.columns
        ]

        if feature_cols_to_scale:
            x_train[feature_cols_to_scale] = x_train[feature_cols_to_scale].astype('float64')
            x_train.loc[:, feature_cols_to_scale] = self.feature_scaler.fit_transform(
                x_train[feature_cols_to_scale]
            )

        if target_cols_to_scale:
            y_train[target_cols_to_scale] = y_train[target_cols_to_scale].astype('float64')
            y_train.loc[:, target_cols_to_scale] = self.target_scaler.fit_transform(
                y_train[target_cols_to_scale]
            )

        return x_train, y_train

    def transform(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to test data using fitted parameters.

        Parameters
        ----------
        x_test : pd.DataFrame
            Test feature set.

        Returns
        -------
        x_test_transformed : pd.DataFrame
            Transformed test feature set.
        """
        if not hasattr(self.feature_scaler, "scale_"):
            raise ValueError("Scaler has not been fitted. Call fit_transform first.")

        x_test = x_test.copy()

        feature_cols_to_scale = [
            col for col in self.feature_numerical_cols if col in x_test.columns
        ]
        if feature_cols_to_scale:
            x_test[feature_cols_to_scale] = x_test[feature_cols_to_scale].astype('float64')
            x_test.loc[:, feature_cols_to_scale] = self.feature_scaler.transform(
                x_test[feature_cols_to_scale]
            )

        return x_test

    def inverse_transform_target(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Invert the scaling transformation on predicted target values.

        Parameters
        ----------
        y_pred : pd.DataFrame
            Scaled predictions to be transformed back to original scale.

        Returns
        -------
        y_pred_original : pd.DataFrame
            Predictions in the original target scale.
        """
        if not hasattr(self.target_scaler, "scale_"):
            raise ValueError("Target scaler has not been fitted. Call fit_transform first.")

        y_pred = y_pred.copy()

        target_cols_to_scale = [
            col for col in self.target_numerical_cols if col in y_pred.columns
        ]

        if target_cols_to_scale:
            y_pred.loc[:, target_cols_to_scale] = self.target_scaler.inverse_transform(
                y_pred[target_cols_to_scale]
            )

        return y_pred
