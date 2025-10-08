import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from src.core.data.schema import Schema, FeatureSchema, TargetSchema

class PostSplitProcessor:
    """
    Applies post-split preprocessing transformations to numerical features and targets,
    including median imputation and robust scaling.

    Parameters
    ----------
    schema : Schema
        Schema object defining features and targets.

    Attributes
    ----------
    target : TargetSchema
        Target variable schema.
    features : list[FeatureSchema]
        List of feature definitions from the schema.
    feature_numerical_cols : list[str]
        List of numerical feature column names (excluding target columns).
    target_numerical_cols : list[str]
        List of target column names.
    feature_imputer : SimpleImputer
        Imputer for numerical features using median strategy.
    feature_scaler : RobustScaler
        Scaler for numerical features using robust scaling.
    target_scaler : RobustScaler
        Scaler for target variables.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema   : Schema              = schema
        self.target   : TargetSchema        = schema.target
        self.features : list[FeatureSchema] = schema.features

        self.feature_numerical_cols : list[str] = [f.name for f in self.features if f.type == "numerical"]
        self.target_numerical_cols  : list[str] = self.target.names

        self.feature_imputer : SimpleImputer = SimpleImputer(strategy="median")
        self.feature_scaler  : RobustScaler  = RobustScaler()
        self.target_scaler   : RobustScaler  = RobustScaler()

    def fit_transform(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the imputers and scalers on training data and apply transformations.

        Parameters
        ----------
        x_train : pd.DataFrame
            Training feature set.
        y_train : pd.DataFrame or None, optional
            Training target values. If provided, target scaling is also applied.

        Returns
        -------
        x_train_transformed : pd.DataFrame
            Transformed training feature set.

        y_train_transformed : pd.DataFrame
            Transformed training target values.
        """
        x_train[self.feature_numerical_cols] = self.feature_imputer.fit_transform(x_train[self.feature_numerical_cols])
        x_train[self.feature_numerical_cols] = self.feature_scaler.fit_transform(x_train[self.feature_numerical_cols])

        y_train[self.target_numerical_cols] = self.target_scaler.fit_transform(y_train[self.target_numerical_cols])

        return x_train, y_train

    def transform(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation and scaling to test data using fitted parameters.

        Parameters
        ----------
        x_test : pd.DataFrame
            Test feature set.

        Returns
        -------
        x_test_transformed : pd.DataFrame
            Transformed test feature set.
        """
        if not hasattr(self.feature_imputer, "statistics_"):
            raise ValueError("Imputer has not been fitted. Call fit_transform first.")
        if not hasattr(self.feature_scaler, "scale_"):
            raise ValueError("Scaler has not been fitted. Call fit_transform first.")

        x_test[self.feature_numerical_cols] = self.feature_imputer.transform(x_test[self.feature_numerical_cols])
        x_test[self.feature_numerical_cols] = self.feature_scaler.transform(x_test[self.feature_numerical_cols])

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
        if self.target_scaler is None:
            raise ValueError("Target scaler has not been fitted. Call fit_transform first.")
        y_pred = pd.DataFrame(
            self.target_scaler.inverse_transform(y_pred),
            columns=self.target_numerical_cols,
            index=y_pred.index
        )
        return y_pred