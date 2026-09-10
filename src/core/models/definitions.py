import time
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Ridge,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ModelConvergenceError(RuntimeError):
    """
    Raised when a regression model does not converge.
    """


# =============================================================================
# BASE MODEL
# =============================================================================


class AbstractModel:
    """
    Base class for the regression models used in the HFI experiment.
    """

    seed: Optional[int] = None

    def __init__(self) -> None:
        self.target_columns: list[str] = []

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """
        Fit the model and return training time.
        """

        self.target_columns = list(
            y.columns
        )

        start = time.perf_counter()

        self._fit(
            x,
            y,
            sample_weight,
        )

        return (
            time.perf_counter()
            - start
        )

    def predict(
        self,
        x: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float]:
        """
        Generate predictions and return prediction time.
        """

        start = time.perf_counter()

        predictions = self._predict(
            x
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        predictions = (
            np.asarray(
                predictions
            )
            .reshape(-1, 1)
        )

        return (
            pd.DataFrame(
                predictions,
                columns=self.target_columns,
                index=x.index,
            ),
            elapsed,
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _target_values(
        y: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert the single-target dataframe to a 1D array.
        """

        return (
            y.iloc[:, 0]
            .to_numpy()
        )

    @property
    def name(self) -> str:
        return (
            self.__class__.__name__
            .replace("Model", "")
            .replace("_", " ")
            .title()
        )


# =============================================================================
# REGULARIZED LINEAR MODELS
# =============================================================================


class RidgeRegressionModel(
    AbstractModel
):
    """
    Ridge Regression.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        self.model = Ridge(
            **params
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:

        self.model.fit(
            x,
            self._target_values(y),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


class ElasticNetModel(
    AbstractModel
):
    """
    Elastic Net Regression.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        params.setdefault(
            "random_state",
            AbstractModel.seed,
        )

        self.model = ElasticNet(
            **params
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:

        self.model.fit(
            x,
            self._target_values(y),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


# =============================================================================
# KERNEL MODEL
# =============================================================================


class SVRModel(
    AbstractModel
):
    """
    Support Vector Regression with RBF kernel.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        params.setdefault(
            "kernel",
            "rbf",
        )

        self.model = SVR(
            **params
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:

        self.model.fit(
            x,
            self._target_values(y),
            sample_weight=sample_weight,
        )

        if self.model.fit_status_ != 0:
            raise ModelConvergenceError(
                "SVR reached the iteration limit "
                "before convergence."
            )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


# =============================================================================
# TREE ENSEMBLES
# =============================================================================


class RandomForestModel(
    AbstractModel
):
    """
    Random Forest Regressor.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        params.setdefault(
            "random_state",
            AbstractModel.seed,
        )

        params.setdefault(
            "n_jobs",
            -1,
        )

        self.model = (
            RandomForestRegressor(
                **params
            )
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:

        self.model.fit(
            x,
            self._target_values(y),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


class ExtraTreesModel(
    AbstractModel
):
    """
    Extra Trees Regressor.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        params.setdefault(
            "random_state",
            AbstractModel.seed,
        )

        params.setdefault(
            "n_jobs",
            -1,
        )

        self.model = (
            ExtraTreesRegressor(
                **params
            )
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:

        self.model.fit(
            x,
            self._target_values(y),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


# =============================================================================
# GRADIENT BOOSTING
# =============================================================================


class XGBoostModel(
    AbstractModel
):
    """
    XGBoost Regressor.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        params.setdefault(
            "random_state",
            AbstractModel.seed,
        )

        params.setdefault(
            "n_jobs",
            -1,
        )

        self.model = XGBRegressor(
            **params
        )

    def _fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:

        self.model.fit(
            x,
            self._target_values(y),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )
