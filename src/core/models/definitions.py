import time
import warnings

from typing import Optional

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
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

        params.setdefault(
            "solver",
            "svd",
        )

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
            self._target_values(
                y
            ),
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

    Non-converged fits are rejected so they cannot receive an Optuna
    validation score.
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

        try:

            with warnings.catch_warnings():

                warnings.filterwarnings(
                    "error",
                    category=ConvergenceWarning,
                )

                self.model.fit(
                    x,
                    self._target_values(
                        y
                    ),
                    sample_weight=sample_weight,
                )

        except ConvergenceWarning as exc:

            raise ModelConvergenceError(
                "Elastic Net reached the iteration limit "
                "before convergence."
            ) from exc

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


class BayesianRidgeModel(
    AbstractModel
):
    """
    Bayesian Ridge Regression.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        self.model = BayesianRidge(
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
            self._target_values(
                y
            ),
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
# KERNEL MODELS
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
            self._target_values(
                y
            ),
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


class KernelRidgeModel(
    AbstractModel
):
    """
    Kernel Ridge Regression.
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

        self.model = KernelRidge(
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
            self._target_values(
                y
            ),
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
# BAGGED TREE ENSEMBLES
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

        self.model = RandomForestRegressor(
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
            self._target_values(
                y
            ),
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

        self.model = ExtraTreesRegressor(
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
            self._target_values(
                y
            ),
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
# BOOSTED TREE ENSEMBLES
# =============================================================================


class GradientBoostingModel(
    AbstractModel
):
    """
    Gradient Boosting Regressor.
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

        self.model = GradientBoostingRegressor(
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
            self._target_values(
                y
            ),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


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
            self._target_values(
                y
            ),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


class LightGBMModel(
    AbstractModel
):
    """
    LightGBM Regressor.
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

        params.setdefault(
            "verbosity",
            -1,
        )

        self.model = LGBMRegressor(
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
            self._target_values(
                y
            ),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )


class CatBoostModel(
    AbstractModel
):
    """
    CatBoost Regressor.
    """

    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()

        params.setdefault(
            "random_seed",
            AbstractModel.seed,
        )

        params.setdefault(
            "thread_count",
            -1,
        )

        params.setdefault(
            "verbose",
            False,
        )

        params.setdefault(
            "allow_writing_files",
            False,
        )

        self.model = CatBoostRegressor(
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
            self._target_values(
                y
            ),
            sample_weight=sample_weight,
        )

    def _predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )