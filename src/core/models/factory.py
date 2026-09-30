from enum import IntEnum, auto

from .definitions import (
    AbstractModel,
    ElasticNetModel,
    ExtraTreesModel,
    GradientBoostingModel,
    RandomForestModel,
    RidgeRegressionModel,
    SVRModel,
    XGBoostModel,
)


# =============================================================================
# REGRESSION MODELS
# =============================================================================


class RegressionModels(IntEnum):
    """
    Regression model families evaluated in the HFI prediction experiment.
    """

    RIDGE_REGRESSION = auto()
    ELASTIC_NET = auto()
    SVR = auto()
    RANDOM_FOREST = auto()
    EXTRA_TREES = auto()
    GRADIENT_BOOSTING = auto()
    XGBOOST = auto()


# =============================================================================
# MODEL FACTORY
# =============================================================================


class ModelFactory:
    """
    Factory responsible for creating regression model instances.
    """

    @staticmethod
    def create_model(
        model: RegressionModels,
        params: dict | None = None,
    ) -> AbstractModel:
        """
        Create a regression model using the supplied hyperparameters.
        """

        params = params or {}

        match model:

            case RegressionModels.RIDGE_REGRESSION:
                return RidgeRegressionModel(
                    **params
                )

            case RegressionModels.ELASTIC_NET:
                return ElasticNetModel(
                    **params
                )

            case RegressionModels.SVR:
                return SVRModel(
                    **params
                )

            case RegressionModels.RANDOM_FOREST:
                return RandomForestModel(
                    **params
                )

            case RegressionModels.EXTRA_TREES:
                return ExtraTreesModel(
                    **params
                )

            case RegressionModels.GRADIENT_BOOSTING:
                return GradientBoostingModel(
                    **params
                )

            case RegressionModels.XGBOOST:
                return XGBoostModel(
                    **params
                )

            case _:
                raise ValueError(
                    f"Unsupported model: {model}"
                )
