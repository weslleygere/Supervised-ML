from enum import IntEnum, auto

from .definitions import (
    AbstractModel,
    ElasticNetModel,
    ExtraTreesModel,
    RandomForestModel,
    RidgeRegressionModel,
    SVRModel,
    XGBoostModel,
)


class RegressionModels(IntEnum):
    """
    Regression models used in the first instance-MIR experiment.
    """

    RIDGE_REGRESSION = auto()
    ELASTIC_NET = auto()
    SVR = auto()
    RANDOM_FOREST = auto()
    EXTRA_TREES = auto()
    XGBOOST = auto()


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

        Parameters
        ----------
        model : RegressionModels
            Model to instantiate.

        params : dict | None
            Hyperparameters selected for the model. If None, the model
            is created using its default parameters.

        Returns
        -------
        AbstractModel
            Instantiated regression model.
        """
        params = params or {}

        match model:
            case RegressionModels.RIDGE_REGRESSION:
                return RidgeRegressionModel(**params)

            case RegressionModels.ELASTIC_NET:
                return ElasticNetModel(**params)

            case RegressionModels.SVR:
                return SVRModel(**params)

            case RegressionModels.RANDOM_FOREST:
                return RandomForestModel(**params)

            case RegressionModels.EXTRA_TREES:
                return ExtraTreesModel(**params)

            case RegressionModels.XGBOOST:
                return XGBoostModel(**params)

            case _:
                raise ValueError(f"Unsupported model: {model}")
