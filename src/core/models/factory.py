from .definitions import * 
from enum import IntEnum, auto

class RegressionModels(IntEnum):
    """
    Enum of supported model types for regression experiments.
    Each enum member corresponds to a distinct regression algorithm.
    """

    LINEAR_REGRESSION = auto()
    RIDGE_REGRESSION  = auto()
    LASSO_REGRESSION  = auto()
    ELASTIC_NET       = auto()
    PLS               = auto()
    K_NEIGHBORS       = auto()
    DECISION_TREE     = auto()
    RANDOM_FOREST     = auto()
    EXTRA_TREES       = auto()
    ADABOOST          = auto()
    XGBOOST           = auto()
    LIGHTGBM          = auto()
    CATBOOST          = auto()
    SVR               = auto()
    MLP               = auto()


class ModelFactory:
    """
    Factory class for creating model instances based on the RegressionModels enum.
    Provides a single static method create_model() that returns a subclass of AbstractModel.
    """
    
    @staticmethod
    def create_model(model: RegressionModels) -> AbstractModel:
        """
        Create and return an instance of a model corresponding to the given enum member.
        
        Parameters
        ----------
        model : RegressionModels
            The model enum member for which to create an instance.
        seed : int
            Random seed for model initialization.
        
        Returns
        -------
        AbstractModel
            An instance of the specified model.
        """
        match model:
            case RegressionModels.LINEAR_REGRESSION:
                return LinearRegressionModel()
            case RegressionModels.RIDGE_REGRESSION:
                return RidgeRegressionModel()
            case RegressionModels.LASSO_REGRESSION:
                return LassoRegressionModel()
            case RegressionModels.ELASTIC_NET:
                return ElasticNetModel()
            case RegressionModels.PLS:
                return PLSRegressorModel()
            case RegressionModels.K_NEIGHBORS:
                return KNeighborsModel()
            case RegressionModels.DECISION_TREE:
                return DecisionTreeModel()
            case RegressionModels.RANDOM_FOREST:
                return RandomForestModel()
            case RegressionModels.EXTRA_TREES:
                return ExtraTreesModel()
            case RegressionModels.ADABOOST:
                return AdaBoostModel()
            case RegressionModels.XGBOOST:
                return XGBoostModel()
            case RegressionModels.LIGHTGBM:
                return LightGBMModel()
            case RegressionModels.CATBOOST:
                return CatBoostModel()
            case RegressionModels.SVR:
                return SVRModel()
            case RegressionModels.MLP:
                return MLPModel()