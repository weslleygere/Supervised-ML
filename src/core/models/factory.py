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
    def create_model(model: RegressionModels, seed: int) -> AbstractModel:
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
                return LinearRegressionModel(seed)
            case RegressionModels.RIDGE_REGRESSION:
                return RidgeRegressionModel(seed)
            case RegressionModels.LASSO_REGRESSION:
                return LassoRegressionModel(seed)
            case RegressionModels.ELASTIC_NET:
                return ElasticNetModel(seed)
            case RegressionModels.PLS:
                return PLSRegressorModel(seed)
            case RegressionModels.K_NEIGHBORS:
                return KNeighborsModel(seed)
            case RegressionModels.DECISION_TREE:
                return DecisionTreeModel(seed)
            case RegressionModels.RANDOM_FOREST:
                return RandomForestModel(seed)
            case RegressionModels.EXTRA_TREES:
                return ExtraTreesModel(seed)
            case RegressionModels.ADABOOST:
                return AdaBoostModel(seed)
            case RegressionModels.XGBOOST:
                return XGBoostModel(seed)
            case RegressionModels.LIGHTGBM:
                return LightGBMModel(seed)
            case RegressionModels.CATBOOST:
                return CatBoostModel(seed)
            case RegressionModels.SVR:
                return SVRModel(seed)
            case RegressionModels.MLP:
                return MLPModel(seed)